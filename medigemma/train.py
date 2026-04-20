"""
train.py — Per-dataset-leaf LoRA adapter fine-tuning for MedGemma
=================================================================

Training design rationale
-------------------------
This code trains one LoRA adapter per dataset leaf, not one adapter per bare
language code. In your dataset, `eng_uga`, `eng_gha`, and `eng_ken` are distinct
training targets because the country-specific leaf is part of the real corpus
identity and may encode different phrasing, examples, and SRH guidance style.

Why separate adapters per leaf?
* It prevents gradient interference between different leaves.
* It keeps evaluation and deployment aligned with the dataset you actually have.
* It makes adapter outputs traceable back to a concrete corpus leaf.

Why a shared dataset loader?
* The original training path assumed `load_from_disk(data/<lang_code>)`.
* Your real data may come from a Hub repo, a local shard tree, or a mirrored
  `save_to_disk()` cache.
* `MultilingualDatasetBuilder` hides those source differences so the training
  loop stays focused on formatting and optimization.
"""

import argparse
import inspect
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import DatasetDict
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer

from config import (
    LanguageConfig,
    SUPPORTED_LANGUAGES,
    TrainingConfig,
    expand_language_selection,
)
from data_utils import MultilingualDatasetBuilder, get_split
from evaluation import MultilingualEvaluator, resolve_adapter_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_base_model(cfg: TrainingConfig):
    """
    Load the quantized base model and processor for QLoRA training.

    The base model is loaded in 4-bit mode so the adapter training path remains
    practical on constrained hardware.
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    logger.info("Loading base model: %s", cfg.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    processor = AutoProcessor.from_pretrained(cfg.model_id)
    processor.tokenizer.padding_side = "right"
    return model, processor


def build_lora_config(lang_cfg: LanguageConfig) -> LoraConfig:
    """
    Build a leaf-specific LoRA configuration.

    Lower-resource leaves use lower-rank settings via the config registry to
    reduce overfitting risk.
    """

    rank = lang_cfg.lora_r
    return LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=["lm_head", "embed_tokens"],
    )


def build_sft_config(
    cfg: TrainingConfig,
    lang_cfg: LanguageConfig,
    adapter_output_dir: Path,
    has_eval: bool,
    run_name: Optional[str],
) -> SFTConfig:
    """
    Handle TRL argument-name differences across releases.

    `trl` has used both `evaluation_strategy` and `eval_strategy`, and likewise
    both `max_seq_length` and `max_length` depending on version. This adapter
    keeps the project tolerant to that variation.
    """

    use_bf16 = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)

    if use_bf16:
        logger.info("Using BF16 mixed precision for training.")
    elif use_fp16:
        logger.info("BF16 is unavailable on this GPU; falling back to FP16 training.")
    else:
        logger.warning(
            "No CUDA device detected. Training MedGemma on CPU is likely impractical."
        )

    kwargs = {
        "output_dir": str(adapter_output_dir),
        "num_train_epochs": lang_cfg.num_epochs,
        "per_device_train_batch_size": lang_cfg.batch_size,
        "per_device_eval_batch_size": lang_cfg.batch_size,
        "gradient_accumulation_steps": lang_cfg.grad_accumulation,
        "learning_rate": lang_cfg.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "logging_steps": 10,
        "eval_steps": 50 if has_eval else None,
        "save_strategy": "steps",
        # Checkpoint writes were failing on the VM when optimizer state was
        # serialized every 100 steps. Save less often and avoid best-model
        # reloading so PEFT's nested checkpoint layout does not block training.
        "save_steps": 500,
        "save_total_limit": 2,
        "load_best_model_at_end": False,
        "report_to": "none",
        "gradient_checkpointing": True,
        "dataset_text_field": "text",
        "packing": False,
        "remove_unused_columns": False,
    }

    params = inspect.signature(SFTConfig).parameters
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps" if has_eval else "no"
    else:
        kwargs["evaluation_strategy"] = "steps" if has_eval else "no"

    if "dataset_kwargs" in params:
        # Gemma 3 requires `token_type_ids` during training. We pretokenize the
        # dataset ourselves so TRL does not drop that field in its generic text
        # preprocessing path.
        kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

    if "save_only_model" in params:
        # Optimizer checkpoints are by far the largest files in this run and
        # were the direct source of serialization failures on disk.
        kwargs["save_only_model"] = True

    if "max_length" in params:
        kwargs["max_length"] = cfg.max_seq_length
    else:
        kwargs["max_seq_length"] = cfg.max_seq_length

    if "run_name" in params and run_name:
        kwargs["run_name"] = run_name

    return SFTConfig(**kwargs)


def train_language_adapter(
    language: str,
    cfg: TrainingConfig,
    lang_cfg: LanguageConfig,
    dataset: DatasetDict,
    output_root: Path,
) -> str:
    """
    Fine-tune one LoRA adapter for a dataset leaf like `eng_uga`.

    The adapter directory name intentionally mirrors the dataset leaf name so
    downstream inference and evaluation remain unambiguous.
    """

    logger.info("═══ Starting adapter training for dataset: %s ═══", language)

    adapter_output_dir = output_root / f"adapter_{language}"
    adapter_output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"train_{language}"

    base_model, processor = load_base_model(cfg)
    model = get_peft_model(base_model, build_lora_config(lang_cfg), adapter_name=language)
    model.print_trainable_parameters()

    train_ds = get_split(dataset, "train")
    eval_ds = get_split(dataset, "dev")

    if train_ds is None or len(train_ds) == 0:
        logger.warning("Empty training set for %s; skipping.", language)
        return str(adapter_output_dir)

    # Gemma-style supervised fine-tuning expects conversational turns. We map
    # the dataset's `input` / `output` columns into a simple SRH chat exchange.
    def format_sample(example):
        messages = [
            {
                "role": "system",
                "content": (
                    """ 
                    You are Hashie, a muliti-lingual medical assistant with expertise in sexual and reproductive health (SRH).
                    You are knowledgeable, supportive, and approachable, capable of communicating with empathy and clarity.
                    You can explain sexually transmitted infections (STIs) and related health topics in simple, everyday language suitable for young adults and the general public.
                    When interacting with medical professionals, you can also provide detailed, evidence-based explanations and use precise clinical terminology when appropriate.
                    Your goal is to ensure that all users — regardless of their medical background — receive accurate, respectful, and easy-to-understand information about sexual and reproductive health.
                    """
                    f"Answer in {lang_cfg.language_name}."
                ),
            },
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_text = train_ds.map(format_sample, remove_columns=train_ds.column_names)
    eval_text = None
    if eval_ds is not None and len(eval_ds) > 0:
        eval_text = eval_ds.map(format_sample, remove_columns=eval_ds.column_names)

    def tokenize_batch(batch):
        eos_token = processor.tokenizer.eos_token or ""
        texts = []
        for text in batch["text"]:
            if eos_token and not text.endswith(eos_token):
                text = f"{text}{eos_token}"
            texts.append(text)

        tokenized = processor.tokenizer(
            texts,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
            return_attention_mask=True,
        )

        if "token_type_ids" not in tokenized:
            tokenized["token_type_ids"] = [
                [0] * len(input_ids) for input_ids in tokenized["input_ids"]
            ]

        return tokenized

    logger.info("[%s] Tokenizing dataset with explicit token_type_ids for Gemma 3.", language)
    train_tokens = train_text.map(tokenize_batch, batched=True, remove_columns=train_text.column_names)
    eval_tokens = None
    if eval_text is not None:
        eval_tokens = eval_text.map(tokenize_batch, batched=True, remove_columns=eval_text.column_names)

    sft_cfg = build_sft_config(
        cfg=cfg,
        lang_cfg=lang_cfg,
        adapter_output_dir=adapter_output_dir,
        has_eval=eval_text is not None,
        run_name=run_name,
    )

    callbacks = []
    use_early_stopping = bool(
        eval_text is not None
        and lang_cfg.early_stopping_patience
        and getattr(sft_cfg, "load_best_model_at_end", False)
    )
    if use_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=lang_cfg.early_stopping_patience,
            )
        )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=processor.tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_tokens,
        eval_dataset=eval_tokens,
        processing_class=processor.tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    logger.info("Training %s adapter with %s train samples", language, len(train_tokens))
    trainer.train()

    model.save_pretrained(str(adapter_output_dir))
    processor.save_pretrained(str(adapter_output_dir))

    metadata = {
        "dataset_id": language,
        "language_code": lang_cfg.language_code,
        "language_name": lang_cfg.language_name,
        "country_code": lang_cfg.country_code,
        "country_name": lang_cfg.country_name,
        "display_name": lang_cfg.display_name,
        "lora_r": lang_cfg.lora_r,
        "num_epochs": lang_cfg.num_epochs,
        "learning_rate": lang_cfg.learning_rate,
        "train_samples": len(train_tokens),
        "dev_samples": len(eval_tokens) if eval_tokens is not None else 0,
        "base_model": cfg.model_id,
    }
    with open(adapter_output_dir / "adapter_meta.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    logger.info("Adapter saved to %s", adapter_output_dir)

    del trainer
    del model
    del base_model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(adapter_output_dir)


def load_adapter_for_inference(
    language: str,
    cfg: TrainingConfig,
    output_root: Path,
):
    """
    Load the base model plus a trained adapter for inference.

    Inference follows the same leaf naming convention used during training.
    """

    base_model, processor = load_base_model(cfg)
    adapter_path = resolve_adapter_path(output_root, language)

    if adapter_path is None:
        raise FileNotFoundError(
            f"No adapter found for dataset '{language}' under {output_root}. Run training first."
        )

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    logger.info("Loaded adapter for '%s' from %s", language, adapter_path)
    return model, processor


def parse_args():
    """Define CLI arguments for training and evaluation against Hub or local data."""
    parser = argparse.ArgumentParser(
        description="Train per-dataset-leaf LoRA adapters on MedGemma"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(SUPPORTED_LANGUAGES.keys()),
        help=(
            "Dataset leaves or base language groups to train. Examples: "
            "`eng_uga`, `swa_ken`, or grouped selections like `eng` and `swa`."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Local mirror root for save_to_disk data or shard trees.",
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        default=None,
        help="Hugging Face dataset repo id containing the multilingual SRH shards.",
    )
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help="Optional dataset revision, branch, or commit for the Hub repo.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for dataset downloads.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./adapters",
        help="Root directory for saving trained adapters.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=200,
        help="Maximum number of test examples per dataset leaf during evaluation.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training and run evaluation on saved adapters only.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face API token. Falls back to HF_TOKEN from the environment.",
    )
    return parser.parse_args()


def main():
    """
    Entry point for training and/or evaluation.

    The CLI accepts either grouped language selections such as `eng` or explicit
    dataset leaves such as `eng_uga`. Grouped selections are expanded before any
    data loading begins.
    """
    args = parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login

        login(hf_token)
    else:
        logger.warning(
            "No HF token provided. This is fine for public datasets, but MedGemma may still "
            "require authentication depending on your model access."
        )

    try:
        selected_languages = expand_language_selection(args.languages)
    except KeyError as exc:
        raise SystemExit(
            f"Unsupported language or dataset selection: {exc.args[0]}. "
            f"Supported values: {', '.join(sorted(set(SUPPORTED_LANGUAGES) | set(['aka', 'amh', 'eng', 'lug', 'swa'])))}"
        ) from exc

    cfg = TrainingConfig()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    builder = MultilingualDatasetBuilder(
        data_root=args.data_root,
        dataset_repo=args.dataset_repo,
        dataset_revision=args.dataset_revision,
        hf_token=hf_token,
        cache_dir=args.cache_dir,
        seed=cfg.seed,
    )

    if not args.eval_only:
        for lang_code in selected_languages:
            lang_cfg = SUPPORTED_LANGUAGES[lang_code]
            try:
                dataset = builder.load_language(lang_code, lang_cfg, augment=True)
            except Exception as exc:
                logger.warning("Could not load dataset for %s: %s", lang_code, exc)
                continue

            train_language_adapter(
                language=lang_code,
                cfg=cfg,
                lang_cfg=lang_cfg,
                dataset=dataset,
                output_root=output_root,
            )

    evaluator = MultilingualEvaluator(cfg, output_root, dataset_builder=builder)
    results = evaluator.evaluate_all(
        selected_languages,
        max_eval_samples=args.max_eval_samples,
    )
    evaluator.save_report(results, output_root / "eval_report.json")
    logger.info("All done.")


if __name__ == "__main__":
    main()
