"""
train.py — Per-dataset-leaf LoRA adapter fine-tuning for epfl-llm/meditron-7b
==============================================================================

Model: epfl-llm/meditron-7b (LLaMA-based, text-only causal LM)
Hardware target: Single GPU (A100 / V100)

Design notes
------------
meditron-7b uses a standard LlamaConfig. It is NOT a vision-language model,
so we use:
  - AutoModelForCausalLM   (not AutoModelForImageTextToText)
  - AutoTokenizer          (not AutoProcessor — there is no processor wrapper)

This removes the entire processor.tokenizer nesting issue: the tokenizer IS
the processor, so we work with `tokenizer` directly throughout.

Training design
---------------
One LoRA adapter is trained per dataset leaf (e.g. eng_eth, eng_uga). This
prevents gradient interference between geographically/culturally distinct
corpora and keeps adapter outputs traceable to a concrete dataset.
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
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

from config import (
    LanguageConfig,
    SUPPORTED_LANGUAGES,
    TrainingConfig,
    expand_language_selection,
)
from data_utils import MultilingualDatasetBuilder, get_split
from evaluation import MultilingualEvaluator, download_adapters, resolve_adapter_path
from prompt_utils import render_meditron_chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model + tokenizer loading
# ---------------------------------------------------------------------------

def load_base_model(cfg: TrainingConfig):
    """
    Load meditron-7b in 4-bit (QLoRA) mode with its tokenizer.

    Returns
    -------
    model : AutoModelForCausalLM
        Quantized base model ready for PEFT wrapping.
    tokenizer : AutoTokenizer
        Tokenizer for meditron-7b (LlamaTokenizerFast).
    """
    logger.info("Loading base model: %s", cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    logger.info("Loading tokenizer: %s", cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_id,
        trust_remote_code=True,
    )

    # LLaMA tokenizers ship without a pad token; use EOS as the pad token.
    # padding_side="right" is required for causal LM training to avoid
    # attention mask misalignment.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("pad_token set to eos_token (%s)", tokenizer.eos_token)

    tokenizer.padding_side = "right"

    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA configuration
# ---------------------------------------------------------------------------

def build_lora_config(lang_cfg: LanguageConfig) -> LoraConfig:
    """
    Build a leaf-specific LoRA configuration.

    Lower-resource leaves use lower rank via the config registry to reduce
    overfitting.  meditron-7b uses standard LLaMA attention and FFN layer
    names, so target_modules targets all linear projections.
    """
    rank = lang_cfg.lora_r
    return LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.0,
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
        # embed_tokens and lm_head are saved so generation is self-contained.
        modules_to_save=["lm_head", "embed_tokens"],
    )


# ---------------------------------------------------------------------------
# SFTConfig builder (handles TRL API variation across releases)
# ---------------------------------------------------------------------------

def build_sft_config(
    cfg: TrainingConfig,
    lang_cfg: LanguageConfig,
    adapter_output_dir: Path,
    has_eval: bool,
    run_name: Optional[str],
) -> SFTConfig:
    """
    Build an SFTConfig that is tolerant to TRL argument-name drift.

    TRL has renamed several arguments across minor releases. This function
    inspects the current SFTConfig signature and uses whichever name is
    present.
    """
    use_bf16 = bool(
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = bool(torch.cuda.is_available() and not use_bf16)

    if use_bf16:
        logger.info("Using BF16 mixed precision.")
    elif use_fp16:
        logger.info("BF16 unavailable; falling back to FP16.")
    else:
        logger.warning("No CUDA device detected. Training on CPU is impractical for a 7B model.")

    kwargs = {
        "output_dir": str(adapter_output_dir),
        "num_train_epochs": lang_cfg.num_epochs,
        "per_device_train_batch_size": lang_cfg.batch_size,
        "per_device_eval_batch_size": lang_cfg.batch_size,
        "gradient_accumulation_steps": lang_cfg.grad_accumulation,
        "learning_rate": lang_cfg.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 10,
        "weight_decay": 0.01,
        "optim": "adamw_8bit",
        "fp16": use_fp16,
        "bf16": use_bf16,
        "logging_steps": 10,
        "eval_steps": 50 if has_eval else None,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 2,
        "load_best_model_at_end": False,
        "report_to": "none",
        "gradient_checkpointing": True,
        # We pre-tokenize the dataset ourselves, so tell TRL not to reprocess.
        "dataset_text_field": "text",
        "packing": False,
        "remove_unused_columns": False,
    }

    params = inspect.signature(SFTConfig).parameters

    # eval_strategy vs evaluation_strategy
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps" if has_eval else "no"
    else:
        kwargs["evaluation_strategy"] = "steps" if has_eval else "no"

    # skip_prepare_dataset — avoids TRL re-tokenising our pre-tokenised data
    if "dataset_kwargs" in params:
        kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

    # Avoid writing optimizer state to disk (major space/time savings on VM)
    if "save_only_model" in params:
        kwargs["save_only_model"] = True

    # max_length vs max_seq_length
    if "max_length" in params:
        kwargs["max_length"] = cfg.max_seq_length
    else:
        kwargs["max_seq_length"] = cfg.max_seq_length

    if "run_name" in params and run_name:
        kwargs["run_name"] = run_name

    return SFTConfig(**kwargs)


# ---------------------------------------------------------------------------
# Chat template formatting
# ---------------------------------------------------------------------------

def make_format_fn(_lang_cfg: LanguageConfig, _tokenizer: AutoTokenizer):
    """
    Return a dataset map function that converts raw input/output pairs into
    a single formatted text string using the model's chat template.

    The Meditron data curation flow used a LLaMA-style chat template with the
    Hashie system prompt, so we render that prompt family directly here to keep
    training aligned with the prepared fine-tuning data.
    """

    def format_sample(example):
        text = render_meditron_chat(
            user_text=example["input"],
            assistant_text=example["output"],
        )
        return {"text": text}

    return format_sample


# ---------------------------------------------------------------------------
# Per-language adapter training
# ---------------------------------------------------------------------------

def train_language_adapter(
    language: str,
    cfg: TrainingConfig,
    lang_cfg: LanguageConfig,
    dataset: DatasetDict,
    output_root: Path,
) -> str:
    """
    Fine-tune one LoRA adapter for a dataset leaf (e.g. eng_eth).

    The adapter directory name mirrors the dataset leaf name so downstream
    inference and evaluation are unambiguous.
    """
    logger.info("═══ Starting adapter training for dataset: %s ═══", language)

    adapter_output_dir = output_root / f"adapter_{language}"
    adapter_output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"train_{language}"

    # -----------------------------------------------------------------------
    # Load model + tokenizer
    # -----------------------------------------------------------------------
    base_model, tokenizer = load_base_model(cfg)
    model = get_peft_model(base_model, build_lora_config(lang_cfg), adapter_name=language)
    model.print_trainable_parameters()

    # -----------------------------------------------------------------------
    # Dataset splits
    # -----------------------------------------------------------------------
    train_ds = get_split(dataset, "train")
    eval_ds  = get_split(dataset, "dev")

    if train_ds is None or len(train_ds) == 0:
        logger.warning("Empty training set for %s; skipping.", language)
        return str(adapter_output_dir)

    # -----------------------------------------------------------------------
    # Format: raw input/output → chat-style text
    # -----------------------------------------------------------------------
    format_sample = make_format_fn(lang_cfg, tokenizer)

    train_text = train_ds.map(format_sample, remove_columns=train_ds.column_names)
    eval_text  = None
    if eval_ds is not None and len(eval_ds) > 0:
        eval_text = eval_ds.map(format_sample, remove_columns=eval_ds.column_names)

    # -----------------------------------------------------------------------
    # Tokenise
    # -----------------------------------------------------------------------
    def tokenize_batch(batch):
        eos_token = tokenizer.eos_token or ""
        texts = []
        for text in batch["text"]:
            if eos_token and not text.endswith(eos_token):
                text = f"{text}{eos_token}"
            texts.append(text)

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
            return_attention_mask=True,
        )

        # meditron-7b does not emit token_type_ids; add zeros so the collator
        # does not complain if any downstream code checks for this field.
        if "token_type_ids" not in tokenized:
            tokenized["token_type_ids"] = [
                [0] * len(ids) for ids in tokenized["input_ids"]
            ]

        return tokenized

    logger.info("[%s] Tokenizing training dataset …", language)
    train_tokens = train_text.map(
        tokenize_batch, batched=True, remove_columns=train_text.column_names
    )
    eval_tokens = None
    if eval_text is not None:
        logger.info("[%s] Tokenizing eval dataset …", language)
        eval_tokens = eval_text.map(
            tokenize_batch, batched=True, remove_columns=eval_text.column_names
        )

    # -----------------------------------------------------------------------
    # SFT training
    # -----------------------------------------------------------------------
    sft_cfg = build_sft_config(
        cfg=cfg,
        lang_cfg=lang_cfg,
        adapter_output_dir=adapter_output_dir,
        has_eval=eval_tokens is not None,
        run_name=run_name,
    )

    callbacks = []
    use_early_stopping = bool(
        eval_tokens is not None
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
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_tokens,
        eval_dataset=eval_tokens,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    logger.info(
        "Training %s adapter with %d train samples", language, len(train_tokens)
    )
    trainer.train()

    # -----------------------------------------------------------------------
    # Save adapter + tokenizer + metadata
    # -----------------------------------------------------------------------
    model.save_pretrained(str(adapter_output_dir))
    tokenizer.save_pretrained(str(adapter_output_dir))

    metadata = {
        "dataset_id":    language,
        "language_code": lang_cfg.language_code,
        "language_name": lang_cfg.language_name,
        "country_code":  lang_cfg.country_code,
        "country_name":  lang_cfg.country_name,
        "display_name":  lang_cfg.display_name,
        "lora_r":        lang_cfg.lora_r,
        "num_epochs":    lang_cfg.num_epochs,
        "learning_rate": lang_cfg.learning_rate,
        "train_samples": len(train_tokens),
        "dev_samples":   len(eval_tokens) if eval_tokens is not None else 0,
        "base_model":    cfg.model_id,
    }
    with open(adapter_output_dir / "adapter_meta.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)

    logger.info("Adapter saved to %s", adapter_output_dir)

    # Free VRAM before the next language
    del trainer
    del model
    del base_model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(adapter_output_dir)


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def load_adapter_for_inference(
    language: str,
    cfg: TrainingConfig,
    output_root: Path,
):
    """
    Load the base model plus a trained adapter for inference.

    Returns the merged PEFT model and the tokenizer in eval mode.
    """
    base_model, tokenizer = load_base_model(cfg)
    adapter_path = resolve_adapter_path(output_root, language)

    if adapter_path is None:
        raise FileNotFoundError(
            f"No adapter found for '{language}' under {output_root}. "
            "Run training first."
        )

    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    model.eval()
    logger.info("Loaded adapter for '%s' from %s", language, adapter_path)
    return model, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train per-dataset-leaf LoRA adapters on meditron-7b"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(SUPPORTED_LANGUAGES.keys()),
        help=(
            "Dataset leaves or base language groups to train. "
            "Examples: eng_uga, swa_ken, or grouped like eng, swa."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./data",
        help="Local root for save_to_disk data or shard trees.",
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        default="AiHub4MSRH-Hash/RAW_HASH_DATASET",
        help="HuggingFace dataset repo id for multilingual SRH shards.",
    )
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help="Optional dataset revision, branch, or commit.",
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
        help="Maximum test examples per dataset leaf during evaluation.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; run evaluation on saved adapters only.",
    )
    parser.add_argument(
        "--no_adapters",
        action="store_true",
        help="Evaluate the base model without any adapters (baseline comparison).",
    )
    parser.add_argument(
        "--adapter_repo",
        type=str,
        default="AiHub4MSRH-Hash/hashie-srh-meditron-adapters-v2",
        help=(
            "HF repo id to download pre-trained adapters from before evaluation. "
            "Set to empty string to skip."
        ),
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace API token. Falls back to HF_TOKEN env variable.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(hf_token)
    else:
        logger.warning(
            "No HF token provided. Fine for public datasets, but some model "
            "repos may require authentication."
        )

    try:
        selected_languages = expand_language_selection(args.languages)
    except KeyError as exc:
        raise SystemExit(
            f"Unsupported language or dataset selection: {exc.args[0]}. "
            f"Supported: {', '.join(sorted(set(SUPPORTED_LANGUAGES) | {'aka', 'amh', 'eng', 'lug', 'swa'}))}"
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

    if not args.no_adapters and args.adapter_repo:
        logger.info("Downloading adapters from %s …", args.adapter_repo)
        download_adapters(
            adapter_repo=args.adapter_repo,
            lang_codes=selected_languages,
            adapter_root=output_root,
            cache_dir=args.cache_dir,
            hf_token=hf_token,
        )

    evaluator = MultilingualEvaluator(
        cfg,
        output_root,
        dataset_builder=builder,
        base_model_only=args.no_adapters,
    )
    results = evaluator.evaluate_all(
        selected_languages,
        max_eval_samples=args.max_eval_samples,
    )
    report_name = "eval_report_base.json" if args.no_adapters else "eval_report.json"
    evaluator.save_report(results, output_root / report_name)
    logger.info("All done.")


if __name__ == "__main__":
    main()
