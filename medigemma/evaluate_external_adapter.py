#!/usr/bin/env python3
"""
evaluate_external_adapter.py - Evaluate an external HF PEFT adapter with our data.

This script is intentionally shaped like `compare_models.py`, but the adapter
side can come from a public Hugging Face PEFT repo such as
`DariusTheGeek/mhqa-itu-adapters`.

It writes a row-level CSV with `baseline_prediction` and `adapter_prediction`
columns so the existing `judge_comparison_with_openai.py` script can score the
outputs without any format conversion.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import concatenate_datasets
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from compare_models import (
    char_ngram_f1,
    compute_aggregate,
    compute_aggregate_delta,
    compute_metric_delta,
    exact_match,
    generate_response as generate_baseline_response,
    latin_leak_ratio,
    length_ratio,
    load_model_and_processor as load_baseline_model_and_processor,
    quality_flag,
    repetition_ngram_rate,
    rouge_l,
    script_match_ratio,
    summarize_metrics,
    token_f1,
    unload_model,
)
from config import SOURCE_DATASETS, SUPPORTED_LANGUAGES, TrainingConfig, expand_language_selection
from data_utils import MultilingualDatasetBuilder, get_split
from prompt_utils import build_hashie_messages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_DARIUS_REPO = "DariusTheGeek/mhqa-itu-adapters"
DEFAULT_DARIUS_SUBFOLDER = "gen7454"
DEFAULT_DARIUS_BASE = "google/gemma-4-31B-it"

SOURCE_TO_DARIUS_SUBSET = {
    "aka_gha": "Aka_Gha",
    "amh_eth": "Amh_Eth",
    "eng_eth": "Eng_Eth",
    "eng_gha": "Eng_Gha",
    "eng_ken": "Eng_Ken",
    "eng_uga": "Eng_Uga",
    "lug_uga": "Lug_Uga",
    "swa_ken": "Swa_Ken",
    # Darius' documented competition subsets do not include Swa_Uga. Use the
    # closest Swahili instruction when evaluating our combined Swahili target.
    "swa_uga": "Swa_Ken",
    "eng": "Eng_Uga",
    "swa": "Swa_Ken",
}

DARIUS_STYLE_DESCRIPTIONS = {
    "Aka_Gha": "Akan/Twi Ghana health question; answer in Twi.",
    "Amh_Eth": "Amharic Ethiopia health question; answer concisely in Ethiopic script.",
    "Eng_Eth": "English Ethiopia health question; answer concisely.",
    "Eng_Gha": "English Ghana health question; answer in a helpful patient-facing style.",
    "Eng_Ken": "English Kenya health question; answer naturally and informatively.",
    "Eng_Uga": "English Uganda health question; answer with clinical detail.",
    "Lug_Uga": "Luganda Uganda health question; answer in Luganda.",
    "Swa_Ken": "Kiswahili Kenya health question; answer in Kiswahili.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an external HF PEFT adapter on our multilingual SRH splits."
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(SUPPORTED_LANGUAGES.keys()),
        help="Adapter targets or grouped selections such as `aka amh eng lug swa`.",
    )
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--dataset_repo", default=None)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--split", choices=("train", "dev", "test"), default="test")
    parser.add_argument("--max_eval_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument(
        "--use_source_leaves",
        action="store_true",
        help=(
            "For combined targets such as eng and swa, load individual source "
            "leaves when available so reports include source-specific rows."
        ),
    )
    parser.add_argument(
        "--require_source_leaves",
        action="store_true",
        help=(
            "When --use_source_leaves is set, fail if any requested source leaf "
            "cannot be loaded instead of falling back to the combined target."
        ),
    )
    parser.add_argument(
        "--max_eval_samples_per_source",
        type=int,
        default=None,
        help=(
            "Sample up to this many examples from each loaded source leaf or "
            "single target before concatenation. If omitted, --max_eval_samples "
            "is applied after concatenating all loaded sources."
        ),
    )
    parser.add_argument(
        "--adapter_repo",
        default=DEFAULT_DARIUS_REPO,
        help="HF PEFT repo id containing the external adapter.",
    )
    parser.add_argument(
        "--adapter_subfolder",
        default=DEFAULT_DARIUS_SUBFOLDER,
        help="Subfolder inside the adapter repo, e.g. gen7454 or mg_2226.",
    )
    parser.add_argument(
        "--adapter_base_model",
        default=DEFAULT_DARIUS_BASE,
        help="Base model to load before attaching the external adapter.",
    )
    parser.add_argument(
        "--adapter_model_class",
        choices=("causal_lm", "image_text_to_text"),
        default="causal_lm",
        help="Transformers AutoModel class for the external adapter base.",
    )
    parser.add_argument(
        "--inference_backend",
        choices=("transformers_peft", "vllm"),
        default="transformers_peft",
        help=(
            "`transformers_peft` uses vanilla Transformers + PEFT. `vllm` "
            "uses Darius' serving pattern with vLLM LoRARequest, which is "
            "needed for Gemma-4 adapters saved with clippable linear wrappers."
        ),
    )
    parser.add_argument(
        "--candidate_name",
        default="darius_gen7454",
        help="Human-readable candidate label stored in the JSON report.",
    )
    parser.add_argument(
        "--prompt_style",
        choices=("hashie", "darius", "darius_quality", "plain"),
        default="hashie",
        help=(
            "`hashie` uses our normal evaluation prompt. `darius` and "
            "`darius_quality` use Darius-style subset instructions. `plain` "
            "sends only the question."
        ),
    )
    parser.add_argument(
        "--baseline_model",
        default="google/medgemma-4b-it",
        help="Optional baseline model for comparison rows.",
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Only generate the external adapter predictions.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help=(
            "Load the external adapter base and optional baseline in 4-bit mode. "
            "For the vLLM backend this maps to bitsandbytes quantization."
        ),
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor_parallel_size when --inference_backend vllm.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="vLLM max_model_len when --inference_backend vllm.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="vLLM GPU memory utilization when --inference_backend vllm.",
    )
    parser.add_argument(
        "--vllm_quantization",
        default=None,
        help="Optional vLLM quantization, e.g. bitsandbytes.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="vLLM max_lora_rank. gen7454 and mg_2226 use rank 64.",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN.",
    )
    parser.add_argument("--output_dir", default="./reports")
    parser.add_argument("--csv_path", default=None)
    parser.add_argument("--report_json", default=None)
    return parser.parse_args()


def select_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_external_model_kwargs(load_in_4bit: bool) -> dict[str, Any]:
    dtype = select_dtype()
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
            ),
        )
    return kwargs


def model_device(model) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def move_to_model_device(inputs: Any, model):
    if not torch.cuda.is_available():
        return inputs
    device = model_device(model)
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }


def tokenizer_like(processor_or_tokenizer):
    return getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)


def attach_adapter_chat_template(
    processor_or_tokenizer,
    *,
    adapter_repo: str,
    adapter_subfolder: str,
    cache_dir: Optional[str],
    hf_token: Optional[str],
) -> None:
    """Use the adapter's shipped chat template when available."""

    try:
        template_path = hf_hub_download(
            repo_id=adapter_repo,
            filename=f"{adapter_subfolder}/chat_template.jinja",
            cache_dir=cache_dir,
            token=hf_token,
        )
    except Exception as exc:
        logger.warning("Could not fetch adapter chat template: %s", exc)
        return

    template = Path(template_path).read_text(encoding="utf-8")
    owner = tokenizer_like(processor_or_tokenizer)
    setattr(owner, "chat_template", template)
    if owner is not processor_or_tokenizer and hasattr(processor_or_tokenizer, "chat_template"):
        setattr(processor_or_tokenizer, "chat_template", template)


def load_external_processor(
    base_model_ref: str,
    *,
    adapter_repo: str,
    adapter_subfolder: str,
    cache_dir: Optional[str],
    hf_token: Optional[str],
):
    try:
        processor = AutoProcessor.from_pretrained(
            base_model_ref,
            cache_dir=cache_dir,
            token=hf_token,
            trust_remote_code=True,
        )
    except Exception:
        processor = AutoTokenizer.from_pretrained(
            base_model_ref,
            cache_dir=cache_dir,
            token=hf_token,
            trust_remote_code=True,
        )

    owner = tokenizer_like(processor)
    if hasattr(owner, "padding_side"):
        owner.padding_side = "right"
    attach_adapter_chat_template(
        processor,
        adapter_repo=adapter_repo,
        adapter_subfolder=adapter_subfolder,
        cache_dir=cache_dir,
        hf_token=hf_token,
    )
    return processor


def load_external_adapter_model(
    *,
    base_model_ref: str,
    adapter_repo: str,
    adapter_subfolder: str,
    adapter_model_class: str,
    load_in_4bit: bool,
    cache_dir: Optional[str],
    hf_token: Optional[str],
):
    from peft import PeftModel

    logger.info(
        "Loading external adapter %s/%s on base %s",
        adapter_repo,
        adapter_subfolder,
        base_model_ref,
    )
    model_cls = (
        AutoModelForImageTextToText
        if adapter_model_class == "image_text_to_text"
        else AutoModelForCausalLM
    )
    base_model = model_cls.from_pretrained(
        base_model_ref,
        cache_dir=cache_dir,
        token=hf_token,
        **build_external_model_kwargs(load_in_4bit),
    )
    model = PeftModel.from_pretrained(
        base_model,
        adapter_repo,
        subfolder=adapter_subfolder,
        cache_dir=cache_dir,
        token=hf_token,
    )
    model.eval()
    processor = load_external_processor(
        base_model_ref,
        adapter_repo=adapter_repo,
        adapter_subfolder=adapter_subfolder,
        cache_dir=cache_dir,
        hf_token=hf_token,
    )
    return model, processor


def resolve_local_adapter_path(
    *,
    adapter_repo: str,
    adapter_subfolder: str,
    cache_dir: Optional[str],
    hf_token: Optional[str],
) -> str:
    """
    Download an HF adapter repo snapshot and return the local subfolder path.

    vLLM's LoRARequest expects a filesystem path, while PEFT can read directly
    from the Hub. Keeping this small helper separate lets the vLLM backend
    follow Darius' own `gen_samples.py` serving pattern.
    """

    repo_root = snapshot_download(
        repo_id=adapter_repo,
        allow_patterns=[f"{adapter_subfolder}/*"],
        cache_dir=cache_dir,
        token=hf_token,
    )
    adapter_path = Path(repo_root) / adapter_subfolder
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Downloaded {adapter_repo}, but {adapter_subfolder} was not found."
        )
    return str(adapter_path)


def load_vllm_adapter_model(
    *,
    base_model_ref: str,
    adapter_repo: str,
    adapter_subfolder: str,
    load_in_4bit: bool,
    cache_dir: Optional[str],
    hf_token: Optional[str],
    tensor_parallel_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    vllm_quantization: Optional[str],
    lora_rank: int,
):
    try:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
    except ImportError as exc:
        raise RuntimeError(
            "The vLLM backend requires `vllm`. Install or use an environment "
            "compatible with Darius' serving code before running this backend."
        ) from exc

    adapter_path = resolve_local_adapter_path(
        adapter_repo=adapter_repo,
        adapter_subfolder=adapter_subfolder,
        cache_dir=cache_dir,
        hf_token=hf_token,
    )

    quantization = vllm_quantization
    if load_in_4bit and not quantization:
        quantization = "bitsandbytes"

    kwargs: dict[str, Any] = {
        "model": base_model_ref,
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": "bfloat16",
        "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization,
        "enforce_eager": True,
        "trust_remote_code": True,
        "enable_lora": True,
        "max_lora_rank": lora_rank,
    }
    if quantization:
        kwargs["quantization"] = quantization
        kwargs["load_format"] = quantization

    logger.info(
        "Loading vLLM base %s with LoRA adapter %s (max_lora_rank=%s)",
        base_model_ref,
        adapter_path,
        lora_rank,
    )
    llm = LLM(**kwargs)
    lora_request = LoRARequest("adapter", 1, adapter_path)
    processor = load_external_processor(
        base_model_ref,
        adapter_repo=adapter_repo,
        adapter_subfolder=adapter_subfolder,
        cache_dir=cache_dir,
        hf_token=hf_token,
    )
    return {"llm": llm, "lora_request": lora_request}, processor


def source_to_darius_subset(source_dataset: str) -> str:
    return SOURCE_TO_DARIUS_SUBSET.get(source_dataset, source_dataset)


def darius_style_system(source_dataset: str, *, quality: bool = False) -> str:
    subset = source_to_darius_subset(source_dataset)
    desc = DARIUS_STYLE_DESCRIPTIONS.get(subset, subset)
    if quality:
        return (
            "You are a careful multilingual SRH assistant for sub-Saharan Africa. "
            "Answer only in the requested subset language. Prioritize clinical "
            f"accuracy, safety, completeness, and usefulness. Subset: {subset} - {desc}"
        )
    return (
        "You are a careful multilingual SRH assistant for sub-Saharan Africa. "
        "Answer only in the requested subset language and match the expected "
        f"answer style. Subset: {subset} - {desc}"
    )


def build_candidate_messages(
    *,
    prompt: str,
    language_name: str,
    source_dataset: str,
    prompt_style: str,
) -> list[dict[str, str]]:
    if prompt_style == "hashie":
        return build_hashie_messages(prompt, language_name)
    if prompt_style == "plain":
        return [{"role": "user", "content": prompt}]

    system_text = darius_style_system(
        source_dataset,
        quality=(prompt_style == "darius_quality"),
    )
    return [{"role": "user", "content": f"{system_text}\n\nQuestion: {prompt}"}]


def render_prompt_text(processor_or_tokenizer, messages: list[dict[str, str]]) -> str:
    owner = tokenizer_like(processor_or_tokenizer)
    try:
        return owner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        return owner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def encode_messages(processor_or_tokenizer, messages: list[dict[str, str]]):
    owner = tokenizer_like(processor_or_tokenizer)
    try:
        return owner.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    except TypeError:
        text = owner.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if callable(processor_or_tokenizer):
            return processor_or_tokenizer(text=text, return_tensors="pt")
        return owner(text, return_tensors="pt")


def decode_new_tokens(processor_or_tokenizer, token_ids) -> str:
    owner = tokenizer_like(processor_or_tokenizer)
    return owner.decode(token_ids, skip_special_tokens=True).strip()


def generate_candidate_response(
    model,
    processor_or_tokenizer,
    *,
    prompt: str,
    language_name: str,
    source_dataset: str,
    prompt_style: str,
    max_new_tokens: int,
) -> str:
    messages = build_candidate_messages(
        prompt=prompt,
        language_name=language_name,
        source_dataset=source_dataset,
        prompt_style=prompt_style,
    )
    inputs = encode_messages(processor_or_tokenizer, messages)
    inputs = move_to_model_device(inputs, model)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    prompt_length = inputs["input_ids"].shape[1]
    return decode_new_tokens(processor_or_tokenizer, output_ids[0][prompt_length:])


def generate_candidate_responses_vllm(
    vllm_state: dict[str, Any],
    processor_or_tokenizer,
    *,
    examples: list[dict[str, Any]],
    language_name: str,
    lang_code: str,
    prompt_style: str,
    max_new_tokens: int,
) -> list[str]:
    from vllm import SamplingParams

    prompts = []
    for example in examples:
        source_dataset = example.get("_source_dataset", lang_code)
        messages = build_candidate_messages(
            prompt=example["input"],
            language_name=language_name,
            source_dataset=source_dataset,
            prompt_style=prompt_style,
        )
        prompts.append(render_prompt_text(processor_or_tokenizer, messages))

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        seed=1234,
    )
    outputs = vllm_state["llm"].generate(
        prompts,
        sampling_params,
        lora_request=vllm_state["lora_request"],
    )
    predictions = []
    for output in outputs:
        if output.outputs:
            predictions.append(output.outputs[0].text.strip())
        else:
            predictions.append("")
    return predictions


def source_ids_for_target(lang_code: str, use_source_leaves: bool) -> tuple[str, ...]:
    lang_cfg = SUPPORTED_LANGUAGES[lang_code]
    if use_source_leaves and lang_cfg.source_datasets:
        return lang_cfg.source_datasets
    return (lang_code,)


def load_eval_examples(
    *,
    lang_code: str,
    split: str,
    max_eval_samples: int,
    max_eval_samples_per_source: Optional[int],
    use_source_leaves: bool,
    require_source_leaves: bool,
    dataset_builder: MultilingualDatasetBuilder,
    seed: int,
) -> list[dict[str, Any]]:
    source_ids = source_ids_for_target(lang_code, use_source_leaves)
    parts = []
    load_errors = []
    loaded_source_ids = []
    requested_source_leaves = (
        use_source_leaves and bool(SUPPORTED_LANGUAGES[lang_code].source_datasets)
    )

    for source_index, source_id in enumerate(source_ids):
        source_cfg = SOURCE_DATASETS.get(source_id, SUPPORTED_LANGUAGES.get(source_id))
        if source_cfg is None:
            load_errors.append(f"{source_id}: no dataset config")
            continue
        try:
            dataset = dataset_builder.load_language(source_id, source_cfg, augment=False)
            split_ds = get_split(dataset, split)
        except Exception as exc:
            load_errors.append(f"{source_id}: {exc}")
            continue

        if split_ds is None:
            load_errors.append(f"{source_id}: no {split} split")
            continue
        if len(split_ds) == 0:
            load_errors.append(f"{source_id}: empty {split} split")
            continue

        columns = [
            column
            for column in ("input", "output", "label")
            if column in split_ds.column_names
        ]
        part = split_ds.select_columns(columns)
        part = part.add_column("_source_dataset", [source_id] * len(part))
        part = part.add_column("_source_example_index", list(range(len(part))))
        if max_eval_samples_per_source is not None:
            n = min(len(part), max_eval_samples_per_source)
            part = part.shuffle(seed=seed + source_index).select(range(n))
        parts.append(part)
        loaded_source_ids.append(source_id)

    if requested_source_leaves and require_source_leaves:
        missing_source_ids = [
            source_id for source_id in source_ids if source_id not in loaded_source_ids
        ]
        if missing_source_ids:
            details = "; ".join(load_errors) if load_errors else "no loader details"
            raise FileNotFoundError(
                f"Required source leaves for {lang_code} were not all available. "
                f"Missing: {missing_source_ids}. Loaded: {loaded_source_ids}. Details: {details}"
            )

    if requested_source_leaves and load_errors and not require_source_leaves:
        logger.warning(
            "Some source leaves for %s could not be loaded and will be omitted: %s",
            lang_code,
            "; ".join(load_errors),
        )

    if not parts and use_source_leaves and SUPPORTED_LANGUAGES[lang_code].source_datasets:
        logger.warning(
            "Could not load source leaves for %s (%s); falling back to combined target.",
            lang_code,
            "; ".join(load_errors),
        )
        return load_eval_examples(
            lang_code=lang_code,
            split=split,
            max_eval_samples=max_eval_samples,
            max_eval_samples_per_source=max_eval_samples_per_source,
            use_source_leaves=False,
            require_source_leaves=False,
            dataset_builder=dataset_builder,
            seed=seed,
        )

    if not parts:
        if load_errors:
            raise FileNotFoundError("; ".join(load_errors))
        raise ValueError(f"No {split} examples found for {lang_code}")

    combined = concatenate_datasets(parts).shuffle(seed=seed)
    if max_eval_samples_per_source is None:
        n = min(len(combined), max_eval_samples)
        selected = combined.select(range(n))
    else:
        selected = combined
    return [selected[idx] for idx in range(len(selected))]


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    default_stem = f"{args.candidate_name}_{args.split}_comparison"
    csv_path = Path(args.csv_path) if args.csv_path else output_dir / f"{default_stem}.csv"
    report_path = (
        Path(args.report_json)
        if args.report_json
        else output_dir / f"{default_stem}_report.json"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    return csv_path, report_path


def blank_baseline_metrics(lang_code: str, details: str = "") -> dict[str, Any]:
    metrics = {"language": lang_code, "error": "baseline_skipped"}
    if details:
        metrics["details"] = details
    return metrics


def main() -> None:
    args = parse_args()
    if args.require_source_leaves and not args.use_source_leaves:
        raise SystemExit("--require_source_leaves requires --use_source_leaves.")
    if args.max_eval_samples <= 0:
        raise SystemExit("--max_eval_samples must be positive.")
    if (
        args.max_eval_samples_per_source is not None
        and args.max_eval_samples_per_source <= 0
    ):
        raise SystemExit("--max_eval_samples_per_source must be positive.")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    cfg = TrainingConfig()
    csv_path, report_json = resolve_output_paths(args)

    try:
        languages = expand_language_selection(args.languages)
    except KeyError as exc:
        raise SystemExit(f"Unknown language selection: {exc.args[0]}") from exc

    dataset_builder = MultilingualDatasetBuilder(
        data_root=args.data_root,
        dataset_repo=args.dataset_repo,
        dataset_revision=args.dataset_revision,
        hf_token=hf_token,
        cache_dir=args.cache_dir,
        seed=cfg.seed,
    )

    examples_by_language: dict[str, list[dict[str, Any]]] = {}
    rows_by_language: dict[str, list[dict[str, Any]]] = {}
    baseline_predictions_by_language: dict[str, list[str]] = {}
    baseline_per_language: dict[str, dict[str, Any]] = {}
    candidate_per_language: dict[str, dict[str, Any]] = {}
    comparison_per_language: dict[str, dict[str, Any]] = {}
    source_counts_by_language: dict[str, dict[str, int]] = {}

    for lang_code in languages:
        lang_cfg = SUPPORTED_LANGUAGES[lang_code]
        try:
            examples = load_eval_examples(
                lang_code=lang_code,
                split=args.split,
                max_eval_samples=args.max_eval_samples,
                max_eval_samples_per_source=args.max_eval_samples_per_source,
                use_source_leaves=args.use_source_leaves,
                require_source_leaves=args.require_source_leaves,
                dataset_builder=dataset_builder,
                seed=cfg.seed,
            )
        except Exception as exc:
            error = {"language": lang_code, "error": "dataset_load_failed", "details": str(exc)}
            baseline_per_language[lang_code] = error
            candidate_per_language[lang_code] = error
            comparison_per_language[lang_code] = {
                "baseline": error,
                "adapter": error,
                "delta": {},
            }
            logger.error("Skipping %s because data could not be loaded: %s", lang_code, exc)
            continue

        examples_by_language[lang_code] = examples
        rows_by_language[lang_code] = []
        source_counts = dict(
            Counter(example.get("_source_dataset", lang_code) for example in examples)
        )
        source_counts_by_language[lang_code] = source_counts
        logger.info(
            "Loaded %s %s examples for %s from sources: %s",
            len(examples),
            args.split,
            lang_code,
            source_counts,
        )

        if args.skip_baseline:
            baseline_predictions_by_language[lang_code] = [""] * len(examples)
            baseline_per_language[lang_code] = blank_baseline_metrics(lang_code)

    if not args.skip_baseline:
        baseline_model = None
        try:
            baseline_model, baseline_processor = load_baseline_model_and_processor(
                args.baseline_model,
                load_in_4bit=args.load_in_4bit,
                cache_dir=args.cache_dir,
            )
            for lang_code, examples in examples_by_language.items():
                lang_cfg = SUPPORTED_LANGUAGES[lang_code]
                predictions = []
                for example in examples:
                    try:
                        prediction = generate_baseline_response(
                            baseline_model,
                            baseline_processor,
                            prompt=example["input"],
                            language_name=lang_cfg.language_name,
                            max_new_tokens=args.max_new_tokens,
                        )
                    except Exception as exc:
                        logger.error("Baseline generation error for %s: %s", lang_code, exc)
                        prediction = ""
                    predictions.append(prediction)

                baseline_predictions_by_language[lang_code] = predictions
                baseline_per_language[lang_code] = {
                    "language": lang_code,
                    "display_name": lang_cfg.display_name,
                    "language_name": lang_cfg.language_name,
                    "resource_level": lang_cfg.resource_level,
                    **summarize_metrics(predictions, examples, target_script=lang_cfg.script),
                }
        finally:
            if baseline_model is not None:
                unload_model(baseline_model)

    candidate_model = None
    try:
        if args.inference_backend == "vllm":
            candidate_model, candidate_processor = load_vllm_adapter_model(
                base_model_ref=args.adapter_base_model,
                adapter_repo=args.adapter_repo,
                adapter_subfolder=args.adapter_subfolder,
                load_in_4bit=args.load_in_4bit,
                cache_dir=args.cache_dir,
                hf_token=hf_token,
                tensor_parallel_size=args.tensor_parallel_size,
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                vllm_quantization=args.vllm_quantization,
                lora_rank=args.lora_rank,
            )
        else:
            candidate_model, candidate_processor = load_external_adapter_model(
                base_model_ref=args.adapter_base_model,
                adapter_repo=args.adapter_repo,
                adapter_subfolder=args.adapter_subfolder,
                adapter_model_class=args.adapter_model_class,
                load_in_4bit=args.load_in_4bit,
                cache_dir=args.cache_dir,
                hf_token=hf_token,
            )

        for lang_code, examples in examples_by_language.items():
            lang_cfg = SUPPORTED_LANGUAGES[lang_code]
            logger.info("Generating external adapter predictions for %s", lang_code)
            if args.inference_backend == "vllm":
                try:
                    candidate_predictions = generate_candidate_responses_vllm(
                        candidate_model,
                        candidate_processor,
                        examples=examples,
                        language_name=lang_cfg.language_name,
                        lang_code=lang_code,
                        prompt_style=args.prompt_style,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as exc:
                    logger.error("vLLM generation error for %s: %s", lang_code, exc)
                    candidate_predictions = [""] * len(examples)
            else:
                candidate_predictions = []
                for example in examples:
                    source_dataset = example.get("_source_dataset", lang_code)
                    try:
                        prediction = generate_candidate_response(
                            candidate_model,
                            candidate_processor,
                            prompt=example["input"],
                            language_name=lang_cfg.language_name,
                            source_dataset=source_dataset,
                            prompt_style=args.prompt_style,
                            max_new_tokens=args.max_new_tokens,
                        )
                    except Exception as exc:
                        logger.error("Candidate generation error for %s: %s", lang_code, exc)
                        prediction = ""
                    candidate_predictions.append(prediction)

            candidate_metrics = {
                "language": lang_code,
                "display_name": lang_cfg.display_name,
                "language_name": lang_cfg.language_name,
                "resource_level": lang_cfg.resource_level,
                "source_counts": source_counts_by_language.get(lang_code, {}),
                **summarize_metrics(
                    candidate_predictions,
                    examples,
                    target_script=lang_cfg.script,
                ),
            }
            candidate_per_language[lang_code] = candidate_metrics

            baseline_metrics = baseline_per_language.get(
                lang_code,
                blank_baseline_metrics(lang_code, "baseline metrics unavailable"),
            )
            delta = (
                compute_metric_delta(baseline_metrics, candidate_metrics)
                if "error" not in baseline_metrics
                else {}
            )
            comparison_per_language[lang_code] = {
                "baseline": baseline_metrics,
                "adapter": candidate_metrics,
                "delta": delta,
            }

            baseline_predictions = baseline_predictions_by_language.get(
                lang_code,
                [""] * len(examples),
            )
            for index, example in enumerate(examples):
                source_dataset = example.get("_source_dataset", lang_code)
                reference_answer = example["output"]
                baseline_prediction = baseline_predictions[index]
                adapter_prediction = candidate_predictions[index]
                rows_by_language[lang_code].append(
                    {
                        "language": lang_code,
                        "source_dataset": source_dataset,
                        "darius_subset": source_to_darius_subset(source_dataset),
                        "display_name": lang_cfg.display_name,
                        "split": args.split,
                        "example_index": index,
                        "source_example_index": example.get("_source_example_index", ""),
                        "question": example["input"],
                        "reference_answer": reference_answer,
                        "baseline_prediction": baseline_prediction,
                        "adapter_prediction": adapter_prediction,
                        "baseline_exact_match": (
                            exact_match(baseline_prediction, reference_answer)
                            if baseline_prediction
                            else ""
                        ),
                        "adapter_exact_match": exact_match(adapter_prediction, reference_answer),
                        "baseline_f1_token": (
                            round(token_f1(baseline_prediction, reference_answer), 4)
                            if baseline_prediction
                            else ""
                        ),
                        "adapter_f1_token": round(token_f1(adapter_prediction, reference_answer), 4),
                        "baseline_rouge_l": (
                            round(rouge_l(baseline_prediction, reference_answer), 4)
                            if baseline_prediction
                            else ""
                        ),
                        "adapter_rouge_l": round(rouge_l(adapter_prediction, reference_answer), 4),
                        "baseline_char_3gram_f1": (
                            round(char_ngram_f1(baseline_prediction, reference_answer), 4)
                            if baseline_prediction
                            else ""
                        ),
                        "adapter_char_3gram_f1": round(
                            char_ngram_f1(adapter_prediction, reference_answer),
                            4,
                        ),
                        "baseline_script_match_ratio": (
                            round(script_match_ratio(baseline_prediction, lang_cfg.script), 4)
                            if baseline_prediction
                            else ""
                        ),
                        "adapter_script_match_ratio": round(
                            script_match_ratio(adapter_prediction, lang_cfg.script),
                            4,
                        ),
                        "baseline_latin_leak_ratio": (
                            round(latin_leak_ratio(baseline_prediction), 4)
                            if baseline_prediction
                            else ""
                        ),
                        "adapter_latin_leak_ratio": round(latin_leak_ratio(adapter_prediction), 4),
                        "baseline_repetition_4gram_rate": (
                            round(repetition_ngram_rate(baseline_prediction), 4)
                            if baseline_prediction
                            else ""
                        ),
                        "adapter_repetition_4gram_rate": round(
                            repetition_ngram_rate(adapter_prediction),
                            4,
                        ),
                        "baseline_length_ratio": (
                            round(length_ratio(baseline_prediction, reference_answer), 4)
                            if baseline_prediction
                            else ""
                        ),
                        "adapter_length_ratio": round(
                            length_ratio(adapter_prediction, reference_answer),
                            4,
                        ),
                        "baseline_quality_flag": (
                            quality_flag(
                                baseline_prediction,
                                reference_answer,
                                lang_cfg.script,
                            )
                            if baseline_prediction
                            else ""
                        ),
                        "adapter_quality_flag": quality_flag(
                            adapter_prediction,
                            reference_answer,
                            lang_cfg.script,
                        ),
                    }
                )
    finally:
        if candidate_model is not None:
            unload_model(candidate_model)

    rows = [
        row
        for lang_code in languages
        for row in rows_by_language.get(lang_code, [])
    ]
    fieldnames = [
        "language",
        "source_dataset",
        "darius_subset",
        "display_name",
        "split",
        "example_index",
        "source_example_index",
        "question",
        "reference_answer",
        "baseline_prediction",
        "adapter_prediction",
        "baseline_exact_match",
        "adapter_exact_match",
        "baseline_f1_token",
        "adapter_f1_token",
        "baseline_rouge_l",
        "adapter_rouge_l",
        "baseline_char_3gram_f1",
        "adapter_char_3gram_f1",
        "baseline_script_match_ratio",
        "adapter_script_match_ratio",
        "baseline_latin_leak_ratio",
        "adapter_latin_leak_ratio",
        "baseline_repetition_4gram_rate",
        "adapter_repetition_4gram_rate",
        "baseline_length_ratio",
        "adapter_length_ratio",
        "baseline_quality_flag",
        "adapter_quality_flag",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    baseline_aggregate = compute_aggregate(baseline_per_language)
    candidate_aggregate = compute_aggregate(candidate_per_language)
    report = {
        "candidate_name": args.candidate_name,
        "adapter_repo": args.adapter_repo,
        "adapter_subfolder": args.adapter_subfolder,
        "adapter_base_model": args.adapter_base_model,
        "adapter_model_class": args.adapter_model_class,
        "inference_backend": args.inference_backend,
        "prompt_style": args.prompt_style,
        "baseline_model": None if args.skip_baseline else args.baseline_model,
        "split": args.split,
        "max_eval_samples": args.max_eval_samples,
        "max_eval_samples_per_source": args.max_eval_samples_per_source,
        "use_source_leaves": args.use_source_leaves,
        "require_source_leaves": args.require_source_leaves,
        "source_counts": source_counts_by_language,
        "csv_path": str(csv_path),
        "per_language": comparison_per_language,
        "aggregate": {
            "baseline": baseline_aggregate,
            "adapter": candidate_aggregate,
            "delta": (
                compute_aggregate_delta(baseline_aggregate, candidate_aggregate)
                if baseline_aggregate and candidate_aggregate
                else {}
            ),
        },
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Wrote row-level comparison CSV to %s", csv_path)
    logger.info("Wrote external adapter report to %s", report_json)


if __name__ == "__main__":
    main()
