"""
compare_models.py — Compare baseline MedGemma generations against LoRA adapters
===============================================================================

This script runs the same evaluation prompts through:
1. a baseline MedGemma model checkpoint
2. the leaf-specific LoRA adapter model

It saves:
* a row-level CSV for manual review
* a baseline-only evaluation report
* a comparison report with adapter deltas
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from config import SUPPORTED_LANGUAGES, TrainingConfig, expand_language_selection
from data_utils import MultilingualDatasetBuilder, get_split
from evaluation import resolve_adapter_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

METRIC_KEYS = ("exact_match", "f1_token", "rouge_l", "invalid_rate", "mcq_accuracy")


def parse_args():
    """Define CLI arguments for baseline-versus-adapter comparison."""

    parser = argparse.ArgumentParser(
        description="Compare baseline MedGemma generations against saved LoRA adapters"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(SUPPORTED_LANGUAGES.keys()),
        help="Dataset leaves or grouped selections such as `eng` and `swa`.",
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
        help="Optional Hugging Face dataset repo id when local data is unavailable.",
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
        help="Optional cache directory for dataset and model downloads.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./adapters",
        help="Root directory containing saved adapter_* folders.",
    )
    parser.add_argument(
        "--baseline_model",
        type=str,
        default=None,
        help=(
            "Baseline model id or local path. Use this for the original model you "
            "want to compare against the adapters."
        ),
    )
    parser.add_argument(
        "--adapter_base_model",
        type=str,
        default=None,
        help=(
            "Optional override for the base model used before attaching each adapter. "
            "Defaults to adapter metadata or TrainingConfig.model_id."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "dev", "test"),
        default="test",
        help="Dataset split to compare on. Defaults to `test`.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=200,
        help="Maximum number of examples per dataset leaf.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per example.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load models in 4-bit mode to reduce GPU memory usage.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN from the environment.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Optional output path for the row-level comparison CSV.",
    )
    parser.add_argument(
        "--baseline_report_path",
        type=str,
        default=None,
        help="Optional output path for the baseline-only evaluation report.",
    )
    parser.add_argument(
        "--comparison_report_path",
        type=str,
        default=None,
        help="Optional output path for the comparison report.",
    )
    return parser.parse_args()


def build_model_kwargs(load_in_4bit: bool) -> dict[str, Any]:
    """Build shared model loading kwargs for baseline and adapter runs."""

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    return kwargs


def load_model_and_processor(
    model_ref: str,
    *,
    processor_ref: Optional[str] = None,
    load_in_4bit: bool = False,
    cache_dir: Optional[str] = None,
):
    """Load a model checkpoint and a matching processor."""

    logger.info("Loading model from %s", model_ref)
    model = AutoModelForImageTextToText.from_pretrained(
        model_ref,
        cache_dir=cache_dir,
        **build_model_kwargs(load_in_4bit),
    )
    processor_source = processor_ref or model_ref
    try:
        processor = AutoProcessor.from_pretrained(processor_source, cache_dir=cache_dir)
    except Exception:
        if processor_source == model_ref:
            raise
        processor = AutoProcessor.from_pretrained(model_ref, cache_dir=cache_dir)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "right"
    model.eval()
    return model, processor


def build_messages(prompt: str, language_name: str) -> list[dict[str, str]]:
    """Build the same evaluation prompt shape used in the repo evaluator."""

    return [
        {
            "role": "system",
            "content": (
                "You are a helpful sexual and reproductive health assistant. "
                f"Answer in {language_name}."
            ),
        },
        {"role": "user", "content": prompt},
    ]


def generate_response(
    model,
    processor,
    prompt: str,
    language_name: str,
    max_new_tokens: int,
) -> str:
    """Generate one response using deterministic decoding."""

    messages = build_messages(prompt, language_name)
    text = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][prompt_length:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact-match score."""

    return float(prediction.strip().lower() == ground_truth.strip().lower())


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1."""

    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """Compute a lightweight ROUGE-L F-score."""

    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    m, n = len(ref_tokens), len(pred_tokens)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    precision = lcs / n
    recall = lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def summarize_metrics(predictions: list[str], examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the same core metrics used by the existing evaluator."""

    references = [example["output"] for example in examples]
    n = len(examples)
    invalid_count = sum(1 for prediction in predictions if not prediction.strip())

    exact_scores = [exact_match(pred, ref) for pred, ref in zip(predictions, references)]
    f1_scores = [token_f1(pred, ref) for pred, ref in zip(predictions, references)]
    rouge_scores = [rouge_l(pred, ref) for pred, ref in zip(predictions, references)]

    summary = {
        "n_evaluated": n,
        "exact_match": round(sum(exact_scores) / n, 4),
        "f1_token": round(sum(f1_scores) / n, 4),
        "rouge_l": round(sum(rouge_scores) / n, 4),
        "invalid_rate": round(invalid_count / n, 4),
    }

    if examples and "label" in examples[0]:
        mcq_hits = 0
        for prediction, example in zip(predictions, examples):
            label = str(example.get("label", "")).strip().lower()
            if label and label in prediction.lower():
                mcq_hits += 1
        summary["mcq_accuracy"] = round(mcq_hits / n, 4)

    return summary


def compute_metric_delta(
    baseline_metrics: dict[str, Any],
    adapter_metrics: dict[str, Any],
) -> dict[str, float]:
    """Compute adapter-minus-baseline deltas for numeric metrics."""

    delta = {}
    for key in METRIC_KEYS:
        if key in baseline_metrics and key in adapter_metrics:
            delta[key] = round(adapter_metrics[key] - baseline_metrics[key], 4)
    return delta


def compute_aggregate(per_language: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compute macro averages over successful per-language results."""

    valid_results = [result for result in per_language.values() if "error" not in result]
    if not valid_results:
        return {}

    aggregate = {
        "macro_avg_exact_match": round(
            sum(result["exact_match"] for result in valid_results) / len(valid_results),
            4,
        ),
        "macro_avg_f1": round(
            sum(result["f1_token"] for result in valid_results) / len(valid_results),
            4,
        ),
        "macro_avg_rouge_l": round(
            sum(result["rouge_l"] for result in valid_results) / len(valid_results),
            4,
        ),
        "macro_avg_invalid_rate": round(
            sum(result["invalid_rate"] for result in valid_results) / len(valid_results),
            4,
        ),
        "n_languages_evaluated": len(valid_results),
    }

    if all("mcq_accuracy" in result for result in valid_results):
        aggregate["macro_avg_mcq_accuracy"] = round(
            sum(result["mcq_accuracy"] for result in valid_results) / len(valid_results),
            4,
        )

    return aggregate


def compute_aggregate_delta(
    baseline_aggregate: dict[str, Any],
    adapter_aggregate: dict[str, Any],
) -> dict[str, float]:
    """Compute aggregate adapter-minus-baseline deltas."""

    key_pairs = {
        "macro_avg_exact_match": "exact_match",
        "macro_avg_f1": "f1_token",
        "macro_avg_rouge_l": "rouge_l",
        "macro_avg_invalid_rate": "invalid_rate",
        "macro_avg_mcq_accuracy": "mcq_accuracy",
    }
    delta = {}
    for aggregate_key, short_name in key_pairs.items():
        if aggregate_key in baseline_aggregate and aggregate_key in adapter_aggregate:
            delta[short_name] = round(
                adapter_aggregate[aggregate_key] - baseline_aggregate[aggregate_key],
                4,
            )
    return delta


def resolve_output_path(explicit_path: Optional[str], default_path: Path) -> Path:
    """Resolve an output path and ensure the parent directory exists."""

    path = Path(explicit_path) if explicit_path else default_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_adapter_metadata(adapter_path: Path) -> dict[str, Any]:
    """Load adapter metadata from flat or nested PEFT saves."""

    candidates = (
        adapter_path / "adapter_meta.json",
        adapter_path.parent / "adapter_meta.json",
    )
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def unload_model(model) -> None:
    """Release model references and clear CUDA cache when available."""

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    """Run baseline-versus-adapter generation and save CSV plus reports."""

    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    cfg = TrainingConfig()

    try:
        languages = expand_language_selection(args.languages)
    except KeyError as exc:
        raise SystemExit(f"Unknown language selection: {exc.args[0]}") from exc

    output_root = Path(args.output_root)
    csv_path = resolve_output_path(
        args.csv_path,
        output_root / "adapter_baseline_comparison.csv",
    )
    baseline_report_path = resolve_output_path(
        args.baseline_report_path,
        output_root / "baseline_eval_report.json",
    )
    comparison_report_path = resolve_output_path(
        args.comparison_report_path,
        output_root / "adapter_comparison_report.json",
    )

    dataset_builder = MultilingualDatasetBuilder(
        data_root=args.data_root,
        dataset_repo=args.dataset_repo,
        dataset_revision=args.dataset_revision,
        hf_token=hf_token,
        cache_dir=args.cache_dir,
        seed=cfg.seed,
    )

    baseline_model_ref = args.baseline_model or cfg.model_id
    baseline_model, baseline_processor = load_model_and_processor(
        baseline_model_ref,
        load_in_4bit=args.load_in_4bit,
        cache_dir=args.cache_dir,
    )

    rows: list[dict[str, Any]] = []
    baseline_per_language: dict[str, dict[str, Any]] = {}
    comparison_per_language: dict[str, dict[str, Any]] = {}

    try:
        for lang_code in languages:
            lang_cfg = SUPPORTED_LANGUAGES[lang_code]
            display_name = lang_cfg.display_name
            language_name = lang_cfg.language_name
            logger.info("Comparing baseline and adapter for %s", lang_code)

            try:
                dataset = dataset_builder.load_language(lang_code, lang_cfg, augment=False)
            except Exception as exc:
                error = {"language": lang_code, "error": "dataset_load_failed", "details": str(exc)}
                baseline_per_language[lang_code] = error
                comparison_per_language[lang_code] = {
                    "baseline": error,
                    "adapter": error,
                    "delta": {},
                }
                logger.error("Skipping %s because the dataset could not be loaded: %s", lang_code, exc)
                continue

            split_ds = get_split(dataset, args.split)
            if split_ds is None or len(split_ds) == 0:
                error = {"language": lang_code, "error": f"no_{args.split}_data"}
                baseline_per_language[lang_code] = error
                comparison_per_language[lang_code] = {
                    "baseline": error,
                    "adapter": error,
                    "delta": {},
                }
                logger.warning("Skipping %s because the %s split is empty.", lang_code, args.split)
                continue

            n = min(len(split_ds), args.max_eval_samples)
            selected = split_ds.shuffle(seed=cfg.seed).select(range(n))
            examples = [selected[idx] for idx in range(len(selected))]

            baseline_predictions: list[str] = []
            for example in examples:
                try:
                    prediction = generate_response(
                        baseline_model,
                        baseline_processor,
                        prompt=example["input"],
                        language_name=language_name,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as exc:
                    logger.error("Baseline generation error for %s: %s", lang_code, exc)
                    prediction = ""
                baseline_predictions.append(prediction)

            baseline_metrics = {
                "language": lang_code,
                "display_name": display_name,
                "language_name": language_name,
                "resource_level": lang_cfg.resource_level,
                **summarize_metrics(baseline_predictions, examples),
            }
            baseline_per_language[lang_code] = baseline_metrics

            adapter_path = resolve_adapter_path(output_root, lang_code)
            if adapter_path is None:
                adapter_metrics = {"language": lang_code, "error": "adapter_not_found"}
                comparison_per_language[lang_code] = {
                    "baseline": baseline_metrics,
                    "adapter": adapter_metrics,
                    "delta": {},
                }
                for index, example in enumerate(examples):
                    rows.append(
                        {
                            "language": lang_code,
                            "display_name": display_name,
                            "split": args.split,
                            "example_index": index,
                            "question": example["input"],
                            "reference_answer": example["output"],
                            "baseline_prediction": baseline_predictions[index],
                            "adapter_prediction": "",
                            "baseline_exact_match": exact_match(
                                baseline_predictions[index],
                                example["output"],
                            ),
                            "adapter_exact_match": "",
                        }
                    )
                logger.warning("Adapter not found for %s; baseline rows were still written.", lang_code)
                continue

            adapter_metadata = load_adapter_metadata(adapter_path)
            adapter_base_model_ref = (
                args.adapter_base_model
                or adapter_metadata.get("base_model")
                or cfg.model_id
            )

            adapter_model = None
            try:
                adapter_model, adapter_processor = load_model_and_processor(
                    adapter_base_model_ref,
                    processor_ref=str(adapter_path),
                    load_in_4bit=args.load_in_4bit,
                    cache_dir=args.cache_dir,
                )
                adapter_model = PeftModel.from_pretrained(adapter_model, str(adapter_path))
                adapter_model.eval()
            except Exception as exc:
                adapter_metrics = {
                    "language": lang_code,
                    "error": "adapter_load_failed",
                    "details": str(exc),
                }
                comparison_per_language[lang_code] = {
                    "baseline": baseline_metrics,
                    "adapter": adapter_metrics,
                    "delta": {},
                }
                for index, example in enumerate(examples):
                    rows.append(
                        {
                            "language": lang_code,
                            "display_name": display_name,
                            "split": args.split,
                            "example_index": index,
                            "question": example["input"],
                            "reference_answer": example["output"],
                            "baseline_prediction": baseline_predictions[index],
                            "adapter_prediction": "",
                            "baseline_exact_match": exact_match(
                                baseline_predictions[index],
                                example["output"],
                            ),
                            "adapter_exact_match": "",
                        }
                    )
                logger.error("Skipping adapter generations for %s: %s", lang_code, exc)
                if adapter_model is not None:
                    unload_model(adapter_model)
                continue

            try:
                adapter_predictions: list[str] = []
                for example in examples:
                    try:
                        prediction = generate_response(
                            adapter_model,
                            adapter_processor,
                            prompt=example["input"],
                            language_name=language_name,
                            max_new_tokens=args.max_new_tokens,
                        )
                    except Exception as exc:
                        logger.error("Adapter generation error for %s: %s", lang_code, exc)
                        prediction = ""
                    adapter_predictions.append(prediction)
            finally:
                unload_model(adapter_model)

            adapter_metrics = {
                "language": lang_code,
                "display_name": display_name,
                "language_name": language_name,
                "resource_level": lang_cfg.resource_level,
                **summarize_metrics(adapter_predictions, examples),
            }
            comparison_per_language[lang_code] = {
                "baseline": baseline_metrics,
                "adapter": adapter_metrics,
                "delta": compute_metric_delta(baseline_metrics, adapter_metrics),
            }

            for index, example in enumerate(examples):
                rows.append(
                    {
                        "language": lang_code,
                        "display_name": display_name,
                        "split": args.split,
                        "example_index": index,
                        "question": example["input"],
                        "reference_answer": example["output"],
                        "baseline_prediction": baseline_predictions[index],
                        "adapter_prediction": adapter_predictions[index],
                        "baseline_exact_match": exact_match(
                            baseline_predictions[index],
                            example["output"],
                        ),
                        "adapter_exact_match": exact_match(
                            adapter_predictions[index],
                            example["output"],
                        ),
                    }
                )
    finally:
        unload_model(baseline_model)

    baseline_report = {
        "model": baseline_model_ref,
        "split": args.split,
        "per_language": baseline_per_language,
        "aggregate": compute_aggregate(baseline_per_language),
    }
    comparison_report = {
        "baseline_model": baseline_model_ref,
        "adapter_base_model_override": args.adapter_base_model,
        "split": args.split,
        "csv_path": str(csv_path),
        "baseline_report_path": str(baseline_report_path),
        "per_language": comparison_per_language,
        "aggregate": {
            "baseline": baseline_report["aggregate"],
            "adapter": compute_aggregate(
                {
                    lang_code: result["adapter"]
                    for lang_code, result in comparison_per_language.items()
                }
            ),
        },
    }
    comparison_report["aggregate"]["delta"] = compute_aggregate_delta(
        comparison_report["aggregate"]["baseline"],
        comparison_report["aggregate"]["adapter"],
    )

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "language",
                "display_name",
                "split",
                "example_index",
                "question",
                "reference_answer",
                "baseline_prediction",
                "adapter_prediction",
                "baseline_exact_match",
                "adapter_exact_match",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(baseline_report_path, "w", encoding="utf-8") as handle:
        json.dump(baseline_report, handle, indent=2, ensure_ascii=False)

    with open(comparison_report_path, "w", encoding="utf-8") as handle:
        json.dump(comparison_report, handle, indent=2, ensure_ascii=False)

    logger.info("Saved comparison CSV to %s", csv_path)
    logger.info("Saved baseline report to %s", baseline_report_path)
    logger.info("Saved comparison report to %s", comparison_report_path)


if __name__ == "__main__":
    main()
