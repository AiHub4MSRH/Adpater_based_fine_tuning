"""
compare_models.py — Compare baseline meditron-7b generations against LoRA adapters
====================================================================================

Model: epfl-llm/meditron-7b (LLaMA-based, text-only causal LM)

Changes from original
---------------------
* AutoModelForImageTextToText → AutoModelForCausalLM
* AutoProcessor               → AutoTokenizer
* torch_dtype=                → dtype=  (deprecation fix)
* pad_token set to eos_token  (LLaMA ships without one)
* processor(text=...) calls   → tokenizer(...) calls
* processor.tokenizer.*       → tokenizer.* throughout
* Prompt rendering now follows the shared Hashie LLaMA-style template

This script runs the same evaluation prompts through:
1. a baseline meditron-7b checkpoint
2. each leaf-specific LoRA adapter

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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import SUPPORTED_LANGUAGES, TrainingConfig, expand_language_selection
from data_utils import MultilingualDatasetBuilder, get_split
from evaluation import resolve_adapter_path
from prompt_utils import render_meditron_chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

METRIC_KEYS = ("exact_match", "f1_token", "rouge_l", "invalid_rate", "mcq_accuracy")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare baseline meditron-7b generations against saved LoRA adapters"
    )
    parser.add_argument("--languages",      nargs="+", default=list(SUPPORTED_LANGUAGES.keys()),
                        help="Dataset leaves or grouped selections such as `eng` and `swa`.")
    parser.add_argument("--data_root",      type=str, default="./data",
                        help="Local mirror root for save_to_disk data or shard trees.")
    parser.add_argument("--dataset_repo",   type=str, default=None,
                        help="Optional HuggingFace dataset repo id when local data is unavailable.")
    parser.add_argument("--dataset_revision", type=str, default=None,
                        help="Optional dataset revision, branch, or commit.")
    parser.add_argument("--cache_dir",      type=str, default=None,
                        help="Optional cache directory for dataset and model downloads.")
    parser.add_argument("--output_root",    type=str, default="./adapters",
                        help="Root directory containing saved adapter_* folders.")
    parser.add_argument("--baseline_model", type=str, default=None,
                        help="Baseline model id or local path to compare against adapters.")
    parser.add_argument("--adapter_base_model", type=str, default=None,
                        help="Optional override for the base model used before attaching each adapter.")
    parser.add_argument("--split",          type=str, choices=("train", "dev", "test"), default="test",
                        help="Dataset split to compare on. Defaults to `test`.")
    parser.add_argument("--max_eval_samples", type=int, default=200,
                        help="Maximum number of examples per dataset leaf.")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate per example.")
    parser.add_argument("--load_in_4bit",   action="store_true",
                        help="Load models in 4-bit mode to reduce GPU memory usage.")
    parser.add_argument("--hf_token",       type=str, default=None,
                        help="HuggingFace token. Falls back to HF_TOKEN env variable.")
    parser.add_argument("--csv_path",       type=str, default=None,
                        help="Optional output path for the row-level comparison CSV.")
    parser.add_argument("--baseline_report_path", type=str, default=None,
                        help="Optional output path for the baseline-only evaluation report.")
    parser.add_argument("--comparison_report_path", type=str, default=None,
                        help="Optional output path for the comparison report.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def build_model_kwargs(load_in_4bit: bool) -> dict[str, Any]:
    """Build shared model loading kwargs for baseline and adapter runs."""
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": dtype,           # replaces deprecated torch_dtype=
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


def load_model_and_tokenizer(
    model_ref: str,
    *,
    tokenizer_ref: Optional[str] = None,
    load_in_4bit: bool = False,
    cache_dir: Optional[str] = None,
):
    """Load a causal LM checkpoint and a matching tokenizer."""
    logger.info("Loading model from %s", model_ref)
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        cache_dir=cache_dir,
        **build_model_kwargs(load_in_4bit),
    )

    tokenizer_source = tokenizer_ref or model_ref
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, cache_dir=cache_dir, trust_remote_code=True
        )
    except Exception:
        if tokenizer_source == model_ref:
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            model_ref, cache_dir=cache_dir, trust_remote_code=True
        )

    # LLaMA tokenizers ship without a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.padding_side = "right"
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_response(
    model,
    tokenizer: AutoTokenizer,
    prompt: str,
    language_name: str,
    max_new_tokens: int,
) -> str:
    """Generate one response using deterministic decoding."""
    _ = language_name  # Kept for API compatibility with existing callers.
    text = render_meditron_chat(
        user_text=prompt,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens    = output_ids[0][prompt_length:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def exact_match(prediction: str, ground_truth: str) -> float:
    return float(prediction.strip().lower() == ground_truth.strip().lower())


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gt_tokens   = ground_truth.lower().split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common    = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    pred_tokens = prediction.lower().split()
    ref_tokens  = reference.lower().split()
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

    lcs       = dp[m][n]
    precision = lcs / n
    recall    = lcs / m
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def summarize_metrics(
    predictions: list[str],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate the same core metrics used by the evaluator."""
    references    = [ex["output"] for ex in examples]
    n             = len(examples)
    invalid_count = sum(1 for p in predictions if not p.strip())

    exact_scores  = [exact_match(p, r) for p, r in zip(predictions, references)]
    f1_scores     = [token_f1(p, r)    for p, r in zip(predictions, references)]
    rouge_scores  = [rouge_l(p, r)     for p, r in zip(predictions, references)]

    summary = {
        "n_evaluated":  n,
        "exact_match":  round(sum(exact_scores)  / n, 4),
        "f1_token":     round(sum(f1_scores)      / n, 4),
        "rouge_l":      round(sum(rouge_scores)   / n, 4),
        "invalid_rate": round(invalid_count        / n, 4),
    }

    if examples and "label" in examples[0]:
        mcq_hits = 0
        for pred, ex in zip(predictions, examples):
            label = str(ex.get("label", "")).strip().lower()
            if label and label in pred.lower():
                mcq_hits += 1
        summary["mcq_accuracy"] = round(mcq_hits / n, 4)

    return summary


def compute_metric_delta(
    baseline_metrics: dict[str, Any],
    adapter_metrics:  dict[str, Any],
) -> dict[str, float]:
    delta = {}
    for key in METRIC_KEYS:
        if key in baseline_metrics and key in adapter_metrics:
            delta[key] = round(adapter_metrics[key] - baseline_metrics[key], 4)
    return delta


def compute_aggregate(per_language: dict[str, dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in per_language.values() if "error" not in r]
    if not valid:
        return {}
    n = len(valid)
    agg = {
        "macro_avg_exact_match":   round(sum(r["exact_match"]  for r in valid) / n, 4),
        "macro_avg_f1":            round(sum(r["f1_token"]     for r in valid) / n, 4),
        "macro_avg_rouge_l":       round(sum(r["rouge_l"]      for r in valid) / n, 4),
        "macro_avg_invalid_rate":  round(sum(r["invalid_rate"] for r in valid) / n, 4),
        "n_languages_evaluated":   n,
    }
    if all("mcq_accuracy" in r for r in valid):
        agg["macro_avg_mcq_accuracy"] = round(
            sum(r["mcq_accuracy"] for r in valid) / n, 4
        )
    return agg


def compute_aggregate_delta(
    baseline_agg: dict[str, Any],
    adapter_agg:  dict[str, Any],
) -> dict[str, float]:
    key_pairs = {
        "macro_avg_exact_match":  "exact_match",
        "macro_avg_f1":           "f1_token",
        "macro_avg_rouge_l":      "rouge_l",
        "macro_avg_invalid_rate": "invalid_rate",
        "macro_avg_mcq_accuracy": "mcq_accuracy",
    }
    return {
        short: round(adapter_agg[agg_key] - baseline_agg[agg_key], 4)
        for agg_key, short in key_pairs.items()
        if agg_key in baseline_agg and agg_key in adapter_agg
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_output_path(explicit_path: Optional[str], default_path: Path) -> Path:
    path = Path(explicit_path) if explicit_path else default_path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_adapter_metadata(adapter_path: Path) -> dict[str, Any]:
    candidates = (
        adapter_path / "adapter_meta.json",
        adapter_path.parent / "adapter_meta.json",
    )
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, encoding="utf-8") as fh:
                return json.load(fh)
    return {}


def unload_model(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Run baseline-versus-adapter generation and save CSV plus reports."""
    args     = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    cfg      = TrainingConfig()

    try:
        languages = expand_language_selection(args.languages)
    except KeyError as exc:
        raise SystemExit(f"Unknown language selection: {exc.args[0]}") from exc

    output_root = Path(args.output_root)
    csv_path = resolve_output_path(
        args.csv_path, output_root / "adapter_baseline_comparison.csv"
    )
    baseline_report_path = resolve_output_path(
        args.baseline_report_path, output_root / "baseline_eval_report.json"
    )
    comparison_report_path = resolve_output_path(
        args.comparison_report_path, output_root / "adapter_comparison_report.json"
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
    baseline_model, baseline_tokenizer = load_model_and_tokenizer(
        baseline_model_ref,
        load_in_4bit=args.load_in_4bit,
        cache_dir=args.cache_dir,
    )

    rows:                  list[dict[str, Any]]       = []
    baseline_per_language: dict[str, dict[str, Any]]  = {}
    comparison_per_language: dict[str, dict[str, Any]] = {}

    try:
        for lang_code in languages:
            lang_cfg      = SUPPORTED_LANGUAGES[lang_code]
            display_name  = lang_cfg.display_name
            language_name = lang_cfg.language_name
            logger.info("Comparing baseline and adapter for %s", lang_code)

            # ── Load dataset ──────────────────────────────────────────────
            try:
                dataset = dataset_builder.load_language(lang_code, lang_cfg, augment=False)
            except Exception as exc:
                error = {"language": lang_code, "error": "dataset_load_failed", "details": str(exc)}
                baseline_per_language[lang_code]    = error
                comparison_per_language[lang_code]  = {"baseline": error, "adapter": error, "delta": {}}
                logger.error("Skipping %s — dataset load failed: %s", lang_code, exc)
                continue

            split_ds = get_split(dataset, args.split)
            if split_ds is None or len(split_ds) == 0:
                error = {"language": lang_code, "error": f"no_{args.split}_data"}
                baseline_per_language[lang_code]   = error
                comparison_per_language[lang_code] = {"baseline": error, "adapter": error, "delta": {}}
                logger.warning("Skipping %s — %s split is empty.", lang_code, args.split)
                continue

            n        = min(len(split_ds), args.max_eval_samples)
            selected = split_ds.shuffle(seed=cfg.seed).select(range(n))
            examples = [selected[i] for i in range(len(selected))]

            # ── Baseline generation ───────────────────────────────────────
            baseline_predictions: list[str] = []
            for example in examples:
                try:
                    prediction = generate_response(
                        baseline_model,
                        baseline_tokenizer,
                        prompt=example["input"],
                        language_name=language_name,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as exc:
                    logger.error("Baseline generation error for %s: %s", lang_code, exc)
                    prediction = ""
                baseline_predictions.append(prediction)

            baseline_metrics = {
                "language":      lang_code,
                "display_name":  display_name,
                "language_name": language_name,
                "resource_level": lang_cfg.resource_level,
                **summarize_metrics(baseline_predictions, examples),
            }
            baseline_per_language[lang_code] = baseline_metrics

            # ── Adapter generation ────────────────────────────────────────
            adapter_path = resolve_adapter_path(output_root, lang_code)
            if adapter_path is None:
                adapter_metrics = {"language": lang_code, "error": "adapter_not_found"}
                comparison_per_language[lang_code] = {
                    "baseline": baseline_metrics,
                    "adapter":  adapter_metrics,
                    "delta":    {},
                }
                for idx, example in enumerate(examples):
                    rows.append({
                        "language":             lang_code,
                        "display_name":         display_name,
                        "split":                args.split,
                        "example_index":        idx,
                        "question":             example["input"],
                        "reference_answer":     example["output"],
                        "baseline_prediction":  baseline_predictions[idx],
                        "adapter_prediction":   "",
                        "baseline_exact_match": exact_match(baseline_predictions[idx], example["output"]),
                        "adapter_exact_match":  "",
                    })
                logger.warning("Adapter not found for %s; baseline rows still written.", lang_code)
                continue

            adapter_metadata      = load_adapter_metadata(adapter_path)
            adapter_base_model_ref = (
                args.adapter_base_model
                or adapter_metadata.get("base_model")
                or cfg.model_id
            )

            adapter_model = None
            try:
                adapter_model, adapter_tokenizer = load_model_and_tokenizer(
                    adapter_base_model_ref,
                    tokenizer_ref=str(adapter_path),
                    load_in_4bit=args.load_in_4bit,
                    cache_dir=args.cache_dir,
                )
                adapter_model = PeftModel.from_pretrained(adapter_model, str(adapter_path))
                adapter_model.eval()
            except Exception as exc:
                adapter_metrics = {
                    "language": lang_code,
                    "error":    "adapter_load_failed",
                    "details":  str(exc),
                }
                comparison_per_language[lang_code] = {
                    "baseline": baseline_metrics,
                    "adapter":  adapter_metrics,
                    "delta":    {},
                }
                for idx, example in enumerate(examples):
                    rows.append({
                        "language":             lang_code,
                        "display_name":         display_name,
                        "split":                args.split,
                        "example_index":        idx,
                        "question":             example["input"],
                        "reference_answer":     example["output"],
                        "baseline_prediction":  baseline_predictions[idx],
                        "adapter_prediction":   "",
                        "baseline_exact_match": exact_match(baseline_predictions[idx], example["output"]),
                        "adapter_exact_match":  "",
                    })
                logger.error("Adapter load failed for %s: %s", lang_code, exc)
                if adapter_model is not None:
                    unload_model(adapter_model)
                continue

            try:
                adapter_predictions: list[str] = []
                for example in examples:
                    try:
                        prediction = generate_response(
                            adapter_model,
                            adapter_tokenizer,
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
                "language":      lang_code,
                "display_name":  display_name,
                "language_name": language_name,
                "resource_level": lang_cfg.resource_level,
                **summarize_metrics(adapter_predictions, examples),
            }
            comparison_per_language[lang_code] = {
                "baseline": baseline_metrics,
                "adapter":  adapter_metrics,
                "delta":    compute_metric_delta(baseline_metrics, adapter_metrics),
            }

            for idx, example in enumerate(examples):
                rows.append({
                    "language":             lang_code,
                    "display_name":         display_name,
                    "split":                args.split,
                    "example_index":        idx,
                    "question":             example["input"],
                    "reference_answer":     example["output"],
                    "baseline_prediction":  baseline_predictions[idx],
                    "adapter_prediction":   adapter_predictions[idx],
                    "baseline_exact_match": exact_match(baseline_predictions[idx], example["output"]),
                    "adapter_exact_match":  exact_match(adapter_predictions[idx],  example["output"]),
                })

    finally:
        unload_model(baseline_model)

    # ── Build and write reports ───────────────────────────────────────────
    baseline_report = {
        "model":        baseline_model_ref,
        "split":        args.split,
        "per_language": baseline_per_language,
        "aggregate":    compute_aggregate(baseline_per_language),
    }

    adapter_per_lang = {
        lc: result["adapter"]
        for lc, result in comparison_per_language.items()
    }
    comparison_report = {
        "baseline_model":              baseline_model_ref,
        "adapter_base_model_override": args.adapter_base_model,
        "split":                       args.split,
        "csv_path":                    str(csv_path),
        "baseline_report_path":        str(baseline_report_path),
        "per_language":                comparison_per_language,
        "aggregate": {
            "baseline": baseline_report["aggregate"],
            "adapter":  compute_aggregate(adapter_per_lang),
        },
    }
    comparison_report["aggregate"]["delta"] = compute_aggregate_delta(
        comparison_report["aggregate"]["baseline"],
        comparison_report["aggregate"]["adapter"],
    )

    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "language", "display_name", "split", "example_index",
            "question", "reference_answer",
            "baseline_prediction", "adapter_prediction",
            "baseline_exact_match", "adapter_exact_match",
        ])
        writer.writeheader()
        writer.writerows(rows)

    with open(baseline_report_path, "w", encoding="utf-8") as fh:
        json.dump(baseline_report, fh, indent=2, ensure_ascii=False)

    with open(comparison_report_path, "w", encoding="utf-8") as fh:
        json.dump(comparison_report, fh, indent=2, ensure_ascii=False)

    logger.info("Saved comparison CSV    → %s", csv_path)
    logger.info("Saved baseline report  → %s", baseline_report_path)
    logger.info("Saved comparison report → %s", comparison_report_path)


if __name__ == "__main__":
    main()
