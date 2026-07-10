"""
evaluation.py — Multilingual evaluation for adapter-target LoRA adapters
========================================================================

Model: epfl-llm/meditron-7b (LLaMA-based, text-only causal LM)

Changes from the original
--------------------------
* AutoModelForImageTextToText  → AutoModelForCausalLM
* AutoProcessor                → AutoTokenizer
* processor.tokenizer.*        → tokenizer.* throughout
* torch_dtype=                 → dtype=  (deprecation fix)
* pad_token set to eos_token   (LLaMA tokenizers ship without a pad token)
* _generate uses plain tokenizer encode/decode instead of processor(text=...)
* Prompt rendering now follows the shared Hashie LLaMA-style template

Evaluation design notes
-----------------------
Evaluation is keyed by adapter target. English and Swahili targets evaluate on
combined source leaves, while targets such as `aka_gha` and `lug_uga` evaluate
on a single source leaf.

Metrics are intentionally lightweight and dependency-minimal:
  * Exact Match
  * Token F1
  * ROUGE-L
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import TrainingConfig, SUPPORTED_LANGUAGES
from data_utils import MultilingualDatasetBuilder, get_split
from prompt_utils import render_meditron_chat

logger = logging.getLogger(__name__)

PRECISION_CHOICES = ("full_lora", "qlora")


def select_model_dtype() -> torch.dtype:
    """Pick the best non-quantized dtype supported by the current hardware."""

    if not torch.cuda.is_available():
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_model_load_kwargs(cfg: TrainingConfig) -> dict:
    """Build evaluation model-loading kwargs for full LoRA or QLoRA adapters."""

    dtype = select_model_dtype()
    kwargs = {
        "device_map": "auto",
        "dtype": dtype,
        "trust_remote_code": True,
    }

    if cfg.training_precision == "qlora":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16
            ),
        )
        logger.info("Evaluating with QLoRA 4-bit base model loading.")
    elif cfg.training_precision == "full_lora":
        logger.info("Evaluating with full-precision base model dtype=%s.", dtype)
    else:
        raise ValueError(
            f"Unsupported training precision '{cfg.training_precision}'. "
            f"Expected one of: {', '.join(PRECISION_CHOICES)}"
        )

    return kwargs


# ---------------------------------------------------------------------------
# Adapter path resolution
# ---------------------------------------------------------------------------

def resolve_adapter_path(adapter_root: Path, lang_code: str) -> Optional[Path]:
    """
    Resolve either a flat PEFT save or a named-adapter nested save.

    PEFT commonly stores non-default adapters under a child folder named after
    the adapter, e.g. `adapter_amh_eth/amh_eth/adapter_config.json`.
    """
    adapter_dir = adapter_root / f"adapter_{lang_code}"
    if not adapter_dir.exists():
        return None

    flat_config   = adapter_dir / "adapter_config.json"
    nested_dir    = adapter_dir / lang_code
    nested_config = nested_dir  / "adapter_config.json"

    if flat_config.exists():
        return adapter_dir
    if nested_config.exists():
        return nested_dir

    return adapter_dir


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _token_f1(prediction: str, ground_truth: str) -> float:
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


def _exact_match(prediction: str, ground_truth: str) -> float:
    return float(prediction.strip().lower() == ground_truth.strip().lower())


def _rouge_l(prediction: str, reference: str) -> float:
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


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class MultilingualEvaluator:
    """
    Evaluate adapters on the `test` split for each configured adapter target.

    The adapter path and dataset path are both resolved from the same target ID,
    which prevents accidental cross-evaluation such as testing `adapter_eng` on
    non-English data.
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        adapter_root: Path,
        dataset_builder: Optional[MultilingualDatasetBuilder] = None,
    ):
        self.cfg             = cfg
        self.adapter_root    = adapter_root
        self.dataset_builder = dataset_builder
        self._base_model: Optional[AutoModelForCausalLM] = None
        self._tokenizer:  Optional[AutoTokenizer]        = None

    # ------------------------------------------------------------------
    # Base model loading (lazy, shared across all adapters)
    # ------------------------------------------------------------------

    def _ensure_base_loaded(self):
        """Lazily load the shared base model once for evaluation."""
        if self._base_model is not None:
            return

        logger.info("Loading base model for evaluation: %s", self.cfg.model_id)
        self._base_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id,
            **build_model_load_kwargs(self.cfg),
        )
        self._base_model.config.use_cache = True  # enable KV cache for inference

        logger.info("Loading tokenizer for evaluation: %s", self.cfg.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_id,
            trust_remote_code=True,
        )

        # LLaMA tokenizers ship without a pad token; use EOS as pad.
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token    = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
            logger.info(
                "pad_token set to eos_token (%s)", self._tokenizer.eos_token
            )

        self._tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    # Adapter loading
    # ------------------------------------------------------------------

    def _load_adapter(self, lang_code: str):
        """Attach a trained leaf-specific adapter onto the shared base model."""
        self._ensure_base_loaded()
        adapter_path = resolve_adapter_path(self.adapter_root, lang_code)
        if adapter_path is None:
            logger.warning(
                "Adapter not found for %s; skipping evaluation.", lang_code
            )
            return None, None

        model = PeftModel.from_pretrained(self._base_model, str(adapter_path))
        model.eval()
        return model, self._tokenizer

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def _generate(
        self,
        model,
        tokenizer: AutoTokenizer,
        prompt: str,
        language_name: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate a response for a single prompt.

        Uses the same Hashie LLaMA-style prompt template that was used during
        Meditron data curation and training.
        """
        _ = language_name  # Kept for API compatibility with existing callers.
        text = render_meditron_chat(
            user_text=prompt,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_seq_length,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # Per-language evaluation
    # ------------------------------------------------------------------

    def evaluate_language(
        self,
        lang_code: str,
        max_eval_samples: int = 200,
    ) -> dict[str, Any]:
        """
        Run evaluation for one adapter target and return aggregate metrics.

        The test split is subsampled for speed when `max_eval_samples` is
        lower than the full test size.
        """
        model, tokenizer = self._load_adapter(lang_code)
        if model is None:
            return {"language": lang_code, "error": "adapter_not_found"}

        if self.dataset_builder is None:
            return {"language": lang_code, "error": "no_dataset_builder"}

        lang_cfg      = SUPPORTED_LANGUAGES.get(lang_code)
        language_name = lang_cfg.language_name if lang_cfg else lang_code
        display_name  = lang_cfg.display_name  if lang_cfg else lang_code

        try:
            dataset = self.dataset_builder.load_language(
                lang_code, lang_cfg, augment=False
            )
        except Exception as exc:
            logger.warning("No evaluation data for %s: %s", lang_code, exc)
            return {
                "language": lang_code,
                "error":    "no_test_data",
                "details":  str(exc),
            }

        test_ds = get_split(dataset, "test")
        if test_ds is None or len(test_ds) == 0:
            logger.warning("No test split for %s. Skipping.", lang_code)
            return {"language": lang_code, "error": "no_test_data"}

        n       = min(len(test_ds), max_eval_samples)
        test_ds = test_ds.shuffle(seed=42).select(range(n))

        predictions   = []
        references    = []
        invalid_count = 0

        for example in test_ds:
            prompt    = example["input"]
            reference = example["output"]
            try:
                prediction = self._generate(
                    model,
                    tokenizer,
                    prompt,
                    language_name=language_name,
                )
            except Exception as exc:
                logger.error("Generation error for %s: %s", lang_code, exc)
                prediction    = ""
                invalid_count += 1

            predictions.append(prediction)
            references.append(reference)

        exact_matches = [
            _exact_match(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        f1_scores = [
            _token_f1(pred, ref)
            for pred, ref in zip(predictions, references)
        ]
        rouge_scores = [
            _rouge_l(pred, ref)
            for pred, ref in zip(predictions, references)
        ]

        results = {
            "language":      lang_code,
            "display_name":  display_name,
            "language_name": language_name,
            "n_evaluated":   n,
            "exact_match":   round(sum(exact_matches) / n, 4),
            "f1_token":      round(sum(f1_scores)     / n, 4),
            "rouge_l":       round(sum(rouge_scores)  / n, 4),
            "invalid_rate":  round(invalid_count       / n, 4),
            "resource_level": lang_cfg.resource_level if lang_cfg else "unknown",
        }

        if "label" in test_ds.column_names:
            mcq_hits = 0
            for prediction, example in zip(predictions, test_ds):
                if example["label"].strip().lower() in prediction.lower():
                    mcq_hits += 1
            results["mcq_accuracy"] = round(mcq_hits / n, 4)

        logger.info(
            "[%s] EM=%.3f  F1=%.3f  ROUGE-L=%.3f  Invalid=%.3f",
            lang_code,
            results["exact_match"],
            results["f1_token"],
            results["rouge_l"],
            results["invalid_rate"],
        )
        return results

    # ------------------------------------------------------------------
    # Aggregate evaluation across all languages
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        languages: list[str],
        max_eval_samples: int = 200,
    ) -> dict[str, Any]:
        """Evaluate all requested dataset leaves and compute macro averages."""
        per_language = {}
        for lang_code in languages:
            per_language[lang_code] = self.evaluate_language(
                lang_code,
                max_eval_samples=max_eval_samples,
            )

        valid_results = [
            r for r in per_language.values() if "error" not in r
        ]
        if valid_results:
            aggregate = {
                "macro_avg_exact_match": round(
                    sum(r["exact_match"]  for r in valid_results) / len(valid_results), 4
                ),
                "macro_avg_f1": round(
                    sum(r["f1_token"]     for r in valid_results) / len(valid_results), 4
                ),
                "macro_avg_rouge_l": round(
                    sum(r["rouge_l"]      for r in valid_results) / len(valid_results), 4
                ),
                "macro_avg_invalid_rate": round(
                    sum(r["invalid_rate"] for r in valid_results) / len(valid_results), 4
                ),
                "n_languages_evaluated": len(valid_results),
            }
        else:
            aggregate = {}

        return {"per_language": per_language, "aggregate": aggregate}

    # ------------------------------------------------------------------
    # Report persistence
    # ------------------------------------------------------------------

    def save_report(self, results: dict, output_path: Path) -> None:
        """Persist the JSON report and print a compact console summary."""
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        logger.info("Evaluation report saved to %s", output_path)

        print("\n" + "═" * 72)
        print("MULTILINGUAL SRH EVALUATION SUMMARY")
        print("═" * 72)
        for lang_code, metrics in results["per_language"].items():
            if "error" in metrics:
                print(f"  {lang_code:<12} ERROR: {metrics['error']}")
            else:
                print(
                    f"  {lang_code:<12} "
                    f"EM={metrics['exact_match']:.3f}  "
                    f"F1={metrics['f1_token']:.3f}  "
                    f"ROUGE-L={metrics['rouge_l']:.3f}  "
                    f"Invalid={metrics['invalid_rate']:.3f}"
                )

        if results.get("aggregate"):
            agg = results["aggregate"]
            print("─" * 72)
            print(
                f"  {'MACRO AVG':<12} "
                f"EM={agg['macro_avg_exact_match']:.3f}  "
                f"F1={agg['macro_avg_f1']:.3f}  "
                f"ROUGE-L={agg['macro_avg_rouge_l']:.3f}  "
                f"Invalid={agg['macro_avg_invalid_rate']:.3f}"
            )
        print("═" * 72 + "\n")
