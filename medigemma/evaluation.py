"""
evaluation.py — Multilingual evaluation for per-dataset-leaf LoRA adapters
===========================================================================

Evaluation design notes
-----------------------
Evaluation is keyed by dataset leaf for the same reason training is: the real
unit of data is `eng_uga`, `swa_ken`, `aka_gha`, and so on, not just `eng` or
`swa`.

This module deliberately reloads test data through the shared dataset builder so
that training and evaluation read from the same source abstraction:
* Hub shard repo
* local shard mirror
* local `save_to_disk()` mirror

Metrics kept here are intentionally lightweight and dependency-minimal:
* Exact Match
* Token F1
* ROUGE-L

That keeps the evaluation path usable even before optional semantic metrics are
added back.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from config import TrainingConfig, SUPPORTED_LANGUAGES
from data_utils import MultilingualDatasetBuilder, get_split

logger = logging.getLogger(__name__)


def resolve_adapter_path(adapter_root: Path, lang_code: str) -> Optional[Path]:
    """
    Resolve either a flat PEFT save or a named-adapter nested save.

    PEFT commonly stores non-default adapters under a child folder named after
    the adapter, e.g. `adapter_amh_eth/amh_eth/adapter_config.json`.
    """

    adapter_dir = adapter_root / f"adapter_{lang_code}"
    if not adapter_dir.exists():
        return None

    flat_config = adapter_dir / "adapter_config.json"
    nested_dir = adapter_dir / lang_code
    nested_config = nested_dir / "adapter_config.json"

    if flat_config.exists():
        return adapter_dir
    if nested_config.exists():
        return nested_dir

    return adapter_dir


def _token_f1(prediction: str, ground_truth: str) -> float:
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


def _exact_match(prediction: str, ground_truth: str) -> float:
    return float(prediction.strip().lower() == ground_truth.strip().lower())


def _rouge_l(prediction: str, reference: str) -> float:
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


class MultilingualEvaluator:
    """
    Evaluate adapters on the `test-*` split for each configured dataset leaf.

    The adapter path and dataset path are both resolved from the same leaf ID,
    which avoids accidental cross-evaluation such as testing `adapter_eng_uga`
    on `eng_gha` data.
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        adapter_root: Path,
        dataset_builder: Optional[MultilingualDatasetBuilder] = None,
    ):
        self.cfg = cfg
        self.adapter_root = adapter_root
        self.dataset_builder = dataset_builder
        self._base_model = None
        self._processor = None

    def _ensure_base_loaded(self):
        """Lazily load the shared quantized base model once for evaluation."""
        if self._base_model is not None:
            return

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self._base_model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self._processor = AutoProcessor.from_pretrained(self.cfg.model_id)
        self._processor.tokenizer.padding_side = "right"

    def _load_adapter(self, lang_code: str):
        """Attach a trained leaf-specific adapter onto the shared base model."""
        self._ensure_base_loaded()
        adapter_path = resolve_adapter_path(self.adapter_root, lang_code)
        if adapter_path is None:
            logger.warning("Adapter not found for %s; skipping evaluation.", lang_code)
            return None, None

        model = PeftModel.from_pretrained(self._base_model, str(adapter_path))
        model.eval()
        return model, self._processor

    def _generate(
        self,
        model,
        processor,
        prompt: str,
        language_name: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate a response for a single prompt using the Gemma chat template.

        The system instruction keeps evaluation generation aligned with the
        target language name recorded in the registry.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful sexual and reproductive health assistant. "
                    f"Answer in {language_name}."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        text = processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = processor(text=text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
        return processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def evaluate_language(
        self,
        lang_code: str,
        max_eval_samples: int = 200,
    ) -> dict[str, Any]:
        """
        Run evaluation for one dataset leaf and return aggregate metrics.

        The test split is subsampled for speed when `max_eval_samples` is lower
        than the full test size.
        """
        model, processor = self._load_adapter(lang_code)
        if model is None:
            return {"language": lang_code, "error": "adapter_not_found"}

        if self.dataset_builder is None:
            return {"language": lang_code, "error": "no_dataset_builder"}

        lang_cfg = SUPPORTED_LANGUAGES.get(lang_code)
        language_name = lang_cfg.language_name if lang_cfg else lang_code
        display_name = lang_cfg.display_name if lang_cfg else lang_code

        try:
            dataset = self.dataset_builder.load_language(lang_code, lang_cfg, augment=False)
        except Exception as exc:
            logger.warning("No evaluation data for %s: %s", lang_code, exc)
            return {"language": lang_code, "error": "no_test_data", "details": str(exc)}

        test_ds = get_split(dataset, "test")
        if test_ds is None or len(test_ds) == 0:
            logger.warning("No test split for %s. Skipping.", lang_code)
            return {"language": lang_code, "error": "no_test_data"}

        n = min(len(test_ds), max_eval_samples)
        test_ds = test_ds.shuffle(seed=42).select(range(n))

        predictions = []
        references = []
        invalid_count = 0

        for example in test_ds:
            prompt = example["input"]
            reference = example["output"]
            try:
                prediction = self._generate(
                    model,
                    processor,
                    prompt,
                    language_name=language_name,
                )
            except Exception as exc:
                logger.error("Generation error for %s: %s", lang_code, exc)
                prediction = ""
                invalid_count += 1

            predictions.append(prediction)
            references.append(reference)

        exact_matches = [_exact_match(pred, ref) for pred, ref in zip(predictions, references)]
        f1_scores = [_token_f1(pred, ref) for pred, ref in zip(predictions, references)]
        rouge_scores = [_rouge_l(pred, ref) for pred, ref in zip(predictions, references)]

        results = {
            "language": lang_code,
            "display_name": display_name,
            "language_name": language_name,
            "n_evaluated": n,
            "exact_match": round(sum(exact_matches) / n, 4),
            "f1_token": round(sum(f1_scores) / n, 4),
            "rouge_l": round(sum(rouge_scores) / n, 4),
            "invalid_rate": round(invalid_count / n, 4),
            "resource_level": lang_cfg.resource_level if lang_cfg else "unknown",
        }

        if "label" in test_ds.column_names:
            mcq_hits = 0
            for prediction, example in zip(predictions, test_ds):
                if example["label"].strip().lower() in prediction.lower():
                    mcq_hits += 1
            results["mcq_accuracy"] = round(mcq_hits / n, 4)

        logger.info(
            "[%s] EM=%.3f F1=%.3f ROUGE-L=%.3f Invalid=%.3f",
            lang_code,
            results["exact_match"],
            results["f1_token"],
            results["rouge_l"],
            results["invalid_rate"],
        )
        return results

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

        valid_results = [result for result in per_language.values() if "error" not in result]
        if valid_results:
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
        else:
            aggregate = {}

        return {"per_language": per_language, "aggregate": aggregate}

    def save_report(self, results: dict, output_path: Path) -> None:
        """Persist the JSON report and print a compact console summary."""
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
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
            aggregate = results["aggregate"]
            print("─" * 72)
            print(
                f"  {'MACRO AVG':<12} "
                f"EM={aggregate['macro_avg_exact_match']:.3f}  "
                f"F1={aggregate['macro_avg_f1']:.3f}  "
                f"ROUGE-L={aggregate['macro_avg_rouge_l']:.3f}  "
                f"Invalid={aggregate['macro_avg_invalid_rate']:.3f}"
            )
        print("═" * 72 + "\n")
