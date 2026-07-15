"""
evaluation.py — Multilingual evaluation for adapter-target LoRA adapters
========================================================================

Evaluation design notes
-----------------------
Evaluation is keyed by adapter target. English and Swahili targets evaluate on
combined source leaves, while targets such as `aka_gha` and `lug_uga` evaluate
on a single source leaf.

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
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from config import TrainingConfig, SUPPORTED_LANGUAGES
from data_utils import MultilingualDatasetBuilder, get_split
from prompt_utils import build_hashie_messages

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
        "torch_dtype": dtype,
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
    common = Counter(pred_tokens) & Counter(gt_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
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


def _normalize_for_char_metrics(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _char_ngram_f1(prediction: str, reference: str, n: int = 3) -> float:
    """Character n-gram F1, useful for morphologically rich languages."""

    def ngrams(text: str) -> Counter[str]:
        normalized = _normalize_for_char_metrics(text)
        if len(normalized) < n:
            return Counter([normalized]) if normalized else Counter()
        return Counter(normalized[i : i + n] for i in range(len(normalized) - n + 1))

    pred_grams = ngrams(prediction)
    ref_grams = ngrams(reference)
    if not pred_grams or not ref_grams:
        return 0.0

    overlap = sum((pred_grams & ref_grams).values())
    precision = overlap / sum(pred_grams.values())
    recall = overlap / sum(ref_grams.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _letters(text: str) -> list[str]:
    return [char for char in text if char.isalpha()]


def _is_latin_letter(char: str) -> bool:
    try:
        return unicodedata.name(char).startswith("LATIN")
    except ValueError:
        return False


def _script_match_ratio(text: str, target_script: str | None) -> float:
    letters = _letters(text)
    if not letters:
        return 0.0

    if target_script == "geez":
        matches = sum(1 for char in letters if "\u1200" <= char <= "\u137f")
    elif target_script == "latin":
        matches = sum(1 for char in letters if _is_latin_letter(char))
    else:
        return 1.0
    return matches / len(letters)


def _latin_leak_ratio(text: str) -> float:
    letters = _letters(text)
    if not letters:
        return 0.0
    latin = sum(1 for char in letters if _is_latin_letter(char))
    return latin / len(letters)


def _repetition_ngram_rate(text: str, n: int = 4) -> float:
    tokens = text.lower().split()
    grams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)


def _length_ratio(prediction: str, reference: str) -> float:
    return len(prediction.split()) / max(len(reference.split()), 1)


def _quality_flag(prediction: str, reference: str, target_script: str | None) -> float:
    if not prediction.strip():
        return 1.0
    if _repetition_ngram_rate(prediction) > 0.2:
        return 1.0
    if _script_match_ratio(prediction, target_script) < 0.85:
        return 1.0
    if _length_ratio(prediction, reference) > 4.0:
        return 1.0
    return 0.0


class MultilingualEvaluator:
    """
    Evaluate adapters on the `test-*` split for each configured adapter target.

    The adapter path and dataset path are both resolved from the same target ID,
    which avoids accidental cross-evaluation such as testing `adapter_eng` on
    non-English data.
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
        """Lazily load the shared base model once for evaluation."""
        if self._base_model is not None:
            return

        self._base_model = AutoModelForImageTextToText.from_pretrained(
            self.cfg.model_id,
            **build_model_load_kwargs(self.cfg),
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
        messages = build_hashie_messages(
            user_text=prompt,
            language_name=language_name,
        )
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
        Run evaluation for one adapter target and return aggregate metrics.

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
        target_script = lang_cfg.script if lang_cfg else None

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
        sample_predictions = []
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
            if len(sample_predictions) < 5:
                sample_predictions.append(
                    {
                        "input": prompt,
                        "reference": reference,
                        "prediction": prediction,
                        "char_3gram_f1": round(
                            _char_ngram_f1(prediction, reference),
                            4,
                        ),
                        "script_match_ratio": round(
                            _script_match_ratio(prediction, target_script),
                            4,
                        ),
                        "repetition_4gram_rate": round(
                            _repetition_ngram_rate(prediction),
                            4,
                        ),
                        "quality_flag": _quality_flag(
                            prediction,
                            reference,
                            target_script,
                        ),
                    }
                )

        exact_matches = [_exact_match(pred, ref) for pred, ref in zip(predictions, references)]
        f1_scores = [_token_f1(pred, ref) for pred, ref in zip(predictions, references)]
        rouge_scores = [_rouge_l(pred, ref) for pred, ref in zip(predictions, references)]
        char_scores = [
            _char_ngram_f1(pred, ref) for pred, ref in zip(predictions, references)
        ]
        script_scores = [
            _script_match_ratio(pred, target_script) for pred in predictions
        ]
        latin_scores = [_latin_leak_ratio(pred) for pred in predictions]
        repetition_scores = [_repetition_ngram_rate(pred) for pred in predictions]
        length_scores = [
            _length_ratio(pred, ref) for pred, ref in zip(predictions, references)
        ]
        quality_flags = [
            _quality_flag(pred, ref, target_script)
            for pred, ref in zip(predictions, references)
        ]

        results = {
            "language": lang_code,
            "display_name": display_name,
            "language_name": language_name,
            "n_evaluated": n,
            "exact_match": round(sum(exact_matches) / n, 4),
            "f1_token": round(sum(f1_scores) / n, 4),
            "rouge_l": round(sum(rouge_scores) / n, 4),
            "char_3gram_f1": round(sum(char_scores) / n, 4),
            "script_match_ratio": round(sum(script_scores) / n, 4),
            "latin_leak_ratio": round(sum(latin_scores) / n, 4),
            "repetition_4gram_rate": round(sum(repetition_scores) / n, 4),
            "length_ratio": round(sum(length_scores) / n, 4),
            "quality_flag_rate": round(sum(quality_flags) / n, 4),
            "invalid_rate": round(invalid_count / n, 4),
            "resource_level": lang_cfg.resource_level if lang_cfg else "unknown",
            "sample_predictions": sample_predictions,
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
                "macro_avg_char_3gram_f1": round(
                    sum(result["char_3gram_f1"] for result in valid_results) / len(valid_results),
                    4,
                ),
                "macro_avg_script_match_ratio": round(
                    sum(result["script_match_ratio"] for result in valid_results) / len(valid_results),
                    4,
                ),
                "macro_avg_latin_leak_ratio": round(
                    sum(result["latin_leak_ratio"] for result in valid_results) / len(valid_results),
                    4,
                ),
                "macro_avg_repetition_4gram_rate": round(
                    sum(result["repetition_4gram_rate"] for result in valid_results) / len(valid_results),
                    4,
                ),
                "macro_avg_length_ratio": round(
                    sum(result["length_ratio"] for result in valid_results) / len(valid_results),
                    4,
                ),
                "macro_avg_quality_flag_rate": round(
                    sum(result["quality_flag_rate"] for result in valid_results) / len(valid_results),
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
                    f"Char3={metrics['char_3gram_f1']:.3f}  "
                    f"Script={metrics['script_match_ratio']:.3f}  "
                    f"Rep={metrics['repetition_4gram_rate']:.3f}  "
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
                f"Char3={aggregate['macro_avg_char_3gram_f1']:.3f}  "
                f"Script={aggregate['macro_avg_script_match_ratio']:.3f}  "
                f"Rep={aggregate['macro_avg_repetition_4gram_rate']:.3f}  "
                f"Invalid={aggregate['macro_avg_invalid_rate']:.3f}"
            )
        print("═" * 72 + "\n")
