#!/usr/bin/env python3
"""
retrieve_select_from_comparison.py - Darius-style answer-bank retrieval.

Darius' winning HASH pipeline is not only LoRA generation. For high-reuse
subsets it builds a same-subset Train+Val answer bank, retrieves candidate
answers, and then selects a bank answer when retrieval is the safer path. His
released code uses cached BGE embeddings, cross-encoder features, and LightGBM
rankers. This script keeps the same evaluation shape inside our repo without
requiring those private/heavy caches:

1. read a generation comparison CSV
2. build same-source Train+Dev answer banks
3. retrieve top-k bank answers using a BM25 question index
4. apply a source-aware selector inspired by Brainiac's direct/frozen policies
5. write a new CSV with retrieval_prediction and selected_prediction columns

The selector is reference-free. Reference answers are used only for evaluation
metrics and an optional retrieval-oracle diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import SOURCE_DATASETS, SUPPORTED_LANGUAGES, TrainingConfig

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

DARIUS_GENERATION_SOURCES = {"aka_gha", "amh_eth", "eng_gha", "swa_uga"}
DARIUS_DIRECT_RETRIEVAL_SOURCES = {"eng_ken", "swa_ken", "lug_uga"}
DARIUS_CONSERVATIVE_RETRIEVAL_SOURCES = {"eng_eth", "eng_uga"}


@dataclass(frozen=True)
class BankEntry:
    source_dataset: str
    split: str
    source_index: int
    question: str
    answer: str


@dataclass
class RetrievedCandidate:
    entry: BankEntry
    rank: int
    bm25_score: float
    question_token_f1: float
    question_char3_f1: float
    answer: str


def exact_match(prediction: str, ground_truth: str) -> float:
    return float(prediction.strip().lower() == ground_truth.strip().lower())


def token_f1(prediction: str, ground_truth: str) -> float:
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


def _normalize_for_char_metrics(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def char_ngram_f1(prediction: str, reference: str, n: int = 3) -> float:
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


def script_match_ratio(text: str, target_script: str | None) -> float:
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


def latin_leak_ratio(text: str) -> float:
    letters = _letters(text)
    if not letters:
        return 0.0
    latin = sum(1 for char in letters if _is_latin_letter(char))
    return latin / len(letters)


def repetition_ngram_rate(text: str, n: int = 4) -> float:
    tokens = text.lower().split()
    grams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
    if not grams:
        return 0.0

    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(grams)


def length_ratio(prediction: str, reference: str) -> float:
    reference_len = max(len(reference.split()), 1)
    return len(prediction.split()) / reference_len


def quality_flag(prediction: str, reference: str, target_script: str | None) -> float:
    if not prediction.strip():
        return 1.0
    if repetition_ngram_rate(prediction) > 0.2:
        return 1.0
    if script_match_ratio(prediction, target_script) < 0.85:
        return 1.0
    if length_ratio(prediction, reference) > 4.0:
        return 1.0
    return 0.0


def rouge_l(prediction: str, reference: str) -> float:
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


class BM25Index:
    """Small dependency-free BM25 index over bank questions."""

    def __init__(self, entries: list[BankEntry], *, k1: float = 1.5, b: float = 0.75):
        self.entries = entries
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(entry.question) for entry in entries]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_id, tokens in enumerate(self.doc_tokens):
            for term, tf in Counter(tokens).items():
                self.inverted[term].append((doc_id, tf))

        n_docs = max(len(entries), 1)
        self.idf = {
            term: math.log(1.0 + (n_docs - len(postings) + 0.5) / (len(postings) + 0.5))
            for term, postings in self.inverted.items()
        }

    def search(self, query: str, top_k: int) -> list[RetrievedCandidate]:
        query_terms = tokenize(query)
        scores: dict[int, float] = defaultdict(float)

        for term in query_terms:
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self.idf[term]
            for doc_id, tf in postings:
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[doc_id] / max(self.avgdl, 1e-9)
                )
                scores[doc_id] += idf * tf * (self.k1 + 1.0) / denom

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        candidates = []
        for rank, (doc_id, score) in enumerate(ranked):
            entry = self.entries[doc_id]
            candidates.append(
                RetrievedCandidate(
                    entry=entry,
                    rank=rank,
                    bm25_score=float(score),
                    question_token_f1=token_f1(query, entry.question),
                    question_char3_f1=char_ngram_f1(query, entry.question),
                    answer=entry.answer,
                )
            )
        return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Darius-style retrieval/selection columns to a comparison CSV."
    )
    parser.add_argument("--comparison_csv", required=True)
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--report_json", default=None)
    parser.add_argument("--data_root", default="./data")
    parser.add_argument("--dataset_repo", default=None)
    parser.add_argument("--dataset_revision", default=None)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN from the environment.",
    )
    parser.add_argument("--bank_splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--require_source_dataset",
        action="store_true",
        help="Fail if rows do not contain source-leaf IDs such as eng_ken or swa_uga.",
    )
    parser.add_argument(
        "--generation_column",
        default="adapter_prediction",
        help="Existing generation column used when the selector keeps generation.",
    )
    parser.add_argument(
        "--direct_char3_threshold",
        type=float,
        default=0.35,
        help="Minimum question char-3 F1 for direct retrieval sources.",
    )
    parser.add_argument(
        "--conservative_char3_threshold",
        type=float,
        default=0.55,
        help="Minimum question char-3 F1 for conservative retrieval sources.",
    )
    parser.add_argument(
        "--conservative_gap_threshold",
        type=float,
        default=0.05,
        help="Minimum top-vs-second question char-3 F1 gap for conservative retrieval.",
    )
    parser.add_argument(
        "--include_oracle",
        action="store_true",
        help="Add a reference-leaking retrieval oracle column for diagnostics only.",
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w']+", str(text).casefold(), flags=re.UNICODE)


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def default_output_paths(comparison_csv: Path) -> tuple[Path, Path]:
    stem = comparison_csv.with_suffix("")
    return (
        stem.with_name(f"{stem.name}_retrieval_selected.csv"),
        stem.with_name(f"{stem.name}_retrieval_selected_report.json"),
    )


def resolve_source_dataset(row: dict[str, str]) -> str:
    return row.get("source_dataset") or row.get("language") or ""


def source_config(source_id: str):
    return SOURCE_DATASETS.get(source_id) or SUPPORTED_LANGUAGES.get(source_id)


def source_script(source_id: str, fallback_language: str = "") -> str | None:
    cfg = source_config(source_id) or SUPPORTED_LANGUAGES.get(fallback_language)
    return cfg.script if cfg else None


def source_policy(source_id: str) -> str:
    if source_id in DARIUS_GENERATION_SOURCES:
        return "generation_passthrough"
    if source_id in DARIUS_DIRECT_RETRIEVAL_SOURCES:
        return "direct_retrieval"
    if source_id in DARIUS_CONSERVATIVE_RETRIEVAL_SOURCES:
        return "conservative_retrieval"
    return "conservative_retrieval"


def validate_source_rows(rows: list[dict[str, str]], *, require_source_dataset: bool) -> None:
    missing = [i for i, row in enumerate(rows, start=1) if not row.get("source_dataset")]
    if require_source_dataset and missing:
        preview = ", ".join(str(i) for i in missing[:10])
        raise SystemExit(
            "Missing source_dataset in comparison CSV rows "
            f"{preview}. Re-run generation with --use_source_leaves "
            "and --require_source_leaves."
        )

    unknown = sorted(
        {
            resolve_source_dataset(row)
            for row in rows
            if resolve_source_dataset(row)
            and source_config(resolve_source_dataset(row)) is None
        }
    )
    if unknown:
        raise SystemExit(f"Unknown source datasets in comparison CSV: {', '.join(unknown)}")


def load_bank_for_source(
    *,
    source_id: str,
    builder: Any,
    bank_splits: list[str],
) -> list[BankEntry]:
    from data_utils import get_split

    cfg = source_config(source_id)
    if cfg is None:
        raise KeyError(f"No dataset config for source '{source_id}'")

    dataset = builder.load_language(source_id, cfg, augment=False)
    entries: list[BankEntry] = []
    seen = set()
    for split_name in bank_splits:
        split_ds = get_split(dataset, split_name)
        if split_ds is None:
            continue
        for index, example in enumerate(split_ds):
            question = str(example.get("input", "")).strip()
            answer = str(example.get("output", "")).strip()
            if not question or not answer:
                continue
            key = (question, answer)
            if key in seen:
                continue
            seen.add(key)
            entries.append(
                BankEntry(
                    source_dataset=source_id,
                    split=split_name,
                    source_index=index,
                    question=question,
                    answer=answer,
                )
            )
    return entries


def should_use_retrieval(
    *,
    source_id: str,
    candidates: list[RetrievedCandidate],
    args: argparse.Namespace,
) -> tuple[bool, str]:
    if not candidates:
        return False, "no_candidates"

    policy = source_policy(source_id)
    top = candidates[0]
    second_char = candidates[1].question_char3_f1 if len(candidates) > 1 else 0.0
    char_gap = top.question_char3_f1 - second_char

    if policy == "generation_passthrough":
        return False, "brainiac_generation_subset"

    if policy == "direct_retrieval":
        if top.question_char3_f1 >= args.direct_char3_threshold:
            return True, "direct_retrieval_match"
        return False, "direct_retrieval_low_similarity"

    if top.question_char3_f1 >= args.conservative_char3_threshold:
        return True, "conservative_high_similarity"
    if (
        top.question_char3_f1 >= args.direct_char3_threshold
        and char_gap >= args.conservative_gap_threshold
    ):
        return True, "conservative_clear_gap"
    return False, "conservative_keep_generation"


def retrieval_oracle(
    candidates: list[RetrievedCandidate],
    reference: str,
) -> tuple[str, float, int]:
    if not candidates:
        return "", 0.0, -1
    scored = []
    for candidate in candidates:
        score = 0.5 * rouge_l(candidate.answer, reference) + 0.5 * char_ngram_f1(
            candidate.answer, reference
        )
        scored.append((score, candidate))
    score, best = max(scored, key=lambda item: item[0])
    return best.answer, float(score), best.rank


def add_prediction_metrics(
    row: dict[str, Any],
    *,
    prefix: str,
    prediction: str,
    reference: str,
    script: str | None,
) -> None:
    row[f"{prefix}_exact_match"] = exact_match(prediction, reference)
    row[f"{prefix}_f1_token"] = round(token_f1(prediction, reference), 4)
    row[f"{prefix}_rouge_l"] = round(rouge_l(prediction, reference), 4)
    row[f"{prefix}_char_3gram_f1"] = round(char_ngram_f1(prediction, reference), 4)
    row[f"{prefix}_script_match_ratio"] = round(script_match_ratio(prediction, script), 4)
    row[f"{prefix}_latin_leak_ratio"] = round(latin_leak_ratio(prediction), 4)
    row[f"{prefix}_repetition_4gram_rate"] = round(repetition_ngram_rate(prediction), 4)
    row[f"{prefix}_length_ratio"] = round(length_ratio(prediction, reference), 4)
    row[f"{prefix}_quality_flag"] = quality_flag(prediction, reference, script)


def summarize_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    metric_names = [
        "exact_match",
        "f1_token",
        "rouge_l",
        "char_3gram_f1",
        "script_match_ratio",
        "latin_leak_ratio",
        "repetition_4gram_rate",
        "length_ratio",
        "quality_flag",
    ]
    if not rows:
        return {"n": 0}
    summary: dict[str, Any] = {"n": len(rows)}
    for metric in metric_names:
        key = f"{prefix}_{metric}"
        values = [float(row[key]) for row in rows if str(row.get(key, "")) != ""]
        if values:
            name = "quality_flag_rate" if metric == "quality_flag" else metric
            summary[name] = round(sum(values) / len(values), 4)
    if prefix == "selected":
        summary["retrieval_selected_rate"] = round(
            sum(1 for row in rows if str(row.get("selected_from_retrieval")) == "true")
            / len(rows),
            4,
        )
    return summary


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top_k must be positive.")
    if not (0.0 <= args.direct_char3_threshold <= 1.0):
        raise SystemExit("--direct_char3_threshold must be between 0 and 1.")
    if not (0.0 <= args.conservative_char3_threshold <= 1.0):
        raise SystemExit("--conservative_char3_threshold must be between 0 and 1.")

    comparison_csv = Path(args.comparison_csv)
    default_csv, default_report = default_output_paths(comparison_csv)
    output_csv = Path(args.output_csv) if args.output_csv else default_csv
    report_json = Path(args.report_json) if args.report_json else default_report
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    report_json.parent.mkdir(parents=True, exist_ok=True)

    rows, fieldnames = read_csv(comparison_csv)
    if not rows:
        raise SystemExit(f"No rows found in {comparison_csv}")
    if args.generation_column not in rows[0]:
        raise SystemExit(f"Missing generation column: {args.generation_column}")
    validate_source_rows(rows, require_source_dataset=args.require_source_dataset)

    cfg = TrainingConfig()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    from data_utils import MultilingualDatasetBuilder

    builder = MultilingualDatasetBuilder(
        data_root=args.data_root,
        dataset_repo=args.dataset_repo,
        dataset_revision=args.dataset_revision,
        hf_token=hf_token,
        cache_dir=args.cache_dir,
        seed=cfg.seed,
    )

    source_ids = sorted({resolve_source_dataset(row) for row in rows if resolve_source_dataset(row)})
    indexes: dict[str, BM25Index] = {}
    bank_sizes: dict[str, int] = {}
    for source_id in source_ids:
        entries = load_bank_for_source(
            source_id=source_id,
            builder=builder,
            bank_splits=args.bank_splits,
        )
        if not entries:
            logger.warning("No retrieval bank entries for %s", source_id)
            continue
        indexes[source_id] = BM25Index(entries)
        bank_sizes[source_id] = len(entries)
        logger.info("[%s] indexed %s bank entries", source_id, len(entries))

    for row_index, row in enumerate(rows):
        source_id = resolve_source_dataset(row)
        reference = row.get("reference_answer", "")
        generation = row.get(args.generation_column, "")
        index = indexes.get(source_id)
        candidates = index.search(row.get("question", ""), args.top_k) if index else []
        top = candidates[0] if candidates else None
        use_retrieval, decision_reason = should_use_retrieval(
            source_id=source_id,
            candidates=candidates,
            args=args,
        )

        retrieval_prediction = top.answer if top else ""
        selected_prediction = retrieval_prediction if use_retrieval else generation
        script = source_script(source_id, row.get("language", ""))

        row["retrieval_prediction"] = retrieval_prediction
        row["selected_prediction"] = selected_prediction
        row["selected_from_retrieval"] = str(bool(use_retrieval)).lower()
        row["selection_policy"] = source_policy(source_id)
        row["selection_reason"] = decision_reason
        row["retrieval_candidate_count"] = len(candidates)
        row["retrieval_bank_size"] = bank_sizes.get(source_id, 0)
        row["retrieval_rank"] = top.rank if top else ""
        row["retrieval_bm25_score"] = round(top.bm25_score, 6) if top else ""
        row["retrieval_question_token_f1"] = round(top.question_token_f1, 4) if top else ""
        row["retrieval_question_char3_f1"] = round(top.question_char3_f1, 4) if top else ""
        row["retrieval_bank_split"] = top.entry.split if top else ""
        row["retrieval_bank_index"] = top.entry.source_index if top else ""
        row["retrieval_bank_question"] = top.entry.question if top else ""

        if args.include_oracle:
            oracle_answer, oracle_score, oracle_rank = retrieval_oracle(candidates, reference)
            row["retrieval_oracle_prediction"] = oracle_answer
            row["retrieval_oracle_score"] = round(oracle_score, 4)
            row["retrieval_oracle_rank"] = oracle_rank

        add_prediction_metrics(
            row,
            prefix="retrieval",
            prediction=retrieval_prediction,
            reference=reference,
            script=script,
        )
        add_prediction_metrics(
            row,
            prefix="selected",
            prediction=selected_prediction,
            reference=reference,
            script=script,
        )

        if row_index and row_index % 500 == 0:
            logger.info("Processed %s rows", row_index)

    extra_fields = [
        "retrieval_prediction",
        "selected_prediction",
        "selected_from_retrieval",
        "selection_policy",
        "selection_reason",
        "retrieval_candidate_count",
        "retrieval_bank_size",
        "retrieval_rank",
        "retrieval_bm25_score",
        "retrieval_question_token_f1",
        "retrieval_question_char3_f1",
        "retrieval_bank_split",
        "retrieval_bank_index",
        "retrieval_bank_question",
    ]
    if args.include_oracle:
        extra_fields += [
            "retrieval_oracle_prediction",
            "retrieval_oracle_score",
            "retrieval_oracle_rank",
        ]
    metric_fields = []
    for prefix in ("retrieval", "selected"):
        metric_fields += [
            f"{prefix}_exact_match",
            f"{prefix}_f1_token",
            f"{prefix}_rouge_l",
            f"{prefix}_char_3gram_f1",
            f"{prefix}_script_match_ratio",
            f"{prefix}_latin_leak_ratio",
            f"{prefix}_repetition_4gram_rate",
            f"{prefix}_length_ratio",
            f"{prefix}_quality_flag",
        ]
    out_fields = fieldnames + [
        field for field in extra_fields + metric_fields if field not in fieldnames
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)

    by_source = {}
    for source_id in sorted({resolve_source_dataset(row) for row in rows}):
        source_rows = [row for row in rows if resolve_source_dataset(row) == source_id]
        by_source[source_id] = {
            "generation": summarize_rows(source_rows, "adapter"),
            "retrieval": summarize_rows(source_rows, "retrieval"),
            "selected": summarize_rows(source_rows, "selected"),
        }

    report = {
        "comparison_csv": str(comparison_csv),
        "output_csv": str(output_csv),
        "bank_splits": args.bank_splits,
        "top_k": args.top_k,
        "generation_column": args.generation_column,
        "direct_char3_threshold": args.direct_char3_threshold,
        "conservative_char3_threshold": args.conservative_char3_threshold,
        "conservative_gap_threshold": args.conservative_gap_threshold,
        "bank_sizes": bank_sizes,
        "aggregate": {
            "generation": summarize_rows(rows, "adapter"),
            "retrieval": summarize_rows(rows, "retrieval"),
            "selected": summarize_rows(rows, "selected"),
        },
        "by_source": by_source,
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote retrieval-selected CSV to %s", output_csv)
    logger.info("Wrote retrieval-selected report to %s", report_json)


if __name__ == "__main__":
    main()
