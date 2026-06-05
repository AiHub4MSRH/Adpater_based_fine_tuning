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

Generation uses a merge-then-infer strategy:
* For each language, the LoRA adapter (including modules_to_save: lm_head,
  embed_tokens) is merged into the base model weights via PEFT merge_and_unload()
  and the merged model is saved to a temp directory on disk.
* vLLM loads that merged model as a plain causal-LM (no enable_lora needed),
  so all fine-tuned weights including lm_head and embed_tokens are fully applied.
* After generation the vLLM instance and temp directory are both cleaned up.

LLM-as-judge — separate vLLM LLM instance (google/gemma-3-12b-it by default);
               all pairs scored in one batched llm.chat() call.
Perplexity   — derived from vLLM prompt_logprobs (no extra model load).

Metrics:
* LLM-as-judge score  (1–5, normalised to 0–1)
* BERTScore F1        (AfroLM / XLM-RoBERTa — language-routed)
* ROUGE-L, ROUGE-1
* Perplexity          (pred_perplexity and ref_perplexity via prompt log-probs)
* Composite score     (0.4·BERTScore + 0.3·ROUGE-L + 0.2·judge_score + 0.1·ROUGE-1)
"""

import json
import logging
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union

from huggingface_hub import snapshot_download
from tqdm import tqdm

import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from transformers import (
    AutoProcessor,
    XLMRobertaModel,
    XLMRobertaTokenizer,
)
from vllm import LLM, SamplingParams

from config import TrainingConfig, SUPPORTED_LANGUAGES
from data_utils import MultilingualDatasetBuilder, get_split

logger = logging.getLogger(__name__)

# ── AfroLM — loaded once at module import ────────────────────────────────────
_AFROLM_NAME = "bonadossou/afrolm_active_learning"
afrolm_tokenizer = XLMRobertaTokenizer.from_pretrained(_AFROLM_NAME)
afrolm_model     = XLMRobertaModel.from_pretrained(_AFROLM_NAME)
afrolm_tokenizer.model_max_length = 256
afrolm_model.eval()

LANGUAGE_MODEL_MAP: dict[str, str] = {
    "twi(akan)": "afrolm",
    "amharic":   "afrolm",
    "luganda":   "afrolm",
    "english":   "afrolm",
    "swahili":   "afrolm",
}
_DEFAULT_BACKEND = "afrolm"


# ── Adapter path resolution ───────────────────────────────────────────────────

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
    nested_config = nested_dir / "adapter_config.json"

    if flat_config.exists():
        return adapter_dir
    if nested_config.exists():
        return nested_dir
    return adapter_dir


def download_adapters(
    adapter_repo: str,
    lang_codes: list[str],
    adapter_root: Path,
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> dict[str, Path]:
    """
    Download per-language adapters from a Hub repo into the layout that
    resolve_adapter_path expects: adapter_root/adapter_{lang_code}/.

    Each adapter is assumed to live under adapters/{lang_code}/ inside the repo.
    Only the files for the requested lang_codes are fetched (allow_patterns).

    Returns:
        Dict mapping lang_code → local adapter Path for every successfully
        downloaded adapter.
    """
    adapter_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path] = {}

    for lang_code in lang_codes:
        dest = adapter_root / f"adapter_{lang_code}"
        if dest.exists() and any(dest.iterdir()):
            logger.info("Adapter %s already present at %s — skipping download.", lang_code, dest)
            results[lang_code] = dest
            continue

        logger.info("Downloading adapter %s from %s …", lang_code, adapter_repo)
        try:
            snapshot_dir = snapshot_download(
                repo_id=adapter_repo,
                repo_type="model",
                revision=revision,
                allow_patterns=[f"adapters/{lang_code}/*"],
                cache_dir=cache_dir,
                token=hf_token,
            )
        except Exception as exc:
            logger.error("Failed to download adapter %s: %s", lang_code, exc)
            continue

        src = Path(snapshot_dir) / "adapters" / lang_code
        if not src.exists():
            logger.error(
                "Adapter '%s' not found under adapters/%s in %s — skipping.",
                lang_code, lang_code, adapter_repo,
            )
            continue

        dest.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            target = dest / f.name
            if not target.exists():
                target.symlink_to(f.resolve())

        logger.info("Adapter %s ready at %s", lang_code, dest)
        results[lang_code] = dest

    return results


# ── Adapter → merged model ────────────────────────────────────────────────────

def _merge_adapter_to_disk(base_model_id: str, adapter_path: Path) -> Path:
    """
    Load the base model, apply the LoRA adapter (including any modules_to_save
    weights such as lm_head and embed_tokens), merge all weights, and save the
    result to a temporary directory on disk.

    vLLM then loads this as a plain causal-LM — no enable_lora needed — so
    all fine-tuned weights are fully applied with zero compromise.

    The caller is responsible for deleting the returned temp directory after
    the vLLM instance that uses it is destroyed.
    """
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    logger.info("Merging adapter %s into base model %s …", adapter_path.name, base_model_id)

    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    peft_model = PeftModel.from_pretrained(base, str(adapter_path))
    merged     = peft_model.merge_and_unload()

    tmp = Path(tempfile.mkdtemp(prefix="merged_model_"))
    logger.info("Saving merged model to %s …", tmp)
    merged.save_pretrained(tmp)

    # Copy the full processor (tokenizer + image processor + preprocessor_config)
    # so vLLM can load it from the temp dir without hitting the Hub.
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    processor.save_pretrained(tmp)

    # Free the CPU copy immediately — vLLM will reload from disk onto GPU
    del merged, peft_model, base
    import gc
    gc.collect()

    logger.info("Adapter merged and saved to %s", tmp)
    return tmp


# ── LLM-as-judge ─────────────────────────────────────────────────────────────

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert multilingual medical evaluator specialising in sexual and "
    "reproductive health (SRH). You will be shown batched reference answers and "
    "predicted answers to health questions. Each pair may be written in any "
    "language (English, Luganda, Kiswahili, Akan, Amharic, or others).\n\n"
    "For each item, evaluate the predicted answer on the following criteria:\n"
    "  1. Factual accuracy         — Is the medical information correct?\n"
    "  2. Completeness             — Does it cover the key points in the reference?\n"
    "  3. Language appropriateness — Is it fluent and clear in the language used?\n\n"
    "Respond with ONLY a JSON array and nothing else. Each element must be an object "
    "in this exact format:\n"
    '{"index": <integer>, "score": <integer 1-5>, "reason": "<one sentence>"}\n\n'
    "The \"index\" must match the input item index exactly.\n\n"
    "Scoring rubric:\n"
    "  5 — Excellent: accurate, complete, and clearly expressed.\n"
    "  4 — Good: mostly accurate and complete with minor gaps.\n"
    "  3 — Adequate: partially correct or missing some key points.\n"
    "  2 — Poor: significant inaccuracies or major omissions.\n"
    "  1 — Very poor: incorrect, irrelevant, or incomprehensible."
)


class LLMJudge:
    """
    Scores prediction/reference pairs using a vLLM-served instruction model.

    All pairs are sent in a single batched llm.chat() call, which is
    dramatically faster than one generate() call per batch.

    Args:
        judge_model_id:         Any HF causal-LM instruct model.
        max_retries:            Retry attempts on malformed JSON.
        tensor_parallel:        Number of GPUs for tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory vLLM may use.
    """

    def __init__(
        self,
        judge_model_id: str = "google/gemma-3-12b-it",
        max_retries: int = 2,
        tensor_parallel: int = 1,
        gpu_memory_utilization: float = 0.4,
    ):
        self.judge_model_id         = judge_model_id
        self.max_retries            = max_retries
        self.tensor_parallel        = tensor_parallel
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm: Optional[LLM]    = None

    def _ensure_loaded(self):
        if self._llm is not None:
            return
        logger.info("Loading judge model via vLLM: %s", self.judge_model_id)
        self._llm = LLM(
            model=self.judge_model_id,
            tensor_parallel_size=self.tensor_parallel,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=2048,  # SRH judge prompts are short; cap to avoid KV OOM
        )

    def _build_conversations(
        self,
        predictions: list[str],
        references: list[str],
    ) -> list[list[dict]]:
        """One conversation per pair — lets vLLM score them all in one batch."""
        conversations = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            conversations.append([
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user",   "content": (
                    f"Item {i}:\n"
                    f"  Reference: {ref}\n"
                    f"  Predicted: {pred}"
                )},
            ])
        return conversations

    def _parse_score(self, raw: str, index: int) -> Optional[float]:
        """Parse a single-item response; return normalised float or None on failure."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        if cleaned.startswith("{"):
            cleaned = f"[{cleaned}]"
        try:
            items = json.loads(cleaned)
            for item in items:
                if "score" in item:
                    return round(max(1, min(5, int(item["score"]))) / 5.0, 4)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        logger.warning("Judge parse failed for item %d; raw: %s", index, raw[:300])
        return None

    def score(
        self,
        predictions: list[str],
        references: list[str],
    ) -> list[float]:
        """
        Score all pairs in one batched vLLM call.
        Returns floats in [0, 1]; failed items default to 0.0.
        """
        self._ensure_loaded()
        n               = len(predictions)
        scores          = [0.0] * n
        pending         = list(range(n))
        sampling_params = SamplingParams(temperature=0.0, max_tokens=128, top_p=1.0)

        for attempt in range(self.max_retries + 1):
            if not pending:
                break
            conversations = self._build_conversations(
                [predictions[i] for i in pending],
                [references[i]  for i in pending],
            )
            outputs = self._llm.chat(conversations, sampling_params=sampling_params)

            still_pending = []
            for global_idx, output in zip(pending, outputs):
                raw   = output.outputs[0].text
                score = self._parse_score(raw, global_idx)
                if score is not None:
                    scores[global_idx] = score
                else:
                    still_pending.append(global_idx)

            if still_pending and attempt < self.max_retries:
                logger.warning(
                    "Judge: %d/%d items failed parsing (attempt %d/%d), retrying …",
                    len(still_pending), n, attempt + 1, self.max_retries + 1,
                )
            pending = still_pending

        return scores


# ── AfroLM BERTScore ─────────────────────────────────────────────────────────

def afrolm_bertscore(
    predictions: list[str],
    references: list[str],
    batch_size: int = 16,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    BERTScore-style F1 using AfroLM (XLM-RoBERTa).
    Mean-pools token embeddings then computes cosine similarity per pair.
    Returns a 1-D tensor of F1 scores clamped to [0, 1].
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    afrolm_model.to(device)

    def _embed(texts: list[str]) -> torch.Tensor:
        all_emb = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = afrolm_tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=afrolm_tokenizer.model_max_length,
            ).to(device)
            with torch.inference_mode():
                outputs = afrolm_model(**encoded)
            mask      = encoded["attention_mask"].unsqueeze(-1).float()
            token_emb = outputs.last_hidden_state
            mean_emb  = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            all_emb.append(mean_emb.cpu())
        return torch.cat(all_emb, dim=0)

    pred_emb = _embed(predictions)
    ref_emb  = _embed(references)
    return F.cosine_similarity(pred_emb, ref_emb, dim=-1).clamp(min=0.0)


def _bertscore_for_language(
    predictions: list[str],
    references: list[str],
    language: str,
) -> torch.Tensor:
    backend = LANGUAGE_MODEL_MAP.get(language, _DEFAULT_BACKEND)
    logger.info("BERTScore | language=%s | backend=%s", language, backend)
    return afrolm_bertscore(predictions, references)


# ── ROUGE ─────────────────────────────────────────────────────────────────────

class _WhitespaceTokenizer:
    """Whitespace tokenizer for rouge_scorer — safe across all target languages."""
    def tokenize(self, text: str) -> list[str]:
        if text is None:
            return []
        return str(text).strip().split()


def _rouge_scores(
    predictions: list[str],
    references: list[str],
) -> tuple[list[float], list[float]]:
    _scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rougeL"],
        use_stemmer=False,
        tokenizer=_WhitespaceTokenizer(),
    )
    r1, rl = [], []
    for pred, ref in zip(predictions, references):
        s = _scorer.score(ref, pred)
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return r1, rl


# ── Perplexity via vLLM prompt log-probs ─────────────────────────────────────

def _compute_perplexity_vllm(
    texts: list[str],
    llm: LLM,
) -> list[float]:
    """
    Compute per-text perplexity using vLLM prompt_logprobs.

    SamplingParams(prompt_logprobs=1) returns the top-1 log-prob for each
    prompt token. We take the mean NLL across all tokens and exponentiate.
    Lower = model assigns higher probability to the text.
    """
    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=1,
        temperature=0.0,
    )
    outputs = llm.generate(texts, sampling_params=sampling_params)
    ppls: list[float] = []

    for output in outputs:
        prompt_logprobs = output.prompt_logprobs
        if not prompt_logprobs:
            ppls.append(float("inf"))
            continue

        nll_sum   = 0.0
        token_cnt = 0
        for token_logprobs in prompt_logprobs:
            if token_logprobs is None:
                continue
            logprob_obj = next(iter(token_logprobs.values()))
            nll_sum   += -logprob_obj.logprob
            token_cnt += 1

        ppls.append(math.exp(nll_sum / token_cnt) if token_cnt > 0 else float("inf"))

    return ppls


# ── score_dataframe (public utility) ─────────────────────────────────────────

def score_dataframe(
    df,
    prediction_col: str,
    reference_col: str,
    judge: LLMJudge,
    lang: Union[str, list[str]] = "english",
):
    """
    Add metric columns to *df* and return the annotated copy.

    Columns added:
        bertscore_f1, rouge_1, rouge_l, judge_score, composite_score

    Note: pred_perplexity / ref_perplexity require the generation LLM and are
    only computed inside MultilingualEvaluator.evaluate_language, not here.
    """
    df          = df.copy()
    predictions = df[prediction_col].astype(str).tolist()
    references  = df[reference_col].astype(str).tolist()

    if isinstance(lang, str):
        bert_f1 = _bertscore_for_language(predictions, references, lang)
        df["bertscore_f1"] = bert_f1.numpy()
    elif isinstance(lang, list):
        if len(lang) != len(df):
            raise ValueError(f"lang list length {len(lang)} must match DataFrame length {len(df)}")
        bert_f1_scores = torch.zeros(len(predictions))
        groups: dict[str, list[int]] = defaultdict(list)
        for idx, language in enumerate(lang):
            groups[language].append(idx)
        for language, indices in groups.items():
            f1 = _bertscore_for_language(
                [predictions[i] for i in indices],
                [references[i]  for i in indices],
                language,
            )
            for idx, score in zip(indices, f1):
                bert_f1_scores[idx] = score
        df["bertscore_f1"] = bert_f1_scores.numpy()
    else:
        raise TypeError(f"lang must be str or list[str], got {type(lang)}")

    rouge1_list, rougeL_list = _rouge_scores(predictions, references)
    df["rouge_1"]     = rouge1_list
    df["rouge_l"]     = rougeL_list
    df["judge_score"] = judge.score(predictions, references)

    df["composite_score"] = (
        0.4 * df["bertscore_f1"]
        + 0.3 * df["rouge_l"]
        + 0.2 * df["judge_score"]
        + 0.1 * df["rouge_1"]
    )
    return df


# ── Evaluator ────────────────────────────────────────────────────────────────

class MultilingualEvaluator:
    """
    Evaluate adapters on the `test-*` split for each configured dataset leaf.

    For each language, the LoRA adapter is merged into the base model via PEFT
    merge_and_unload() and saved to a temp directory. vLLM loads that merged
    model as a plain causal-LM so all fine-tuned weights (including lm_head and
    embed_tokens from modules_to_save) are fully applied. The vLLM instance and
    temp directory are cleaned up after each language to keep GPU memory flat.
    """

    def __init__(
        self,
        cfg: TrainingConfig,
        adapter_root: Path,
        judge: Optional[LLMJudge] = None,
        dataset_builder: Optional[MultilingualDatasetBuilder] = None,
        tensor_parallel: int = 1,
        gpu_memory_utilization: float = 0.5,
    ):
        self.cfg                    = cfg
        self.adapter_root           = adapter_root
        self.judge                  = judge if judge is not None else LLMJudge()
        self.dataset_builder        = dataset_builder
        self.tensor_parallel        = tensor_parallel
        self.gpu_memory_utilization = gpu_memory_utilization
        self._processor             = AutoProcessor.from_pretrained(cfg.model_id)

    def _load_merged_llm(self, merged_model_path: Path) -> LLM:
        """Load a merged model from disk into vLLM."""
        logger.info("Loading merged model via vLLM: %s …", merged_model_path)
        return LLM(
            model=str(merged_model_path),
            tensor_parallel_size=self.tensor_parallel,
            gpu_memory_utilization=self.gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True,
        )

    @staticmethod
    def _release_llm(llm: LLM) -> None:
        """Destroy a vLLM instance and free GPU memory."""
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        del llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    def _build_prompt(self, prompt: str, language_name: str) -> str:
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
        return self._processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _generate_and_score(
        self,
        lang_code: str,
        max_eval_samples: int = 200,
    ) -> dict[str, Any]:
        """
        Phase-1 worker: merge adapter → load vLLM → generate → compute metrics
        except judge_score. Tears down vLLM and deletes the merged temp model
        before returning so GPU memory is free for the next language.

        Returns the result dict with "predictions" and "references" keys so
        evaluate_all can run the judge separately in phase 2.
        """
        adapter_path = resolve_adapter_path(self.adapter_root, lang_code)
        if adapter_path is None:
            logger.warning("Adapter not found for %s; skipping evaluation.", lang_code)
            return {"language": lang_code, "error": "adapter_not_found"}

        if self.dataset_builder is None:
            return {"language": lang_code, "error": "no_dataset_builder"}

        lang_cfg      = SUPPORTED_LANGUAGES.get(lang_code)
        language_name = lang_cfg.language_name if lang_cfg else lang_code
        display_name  = lang_cfg.display_name  if lang_cfg else lang_code

        try:
            dataset = self.dataset_builder.load_language(lang_code, lang_cfg, augment=False)
        except Exception as exc:
            logger.warning("No evaluation data for %s: %s", lang_code, exc)
            return {"language": lang_code, "error": "no_test_data", "details": str(exc)}

        test_ds = get_split(dataset, "test")
        if test_ds is None or len(test_ds) == 0:
            logger.warning("No test split for %s. Skipping.", lang_code)
            return {"language": lang_code, "error": "no_test_data"}

        n       = min(len(test_ds), max_eval_samples)
        test_ds = test_ds.shuffle(seed=42).select(range(n))

        prompts    = [self._build_prompt(ex["input"], language_name) for ex in test_ds]
        references = [ex["output"] for ex in test_ds]

        # ── Merge adapter into base model, load into vLLM ─────────
        merged_path = _merge_adapter_to_disk(self.cfg.model_id, adapter_path)
        llm = self._load_merged_llm(merged_path)

        # ── Batched generation ────────────────────────────────────
        logger.info("[%s] Generating %d samples (batched) …", lang_code, n)
        gen_params  = SamplingParams(temperature=0.3, max_tokens=256)
        gen_outputs = llm.generate(prompts, gen_params)
        predictions   = [out.outputs[0].text.strip() for out in gen_outputs]
        invalid_count = sum(1 for p in predictions if not p)

        # ── Perplexity via prompt log-probs ───────────────────────
        logger.info("[%s] Computing perplexity …", lang_code)
        pred_ppl_list = _compute_perplexity_vllm(predictions, llm)
        ref_ppl_list  = _compute_perplexity_vllm(references,  llm)

        # ── Release vLLM and merged model temp dir ────────────────
        self._release_llm(llm)
        shutil.rmtree(merged_path, ignore_errors=True)
        logger.info("[%s] Released merged model and freed GPU memory.", lang_code)

        # ── BERTScore ─────────────────────────────────────────────
        logger.info("[%s] Computing BERTScore …", lang_code)
        bert_f1      = _bertscore_for_language(predictions, references, language_name.lower())
        bert_f1_list = bert_f1.tolist()

        # ── ROUGE ─────────────────────────────────────────────────
        rouge1_list, rougeL_list = _rouge_scores(predictions, references)

        avg_rouge1   = sum(rouge1_list)   / n
        avg_rougeL   = sum(rougeL_list)   / n
        avg_bert_f1  = sum(bert_f1_list)  / n
        avg_pred_ppl = sum(pred_ppl_list) / n
        avg_ref_ppl  = sum(ref_ppl_list)  / n

        result = {
            "language":        lang_code,
            "display_name":    display_name,
            "language_name":   language_name,
            "n_evaluated":     n,
            "judge_score":     0.0,          # filled in by phase 2
            "bertscore_f1":    round(avg_bert_f1,  4),
            "rouge_1":         round(avg_rouge1,   4),
            "rouge_l":         round(avg_rougeL,   4),
            "pred_perplexity": round(avg_pred_ppl, 4),
            "ref_perplexity":  round(avg_ref_ppl,  4),
            "composite_score": 0.0,          # filled in by phase 2
            "invalid_rate":    round(invalid_count / n, 4),
            "resource_level":  lang_cfg.resource_level if lang_cfg else "unknown",
            # carried for phase 2, removed before final report
            "predictions":     predictions,
            "references":      references,
        }

        if "label" in test_ds.column_names:
            mcq_hits = sum(
                1 for pred, ex in zip(predictions, test_ds)
                if ex["label"].strip().lower() in pred.lower()
            )
            result["mcq_accuracy"] = round(mcq_hits / n, 4)

        return result

    def evaluate_language(
        self,
        lang_code: str,
        max_eval_samples: int = 200,
    ) -> dict[str, Any]:
        """
        Evaluate a single language end-to-end (merge → generate → judge).
        The merged vLLM instance is created and destroyed within this call.
        """
        result = self._generate_and_score(lang_code, max_eval_samples)
        if "error" in result:
            return result

        predictions = result.pop("predictions")
        references  = result.pop("references")
        n           = result["n_evaluated"]

        logger.info("[%s] Running LLM judge (%d samples) …", lang_code, n)
        judge_list        = self.judge.score(predictions, references)
        result["judge_score"] = round(sum(judge_list) / n, 4)
        result["composite_score"] = round(
            0.4 * result["bertscore_f1"]
            + 0.3 * result["rouge_l"]
            + 0.2 * result["judge_score"]
            + 0.1 * result["rouge_1"],
            4,
        )
        logger.info(
            "[%s] Judge=%.3f BERT=%.3f ROUGE-L=%.3f PPL(pred/ref)=%.1f/%.1f Comp=%.3f Inv=%.3f",
            lang_code,
            result["judge_score"], result["bertscore_f1"], result["rouge_l"],
            result["pred_perplexity"], result["ref_perplexity"],
            result["composite_score"], result["invalid_rate"],
        )
        return result

    def evaluate_all(
        self,
        languages: list[str],
        max_eval_samples: int = 200,
    ) -> dict[str, Any]:
        """
        Evaluate all requested dataset leaves and compute macro averages.

        Two-phase to avoid GPU OOM:
          Phase 1 — for each language: merge adapter → load vLLM → generate →
                    compute BERTScore / ROUGE / PPL → unload vLLM → delete temp.
                    GPU is fully free between languages.
          Phase 2 — run the judge over all collected predictions in one pass.
        """
        # ── Phase 1: per-language merge + generate ────────────────
        per_language: dict[str, Any] = {}
        pending_judge: dict[str, tuple[list[str], list[str]]] = {}

        for lang_code in tqdm(languages, desc="generating", unit="lang"):
            result = self._generate_and_score(lang_code, max_eval_samples)
            per_language[lang_code] = result
            if "predictions" in result:
                pending_judge[lang_code] = (
                    result.pop("predictions"),
                    result.pop("references"),
                )

        # ── Phase 2: judge all languages ──────────────────────────
        for lang_code, (predictions, references) in tqdm(
            pending_judge.items(), desc="judging", unit="lang"
        ):
            n = per_language[lang_code]["n_evaluated"]
            logger.info("[%s] Running LLM judge (%d samples) …", lang_code, n)
            judge_list  = self.judge.score(predictions, references)
            avg_judge   = sum(judge_list) / n

            per_language[lang_code]["judge_score"] = round(avg_judge, 4)

            # Recompute composite now that judge_score is available
            r = per_language[lang_code]
            r["composite_score"] = round(
                0.4 * r["bertscore_f1"]
                + 0.3 * r["rouge_l"]
                + 0.2 * r["judge_score"]
                + 0.1 * r["rouge_1"],
                4,
            )
            logger.info(
                "[%s] Judge=%.3f BERT=%.3f ROUGE-L=%.3f PPL(pred/ref)=%.1f/%.1f Comp=%.3f Inv=%.3f",
                lang_code,
                r["judge_score"], r["bertscore_f1"], r["rouge_l"],
                r["pred_perplexity"], r["ref_perplexity"],
                r["composite_score"], r["invalid_rate"],
            )

        valid = [r for r in per_language.values() if "error" not in r]
        if valid:
            aggregate = {
                "macro_avg_judge_score":  round(sum(r["judge_score"]     for r in valid) / len(valid), 4),
                "macro_avg_bertscore_f1": round(sum(r["bertscore_f1"]    for r in valid) / len(valid), 4),
                "macro_avg_rouge_1":      round(sum(r["rouge_1"]         for r in valid) / len(valid), 4),
                "macro_avg_rouge_l":      round(sum(r["rouge_l"]         for r in valid) / len(valid), 4),
                "macro_avg_pred_ppl":     round(sum(r["pred_perplexity"] for r in valid) / len(valid), 4),
                "macro_avg_ref_ppl":      round(sum(r["ref_perplexity"]  for r in valid) / len(valid), 4),
                "macro_avg_composite":    round(sum(r["composite_score"] for r in valid) / len(valid), 4),
                "macro_avg_invalid_rate": round(sum(r["invalid_rate"]    for r in valid) / len(valid), 4),
                "n_languages_evaluated":  len(valid),
            }
        else:
            aggregate = {}

        return {"per_language": per_language, "aggregate": aggregate}

    def save_report(self, results: dict, output_path: Path) -> None:
        """Persist the JSON report and print a compact console summary."""
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, ensure_ascii=False)
        logger.info("Evaluation report saved to %s", output_path)

        print("\n" + "═" * 80)
        print("MULTILINGUAL SRH EVALUATION SUMMARY")
        print("═" * 80)
        for lang_code, metrics in results["per_language"].items():
            if "error" in metrics:
                print(f"  {lang_code:<12} ERROR: {metrics['error']}")
            else:
                print(
                    f"  {lang_code:<12} "
                    f"Judge={metrics['judge_score']:.3f}  "
                    f"BERT={metrics['bertscore_f1']:.3f}  "
                    f"R1={metrics['rouge_1']:.3f}  "
                    f"RL={metrics['rouge_l']:.3f}  "
                    f"PPL(p/r)={metrics['pred_perplexity']:.1f}/{metrics['ref_perplexity']:.1f}  "
                    f"Comp={metrics['composite_score']:.3f}  "
                    f"Inv={metrics['invalid_rate']:.3f}"
                )

        if results.get("aggregate"):
            agg = results["aggregate"]
            print("─" * 80)
            print(
                f"  {'MACRO AVG':<12} "
                f"Judge={agg['macro_avg_judge_score']:.3f}  "
                f"BERT={agg['macro_avg_bertscore_f1']:.3f}  "
                f"R1={agg['macro_avg_rouge_1']:.3f}  "
                f"RL={agg['macro_avg_rouge_l']:.3f}  "
                f"PPL(p/r)={agg['macro_avg_pred_ppl']:.1f}/{agg['macro_avg_ref_ppl']:.1f}  "
                f"Comp={agg['macro_avg_composite']:.3f}  "
                f"Inv={agg['macro_avg_invalid_rate']:.3f}"
            )
        print("═" * 80 + "\n")
