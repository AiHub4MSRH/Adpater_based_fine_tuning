"""
data_utils.py — Dataset access, validation, and augmentation for SRH shards
============================================================================

Expected dataset layouts
------------------------
This module is written around your multilingual SRH dataset. Different mirrors
have used either nested leaves:

    eng/eng_uga/train-*
    eng/eng_uga/dev-*
    eng/eng_uga/test-*

or flatter leaf directories:

    eng_uga/train-*
    eng_uga/dev-*
    eng_uga/test-*

Supported sources:
1. A Hugging Face dataset repo containing shard globs such as
   `eng/eng_uga/train-*` or `eng_uga/train-*`
2. A local mirror of that same shard tree
3. A local `DatasetDict.save_to_disk()` mirror created after download

Column contract
---------------
Each split must contain at least:
* `input`  — source SRH text or prompt
* `output` — target text or answer

Why this module exists
----------------------
The original code assumed:
* ISO 639-1 language IDs
* one folder per language
* `validation` as the canonical eval split
* `load_from_disk()` data prepared in advance

Your real dataset violates all four assumptions. This module centralises the
translation from "user selected `eng` or `eng_uga`" into "load the correct data
from either Hub, local shard files, or local cached DatasetDicts", and it does
so once so the training and evaluation code do not need special cases.
"""

import logging
import random
from pathlib import Path
from typing import Optional

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, load_from_disk

from config import LanguageConfig, SUPPORTED_LANGUAGES

logger = logging.getLogger(__name__)

CANONICAL_SPLITS = ("train", "dev", "test")
SPLIT_ALIASES = {
    "train": ("train",),
    "dev": ("dev", "validation", "val"),
    "test": ("test",),
}
FILE_BUILDERS = {
    ".arrow": "arrow",
    ".csv": "csv",
    ".json": "json",
    ".jsonl": "json",
    ".parquet": "parquet",
}


def normalize_dataset_splits(dataset: DatasetDict) -> DatasetDict:
    """
    Map legacy split aliases onto canonical `train` / `dev` / `test` keys.

    We keep `dev` as canonical because that matches your dataset naming, but
    still accept older `validation` / `val` inputs to avoid breaking mirrored
    or intermediate local datasets.
    """

    normalized = {}
    for canonical, aliases in SPLIT_ALIASES.items():
        for alias in aliases:
            if alias in dataset:
                normalized[canonical] = dataset[alias]
                break
    return DatasetDict(normalized)


def get_split(dataset: DatasetDict, split_name: str) -> Optional[Dataset]:
    """Fetch a split using canonical or legacy alias names."""

    for alias in SPLIT_ALIASES[split_name]:
        if alias in dataset:
            return dataset[alias]
    return None


def validate_required_columns(dataset: DatasetDict, lang_code: str) -> None:
    """
    Validate that every present split exposes the SRH text columns.

    This fails fast before training or evaluation so schema mismatches show up
    at dataset load time rather than mid-run inside the trainer.
    """

    required = {"input", "output"}
    for split_name, split_ds in dataset.items():
        missing = required - set(split_ds.column_names)
        if missing:
            raise ValueError(
                f"Split '{split_name}' for '{lang_code}' is missing columns: {sorted(missing)}"
            )


class MultilingualDatasetBuilder:
    """
    Load, validate, and optionally augment per-variant SRH datasets.

    Resolution order:
    1. `save_to_disk()` mirror under `<data_root>/<dataset_id>`
    2. local shard tree under `<data_root>/<language_code>/<dataset_id>/`
    3. Hugging Face Hub repo referenced by `dataset_repo`
    """

    def __init__(
        self,
        data_root: Optional[str | Path] = None,
        dataset_repo: Optional[str] = None,
        dataset_revision: Optional[str] = None,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        seed: int = 42,
    ):
        self.data_root = Path(data_root) if data_root else None
        self.dataset_repo = dataset_repo
        self.dataset_revision = dataset_revision
        self.hf_token = hf_token
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.seed = seed

    def load_language(
        self,
        lang_code: str,
        lang_cfg: LanguageConfig,
        augment: bool = True,
    ) -> DatasetDict:
        """
        Load one dataset leaf such as `eng_uga` or `swa_ken`.

        The returned dataset is normalized so downstream code can always ask for
        `train`, `dev`, and `test` regardless of whether the source used legacy
        alias names.
        """

        dataset = self._load_dataset(lang_code, lang_cfg)
        dataset = normalize_dataset_splits(dataset)
        validate_required_columns(dataset, lang_code)

        train_ds = get_split(dataset, "train")
        if train_ds is None:
            raise ValueError(f"No train split found for '{lang_code}'.")

        n_train = len(train_ds)
        logger.info("[%s] Loaded %s training samples.", lang_code, n_train)

        if augment and n_train < 2000:
            logger.info(
                "[%s] Low-resource mode enabled (%s samples < 2000 threshold).",
                lang_code,
                n_train,
            )
            dataset["train"] = self._augment_low_resource(
                train_ds,
                lang_code,
                lang_cfg,
                n_train,
            )

        return dataset

    def _load_dataset(self, lang_code: str, lang_cfg: LanguageConfig) -> DatasetDict:
        """
        Resolve a dataset leaf from local caches first, then from the Hub.

        This keeps repeated experiments cheap while still allowing direct Hub
        training when no local mirror exists.
        """
        saved_to_disk = self._load_saved_to_disk(lang_code)
        if saved_to_disk is not None:
            return saved_to_disk

        local_shards = self._load_local_shards(lang_cfg)
        if local_shards is not None:
            return local_shards

        if self.dataset_repo:
            return self._load_hub_shards(lang_cfg)

        if self.data_root:
            expected = " or ".join(
                f"{subdir}/train-* under {self.data_root}"
                for subdir in lang_cfg.shard_subdirs
            )
        else:
            expected = " or ".join(lang_cfg.shard_subdirs)
        raise FileNotFoundError(
            f"Could not locate dataset '{lang_code}'. Expected local save_to_disk data, "
            f"local shard files, or a Hub dataset repo. Last expected location: {expected}"
        )

    def _load_saved_to_disk(self, lang_code: str) -> Optional[DatasetDict]:
        """Load a previously mirrored `DatasetDict.save_to_disk()` dataset."""
        if not self.data_root:
            return None

        dataset_dir = self.data_root / lang_code
        manifest = dataset_dir / "dataset_dict.json"
        if dataset_dir.exists() and manifest.exists():
            logger.info("[%s] Loading local DatasetDict from %s", lang_code, dataset_dir)
            return load_from_disk(str(dataset_dir))

        return None

    def _load_local_shards(self, lang_cfg: LanguageConfig) -> Optional[DatasetDict]:
        """
        Load a local tree of shard files that preserves the Hub directory layout.

        Examples:
            <data_root>/eng/eng_uga/train-00000-of-00001.parquet
            <data_root>/eng_uga/train-00000-of-00001.parquet
        """
        if not self.data_root:
            return None

        for subdir in lang_cfg.shard_subdirs:
            shard_dir = self.data_root / subdir
            if not shard_dir.exists():
                continue

            matches_by_split: dict[str, list[Path]] = {}
            for split_name in CANONICAL_SPLITS:
                matches = sorted(shard_dir.glob(f"{split_name}-*"))
                if matches:
                    matches_by_split[split_name] = matches

            if not matches_by_split:
                continue

            builder_name = self._infer_builder(matches_by_split)
            data_files = {
                split_name: [str(path) for path in matches]
                for split_name, matches in matches_by_split.items()
            }
            logger.info("[%s] Loading local shard files from %s", lang_cfg.dataset_id, shard_dir)
            return load_dataset(builder_name, data_files=data_files, cache_dir=self.cache_dir)

        return None

    def _infer_builder(self, matches_by_split: dict[str, list[Path]]) -> str:
        """
        Infer the `datasets` loading builder from shard file extensions.

        We require a single file format per leaf because mixing builders across
        splits usually signals a malformed export.
        """
        suffixes = {
            path.suffix.lower()
            for matches in matches_by_split.values()
            for path in matches
        }

        if len(suffixes) != 1:
            raise ValueError(
                "Shard files must use a single format per dataset leaf. "
                f"Found extensions: {sorted(suffixes)}"
            )

        suffix = next(iter(suffixes))
        if suffix not in FILE_BUILDERS:
            raise ValueError(
                f"Unsupported shard format '{suffix}'. Supported formats: {sorted(FILE_BUILDERS)}"
            )
        return FILE_BUILDERS[suffix]

    def _load_hub_shards(self, lang_cfg: LanguageConfig) -> DatasetDict:
        """
        Load shard globs directly from a dataset repository on the Hub.

        The `datasets` library can read generic data files from dataset repos via
        `data_files`, which lets us target the real nested shard paths without
        requiring a bespoke dataset loading script in this project.
        """
        data_files = {
            split_name: lang_cfg.split_globs(split_name)
            for split_name in CANONICAL_SPLITS
        }

        logger.info(
            "[%s] Loading shard globs from Hub repo %s",
            lang_cfg.dataset_id,
            self.dataset_repo,
        )
        return load_dataset(
            self.dataset_repo,
            data_files=data_files,
            revision=self.dataset_revision,
            cache_dir=self.cache_dir,
            token=self.hf_token,
        )

    def _augment_low_resource(
        self,
        dataset: Dataset,
        lang_code: str,
        lang_cfg: LanguageConfig,
        original_size: int,
    ) -> Dataset:
        """
        Three-stage augmentation for low-resource dataset leaves:
        1. Repeat tiny datasets
        2. Add a capped donor sample from a configured related leaf
        3. Add a small synthetic SRH template set

        This remains intentionally conservative. The synthetic stage is meant to
        stabilize training on tiny leaves, not replace real SRH data.
        """

        augmented_parts = [dataset]

        if original_size < 500:
            repeats = max(2, 1000 // max(original_size, 1))
            logger.info("[%s] Repeating training set x%s", lang_code, repeats)
            augmented_parts.extend([dataset] * (repeats - 1))

        donor_code = lang_cfg.transfer_from
        if donor_code and donor_code in SUPPORTED_LANGUAGES:
            donor_cfg = SUPPORTED_LANGUAGES[donor_code]
            try:
                donor_ds = self.load_language(donor_code, donor_cfg, augment=False)["train"]
                n_donor = min(len(donor_ds), max(original_size // 2, 200))
                donor_sample = donor_ds.shuffle(seed=self.seed).select(range(n_donor))
                logger.info(
                    "[%s] Adding %s donor samples from %s",
                    lang_code,
                    n_donor,
                    donor_code,
                )
                augmented_parts.append(donor_sample)
            except Exception as exc:
                logger.warning(
                    "[%s] Could not load donor dataset '%s': %s",
                    lang_code,
                    donor_code,
                    exc,
                )

        synthetic = self._generate_synthetic_samples(lang_code, n=min(200, original_size))
        if synthetic:
            augmented_parts.append(Dataset.from_list(synthetic))
            logger.info("[%s] Added %s synthetic samples", lang_code, len(synthetic))

        combined = concatenate_datasets(augmented_parts).shuffle(seed=self.seed)
        logger.info("[%s] Augmented dataset size: %s -> %s", lang_code, original_size, len(combined))
        return combined

    def _generate_synthetic_samples(self, lang_code: str, n: int = 100) -> list[dict]:
        """
        Generate lightweight SRH seed pairs to stabilize tiny training sets.

        These are generic English scaffolds by design. The purpose is to give
        the adapter additional supervised turns, not to fabricate fluent target-
        language data that could pollute the training signal.
        """

        templates = [
            {
                "input": "What are common signs of a sexually transmitted infection?",
                "output": (
                    "Common signs can include unusual discharge, genital sores, pain when "
                    "urinating, itching, lower abdominal pain, or no symptoms at all."
                ),
            },
            {
                "input": "When should someone seek urgent care during pregnancy?",
                "output": (
                    "Urgent care is needed for heavy bleeding, severe abdominal pain, "
                    "convulsions, difficulty breathing, severe headache, or reduced fetal movement."
                ),
            },
            {
                "input": "Why is family planning counseling important?",
                "output": (
                    "Family planning counseling helps people understand available methods, "
                    "side effects, benefits, and how to choose an option that fits their needs."
                ),
            },
            {
                "input": "How can HIV transmission be reduced?",
                "output": (
                    "HIV transmission risk can be reduced through condom use, testing, "
                    "treatment adherence, PrEP when appropriate, and avoiding needle sharing."
                ),
            },
        ]

        synthetic = (templates * ((n // len(templates)) + 1))[:n]
        random.Random(self.seed).shuffle(synthetic)
        for sample in synthetic:
            sample["source"] = "synthetic_template"
            sample["language"] = lang_code
        return synthetic


def collate_fn_factory(processor, max_length: int = 1024):
    """
    Return a simple text-only collator suitable for SFT training.

    The current project path uses text-only SRH examples. This helper is kept in
    place in case you later want to switch back to an explicit custom collator.
    """

    def collate_fn(examples: list[dict]) -> dict:
        texts = [ex["text"] for ex in examples]

        batch = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


def print_dataset_statistics(datasets: dict[str, DatasetDict]) -> None:
    """Print a compact size summary using canonical `train` / `dev` / `test` keys."""

    print("\n" + "═" * 60)
    print(f"{'Dataset':<12} {'Train':>8} {'Dev':>8} {'Test':>8}")
    print("─" * 60)
    for lang_code, dataset in datasets.items():
        train_split = get_split(dataset, "train")
        dev_split = get_split(dataset, "dev")
        test_split = get_split(dataset, "test")
        train_n = len(train_split) if train_split is not None else 0
        dev_n = len(dev_split) if dev_split is not None else 0
        test_n = len(test_split) if test_split is not None else 0
        print(f"{lang_code:<12} {train_n:>8,} {dev_n:>8,} {test_n:>8,}")
    print("═" * 60 + "\n")
