"""
config.py — Training and dataset configuration for multilingual SRH adapters
============================================================================

Design notes
------------
This project no longer treats a bare language code such as `eng` or `swa` as
the unit of training. Your real Hugging Face dataset is organised into leaves
such as `eng_uga`, `eng_ken`, `swa_uga`, and `aka_gha`, so the registry below
models those leaves directly.

Why this matters:
1. The country suffix is part of the dataset identity, not incidental metadata.
   `eng_uga` and `eng_gha` may differ in terminology, orthography, examples,
   and SRH phrasing conventions even though both are English.
2. Adapter naming should stay aligned with the actual data leaf to avoid
   ambiguity during training, evaluation, and deployment.
3. The CLI still supports grouped selections such as `eng` or `swa`, but those
   are now just conveniences that expand to the concrete dataset leaves.

Configuration philosophy
------------------------
* `TrainingConfig` holds global model-level defaults shared across all runs.
* `LanguageConfig` is really a per-dataset-leaf config. The historical name is
  retained to minimise churn across the rest of the codebase.
* Resource-level tuning still exists because some leaves are lower-resource than
  others and benefit from lower-rank LoRA, more epochs, and donor augmentation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainingConfig:
    """Global settings shared across all adapter runs."""

    model_id: str = "google/medgemma-1.5-4b-it"
    max_seq_length: int = 1536
    max_length: int = 512
    seed: int = 42


@dataclass(frozen=True)
class LanguageConfig:
    """Per-variant training configuration for a Hub-hosted dataset leaf."""

    dataset_id: str
    language_code: str
    language_name: str
    country_code: str
    country_name: str
    script: str
    resource_level: str

    lora_r: int = 64
    num_epochs: int = 3
    batch_size: int = 4
    grad_accumulation: int = 8
    learning_rate: float = 1e-5
    early_stopping_patience: Optional[int] = 3
    transfer_from: Optional[str] = None

    @property
    def display_name(self) -> str:
        if self.country_name:
            return f"{self.language_name} ({self.country_name})"
        return self.language_name

    @property
    def hub_subdir(self) -> str:
        return f"{self.language_code}/{self.dataset_id}"

    @property
    def shard_subdirs(self) -> tuple[str, ...]:
        """
        Supported shard directory layouts for this dataset leaf.

        We prefer the nested `<language>/<dataset_id>/` layout because that is
        what the original project documentation described. Some mirrors differ
        in both nesting and casing, for example `aka/aka_gha/` versus
        `Aka/Aka_Gha/`, so we accept both to keep data loading resilient across
        local exports and Hub repos.
        """

        candidates = (
            self.hub_subdir,
            self.dataset_id,
            f"{self._title_case_token(self.language_code)}/{self._title_case_token(self.dataset_id)}",
            self._title_case_token(self.dataset_id),
        )
        return tuple(dict.fromkeys(candidates))

    def split_glob(self, split_name: str) -> str:
        return f"{self.hub_subdir}/{split_name}-*"

    def split_globs(self, split_name: str) -> list[str]:
        return [f"{subdir}/{split_name}-*" for subdir in self.shard_subdirs]

    @staticmethod
    def _title_case_token(value: str) -> str:
        return "_".join(part.capitalize() for part in value.split("_"))


SUPPORTED_LANGUAGES: dict[str, LanguageConfig] = {
    # ── Akan (all countries combined) ───────────────────────────────────────
    "aka": LanguageConfig(
        dataset_id="aka",
        language_code="aka",
        language_name="Akan",
        country_code="",
        country_name="",
        script="latin",
        resource_level="low",
        transfer_from="eng",
    ),
    # ── Amharic (all countries combined) ────────────────────────────────────
    "amh": LanguageConfig(
        dataset_id="amh",
        language_code="amh",
        language_name="Amharic",
        country_code="",
        country_name="",
        script="geez",
        resource_level="low",
        transfer_from="eng",
    ),
    # ── English (all countries combined) ────────────────────────────────────
    "eng": LanguageConfig(
        dataset_id="eng",
        language_code="eng",
        language_name="English",
        country_code="",
        country_name="",
        script="latin",
        resource_level="high",
    ),
    # ── Luganda (all countries combined) ────────────────────────────────────
    "lug": LanguageConfig(
        dataset_id="lug",
        language_code="lug",
        language_name="Luganda",
        country_code="",
        country_name="",
        script="latin",
        resource_level="low",
        transfer_from="eng",
    ),
    # ── Swahili (all countries combined) ────────────────────────────────────
    "swa": LanguageConfig(
        dataset_id="swa",
        language_code="swa",
        language_name="Swahili",
        country_code="",
        country_name="",
        script="latin",
        resource_level="medium",
    ),
}


LANGUAGE_GROUPS: dict[str, list[str]] = {
    "aka": ["aka"],
    "amh": ["amh"],
    "eng": ["eng"],
    "lug": ["lug"],
    "swa": ["swa"],
}


def expand_language_selection(selections: list[str]) -> list[str]:
    """
    Expand CLI selections so users can pass either dataset leaves (`eng_uga`)
    or base language groups (`eng`).

    Examples:
    * `["eng"]` expands to all English dataset leaves.
    * `["swa_ken", "eng"]` preserves the explicitly requested leaf and then
      appends the grouped English leaves without duplicates.
    """

    expanded: list[str] = []
    seen: set[str] = set()

    for selection in selections:
        if selection in SUPPORTED_LANGUAGES:
            candidates = [selection]
        elif selection in LANGUAGE_GROUPS:
            candidates = LANGUAGE_GROUPS[selection]
        else:
            raise KeyError(selection)

        for candidate in candidates:
            if candidate not in seen:
                expanded.append(candidate)
                seen.add(candidate)

    return expanded
