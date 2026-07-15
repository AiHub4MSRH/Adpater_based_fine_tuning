"""
config.py — Training and dataset configuration for multilingual SRH adapters
============================================================================

Design notes
------------
This project trains one adapter for each deployment language target. Some
targets map directly to a single dataset leaf such as `aka_gha` or `lug_uga`.
English and Swahili intentionally combine multiple country leaves into one
adapter each: `eng` and `swa`.

Why this matters:
1. Dataset leaves remain the source-of-truth for loading local or Hub shards.
2. Adapter naming stays aligned with deployment targets, so English produces
   `adapter_eng` and Swahili produces `adapter_swa`.
3. Legacy leaf selections such as `eng_uga` and `swa_ken` are accepted by the
   CLI, but they resolve to the combined adapter target.

Configuration philosophy
------------------------
* `TrainingConfig` holds global model-level defaults shared across all runs.
* `LanguageConfig` can describe either a concrete source leaf or a combined
  adapter target.
* Resource-level tuning still exists because some leaves are lower-resource than
  others and benefit from lower-rank LoRA, more epochs, and donor augmentation.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TrainingConfig:
    """Global settings shared across all adapter runs."""

    model_id: str = "epfl-llm/meditron-7b"
    max_seq_length: int = 1024
    seed: int = 42
    training_precision: str = "full_lora"


@dataclass(frozen=True)
class LanguageConfig:
    """Training configuration for a source leaf or adapter target."""

    dataset_id: str
    language_code: str
    language_name: str
    country_code: str
    country_name: str
    script: str
    resource_level: str

    lora_r: int = 32
    num_epochs: int = 1
    batch_size: int = 4
    grad_accumulation: int = 4
    learning_rate: float = 2e-4
    early_stopping_patience: Optional[int] = 3
    early_stopping_threshold: float = 0.001
    transfer_from: Optional[str] = None
    source_datasets: tuple[str, ...] = ()

    @property
    def display_name(self) -> str:
        return f"{self.language_name} ({self.country_name})"

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


SOURCE_DATASETS: dict[str, LanguageConfig] = {
    # ── Akan / Ghana ────────────────────────────────────────────────────────
    "aka_gha": LanguageConfig(
        dataset_id="aka_gha",
        language_code="aka",
        language_name="Akan",
        country_code="gha",
        country_name="Ghana",
        script="latin",
        resource_level="low",
        lora_r=16,
        num_epochs=10,
        learning_rate=1e-4,
        early_stopping_patience=5,
        transfer_from="eng",
    ),
    # ── Amharic / Ethiopia ──────────────────────────────────────────────────
    "amh_eth": LanguageConfig(
        dataset_id="amh_eth",
        language_code="amh",
        language_name="Amharic",
        country_code="eth",
        country_name="Ethiopia",
        script="geez",
        resource_level="low",
        lora_r=16,
        num_epochs=10,
        learning_rate=1e-4,
        early_stopping_patience=5,
        transfer_from="eng",
    ),
    # ── English variants ────────────────────────────────────────────────────
    "eng_eth": LanguageConfig(
        dataset_id="eng_eth",
        language_code="eng",
        language_name="English",
        country_code="eth",
        country_name="Ethiopia",
        script="latin",
        resource_level="high",
        lora_r=32,
        num_epochs=5,
        learning_rate=2e-4,
        early_stopping_patience=3,
    ),
    "eng_gha": LanguageConfig(
        dataset_id="eng_gha",
        language_code="eng",
        language_name="English",
        country_code="gha",
        country_name="Ghana",
        script="latin",
        resource_level="high",
        lora_r=32,
        num_epochs=5,
        learning_rate=2e-4,
        early_stopping_patience=3,
    ),
    "eng_ken": LanguageConfig(
        dataset_id="eng_ken",
        language_code="eng",
        language_name="English",
        country_code="ken",
        country_name="Kenya",
        script="latin",
        resource_level="high",
        lora_r=32,
        num_epochs=5,
        learning_rate=2e-4,
        early_stopping_patience=3,
    ),
    "eng_uga": LanguageConfig(
        dataset_id="eng_uga",
        language_code="eng",
        language_name="English",
        country_code="uga",
        country_name="Uganda",
        script="latin",
        resource_level="high",
        lora_r=32,
        num_epochs=5,
        learning_rate=2e-4,
        early_stopping_patience=3,
    ),
    # ── Luganda / Uganda ────────────────────────────────────────────────────
    "lug_uga": LanguageConfig(
        dataset_id="lug_uga",
        language_code="lug",
        language_name="Luganda",
        country_code="uga",
        country_name="Uganda",
        script="latin",
        resource_level="low",
        lora_r=16,
        num_epochs=10,
        learning_rate=1e-4,
        early_stopping_patience=5,
        transfer_from="eng",
    ),
    # ── Swahili variants ────────────────────────────────────────────────────
    "swa_ken": LanguageConfig(
        dataset_id="swa_ken",
        language_code="swa",
        language_name="Swahili",
        country_code="ken",
        country_name="Kenya",
        script="latin",
        resource_level="medium",
        lora_r=32,
        num_epochs=6,
        learning_rate=2e-4,
        early_stopping_patience=3,
    ),
    "swa_uga": LanguageConfig(
        dataset_id="swa_uga",
        language_code="swa",
        language_name="Swahili",
        country_code="uga",
        country_name="Uganda",
        script="latin",
        resource_level="medium",
        lora_r=32,
        num_epochs=6,
        learning_rate=2e-4,
        early_stopping_patience=3,
        transfer_from="swa_ken",
    ),
}


SUPPORTED_LANGUAGES: dict[str, LanguageConfig] = {
    "aka_gha": SOURCE_DATASETS["aka_gha"],
    "amh_eth": SOURCE_DATASETS["amh_eth"],
    "eng": LanguageConfig(
        dataset_id="eng",
        language_code="eng",
        language_name="English",
        country_code="multi",
        country_name="Ethiopia, Ghana, Kenya, Uganda",
        script="latin",
        resource_level="high",
        lora_r=32,
        num_epochs=5,
        learning_rate=2e-4,
        early_stopping_patience=3,
        source_datasets=("eng_eth", "eng_gha", "eng_ken", "eng_uga"),
    ),
    "lug_uga": SOURCE_DATASETS["lug_uga"],
    "swa": LanguageConfig(
        dataset_id="swa",
        language_code="swa",
        language_name="Swahili",
        country_code="multi",
        country_name="Kenya, Uganda",
        script="latin",
        resource_level="medium",
        lora_r=32,
        num_epochs=6,
        learning_rate=2e-4,
        early_stopping_patience=3,
        source_datasets=("swa_ken", "swa_uga"),
    ),
}


LANGUAGE_GROUPS: dict[str, list[str]] = {
    "aka": ["aka_gha"],
    "amh": ["amh_eth"],
    "eng": ["eng"],
    "eng_eth": ["eng"],
    "eng_gha": ["eng"],
    "eng_ken": ["eng"],
    "eng_uga": ["eng"],
    "lug": ["lug_uga"],
    "swa": ["swa"],
    "swa_ken": ["swa"],
    "swa_uga": ["swa"],
}


def expand_language_selection(selections: list[str]) -> list[str]:
    """
    Expand CLI selections into adapter targets.

    Examples:
    * `["eng"]` resolves to the combined English adapter target.
    * `["swa_ken"]` resolves to the combined Swahili adapter target.
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
