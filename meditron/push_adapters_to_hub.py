"""
push_adapters_to_hub.py — Publish local adapter folders into one Hugging Face repo
===============================================================================

Purpose
-------
Training writes adapters locally as:

    adapters/
      adapter_eng_uga/
      adapter_swa_ken/
      ...

This script publishes those folders into a single Hugging Face model repo under:

    adapters/<dataset_id>/

Why a single repo?
------------------
* It keeps all multilingual adapter leaves versioned together.
* It makes discovery easier for downstream inference scripts.
* It lets us publish a shared manifest and model card once.

Repository layout on the Hub
----------------------------
<repo root>/
  README.md
  adapters/
    manifest.json
    eng_uga/
      adapter_config.json
      adapter_model.safetensors
      adapter_meta.json
      ...
    swa_ken/
      ...
  reports/
    eval_report.json   (optional)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi

from config import expand_language_selection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def resolve_publish_dir(adapter_root: Path, dataset_id: str) -> Path:
    """
    Resolve the folder that actually contains the PEFT adapter files.

    Training with a named adapter can produce either:
    * adapter_<id>/adapter_config.json
    * adapter_<id>/<id>/adapter_config.json
    """

    flat_config = adapter_root / "adapter_config.json"
    nested_dir = adapter_root / dataset_id
    nested_config = nested_dir / "adapter_config.json"

    if flat_config.exists():
        return adapter_root
    if nested_config.exists():
        return nested_dir
    return adapter_root


def parse_args():
    """Define CLI arguments for publishing adapters to a model repository."""

    parser = argparse.ArgumentParser(
        description="Push one or more local adapter folders to a Hugging Face model repo"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="Target Hugging Face model repo, for example `org/hashie-srh-adapters`.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./adapters",
        help="Local directory containing adapter_* folders.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Optional dataset leaves or grouped language selections to publish.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional branch or PR revision to upload into.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private if it does not already exist.",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default="Upload multilingual SRH adapters",
        help="Commit message to use for Hub uploads.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN from the environment.",
    )
    return parser.parse_args()


def resolve_adapter_dirs(output_root: Path, selections: list[str] | None) -> list[tuple[str, Path]]:
    """
    Resolve which local adapter folders should be uploaded.

    If no explicit selections are provided, all `adapter_*` directories under
    `output_root` are published.
    """

    if selections:
        dataset_ids = expand_language_selection(selections)
        adapter_dirs = [(dataset_id, output_root / f"adapter_{dataset_id}") for dataset_id in dataset_ids]
    else:
        adapter_dirs = []
        for path in sorted(output_root.glob("adapter_*")):
            if path.is_dir():
                dataset_id = path.name.removeprefix("adapter_")
                adapter_dirs.append((dataset_id, path))

    missing = [str(path) for _, path in adapter_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "The following adapter directories were not found:\n" + "\n".join(missing)
        )

    return [
        (dataset_id, resolve_publish_dir(adapter_dir, dataset_id))
        for dataset_id, adapter_dir in adapter_dirs
    ]


def load_adapter_metadata(adapter_dir: Path, dataset_id: str) -> dict:
    """Read `adapter_meta.json` if present so it can be added to the manifest."""

    meta_path = adapter_dir / "adapter_meta.json"
    if not meta_path.exists():
        return {"dataset_id": dataset_id}

    with open(meta_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata.setdefault("dataset_id", dataset_id)
    return metadata


def build_manifest(repo_id: str, adapters: list[tuple[str, Path]]) -> dict:
    """Build a lightweight machine-readable index of published adapters."""

    manifest = {
        "repo_id": repo_id,
        "repo_format": "multilingual_srh_adapter_bundle",
        "adapters": {},
    }
    for dataset_id, adapter_dir in adapters:
        manifest["adapters"][dataset_id] = load_adapter_metadata(adapter_dir, dataset_id)
    return manifest


def build_model_card(repo_id: str, manifest: dict) -> str:
    """Create a simple Hub README describing the adapter bundle."""

    adapter_names = sorted(manifest["adapters"])
    adapter_lines = "\n".join(f"- `{name}`" for name in adapter_names)

    return f"""# Multilingual SRH LoRA Adapters

This repository bundles language- and country-specific LoRA adapters for the SRH
assistant pipeline.

## Repository layout

Adapters are stored under:

```text
adapters/<dataset_id>/
```

Available adapters:

{adapter_lines}

## Example usage

```bash
python3 run_inference_from_hub.py \\
  --adapter_repo {repo_id} \\
  --adapter_name {adapter_names[0] if adapter_names else "eng_uga"} \\
  --prompt "What are common symptoms of an STI?"
```
"""


def main():
    """Create the repo if needed, then upload the selected adapter folders."""

    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    output_root = Path(args.output_root)
    adapter_dirs = resolve_adapter_dirs(output_root, args.languages)

    api = HfApi(token=hf_token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    for dataset_id, adapter_dir in adapter_dirs:
        logger.info("Uploading %s from %s", dataset_id, adapter_dir)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(adapter_dir),
            path_in_repo=f"adapters/{dataset_id}",
            commit_message=f"{args.commit_message}: {dataset_id}",
            revision=args.revision,
        )

    manifest = build_manifest(args.repo_id, adapter_dirs)
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"),
        path_in_repo="adapters/manifest.json",
        commit_message=args.commit_message,
        revision=args.revision,
    )

    model_card = build_model_card(args.repo_id, manifest)
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="model",
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        commit_message=args.commit_message,
        revision=args.revision,
    )

    eval_report = output_root / "eval_report.json"
    if eval_report.exists():
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(eval_report),
            path_in_repo="reports/eval_report.json",
            commit_message=args.commit_message,
            revision=args.revision,
        )

    logger.info("Finished publishing adapters to https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
