"""
prepare_data.py — Mirror Hub-hosted multilingual SRH shards into local DatasetDicts
===============================================================================

Purpose
-------
The old version of this file converted CSV/JSONL inputs into Hugging Face
datasets. That no longer matches your real workflow because the authoritative
source is already a Hugging Face-hosted multilingual SRH dataset with shard
files arranged by language and country leaf.

This script now exists for a different purpose:
* download selected dataset leaves from the Hub
* normalize them through the shared loader
* save them locally as `DatasetDict.save_to_disk()` caches

This is optional because `train.py` can read directly from the Hub, but local
mirrors remain useful for reproducibility, offline reruns, and faster iteration.
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

from config import SUPPORTED_LANGUAGES, expand_language_selection
from data_utils import MultilingualDatasetBuilder, get_split

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def prepare_language(
    lang_code: str,
    builder: MultilingualDatasetBuilder,
    output_root: Path,
    overwrite: bool = False,
) -> None:
    """
    Download one dataset leaf from the Hub and save it locally.

    The output directory uses the dataset leaf name directly, e.g. `eng_uga/`.
    """

    output_dir = output_root / lang_code
    if output_dir.exists():
        if not overwrite:
            logger.info("[%s] Output already exists at %s; skipping.", lang_code, output_dir)
            return
        shutil.rmtree(output_dir)

    lang_cfg = SUPPORTED_LANGUAGES[lang_code]
    dataset = builder.load_language(lang_code, lang_cfg, augment=False)
    dataset.save_to_disk(str(output_dir))

    train_split = get_split(dataset, "train")
    dev_split = get_split(dataset, "dev")
    test_split = get_split(dataset, "test")
    train_n = len(train_split) if train_split is not None else 0
    dev_n = len(dev_split) if dev_split is not None else 0
    test_n = len(test_split) if test_split is not None else 0
    logger.info(
        "[%s] Mirrored train=%s dev=%s test=%s -> %s",
        lang_code,
        train_n,
        dev_n,
        test_n,
        output_dir,
    )


def parse_args():
    """Define CLI arguments for mirroring selected dataset leaves."""
    parser = argparse.ArgumentParser(
        description="Mirror multilingual SRH dataset leaves from Hugging Face to local disk"
    )
    parser.add_argument(
        "--dataset_repo",
        type=str,
        required=True,
        help="Hugging Face dataset repo id containing the shard layout.",
    )
    parser.add_argument(
        "--dataset_revision",
        type=str,
        default=None,
        help="Optional Hub revision, branch, or commit.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=list(SUPPORTED_LANGUAGES.keys()),
        help="Dataset leaves or language groups to mirror locally.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data",
        help="Destination root for local `save_to_disk()` datasets.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for downloads.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate existing output directories.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN from the environment.",
    )
    return parser.parse_args()


def main():
    """Entry point for mirroring Hub-hosted SRH leaves into local caches."""
    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    try:
        selected_languages = expand_language_selection(args.languages)
    except KeyError as exc:
        raise SystemExit(
            f"Unsupported language or dataset selection: {exc.args[0]}"
        ) from exc

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    builder = MultilingualDatasetBuilder(
        dataset_repo=args.dataset_repo,
        dataset_revision=args.dataset_revision,
        hf_token=hf_token,
        cache_dir=args.cache_dir,
    )

    for lang_code in selected_languages:
        prepare_language(
            lang_code=lang_code,
            builder=builder,
            output_root=output_root,
            overwrite=args.overwrite,
        )

    logger.info("Dataset mirroring complete.")


if __name__ == "__main__":
    main()
