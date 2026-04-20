"""
run_inference_from_hub.py — Sample inference script for Hub-hosted adapters
===========================================================================

Purpose
-------
This script downloads a specific adapter leaf from a Hugging Face repo created
by `push_adapters_to_hub.py`, loads the base model referenced by the adapter
metadata, and runs a single SRH generation example.

Expected Hub layout
-------------------
<repo root>/
  adapters/
    eng_uga/
      adapter_meta.json
      adapter_config.json
      adapter_model.safetensors
      ...

Example
-------
python3 run_inference_from_hub.py \
  --adapter_repo your-org/hashie-srh-adapters \
  --adapter_name eng_uga \
  --prompt "What are common symptoms of an STI?"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from config import TrainingConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def resolve_adapter_dir(adapter_root: Path, adapter_name: str) -> Path:
    """
    Resolve either a flat published adapter folder or a nested named-adapter folder.
    """

    flat_config = adapter_root / "adapter_config.json"
    nested_dir = adapter_root / adapter_name
    nested_config = nested_dir / "adapter_config.json"

    if flat_config.exists():
        return adapter_root
    if nested_config.exists():
        return nested_dir
    return adapter_root


def parse_args():
    """Define CLI arguments for downloading and running one adapter leaf."""

    parser = argparse.ArgumentParser(
        description="Run inference against an adapter that was published to the Hugging Face Hub"
    )
    parser.add_argument(
        "--adapter_repo",
        type=str,
        required=True,
        help="Hub repo containing the published adapters.",
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        required=True,
        help="Dataset leaf name, for example `eng_uga` or `swa_ken`.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt to run against the selected adapter.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hub revision, branch, or commit.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help="Optional override for the base model id. Defaults to adapter metadata.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load the base model in 4-bit mode for lower memory usage.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for Hub downloads.",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="Hugging Face token. Falls back to HF_TOKEN from the environment.",
    )
    return parser.parse_args()


def download_adapter(adapter_repo: str, adapter_name: str, revision: str | None, cache_dir: str | None, hf_token: str | None) -> Path:
    """
    Download only the requested adapter subtree from the Hub repo.

    The script uses `snapshot_download` with `allow_patterns` so it does not
    need to fetch every adapter in the repository.
    """

    snapshot_dir = snapshot_download(
        repo_id=adapter_repo,
        repo_type="model",
        revision=revision,
        allow_patterns=[f"adapters/{adapter_name}/*"],
        cache_dir=cache_dir,
        token=hf_token,
    )
    adapter_root = Path(snapshot_dir) / "adapters" / adapter_name
    if not adapter_root.exists():
        raise FileNotFoundError(
            f"Adapter '{adapter_name}' was not found under adapters/{adapter_name} in {adapter_repo}"
        )
    return resolve_adapter_dir(adapter_root, adapter_name)


def load_adapter_metadata(adapter_dir: Path) -> dict:
    """Read adapter metadata to recover the intended base model and language."""

    meta_path = adapter_dir / "adapter_meta.json"
    if not meta_path.exists():
        return {}
    with open(meta_path, encoding="utf-8") as handle:
        return json.load(handle)


def load_model_and_processor(base_model_id: str, adapter_dir: Path, load_in_4bit: bool):
    """Load the base model, then attach the downloaded LoRA adapter."""

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    logger.info("Loading base model %s", base_model_id)
    base_model = AutoModelForImageTextToText.from_pretrained(base_model_id, **kwargs)

    try:
        processor = AutoProcessor.from_pretrained(str(adapter_dir))
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model_id)

    model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    model.eval()
    return model, processor


def build_prompt(prompt: str, language_name: str) -> list[dict[str, str]]:
    """Create the same chat-style prompt family used during training."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are Hashie, a multi-lingual medical assistant with expertise in "
                "sexual and reproductive health. Provide accurate, respectful, and "
                f"easy-to-understand information. Answer in {language_name}."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return messages


def main():
    """Download the adapter, run one generation call, and print the response."""

    args = parse_args()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    adapter_dir = download_adapter(
        adapter_repo=args.adapter_repo,
        adapter_name=args.adapter_name,
        revision=args.revision,
        cache_dir=args.cache_dir,
        hf_token=hf_token,
    )

    metadata = load_adapter_metadata(adapter_dir)
    base_model_id = args.base_model or metadata.get("base_model") or TrainingConfig().model_id
    language_name = metadata.get("language_name", args.adapter_name)

    model, processor = load_model_and_processor(
        base_model_id=base_model_id,
        adapter_dir=adapter_dir,
        load_in_4bit=args.load_in_4bit,
    )

    messages = build_prompt(args.prompt, language_name)
    text = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = processor(text=text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output_ids[0][prompt_length:]
    response = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"Adapter repo: {args.adapter_repo}")
    print(f"Adapter name: {args.adapter_name}")
    print(f"Base model:   {base_model_id}")
    print()
    print(response)


if __name__ == "__main__":
    main()
