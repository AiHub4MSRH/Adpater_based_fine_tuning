"""
prompt_utils.py — Shared Meditron prompt rendering helpers
==========================================================

The Meditron fine-tuning data was curated with the Hashie system prompt and a
LLaMA-style chat template. We keep that formatting in one place so training,
evaluation, comparison, and inference all render the same prompt family.
"""

from __future__ import annotations

HASHIE_SYSTEM_PROMPT = """You are Hashie, a multilingual medical assistant with expertise in sexual and reproductive health (SRH).
You are knowledgeable, supportive, and approachable, capable of communicating with empathy and clarity.
You can explain sexually transmitted infections (STIs) and related health topics in simple, everyday language suitable for young adults and the general public.
When interacting with medical professionals, you can also provide detailed, evidence-based explanations and use precise clinical terminology when appropriate.
Your goal is to ensure that all users - regardless of their medical background - receive accurate, respectful, and easy-to-understand information about sexual and reproductive health.
"""


def build_hashie_messages(user_text: str, assistant_text: str | None = None) -> list[dict[str, str]]:
    """Build the conversational turns used for Meditron fine-tuning."""

    messages = [
        {"role": "system", "content": HASHIE_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


def render_meditron_chat(
    user_text: str,
    assistant_text: str | None = None,
    *,
    add_generation_prompt: bool = False,
) -> str:
    """
    Render the LLaMA-style chat template used by the dataset curation notebook.

    This keeps Meditron aligned with the prompt family used to prepare the
    fine-tuning dataset instead of falling back to a generic instruction block.
    """

    messages = build_hashie_messages(user_text, assistant_text)
    parts = ["<|begin_of_text|>"]

    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            f"{content}<|eot_id|>"
        )

    if add_generation_prompt:
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")

    return "".join(parts)
