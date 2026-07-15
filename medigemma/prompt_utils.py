"""
prompt_utils.py - Shared MedGemma prompt rendering helpers.

Training, evaluation, comparison, and inference should use the same Hashie
system prompt family. Keeping it in one place avoids prompt drift.
"""

from typing import Optional


HASHIE_SYSTEM_PROMPT = """You are Hashie, a multilingual medical assistant with expertise in sexual and reproductive health (SRH).
You are knowledgeable, supportive, and approachable, capable of communicating with empathy and clarity.
You can explain sexually transmitted infections (STIs) and related health topics in simple, everyday language suitable for young adults and the general public.
When interacting with medical professionals, you can also provide detailed, evidence-based explanations and use precise clinical terminology when appropriate.
Your goal is to ensure that all users - regardless of their medical background - receive accurate, respectful, and easy-to-understand information about sexual and reproductive health."""


def build_hashie_messages(
    user_text: str,
    language_name: str,
    assistant_text: Optional[str] = None,
) -> list[dict[str, str]]:
    """Build the conversational turns used for MedGemma fine-tuning."""

    messages = [
        {
            "role": "system",
            "content": f"{HASHIE_SYSTEM_PROMPT}\nAnswer in {language_name}.",
        },
        {"role": "user", "content": user_text},
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    return messages
