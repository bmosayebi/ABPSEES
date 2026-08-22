"""Prompt templates for Qwen2.5 instruction fine-tuning."""

from __future__ import annotations

import json
from typing import Any

from project.constants import ASPECTS

SYSTEM_PROMPT_FA = (
    "شما یک سیستم استخراج ترجیحات لپ‌تاپ هستید. "
    "متن فارسی کاربر را دریافت کنید و فقط یک شیء JSON خروجی دهید. "
    "برای هر جنبه (performance, portability, design, durability, cost_effectiveness) "
    "یک امتیاز بین ۰ و ۱ و یک متن evidence برگردانید. "
    "evidence باید دقیقاً زیررشته‌ای از متن ورودی باشد و هرگز نباید بازنویسی یا توضیح اضافه شود. "
    "هیچ توضیح، استدلال یا متن اضافه‌ای خارج از JSON ننویسید."
)

USER_PROMPT_TEMPLATE = "متن کاربر:\n{text}"


def format_training_labels(labels: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Strip span metadata, keeping only score and evidence for the training target.

    Args:
        labels: Full labels dict possibly containing span fields.

    Returns:
        Labels dict with only ``score`` and ``evidence`` per aspect.
    """
    output: dict[str, dict[str, Any]] = {}
    for aspect in ASPECTS:
        label = labels[aspect]
        output[aspect] = {
            "score": label["score"],
            "evidence": label.get("evidence", ""),
        }
    return output


def labels_to_json_string(labels: dict[str, Any]) -> str:
    """Serialize training labels to a compact JSON string.

    Args:
        labels: Labels dict (may include span metadata).

    Returns:
        JSON string with fixed aspect key order.
    """
    training_labels = format_training_labels(labels)
    return json.dumps(training_labels, ensure_ascii=False, separators=(",", ": "))


def build_chat_messages(text: str, labels: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Build Qwen chat messages for SFT or inference.

    Args:
        text: User input Persian text.
        labels: Optional ground-truth labels for training (assistant turn).

    Returns:
        List of chat message dicts with ``role`` and ``content``.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT_FA},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
    ]
    if labels is not None:
        messages.append({"role": "assistant", "content": labels_to_json_string(labels)})
    return messages


def format_prompt(tokenizer: Any, text: str, labels: dict[str, Any] | None = None) -> str:
    """Apply the tokenizer chat template to build a full prompt string.

    Args:
        tokenizer: HuggingFace tokenizer with chat template.
        text: User input text.
        labels: Optional labels for training.

    Returns:
        Formatted prompt string.
    """
    messages = build_chat_messages(text, labels=labels)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=labels is None,
    )


def get_response_template(tokenizer: Any) -> str:
    """Return the substring marking the start of the assistant response.

    Used to mask prompt tokens during training.

    Args:
        tokenizer: HuggingFace tokenizer.

    Returns:
        Response template marker string.
    """
    # Qwen chat template uses this marker before assistant content.
    marker = "<|im_start|>assistant\n"
    if marker not in tokenizer.chat_template and hasattr(tokenizer, "apply_chat_template"):
        # Fallback: build a minimal prompt and extract assistant prefix.
        dummy = tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if "assistant" in dummy:
            idx = dummy.rfind("assistant")
            return dummy[idx - 10 :] if idx >= 10 else "assistant"
    return marker
