"""Evidence validation, span conversion, and SFT dataset preparation."""

from __future__ import annotations

import logging
import unicodedata
from copy import deepcopy
from typing import Any

from project.config import AppConfig
from project.constants import ASPECTS, INVALID_CHAR_SPAN, INVALID_TOKEN_SPAN
from project.prompts import build_chat_messages
from project.utils import load_json, save_json

logger = logging.getLogger(__name__)


def normalize_text(text: str, normalize_unicode: bool) -> str:
    """Optionally apply NFC Unicode normalization.

    Args:
        text: Input string.
        normalize_unicode: Whether to normalize.

    Returns:
        Normalized text.
    """
    if normalize_unicode:
        return unicodedata.normalize("NFC", text)
    return text


def find_evidence_char_span(text: str, evidence: str) -> tuple[int, int]:
    """Find the leftmost character span of evidence in text.

    Args:
        text: Source text.
        evidence: Evidence substring to locate.

    Returns:
        ``(start, end)`` char span with end exclusive, or ``(-1, -1)`` if empty.

    Raises:
        ValueError: If evidence is non-empty but not found in text.
    """
    if evidence == "":
        return INVALID_CHAR_SPAN

    start = text.find(evidence)
    if start == -1:
        raise ValueError(f"Evidence not found in text: {evidence!r}")
    return start, start + len(evidence)


def char_span_to_token_span(
    text: str,
    char_span: tuple[int, int],
    tokenizer: Any,
) -> tuple[int, int]:
    """Convert a character span to inclusive token indices.

    Args:
        text: Source text tokenized.
        char_span: ``(start_char, end_char)`` with end exclusive.
        tokenizer: HuggingFace tokenizer.

    Returns:
        ``(start_tok, end_tok)`` inclusive token indices, or ``(-1, -1)`` if empty.
    """
    if char_span == INVALID_CHAR_SPAN:
        return INVALID_TOKEN_SPAN

    start_char, end_char = char_span
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoding["offset_mapping"]
    if not offsets:
        return INVALID_TOKEN_SPAN

    token_indices: list[int] = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= start_char:
            continue
        if tok_start >= end_char:
            break
        token_indices.append(idx)

    if not token_indices:
        raise ValueError(
            f"Could not map char span {char_span} to tokens for text snippet: {text[:80]!r}"
        )
    return token_indices[0], token_indices[-1]


def validate_and_enrich_sample(
    sample: dict[str, Any],
    tokenizer: Any,
    normalize_unicode: bool,
    on_invalid: str,
) -> dict[str, Any] | None:
    """Validate evidence substrings and attach char/token spans.

    Args:
        sample: Raw sample with ``text`` and ``labels``.
        tokenizer: HuggingFace tokenizer.
        normalize_unicode: Whether to NFC-normalize text and evidence.
        on_invalid: ``discard`` or ``raise`` when evidence is invalid.

    Returns:
        Enriched sample or ``None`` if discarded.
    """
    text = normalize_text(sample["text"], normalize_unicode)
    enriched = deepcopy(sample)
    enriched["text"] = text
    enriched_labels: dict[str, Any] = {}

    for aspect in ASPECTS:
        label = deepcopy(sample["labels"][aspect])
        evidence = normalize_text(str(label.get("evidence", "")), normalize_unicode)
        label["evidence"] = evidence

        try:
            char_span = find_evidence_char_span(text, evidence)
            token_span = char_span_to_token_span(text, char_span, tokenizer)
        except ValueError as exc:
            msg = f"Invalid evidence for aspect '{aspect}': {exc}"
            if on_invalid == "raise":
                raise ValueError(msg) from exc
            logger.warning("%s — discarding sample.", msg)
            return None

        label["char_span"] = list(char_span)
        label["token_span"] = list(token_span)
        enriched_labels[aspect] = label

    enriched["labels"] = enriched_labels
    return enriched


def preprocess_samples(
    samples: list[dict[str, Any]],
    tokenizer: Any,
    config: AppConfig,
) -> list[dict[str, Any]]:
    """Validate and enrich a batch of samples.

    Args:
        samples: Raw samples.
        tokenizer: HuggingFace tokenizer.
        config: Application configuration.

    Returns:
        List of enriched samples (invalid samples discarded).
    """
    enriched: list[dict[str, Any]] = []
    discarded = 0
    for sample in samples:
        result = validate_and_enrich_sample(
            sample,
            tokenizer,
            config.data.normalize_unicode,
            config.data.evidence_on_invalid,
        )
        if result is None:
            discarded += 1
        else:
            enriched.append(result)

    logger.info(
        "Preprocessed %d samples (%d discarded).",
        len(enriched),
        discarded,
    )
    return enriched


def save_enriched_split(
    samples: list[dict[str, Any]],
    tokenizer: Any,
    config: AppConfig,
    split_name: str,
) -> list[dict[str, Any]]:
    """Preprocess and save an enriched split.

    Args:
        samples: Raw split samples.
        tokenizer: HuggingFace tokenizer.
        config: Application configuration.
        split_name: Split identifier (e.g. ``train``).

    Returns:
        Enriched samples.
    """
    enriched = preprocess_samples(samples, tokenizer, config)
    out_path = config.paths.processed_dir / f"{split_name}_enriched.json"
    save_json(enriched, out_path)
    return enriched


def build_sft_record(sample: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    """Build a single SFT text record for the HuggingFace datasets library.

    Args:
        sample: Enriched sample.
        tokenizer: HuggingFace tokenizer.

    Returns:
        Dict with ``text`` field containing the full chat-formatted example.
    """
    messages = build_chat_messages(sample["text"], labels=sample["labels"])
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": text}


def build_sft_dataset(
    samples: list[dict[str, Any]],
    tokenizer: Any,
) -> list[dict[str, str]]:
    """Convert enriched samples to SFT records.

    Args:
        samples: Enriched samples.
        tokenizer: HuggingFace tokenizer.

    Returns:
        List of dicts with ``text`` key.
    """
    return [build_sft_record(sample, tokenizer) for sample in samples]


class CompletionOnlyCollator:
    """Data collator that pads batches and masks prompt tokens for causal LM training."""

    def __init__(self, tokenizer: Any, response_template: str) -> None:
        """Initialize collator with tokenizer and response marker.

        Args:
            tokenizer: HuggingFace tokenizer.
            response_template: Substring marking assistant response start.
        """
        self.tokenizer = tokenizer
        self.response_template = response_template
        self.pad_token_id = tokenizer.pad_token_id
        self.response_token_ids = tokenizer.encode(
            response_template, add_special_tokens=False
        )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad, collate, and mask labels before the assistant response template.

        Args:
            features: List of feature dicts with ``input_ids`` and ``labels``.

        Returns:
            Batched tensors with dynamic padding per batch.
        """
        import torch

        if not features:
            raise ValueError("Cannot collate an empty feature list")

        max_len = max(len(feature["input_ids"]) for feature in features)
        input_ids_batch: list[list[int]] = []
        attention_mask_batch: list[list[int]] = []
        labels_batch: list[list[int]] = []

        for feature in features:
            input_ids = list(feature["input_ids"])
            labels = list(feature.get("labels", input_ids))
            attention_mask = list(
                feature.get("attention_mask", [1] * len(input_ids))
            )
            pad_len = max_len - len(input_ids)

            input_ids_batch.append(input_ids + [self.pad_token_id] * pad_len)
            attention_mask_batch.append(attention_mask + [0] * pad_len)
            labels_batch.append(labels + [-100] * pad_len)

        batch = {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
        }

        for idx in range(batch["labels"].size(0)):
            input_ids = batch["input_ids"][idx].tolist()
            resp_start = self._find_response_start(input_ids)
            if resp_start is not None:
                batch["labels"][idx, :resp_start] = -100

        return batch

    def _find_response_start(self, input_ids: list[int]) -> int | None:
        """Find token index where the assistant response begins."""
        template_len = len(self.response_token_ids)
        for i in range(len(input_ids) - template_len + 1):
            if input_ids[i : i + template_len] == self.response_token_ids:
                return i + template_len
        return None


def tokenize_sft_example(
    example: dict[str, str],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    """Tokenize an SFT example with truncation.

    Args:
        example: Dict with ``text`` field.
        tokenizer: HuggingFace tokenizer.
        max_seq_length: Maximum sequence length.

    Returns:
        Tokenized example with ``input_ids``, ``attention_mask``, ``labels``.
    """
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_tensors=None,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def load_enriched_split(config: AppConfig, split_name: str) -> list[dict[str, Any]]:
    """Load enriched split from disk if available.

    Args:
        config: Application configuration.
        split_name: Split name without suffix.

    Returns:
        Enriched samples.
    """
    path = config.paths.processed_dir / f"{split_name}_enriched.json"
    if path.exists():
        return load_json(path)
    raw_path = config.paths.processed_dir / f"{split_name}.json"
    raise FileNotFoundError(
        f"Enriched split not found at {path}. Run preprocessing on {raw_path} first."
    )
