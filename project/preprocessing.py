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


def map_char_span_to_tokens(
    offsets: list[tuple[int, int]],
    char_span: tuple[int, int],
) -> tuple[int, int]:
    """Map a character span to inclusive token indices given an offset mapping.

    Pure function (no tokenizer call) so it can be reused against offsets
    produced from tokenizing a *larger enclosing string* -- e.g. mapping an
    evidence span, originally located within the raw user text, onto the
    tokenization of the full chat-formatted training sequence (see
    :func:`build_aspect_sequence_features`).

    Args:
        offsets: Per-token ``(start_char, end_char)`` offsets (end exclusive)
            as returned by a fast tokenizer's ``return_offsets_mapping``.
        char_span: ``(start_char, end_char)`` with end exclusive, in the
            same coordinate system as ``offsets``.

    Returns:
        ``(start_tok, end_tok)`` inclusive token indices, or ``(-1, -1)`` if
        ``char_span`` is the empty-evidence sentinel or no token overlaps it.
    """
    if char_span == INVALID_CHAR_SPAN:
        return INVALID_TOKEN_SPAN

    start_char, end_char = char_span
    token_indices: list[int] = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_end <= start_char:
            continue
        if tok_start >= end_char:
            break
        token_indices.append(idx)

    if not token_indices:
        return INVALID_TOKEN_SPAN
    return token_indices[0], token_indices[-1]


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

    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = encoding["offset_mapping"]
    if not offsets:
        return INVALID_TOKEN_SPAN

    token_span = map_char_span_to_tokens(offsets, char_span)
    if token_span == INVALID_TOKEN_SPAN:
        raise ValueError(
            f"Could not map char span {char_span} to tokens for text snippet: {text[:80]!r}"
        )
    return token_span


def locate_user_text_offset(prompt: str, user_text: str) -> int:
    """Find the character offset of the raw user text within a chat prompt.

    The Qwen chat template renders message contents verbatim, so the raw
    user ``text`` appears as a contiguous substring of the full prompt
    (inside the ``متن کاربر:\\n{text}`` user turn). We search *after* the
    ``USER_PROMPT_TEMPLATE`` marker so a coincidental match inside the fixed
    system prompt can never be selected.

    Args:
        prompt: Full chat-formatted prompt (or training sequence) string.
        user_text: Raw Persian input text.

    Returns:
        Character offset of ``user_text`` within ``prompt``.

    Raises:
        ValueError: If ``user_text`` cannot be located in ``prompt``.
    """
    from project.prompts import USER_PROMPT_TEMPLATE

    marker = USER_PROMPT_TEMPLATE.split("{text}")[0]
    marker_idx = prompt.find(marker)
    search_start = marker_idx + len(marker) if marker_idx != -1 else 0
    offset = prompt.find(user_text, search_start)
    if offset == -1:
        raise ValueError("Could not locate user text within the chat-formatted prompt.")
    return offset


def build_user_token_mask(
    offsets: list[tuple[int, int]],
    text_char_start: int,
    text_char_end: int,
) -> list[int]:
    """Build a binary token mask marking tokens that fall within a char range.

    Used to restrict aspect-conditioned attention to the raw user-text
    tokens, excluding the system prompt, chat-template markers, the
    assistant's JSON completion, and special/padding tokens (which have the
    degenerate ``(0, 0)`` offset for fast tokenizers).

    Args:
        offsets: Per-token ``(start_char, end_char)`` offsets.
        text_char_start: Inclusive character start of the target range.
        text_char_end: Exclusive character end of the target range.

    Returns:
        List of ``0``/``1`` ints, one per token in ``offsets``.
    """
    mask: list[int] = []
    for tok_start, tok_end in offsets:
        if tok_start == 0 and tok_end == 0:
            mask.append(0)
            continue
        overlaps = tok_end > text_char_start and tok_start < text_char_end
        mask.append(1 if overlaps else 0)
    return mask


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


def build_aspect_sft_example(
    sample: dict[str, Any],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, Any]:
    """Tokenize one enriched sample into full aspect-aware training features.

    Produces everything :class:`AspectAwareCollator` needs to batch: the
    standard causal-LM ``input_ids``/``attention_mask``/``labels`` (labels
    are left un-masked here; prompt-masking happens at collate time, exactly
    like :class:`CompletionOnlyCollator`), plus ``user_token_mask``,
    ``aspect_scores``, ``aspect_has_evidence``, ``aspect_start``, and
    ``aspect_end`` -- all aligned to the *full* chat-formatted sequence.

    The evidence ``char_span``/``token_span`` stored on
    ``sample["labels"][aspect]`` by :func:`validate_and_enrich_sample` are
    relative to the raw ``text`` alone (tokenized without the chat
    template). This function re-locates ``text`` inside the full rendered
    training sequence and re-maps each ``char_span`` onto that sequence's
    token offsets, so the resulting ``aspect_start``/``aspect_end`` indices
    are valid positions into the *same* ``input_ids`` used for the causal-LM
    loss.

    Args:
        sample: Enriched sample (``text``, ``labels`` with ``char_span``).
        tokenizer: HuggingFace tokenizer with a chat template.
        max_seq_length: Maximum sequence length (truncation).

    Returns:
        Dict with ``input_ids``, ``attention_mask``, ``labels``,
        ``user_token_mask`` (all ``list[int]``, length ``T``), and
        ``aspect_scores`` (``list[float]``, length ``len(ASPECTS)``),
        ``aspect_has_evidence`` (``list[int]``), ``aspect_start``,
        ``aspect_end`` (``list[int]``, ``-100`` sentinel where there is no
        evidence, it was truncated away, or the user text could not be
        re-located).
    """
    text = sample["text"]
    messages = build_chat_messages(text, labels=sample["labels"])
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)

    encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    offsets = encoding["offset_mapping"]
    seq_len = len(input_ids)

    try:
        base_offset: int | None = locate_user_text_offset(full_text, text)
    except ValueError:
        logger.warning(
            "Could not locate user text inside training sequence; "
            "disabling aspect labels for this sample."
        )
        base_offset = None

    if base_offset is not None:
        user_token_mask = build_user_token_mask(offsets, base_offset, base_offset + len(text))
    else:
        user_token_mask = [0] * seq_len

    # Truncation can cut the sequence off partway through an evidence span,
    # in which case ``map_char_span_to_tokens`` would silently return a
    # *shorter* span covering only the surviving tokens. Detect this by
    # comparing the evidence's end char offset against the char range
    # actually covered by the (already truncated) offsets, and treat any
    # truncated-away evidence as no-evidence rather than teach the model a
    # mangled partial span.
    max_covered_char = max((end for _, end in offsets), default=0)

    aspect_scores: list[float] = []
    aspect_has_evidence: list[int] = []
    aspect_start: list[int] = []
    aspect_end: list[int] = []

    for aspect in ASPECTS:
        label = sample["labels"][aspect]
        aspect_scores.append(float(label["score"]))
        char_span = tuple(label.get("char_span", list(INVALID_CHAR_SPAN)))

        start_tok, end_tok = -100, -100
        has_evidence = 0
        if base_offset is not None and char_span != INVALID_CHAR_SPAN:
            full_char_span = (base_offset + char_span[0], base_offset + char_span[1])
            not_truncated = full_char_span[1] <= max_covered_char
            tok_span = map_char_span_to_tokens(offsets, full_char_span)
            if tok_span != INVALID_TOKEN_SPAN and tok_span[1] < seq_len and not_truncated:
                start_tok, end_tok = tok_span
                has_evidence = 1
        aspect_has_evidence.append(has_evidence)
        aspect_start.append(start_tok)
        aspect_end.append(end_tok)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": list(input_ids),
        "user_token_mask": user_token_mask,
        "aspect_scores": aspect_scores,
        "aspect_has_evidence": aspect_has_evidence,
        "aspect_start": aspect_start,
        "aspect_end": aspect_end,
    }


def build_aspect_sft_dataset(
    samples: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> list[dict[str, Any]]:
    """Convert enriched samples to aspect-aware tokenized training features.

    Args:
        samples: Enriched samples.
        tokenizer: HuggingFace tokenizer.
        max_seq_length: Maximum sequence length (truncation).

    Returns:
        List of feature dicts (see :func:`build_aspect_sft_example`).
    """
    return [build_aspect_sft_example(sample, tokenizer, max_seq_length) for sample in samples]


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


class AspectAwareCollator(CompletionOnlyCollator):
    """Adds aspect-guided-attention supervision fields on top of the base collator.

    Reuses :class:`CompletionOnlyCollator` verbatim for ``input_ids``,
    ``attention_mask``, and response-template label masking, then pads
    ``user_token_mask`` (0-pad, matching ``attention_mask``'s convention)
    and stacks the fixed-length (``len(ASPECTS)``) ``aspect_scores``,
    ``aspect_has_evidence``, ``aspect_start``, and ``aspect_end`` fields
    produced by :func:`build_aspect_sft_example` -- these need no padding
    since every example already has exactly ``len(ASPECTS)`` entries.
    """

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Pad/collate base fields, then attach aspect supervision tensors.

        Args:
            features: List of feature dicts as returned by
                :func:`build_aspect_sft_example`.

        Returns:
            Batched tensors: the base collator's keys plus
            ``user_token_mask``, ``aspect_scores``, ``aspect_has_evidence``,
            ``aspect_start``, ``aspect_end``.
        """
        import torch

        batch = super().__call__(features)
        max_len = batch["input_ids"].shape[1]

        user_mask_batch: list[list[int]] = []
        for feature in features:
            mask = list(feature["user_token_mask"])
            pad_len = max_len - len(mask)
            user_mask_batch.append(mask + [0] * pad_len)

        batch["user_token_mask"] = torch.tensor(user_mask_batch, dtype=torch.long)
        batch["aspect_scores"] = torch.tensor(
            [feature["aspect_scores"] for feature in features], dtype=torch.float
        )
        batch["aspect_has_evidence"] = torch.tensor(
            [feature["aspect_has_evidence"] for feature in features], dtype=torch.long
        )
        batch["aspect_start"] = torch.tensor(
            [feature["aspect_start"] for feature in features], dtype=torch.long
        )
        batch["aspect_end"] = torch.tensor(
            [feature["aspect_end"] for feature in features], dtype=torch.long
        )
        return batch


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
