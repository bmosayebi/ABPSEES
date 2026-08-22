"""Inference utilities with faithfulness validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from project.config import AppConfig
from project.evaluation import generate_prediction
from project.losses import check_faithfulness_consistency
from project.metrics import normalize_parsed_labels, parse_model_output
from project.model import load_model_for_inference
from project.preprocessing import normalize_text

logger = logging.getLogger(__name__)


def validate_evidence_in_text(
    text: str,
    labels: dict[str, Any],
    normalize_unicode: bool = True,
) -> list[str]:
    """Check that all non-empty evidence strings are substrings of the input.

    Args:
        text: Original user text.
        labels: Parsed prediction labels.
        normalize_unicode: Whether to NFC-normalize.

    Returns:
        List of warning messages for invalid evidence.
    """
    warnings: list[str] = []
    text_n = normalize_text(text, normalize_unicode)
    for aspect, entry in labels.items():
        if aspect.startswith("_"):
            continue
        evidence = normalize_text(str(entry.get("evidence", "")), normalize_unicode)
        if evidence and evidence not in text_n:
            warnings.append(
                f"{aspect}: evidence not found in input text: {evidence!r}"
            )
    return warnings


def predict(
    text: str,
    config: AppConfig,
    adapter_path: str | Path | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
) -> dict[str, Any]:
    """Run inference on a single Persian text input.

    Args:
        text: User description text.
        config: Application configuration.
        adapter_path: Optional LoRA adapter path.
        model: Optional pre-loaded model.
        tokenizer: Optional pre-loaded tokenizer.

    Returns:
        Dict with ``labels`` key and optional ``_warnings``.
    """
    if model is None or tokenizer is None:
        if adapter_path is None:
            adapter_path = config.paths.checkpoint_dir / "best"
        model, tokenizer = load_model_for_inference(config, str(adapter_path))

    raw = generate_prediction(model, tokenizer, text, config)
    parsed, error = parse_model_output(raw)
    labels = normalize_parsed_labels(parsed)

    result: dict[str, Any] = {"labels": labels, "raw_output": raw}
    if parsed is None:
        result["_warnings"] = [f"JSON parse error: {error}"]

    if config.inference.faithfulness_warn:
        warnings = validate_evidence_in_text(
            text, labels, config.data.normalize_unicode
        )
        warnings.extend(check_faithfulness_consistency(labels))
        if warnings:
            result["_warnings"] = result.get("_warnings", []) + warnings
            for w in warnings:
                logger.warning(w)

    return result


def pretty_print_prediction(result: dict[str, Any]) -> str:
    """Format prediction result as pretty-printed JSON.

    Args:
        result: Output of :func:`predict`.

    Returns:
        Formatted JSON string.
    """
    output = result.get("labels", result)
    return json.dumps(output, ensure_ascii=False, indent=2)


def predict_batch(
    texts: list[str],
    config: AppConfig,
    adapter_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run inference on multiple texts reusing a single loaded model.

    Args:
        texts: List of input texts.
        config: Application configuration.
        adapter_path: Optional adapter path.

    Returns:
        List of prediction result dicts.
    """
    if adapter_path is None:
        adapter_path = config.paths.checkpoint_dir / "best"
    model, tokenizer = load_model_for_inference(config, str(adapter_path))
    return [predict(t, config, model=model, tokenizer=tokenizer) for t in texts]
