"""Score and evidence evaluation metrics."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from project.constants import ASPECTS, INVALID_TOKEN_SPAN
from project.preprocessing import char_span_to_token_span, find_evidence_char_span, normalize_text


def parse_model_output(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """Robustly parse JSON from model output text.

    Args:
        raw: Raw model generation string.

    Returns:
        Tuple of (parsed dict or None, error message or None).
    """
    text = raw.strip()
    # Strip markdown code fences.
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Try direct parse.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, None
    except json.JSONDecodeError:
        pass

    # Try to extract first JSON object.
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            parsed = json.loads(brace_match.group(0))
            if isinstance(parsed, dict):
                return parsed, None
        except json.JSONDecodeError as exc:
            return None, str(exc)

    return None, "Could not parse JSON from model output"


def normalize_parsed_labels(
    parsed: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize parsed labels to standard schema with defaults.

    Args:
        parsed: Parsed JSON dict or None.

    Returns:
        Labels dict with all aspects present.
    """
    labels: dict[str, dict[str, Any]] = {}
    for aspect in ASPECTS:
        if parsed and aspect in parsed:
            entry = parsed[aspect]
            score = entry.get("score", float("nan"))
            try:
                score = float(score)
            except (TypeError, ValueError):
                score = float("nan")
            evidence = str(entry.get("evidence", ""))
            labels[aspect] = {"score": score, "evidence": evidence}
        else:
            labels[aspect] = {"score": float("nan"), "evidence": ""}
    return labels


def compute_score_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics for score arrays.

    Args:
        y_true: Ground-truth scores.
        y_pred: Predicted scores.

    Returns:
        Dict with MAE, RMSE, Pearson r, Spearman rho.
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "pearson": float("nan"), "spearman": float("nan")}

    yt = y_true[mask]
    yp = y_pred[mask]
    mae = float(mean_absolute_error(yt, yp))
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))

    if len(yt) < 2 or np.std(yt) == 0 or np.std(yp) == 0:
        pearson = float("nan")
        spearman = float("nan")
    else:
        pearson = float(pearsonr(yt, yp).statistic)
        spearman = float(spearmanr(yt, yp).statistic)

    return {"mae": mae, "rmse": rmse, "pearson": pearson, "spearman": spearman}


def token_span_to_set(token_span: tuple[int, int]) -> set[int]:
    """Convert inclusive token span to a set of token indices.

    Args:
        token_span: ``(start, end)`` inclusive indices.

    Returns:
        Set of token indices.
    """
    if token_span == INVALID_TOKEN_SPAN:
        return set()
    start, end = token_span
    return set(range(start, end + 1))


def compute_token_f1(
    pred_tokens: set[int],
    gold_tokens: set[int],
) -> dict[str, float]:
    """Compute token-level precision, recall, F1, and exact match.

    Args:
        pred_tokens: Predicted token index set.
        gold_tokens: Ground-truth token index set.

    Returns:
        Dict with precision, recall, f1, exact_match.
    """
    if not pred_tokens and not gold_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "exact_match": 1.0}
    if not pred_tokens or not gold_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0}

    overlap = pred_tokens & gold_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gold_tokens)
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    exact_match = 1.0 if pred_tokens == gold_tokens else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match": exact_match,
    }


def evidence_to_token_span(
    text: str,
    evidence: str,
    tokenizer: Any,
    normalize_unicode: bool = True,
) -> tuple[int, int]:
    """Locate evidence in text and return inclusive token span.

    Args:
        text: Source text.
        evidence: Evidence substring.
        tokenizer: HuggingFace tokenizer.
        normalize_unicode: Whether to NFC-normalize.

    Returns:
        Inclusive token span or ``(-1, -1)`` if empty/not found.
    """
    text = normalize_text(text, normalize_unicode)
    evidence = normalize_text(evidence, normalize_unicode)
    if evidence == "":
        return INVALID_TOKEN_SPAN
    try:
        char_span = find_evidence_char_span(text, evidence)
        return char_span_to_token_span(text, char_span, tokenizer)
    except ValueError:
        return INVALID_TOKEN_SPAN


def compute_evidence_metrics_for_sample(
    text: str,
    gold_labels: dict[str, Any],
    pred_labels: dict[str, Any],
    tokenizer: Any,
    normalize_unicode: bool = True,
) -> dict[str, dict[str, float]]:
    """Compute per-aspect evidence metrics for one sample.

    Args:
        text: Input text.
        gold_labels: Ground-truth labels (may include token_span).
        pred_labels: Predicted labels.
        tokenizer: HuggingFace tokenizer.
        normalize_unicode: Whether to normalize Unicode.

    Returns:
        Per-aspect evidence metric dicts.
    """
    results: dict[str, dict[str, float]] = {}
    for aspect in ASPECTS:
        gold = gold_labels[aspect]
        pred = pred_labels[aspect]

        if "token_span" in gold:
            gold_span = tuple(gold["token_span"])
        else:
            gold_span = evidence_to_token_span(
                text, str(gold.get("evidence", "")), tokenizer, normalize_unicode
            )

        pred_span = evidence_to_token_span(
            text, str(pred.get("evidence", "")), tokenizer, normalize_unicode
        )

        gold_set = token_span_to_set(gold_span)
        pred_set = token_span_to_set(pred_span)
        results[aspect] = compute_token_f1(pred_set, gold_set)

    return results


def aggregate_evidence_metrics(
    per_sample: list[dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """Average evidence metrics across samples per aspect.

    Args:
        per_sample: List of per-sample per-aspect metric dicts.

    Returns:
        Per-aspect averaged metrics plus ``overall`` macro average.
    """
    aggregated: dict[str, dict[str, float]] = {}
    for aspect in ASPECTS:
        keys = ["precision", "recall", "f1", "exact_match"]
        aggregated[aspect] = {
            k: float(np.mean([s[aspect][k] for s in per_sample])) for k in keys
        }

    keys = ["precision", "recall", "f1", "exact_match"]
    aggregated["overall"] = {
        k: float(np.mean([aggregated[a][k] for a in ASPECTS])) for k in keys
    }
    return aggregated


def compute_all_score_metrics(
    gold_samples: list[dict[str, Any]],
    pred_labels_list: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """Compute score metrics per aspect and overall.

    Args:
        gold_samples: Ground-truth samples.
        pred_labels_list: Predicted labels per sample.

    Returns:
        Per-aspect and overall score metrics.
    """
    results: dict[str, dict[str, float]] = {}
    for aspect in ASPECTS:
        y_true = np.array([s["labels"][aspect]["score"] for s in gold_samples], dtype=float)
        y_pred = np.array([p[aspect]["score"] for p in pred_labels_list], dtype=float)
        results[aspect] = compute_score_metrics(y_true, y_pred)

    keys = ["mae", "rmse", "pearson", "spearman"]
    results["overall"] = {
        k: float(np.mean([results[a][k] for a in ASPECTS if not np.isnan(results[a][k])]))
        for k in keys
    }
    return results


def evidence_validity_rate(
    texts: list[str],
    pred_labels_list: list[dict[str, Any]],
    normalize_unicode: bool = True,
) -> float:
    """Fraction of non-empty predicted evidence strings found in input text.

    Args:
        texts: Input texts per sample.
        pred_labels_list: Predicted labels.
        normalize_unicode: Whether to normalize Unicode.

    Returns:
        Validity rate in [0, 1].
    """
    total = 0
    valid = 0
    for text, pred in zip(texts, pred_labels_list):
        text_n = normalize_text(text, normalize_unicode)
        for aspect in ASPECTS:
            evidence = normalize_text(str(pred[aspect].get("evidence", "")), normalize_unicode)
            if evidence:
                total += 1
                if evidence in text_n:
                    valid += 1
    return valid / total if total > 0 else 1.0
