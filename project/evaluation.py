"""End-to-end evaluation pipeline and report generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from project.config import AppConfig
from project.constants import ASPECTS, INVALID_CHAR_SPAN, INVALID_TOKEN_SPAN
from project.dataset import load_split
from project.metrics import (
    aggregate_evidence_metrics,
    compute_all_score_metrics,
    compute_evidence_metrics_for_sample,
    compute_faithfulness_rates,
    compute_head_span_metrics,
    compute_score_metrics_from_arrays,
    evidence_validity_rate,
    normalize_parsed_labels,
    parse_model_output,
)
from project.model import load_model_for_inference
from project.prompts import format_prompt
from project.utils import save_json

logger = logging.getLogger(__name__)


def generate_prediction(
    model: Any,
    tokenizer: Any,
    text: str,
    config: AppConfig,
) -> str:
    """Generate raw model output for a single input text.

    Args:
        model: Loaded model (with optional LoRA adapter).
        tokenizer: HuggingFace tokenizer.
        text: User input Persian text.
        config: Application configuration.

    Returns:
        Decoded generated string (assistant response only).
    """
    prompt = format_prompt(tokenizer, text, labels=None)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    inf = config.inference
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=inf.max_new_tokens,
            temperature=inf.temperature if inf.do_sample else None,
            do_sample=inf.do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def evaluate_samples(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    config: AppConfig,
) -> dict[str, Any]:
    """Run inference and compute all metrics on a sample list.

    Args:
        model: Loaded model.
        tokenizer: Tokenizer.
        samples: Ground-truth enriched samples.
        config: Application configuration.

    Returns:
        Full evaluation results dict.
    """
    pred_labels_list: list[dict[str, Any]] = []
    raw_outputs: list[str] = []
    parse_failures = 0
    per_sample_evidence: list[dict[str, dict[str, float]]] = []

    for sample in samples:
        raw = generate_prediction(model, tokenizer, sample["text"], config)
        raw_outputs.append(raw)
        parsed, error = parse_model_output(raw)
        if parsed is None:
            parse_failures += 1
            logger.debug("Parse failure: %s | raw=%s", error, raw[:200])
        labels = normalize_parsed_labels(parsed)
        pred_labels_list.append(labels)

        ev_metrics = compute_evidence_metrics_for_sample(
            sample["text"],
            sample["labels"],
            labels,
            tokenizer,
            config.data.normalize_unicode,
        )
        per_sample_evidence.append(ev_metrics)

    score_metrics = compute_all_score_metrics(samples, pred_labels_list)
    evidence_metrics = aggregate_evidence_metrics(per_sample_evidence)
    validity = evidence_validity_rate(
        [s["text"] for s in samples],
        pred_labels_list,
        config.data.normalize_unicode,
    )

    return {
        "num_samples": len(samples),
        "parse_failures": parse_failures,
        "score_metrics": score_metrics,
        "evidence_metrics": evidence_metrics,
        "evidence_validity_rate": validity,
        "predictions": [
            {
                "text": s["text"],
                "gold": s["labels"],
                "pred": p,
                "raw_output": r,
            }
            for s, p, r in zip(samples, pred_labels_list, raw_outputs)
        ],
        "per_sample_evidence": per_sample_evidence,
    }


def _gold_head_token_span(
    sample: dict[str, Any],
    aspect: str,
    prompt: str,
    offsets: list[tuple[int, int]],
) -> tuple[int, int]:
    """Re-map one aspect's raw-text char span onto the head-inference tokenization.

    ``sample["labels"][aspect]["char_span"]`` is relative to the raw
    ``sample["text"]`` alone; ``offsets`` come from tokenizing the full
    generation ``prompt`` (system + user turn, no assistant completion), so
    the char span must first be shifted by the offset of the raw text
    within that prompt.

    Args:
        sample: Enriched ground-truth sample.
        aspect: Aspect name.
        prompt: Full generation prompt string (as returned by
            :func:`project.hybrid_model.build_aspect_inference_inputs`).
        offsets: Per-token ``(start_char, end_char)`` offsets for ``prompt``.

    Returns:
        Inclusive gold token span, or ``INVALID_TOKEN_SPAN`` if there is no
        evidence or the text could not be located.
    """
    from project.preprocessing import locate_user_text_offset, map_char_span_to_tokens

    char_span = tuple(sample["labels"][aspect].get("char_span", list(INVALID_CHAR_SPAN)))
    if char_span == INVALID_CHAR_SPAN:
        return INVALID_TOKEN_SPAN
    try:
        base_offset = locate_user_text_offset(prompt, sample["text"])
    except ValueError:
        return INVALID_TOKEN_SPAN
    full_char_span = (base_offset + char_span[0], base_offset + char_span[1])
    return map_char_span_to_tokens(offsets, full_char_span)


def evaluate_aspect_heads(
    hybrid_model: Any,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    config: AppConfig,
) -> dict[str, Any]:
    """Evaluate the auxiliary aspect heads (Phase 1 score + Phase 2 span/faithfulness).

    This is separate from :func:`evaluate_samples`, which evaluates the
    primary *generative* JSON pipeline: here, per-aspect scores and evidence
    spans come directly from the score/span heads via
    :func:`project.hybrid_model.run_aspect_heads_on_text`, without going
    through JSON generation or parsing at all.

    Args:
        hybrid_model: Loaded ``AspectGuidedModel``.
        tokenizer: HuggingFace tokenizer.
        samples: Ground-truth enriched samples.
        config: Application configuration (``config.aspect`` used for
            faithfulness thresholds).

    Returns:
        Dict with ``score_metrics``, ``span_metrics``, ``faithfulness_rates``,
        and ``predictions`` (per-sample head outputs, for visualization).
    """
    from project.hybrid_model import run_aspect_heads_on_text

    per_aspect_scores_pred: dict[str, list[float]] = {a: [] for a in ASPECTS}
    per_aspect_scores_gt: dict[str, list[float]] = {a: [] for a in ASPECTS}
    per_aspect_has_evidence_prob: dict[str, list[float]] = {a: [] for a in ASPECTS}
    gold_token_spans: list[dict[str, tuple[int, int]]] = []
    pred_token_spans: list[dict[str, tuple[int, int] | None]] = []
    head_predictions: list[dict[str, Any]] = []

    for sample in samples:
        result = run_aspect_heads_on_text(hybrid_model, tokenizer, sample["text"], config)
        prompt = result["prompt"]
        offsets = result["offsets"]

        gold_spans: dict[str, tuple[int, int]] = {}
        pred_spans: dict[str, tuple[int, int] | None] = {}
        for aspect in ASPECTS:
            entry = result["per_aspect"][aspect]
            per_aspect_scores_pred[aspect].append(entry["score_head"])
            per_aspect_scores_gt[aspect].append(float(sample["labels"][aspect]["score"]))
            per_aspect_has_evidence_prob[aspect].append(entry["has_evidence_prob"])
            gold_spans[aspect] = _gold_head_token_span(sample, aspect, prompt, offsets)
            pred_spans[aspect] = entry["span_token"]

        gold_token_spans.append(gold_spans)
        pred_token_spans.append(pred_spans)
        head_predictions.append(
            {
                "text": sample["text"],
                "gold_token_spans": gold_spans,
                "per_aspect": result["per_aspect"],
                "tokens": result["tokens"],
                "attn_weights": result["attn_weights"],
            }
        )

    score_metrics = compute_score_metrics_from_arrays(per_aspect_scores_gt, per_aspect_scores_pred)
    span_metrics = compute_head_span_metrics(gold_token_spans, pred_token_spans)
    faithfulness_rates = compute_faithfulness_rates(
        per_aspect_scores_pred,
        per_aspect_has_evidence_prob,
        low_threshold=config.aspect.low_score_threshold,
        high_threshold=config.aspect.high_score_threshold,
    )

    return {
        "num_samples": len(samples),
        "score_metrics": score_metrics,
        "span_metrics": span_metrics,
        "faithfulness_rates": faithfulness_rates,
        "predictions": head_predictions,
    }


def format_aspect_report_markdown(aspect_results: dict[str, Any]) -> str:
    """Format head-based evaluation results as a Markdown section.

    Args:
        aspect_results: Output of :func:`evaluate_aspect_heads`.

    Returns:
        Markdown string, meant to be appended after
        :func:`format_report_markdown`'s output.
    """
    lines = [
        "",
        "## Aspect Heads (Phase 1 + Phase 2)",
        "",
        f"- Samples: **{aspect_results['num_samples']}**",
        "",
        "### Head-Based Score Metrics",
        "",
        "| Aspect | MAE | RMSE | Pearson | Spearman |",
        "|--------|-----|------|---------|----------|",
    ]
    for aspect, metrics in aspect_results["score_metrics"].items():
        lines.append(
            f"| {aspect} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | "
            f"{metrics['pearson']:.4f} | {metrics['spearman']:.4f} |"
        )

    lines.extend(
        ["", "### Head-Based Span Metrics", "", "| Aspect | P | R | F1 | EM |", "|--------|---|---|----|----|"]
    )
    for aspect, metrics in aspect_results["span_metrics"].items():
        lines.append(
            f"| {aspect} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['exact_match']:.4f} |"
        )

    lines.extend(
        [
            "",
            "### Faithfulness Rates",
            "",
            "| Aspect | High-score/empty-evidence | Low-score/non-empty-evidence |",
            "|--------|---------------------------|-------------------------------|",
        ]
    )
    for aspect, rates in aspect_results["faithfulness_rates"].items():
        lines.append(
            f"| {aspect} | {rates['high_score_empty_evidence_rate']:.4f} | "
            f"{rates['low_score_nonempty_evidence_rate']:.4f} |"
        )

    return "\n".join(lines)


def format_report_markdown(results: dict[str, Any]) -> str:
    """Format evaluation results as a Markdown report.

    Args:
        results: Output of :func:`evaluate_samples`.

    Returns:
        Markdown string.
    """
    lines = [
        "# ABPSEES Evaluation Report",
        "",
        f"- Samples: **{results['num_samples']}**",
        f"- JSON parse failures: **{results['parse_failures']}**",
        f"- Evidence validity rate: **{results['evidence_validity_rate']:.3f}**",
        "",
        "## Score Metrics",
        "",
        "| Aspect | MAE | RMSE | Pearson | Spearman |",
        "|--------|-----|------|---------|----------|",
    ]

    for aspect, metrics in results["score_metrics"].items():
        lines.append(
            f"| {aspect} | {metrics['mae']:.4f} | {metrics['rmse']:.4f} | "
            f"{metrics['pearson']:.4f} | {metrics['spearman']:.4f} |"
        )

    lines.extend(["", "## Evidence Metrics", "", "| Aspect | P | R | F1 | EM |", "|--------|---|---|----|----|"])
    for aspect, metrics in results["evidence_metrics"].items():
        lines.append(
            f"| {aspect} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | {metrics['exact_match']:.4f} |"
        )

    return "\n".join(lines)


def run_evaluation(
    config: AppConfig,
    adapter_path: str | Path | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Run full evaluation on a data split and save reports.

    Args:
        config: Application configuration.
        adapter_path: Path to LoRA adapter; defaults to ``checkpoints/best``.
        split: Data split name.

    Returns:
        Evaluation results dict.
    """
    if adapter_path is None:
        adapter_path = config.paths.checkpoint_dir / "best"

    enriched_path = config.paths.processed_dir / f"{split}_enriched.json"
    if enriched_path.exists():
        from project.utils import load_json

        samples = load_json(enriched_path)
    else:
        from project.model import load_tokenizer
        from project.preprocessing import save_enriched_split

        raw = load_split(config, split)
        tokenizer = load_tokenizer(config)
        samples = save_enriched_split(raw, tokenizer, config, split)

    model, tokenizer = load_model_for_inference(config, str(adapter_path))
    results = evaluate_samples(model, tokenizer, samples, config)

    reports_dir = config.paths.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save without per-sample raw outputs in JSON for size (keep in separate file).
    summary = {k: v for k, v in results.items() if k not in {"predictions", "per_sample_evidence"}}
    save_json(summary, reports_dir / f"{split}_metrics.json")

    predictions_path = reports_dir / f"{split}_predictions.json"
    save_json(results["predictions"], predictions_path)

    md = format_report_markdown(results)

    if config.aspect.enabled:
        from project.hybrid_model import load_hybrid_model_for_inference

        try:
            hybrid_model, hybrid_tokenizer = load_hybrid_model_for_inference(config, str(adapter_path))
            aspect_results = evaluate_aspect_heads(hybrid_model, hybrid_tokenizer, samples, config)
            aspect_summary = {k: v for k, v in aspect_results.items() if k != "predictions"}
            save_json(aspect_summary, reports_dir / f"{split}_metrics_aspect.json")
            save_json(aspect_results["predictions"], reports_dir / f"{split}_predictions_aspect.json")
            md += "\n" + format_aspect_report_markdown(aspect_results)
            results["aspect_results"] = aspect_results
        except Exception:
            logger.exception(
                "Aspect-head evaluation failed; continuing with generative-only metrics."
            )

    (reports_dir / f"{split}_metrics.md").write_text(md, encoding="utf-8")
    logger.info("Evaluation report saved to %s", reports_dir)
    return results
