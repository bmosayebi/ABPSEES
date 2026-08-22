"""End-to-end evaluation pipeline and report generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from project.config import AppConfig
from project.dataset import load_split
from project.metrics import (
    aggregate_evidence_metrics,
    compute_all_score_metrics,
    compute_evidence_metrics_for_sample,
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
    (reports_dir / f"{split}_metrics.md").write_text(md, encoding="utf-8")
    logger.info("Evaluation report saved to %s", reports_dir)
    return results
