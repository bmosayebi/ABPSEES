"""Error analysis and evaluation visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from project.constants import ASPECTS, ASPECT_DISPLAY_NAMES


def _ensure_dir(path: Path) -> Path:
    """Create parent directory and return path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_score_distributions(
    gold_samples: list[dict[str, Any]],
    pred_labels_list: list[dict[str, Any]],
    save_path: Path,
) -> Path:
    """Plot ground-truth and predicted score distributions per aspect.

    Args:
        gold_samples: Ground-truth samples.
        pred_labels_list: Predicted labels.
        save_path: Output figure path.

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, aspect in enumerate(ASPECTS):
        ax = axes[idx]
        y_true = [s["labels"][aspect]["score"] for s in gold_samples]
        y_pred = [p[aspect]["score"] for p in pred_labels_list]
        ax.hist(y_true, bins=15, alpha=0.6, label="Ground Truth", color="steelblue")
        ax.hist(y_pred, bins=15, alpha=0.6, label="Predicted", color="coral")
        ax.set_title(ASPECT_DISPLAY_NAMES.get(aspect, aspect))
        ax.set_xlabel("Score")
        ax.legend(fontsize=8)

    axes[-1].axis("off")
    fig.suptitle("Score Distributions: Ground Truth vs Predicted", fontsize=14)
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_prediction_vs_ground_truth(
    gold_samples: list[dict[str, Any]],
    pred_labels_list: list[dict[str, Any]],
    save_path: Path,
) -> Path:
    """Scatter plot of predicted vs ground-truth scores per aspect.

    Args:
        gold_samples: Ground-truth samples.
        pred_labels_list: Predicted labels.
        save_path: Output figure path.

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for idx, aspect in enumerate(ASPECTS):
        ax = axes[idx]
        y_true = np.array([s["labels"][aspect]["score"] for s in gold_samples])
        y_pred = np.array([p[aspect]["score"] for p in pred_labels_list])
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        ax.scatter(y_true[mask], y_pred[mask], alpha=0.6, s=20)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.set_title(ASPECT_DISPLAY_NAMES.get(aspect, aspect))

    axes[-1].axis("off")
    fig.suptitle("Predicted vs Ground Truth Scores", fontsize=14)
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_mae_per_aspect(
    score_metrics: dict[str, dict[str, float]],
    save_path: Path,
) -> Path:
    """Bar chart of MAE per aspect.

    Args:
        score_metrics: Output of :func:`compute_all_score_metrics`.
        save_path: Output figure path.

    Returns:
        Path to saved figure.
    """
    aspects = [a for a in ASPECTS if a in score_metrics]
    maes = [score_metrics[a]["mae"] for a in aspects]
    labels = [ASPECT_DISPLAY_NAMES.get(a, a) for a in aspects]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, maes, color="steelblue")
    ax.set_ylabel("MAE")
    ax.set_title("Mean Absolute Error per Aspect")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_token_f1_histogram(
    per_sample_evidence: list[dict[str, dict[str, float]]],
    save_path: Path,
) -> Path:
    """Histogram of per-sample macro-averaged token F1.

    Args:
        per_sample_evidence: Per-sample evidence metrics.
        save_path: Output figure path.

    Returns:
        Path to saved figure.
    """
    macro_f1 = [
        float(np.mean([per_sample_evidence[i][a]["f1"] for a in ASPECTS]))
        for i in range(len(per_sample_evidence))
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(macro_f1, bins=20, color="seagreen", edgecolor="white")
    ax.set_xlabel("Macro Token F1")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Per-Sample Macro Token F1")
    fig.tight_layout()
    fig.savefig(_ensure_dir(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _combined_error(
    sample_idx: int,
    gold: dict[str, Any],
    pred: dict[str, Any],
    evidence_metrics: dict[str, dict[str, float]],
) -> float:
    """Compute combined error score for ranking samples."""
    mae = np.mean(
        [
            abs(gold["labels"][a]["score"] - pred[a]["score"])
            for a in ASPECTS
            if not np.isnan(pred[a]["score"])
        ]
    )
    f1 = np.mean([evidence_metrics[a]["f1"] for a in ASPECTS])
    return float(mae + (1.0 - f1))


def plot_worst_best_examples(
    gold_samples: list[dict[str, Any]],
    pred_labels_list: list[dict[str, Any]],
    per_sample_evidence: list[dict[str, dict[str, float]]],
    save_path: Path,
    n: int = 3,
) -> Path:
    """Save a text summary of worst and best predictions.

    Args:
        gold_samples: Ground-truth samples.
        pred_labels_list: Predicted labels.
        per_sample_evidence: Per-sample evidence metrics.
        save_path: Output text file path.
        n: Number of examples per category.

    Returns:
        Path to saved file.
    """
    error_by_idx = {
        i: _combined_error(i, gold_samples[i], pred_labels_list[i], per_sample_evidence[i])
        for i in range(len(gold_samples))
    }
    ranked = sorted(error_by_idx.items(), key=lambda x: x[1])
    best_idxs = [i for i, _ in ranked[:n]]
    worst_idxs = [i for i, _ in ranked[-n:]][::-1]

    lines = ["# Best and Worst Predictions", ""]
    for title, idxs in [("Best", best_idxs), ("Worst", worst_idxs)]:
        lines.append(f"## {title} Examples")
        for idx in idxs:
            lines.append(f"### Sample {idx} (error={error_by_idx[idx]:.3f})")
            lines.append(f"**Text:** {gold_samples[idx]['text'][:200]}...")
            lines.append(f"**Gold performance score:** {gold_samples[idx]['labels']['performance']['score']}")
            lines.append(f"**Pred performance score:** {pred_labels_list[idx]['performance']['score']}")
            lines.append("")

    _ensure_dir(save_path).write_text("\n".join(lines), encoding="utf-8")
    return save_path


def plot_confusion_examples(
    gold_samples: list[dict[str, Any]],
    pred_labels_list: list[dict[str, Any]],
    save_path: Path,
) -> Path:
    """Identify and save faithfulness confusion examples.

    Args:
        gold_samples: Ground-truth samples.
        pred_labels_list: Predicted labels.
        save_path: Output text file path.

    Returns:
        Path to saved file.
    """
    lines = ["# Faithfulness Confusion Examples", ""]
    categories = {
        "high_score_empty_evidence": [],
        "low_score_nonempty_evidence": [],
        "evidence_not_in_text": [],
    }

    for idx, (gold, pred) in enumerate(zip(gold_samples, pred_labels_list)):
        text = gold["text"]
        for aspect in ASPECTS:
            score = pred[aspect]["score"]
            evidence = pred[aspect]["evidence"]
            if not np.isnan(score) and score >= 0.5 and evidence == "":
                categories["high_score_empty_evidence"].append((idx, aspect, score))
            if not np.isnan(score) and score < 0.25 and evidence != "":
                categories["low_score_nonempty_evidence"].append((idx, aspect, score, evidence))
            if evidence and evidence not in text:
                categories["evidence_not_in_text"].append((idx, aspect, evidence))

    for cat, items in categories.items():
        lines.append(f"## {cat} ({len(items)} cases)")
        for item in items[:10]:
            lines.append(f"- {item}")
        lines.append("")

    _ensure_dir(save_path).write_text("\n".join(lines), encoding="utf-8")
    return save_path


def generate_all_visualizations(
    results: dict[str, Any],
    figures_dir: Path,
) -> dict[str, Path]:
    """Generate all error analysis outputs.

    Args:
        results: Full evaluation results from :func:`run_evaluation`.
        figures_dir: Directory for figures and text reports.

    Returns:
        Mapping of artifact name to file path.
    """
    gold_samples = [
        {"text": p["text"], "labels": p["gold"]} for p in results["predictions"]
    ]
    pred_labels_list = [p["pred"] for p in results["predictions"]]
    per_sample_evidence = results["per_sample_evidence"]

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "score_distributions": plot_score_distributions(
            gold_samples, pred_labels_list, figures_dir / "score_distributions.png"
        ),
        "pred_vs_gt": plot_prediction_vs_ground_truth(
            gold_samples, pred_labels_list, figures_dir / "pred_vs_gt.png"
        ),
        "mae_per_aspect": plot_mae_per_aspect(
            results["score_metrics"], figures_dir / "mae_per_aspect.png"
        ),
        "token_f1_hist": plot_token_f1_histogram(
            per_sample_evidence, figures_dir / "token_f1_histogram.png"
        ),
        "worst_best": plot_worst_best_examples(
            gold_samples, pred_labels_list, per_sample_evidence,
            figures_dir / "worst_best_examples.md",
        ),
        "confusion": plot_confusion_examples(
            gold_samples, pred_labels_list, figures_dir / "confusion_examples.md"
        ),
    }
    return paths
