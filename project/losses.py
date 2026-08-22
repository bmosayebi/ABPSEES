"""Training objective documentation and evaluation-side faithfulness helpers.

Training uses the standard HuggingFace causal language modeling cross-entropy loss
on assistant-response tokens only. No custom multi-objective loss is applied.
"""

from __future__ import annotations

from typing import Any

from project.constants import ASPECTS


def check_faithfulness_consistency(
    labels: dict[str, Any],
    score_threshold: float = 0.5,
) -> list[str]:
    """Flag inconsistent score/evidence pairs for analysis (not training).

    Examples of inconsistency:
    - High score with empty evidence
    - Non-empty evidence with very low score

    Args:
        labels: Parsed prediction or ground-truth labels.
        score_threshold: Boundary for high/low score checks.

    Returns:
        List of warning message strings.
    """
    warnings: list[str] = []
    for aspect in ASPECTS:
        if aspect not in labels:
            warnings.append(f"Missing aspect: {aspect}")
            continue
        entry = labels[aspect]
        score = float(entry.get("score", 0.0))
        evidence = str(entry.get("evidence", ""))
        if score >= score_threshold and evidence == "":
            warnings.append(f"{aspect}: high score ({score}) but empty evidence")
        if score < 0.25 and evidence != "":
            warnings.append(f"{aspect}: low score ({score}) but non-empty evidence")
    return warnings
