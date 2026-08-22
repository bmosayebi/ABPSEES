"""Fixed aspect definitions for the ABPSEES project."""

from __future__ import annotations

ASPECTS: list[str] = [
    "performance",
    "portability",
    "design",
    "durability",
    "cost_effectiveness",
]

ASPECT_DISPLAY_NAMES: dict[str, str] = {
    "performance": "Performance",
    "portability": "Portability",
    "design": "Design & Appearance",
    "durability": "Durability",
    "cost_effectiveness": "Cost-effectiveness",
}

# Scores below this threshold typically have empty evidence in synthetic data.
LOW_SCORE_EVIDENCE_THRESHOLD: float = 0.25

# Invalid span sentinel values.
INVALID_CHAR_SPAN: tuple[int, int] = (-1, -1)
INVALID_TOKEN_SPAN: tuple[int, int] = (-1, -1)
