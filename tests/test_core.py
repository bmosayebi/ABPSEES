"""Lightweight unit tests for ABPSEES core logic."""

from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from project.constants import ASPECTS, INVALID_CHAR_SPAN, INVALID_TOKEN_SPAN
from project.config import load_config
from project.dataset import generate_synthetic_dataset
from project.metrics import (
    compute_score_metrics,
    compute_token_f1,
    normalize_parsed_labels,
    parse_model_output,
    token_span_to_set,
)
from project.preprocessing import find_evidence_char_span, normalize_text
from project.prompts import format_training_labels, labels_to_json_string


class TestEvidenceValidation:
    def test_empty_evidence(self) -> None:
        assert find_evidence_char_span("hello world", "") == INVALID_CHAR_SPAN

    def test_valid_substring(self) -> None:
        text = "من دانشجو هستم و لپ‌تاپ می‌خواهم"
        evidence = "لپ‌تاپ می‌خواهم"
        start, end = find_evidence_char_span(text, evidence)
        assert text[start:end] == evidence

    def test_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            find_evidence_char_span("hello", "missing")

    def test_leftmost_match(self) -> None:
        text = "abc abc def"
        start, end = find_evidence_char_span(text, "abc")
        assert start == 0
        assert text[start:end] == "abc"


class TestMetrics:
    def test_token_f1_both_empty(self) -> None:
        result = compute_token_f1(set(), set())
        assert result["f1"] == 1.0
        assert result["exact_match"] == 1.0

    def test_token_f1_no_overlap(self) -> None:
        result = compute_token_f1({1, 2}, {3, 4})
        assert result["f1"] == 0.0

    def test_token_f1_partial_overlap(self) -> None:
        result = compute_token_f1({1, 2, 3}, {2, 3, 4})
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == pytest.approx(2 / 3)

    def test_score_metrics(self) -> None:
        y_true = np.array([0.9, 0.5, 0.1])
        y_pred = np.array([0.8, 0.6, 0.2])
        m = compute_score_metrics(y_true, y_pred)
        assert m["mae"] == pytest.approx(0.1)
        assert m["rmse"] > 0

    def test_parse_json_with_fences(self) -> None:
        raw = '```json\n{"performance": {"score": 0.9, "evidence": ""}}\n```'
        parsed, err = parse_model_output(raw)
        assert err is None
        assert parsed is not None
        assert "performance" in parsed

    def test_normalize_parsed_labels(self) -> None:
        labels = normalize_parsed_labels(None)
        assert set(labels.keys()) == set(ASPECTS)
        assert all(np.isnan(labels[a]["score"]) for a in ASPECTS)


class TestPrompts:
    def test_labels_json_order(self) -> None:
        sample = generate_synthetic_dataset(1)[0]
        out = labels_to_json_string(sample["labels"])
        parsed = json.loads(out)
        assert list(parsed.keys()) == ASPECTS

    def test_format_training_labels_strips_spans(self) -> None:
        labels = {
            a: {"score": 0.5, "evidence": "", "char_span": [0, 1], "token_span": [0, 1]}
            for a in ASPECTS
        }
        formatted = format_training_labels(labels)
        assert "char_span" not in formatted["performance"]


class TestSyntheticData:
    def test_generate_valid_samples(self) -> None:
        from project.config import validate_sample_scores

        samples = generate_synthetic_dataset(10, seed=42)
        assert len(samples) == 10
        for s in samples:
            validate_sample_scores(s["labels"])
            for aspect in ASPECTS:
                evidence = s["labels"][aspect]["evidence"]
                if evidence:
                    assert evidence in s["text"]

    def test_unicode_normalize(self) -> None:
        text = "café"
        assert normalize_text(text, normalize_unicode=True) == text


class TestTokenSpanSet:
    def test_invalid_span(self) -> None:
        assert token_span_to_set(INVALID_TOKEN_SPAN) == set()

    def test_inclusive_range(self) -> None:
        assert token_span_to_set((2, 4)) == {2, 3, 4}


class TestConfig:
    def test_default_config_load(self) -> None:
        config = load_config()
        assert config.seed == 42
        assert config.model.max_seq_length == 2048
        assert config.colab.enabled is False

    def test_smoke_imports(self) -> None:
        import project.config  # noqa: F401
        import project.dataset  # noqa: F401
        import project.metrics  # noqa: F401
        import project.preprocessing  # noqa: F401
        import project.prompts  # noqa: F401
        import project.utils  # noqa: F401
