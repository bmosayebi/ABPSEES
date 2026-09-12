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

    def test_evidence_spans_are_locatable_by_char_offset(self) -> None:
        """Every non-empty evidence string must resolve to a real char span,
        not just a python ``in`` substring match (mirrors how the real
        training pipeline locates evidence via find_evidence_char_span)."""
        samples = generate_synthetic_dataset(200, seed=7)
        for s in samples:
            for aspect in ASPECTS:
                evidence = s["labels"][aspect]["evidence"]
                if not evidence:
                    continue
                span = find_evidence_char_span(s["text"], evidence)
                assert span != INVALID_CHAR_SPAN, (
                    f"could not locate evidence for {aspect!r} in text {s['text']!r}"
                )

    def test_performance_is_not_always_present_or_first(self) -> None:
        """Regression guard for the original evidence-misattribution bug:
        `performance` must no longer be unconditionally present, nor pinned
        to the same leading character offset in every sample -- both were
        positional shortcuts the model could exploit instead of learning
        real semantic disambiguation."""
        samples = generate_synthetic_dataset(300, seed=123)

        has_evidence_flags = [bool(s["labels"]["performance"]["evidence"]) for s in samples]
        assert any(has_evidence_flags), "expected some samples with performance evidence"
        assert not all(has_evidence_flags), (
            "performance must sometimes have empty evidence, like every other aspect"
        )

        start_offsets = {
            s["text"].find(s["labels"]["performance"]["evidence"])
            for s in samples
            if s["labels"]["performance"]["evidence"]
        }
        assert len(start_offsets) > 1, (
            "performance evidence should not always start at the same char offset"
        )

    def test_all_aspects_have_some_no_evidence_samples(self) -> None:
        """Every aspect (not just the 'extra' ones) should have a non-empty
        no-evidence branch across a large enough sample, so the has-evidence
        gate gets balanced supervision for all five aspects."""
        samples = generate_synthetic_dataset(300, seed=99)
        for aspect in ASPECTS:
            evidences = [s["labels"][aspect]["evidence"] for s in samples]
            assert any(e == "" for e in evidences), f"{aspect} never has empty evidence"
            assert any(e != "" for e in evidences), f"{aspect} never has non-empty evidence"

    def test_combo_hard_negatives_produce_disjoint_adjacent_spans(self) -> None:
        """When a hard-negative combo sentence fires, the two aspects it
        covers must get distinct, non-overlapping evidence substrings
        within the same sample (the whole point of the hard negative is
        forcing content-based, not positional, disambiguation)."""
        samples = generate_synthetic_dataset(400, seed=17)

        found_combo = False
        for s in samples:
            present = {a: s["labels"][a]["evidence"] for a in ASPECTS if s["labels"][a]["evidence"]}
            for a1 in present:
                for a2 in present:
                    if a1 >= a2:
                        continue
                    ev1, ev2 = present[a1], present[a2]
                    if ev1 in ev2 or ev2 in ev1:
                        continue
                    start1 = s["text"].find(ev1)
                    start2 = s["text"].find(ev2)
                    span1 = (start1, start1 + len(ev1))
                    span2 = (start2, start2 + len(ev2))
                    overlap = span1[0] < span2[1] and span2[0] < span1[1]
                    assert not overlap, f"overlapping spans for {a1}/{a2} in {s['text']!r}"
                    found_combo = True
        assert found_combo, "expected at least one sample with 2+ distinct-evidence aspects"

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
