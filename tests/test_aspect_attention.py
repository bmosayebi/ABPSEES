"""Tests for the Aspect-Guided Attention module (Phase 1 + Phase 2).

Pure-Python span/offset helpers (in ``project.preprocessing``) are tested
unconditionally, matching the rest of the test suite's light dependency
footprint. Anything that touches ``torch``/``transformers`` tensors is
guarded with ``pytest.importorskip`` so this file still collects cleanly
in minimal environments, while exercising the real tensor math whenever
the full training stack is installed (as it always is on Colab/CI).
"""

from __future__ import annotations

import pytest

from project.constants import ASPECTS, INVALID_CHAR_SPAN, INVALID_TOKEN_SPAN
from project.preprocessing import (
    build_user_token_mask,
    locate_user_text_offset,
    map_char_span_to_tokens,
)
from project.prompts import USER_PROMPT_TEMPLATE

torch = pytest.importorskip("torch")


class TestMapCharSpanToTokens:
    """Pure-function char-span -> token-span mapping used for null-span handling."""

    def test_invalid_span_returns_sentinel(self) -> None:
        offsets = [(0, 1), (1, 3), (3, 5)]
        assert map_char_span_to_tokens(offsets, INVALID_CHAR_SPAN) == INVALID_TOKEN_SPAN

    def test_basic_overlap(self) -> None:
        offsets = [(0, 3), (3, 6), (6, 9), (9, 12)]
        assert map_char_span_to_tokens(offsets, (3, 9)) == (1, 2)

    def test_no_overlap_returns_sentinel(self) -> None:
        offsets = [(0, 3), (3, 6)]
        assert map_char_span_to_tokens(offsets, (10, 12)) == INVALID_TOKEN_SPAN

    def test_partial_overlap_includes_boundary_token(self) -> None:
        offsets = [(0, 4), (4, 8), (8, 12)]
        # span [2, 6) starts mid-token-0 and ends mid-token-1: both included.
        assert map_char_span_to_tokens(offsets, (2, 6)) == (0, 1)


class TestLocateUserTextOffset:
    """Locating the raw user text inside a full chat-formatted sequence."""

    def test_locates_after_marker(self) -> None:
        text = "سلام دنیا"
        prompt = "SYSTEM PROMPT " + USER_PROMPT_TEMPLATE.format(text=text) + " <|im_end|>"
        offset = locate_user_text_offset(prompt, text)
        assert prompt[offset : offset + len(text)] == text

    def test_ignores_coincidental_match_before_marker(self) -> None:
        text = "abc"
        prompt = "abc در سیستم پرامپت" + USER_PROMPT_TEMPLATE.format(text=text)
        offset = locate_user_text_offset(prompt, text)
        marker_idx = prompt.find(USER_PROMPT_TEMPLATE.split("{text}")[0])
        assert offset > marker_idx

    def test_raises_when_not_found(self) -> None:
        with pytest.raises(ValueError):
            locate_user_text_offset("some unrelated prompt string", "missing text")


class TestBuildUserTokenMask:
    """Restricting aspect attention to raw user-text tokens only."""

    def test_masks_overlapping_tokens(self) -> None:
        offsets = [(0, 0), (0, 3), (3, 6), (6, 9), (0, 0)]
        mask = build_user_token_mask(offsets, 3, 9)
        assert mask == [0, 0, 1, 1, 0]

    def test_special_token_offsets_excluded(self) -> None:
        offsets = [(0, 0), (0, 5)]
        mask = build_user_token_mask(offsets, 0, 5)
        assert mask == [0, 1]

    def test_no_overlap_yields_all_zero(self) -> None:
        offsets = [(0, 3), (3, 6)]
        mask = build_user_token_mask(offsets, 10, 12)
        assert mask == [0, 0]


class TestAspectConditionedAttention:
    """Attention masking correctness (project.aspect_attention)."""

    def test_attention_sums_to_one_and_respects_mask(self) -> None:
        from project.aspect_attention import AspectConditionedAttention

        torch.manual_seed(0)
        b, t, d, a = 2, 8, 16, len(ASPECTS)
        attn = AspectConditionedAttention(d_model=d, num_aspects=a, dropout=0.0)
        hidden = torch.randn(b, t, d)
        mask = torch.zeros(b, t, dtype=torch.long)
        mask[:, 2:5] = 1

        context, weights = attn(hidden, mask)

        assert context.shape == (b, a, d)
        assert weights.shape == (b, a, t)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(b, a), atol=1e-5)
        masked_out = weights[:, :, mask[0] == 0]
        assert torch.allclose(masked_out, torch.zeros_like(masked_out), atol=1e-6)

    def test_safe_token_mask_fallback_for_empty_rows(self) -> None:
        from project.aspect_attention import safe_token_mask

        mask = torch.zeros(3, 5, dtype=torch.long)
        mask[1, 2] = 1  # only row 1 has an attendable token

        safe = safe_token_mask(mask)

        assert safe[0].tolist() == [1, 1, 1, 1, 1]
        assert safe[1].tolist() == [0, 0, 1, 0, 0]
        assert safe[2].tolist() == [1, 1, 1, 1, 1]


class TestAspectHeads:
    """End-to-end Phase 1 + Phase 2 head forward pass."""

    def test_forward_shapes_and_score_range(self) -> None:
        from project.aspect_attention import AspectHeads

        b, t, d, a = 2, 6, 12, len(ASPECTS)
        heads = AspectHeads(d_model=d, num_aspects=a, attn_dropout=0.0, score_hidden=8, enable_span=True)
        hidden = torch.randn(b, t, d)
        mask = torch.ones(b, t, dtype=torch.long)

        out = heads(hidden, mask)

        assert out.scores.shape == (b, a)
        assert bool((out.scores >= 0).all()) and bool((out.scores <= 1).all())
        assert out.start_logits.shape == (b, a, t)
        assert out.end_logits.shape == (b, a, t)
        assert out.has_evidence_logits.shape == (b, a)

    def test_phase1_only_when_span_disabled(self) -> None:
        from project.aspect_attention import AspectHeads

        heads = AspectHeads(d_model=8, num_aspects=len(ASPECTS), enable_span=False)
        hidden = torch.randn(1, 4, 8)
        mask = torch.ones(1, 4, dtype=torch.long)

        out = heads(hidden, mask)

        assert out.start_logits is None
        assert out.end_logits is None
        assert out.has_evidence_logits is None


class TestDecodeSpan:
    """Span decoding, including null-span (no-evidence) handling."""

    def test_decode_best_span_picks_max_scoring_pair(self) -> None:
        from project.aspect_attention import decode_best_span

        start_logits = torch.tensor([0.0, 5.0, 1.0, 0.0])
        end_logits = torch.tensor([0.0, 0.0, 4.0, 1.0])

        start, end, score = decode_best_span(start_logits, end_logits, max_answer_len=4)

        assert (start, end) == (1, 2)
        assert score == pytest.approx(9.0)

    def test_decode_best_span_respects_max_answer_len(self) -> None:
        from project.aspect_attention import decode_best_span

        start_logits = torch.tensor([5.0, 0.0, 0.0, 0.0])
        end_logits = torch.tensor([0.0, 0.0, 0.0, 5.0])

        start, end, _ = decode_best_span(start_logits, end_logits, max_answer_len=2)

        assert end - start < 2

    def test_decode_spans_for_sample_null_below_threshold(self) -> None:
        from project.aspect_attention import AspectHeads, decode_spans_for_sample

        heads = AspectHeads(d_model=8, num_aspects=len(ASPECTS), enable_span=True)
        hidden = torch.randn(1, 5, 8)
        mask = torch.ones(1, 5, dtype=torch.long)
        out = heads(hidden, mask)

        out.has_evidence_logits = torch.full_like(out.has_evidence_logits, -10.0)
        spans = decode_spans_for_sample(out, 0, has_evidence_threshold=0.5)
        assert all(span is None for span in spans)

        out.has_evidence_logits = torch.full_like(out.has_evidence_logits, 10.0)
        spans = decode_spans_for_sample(out, 0, has_evidence_threshold=0.5)
        assert all(span is not None for span in spans)

    def test_decode_spans_for_sample_requires_span_head(self) -> None:
        from project.aspect_attention import AspectHeads, decode_spans_for_sample

        heads = AspectHeads(d_model=8, num_aspects=len(ASPECTS), enable_span=False)
        hidden = torch.randn(1, 5, 8)
        mask = torch.ones(1, 5, dtype=torch.long)
        out = heads(hidden, mask)

        with pytest.raises(ValueError):
            decode_spans_for_sample(out, 0)


class TestLossFunctions:
    """Loss composition: score + span (with null-span masking) + faithfulness."""

    def test_score_loss_fn_zero_when_equal(self) -> None:
        from project.losses import score_loss_fn

        scores = torch.rand(3, 5)
        assert score_loss_fn(scores, scores).item() == pytest.approx(0.0, abs=1e-6)

    def test_span_loss_fn_all_invalid_returns_zero_without_nan(self) -> None:
        from project.losses import span_loss_fn

        b, a, t = 2, 5, 6
        start_logits = torch.randn(b, a, t, requires_grad=True)
        end_logits = torch.randn(b, a, t, requires_grad=True)
        start_labels = torch.full((b, a), -100, dtype=torch.long)
        end_labels = torch.full((b, a), -100, dtype=torch.long)

        loss = span_loss_fn(start_logits, end_logits, start_labels, end_labels)

        assert loss.item() == pytest.approx(0.0)
        assert not bool(torch.isnan(loss))

    def test_span_loss_fn_valid_spans_are_supervised(self) -> None:
        from project.losses import span_loss_fn

        torch.manual_seed(0)
        start_logits = torch.randn(2, 1, 6)
        end_logits = torch.randn(2, 1, 6)
        start_labels = torch.tensor([[2], [-100]])
        end_labels = torch.tensor([[4], [-100]])

        loss = span_loss_fn(start_logits, end_logits, start_labels, end_labels)

        assert loss.item() > 0
        assert not bool(torch.isnan(loss))

    def test_has_evidence_loss_fn_matches_bce(self) -> None:
        import torch.nn.functional as F

        from project.losses import has_evidence_loss_fn

        logits = torch.randn(4, 5)
        labels = torch.randint(0, 2, (4, 5)).float()

        expected = F.binary_cross_entropy_with_logits(logits, labels)
        actual = has_evidence_loss_fn(logits, labels)

        assert actual.item() == pytest.approx(expected.item())

    def test_faithfulness_loss_returns_named_components(self) -> None:
        from project.losses import faithfulness_loss_fn

        b, a, t = 2, 5, 8
        pred_scores = torch.rand(b, a)
        has_evidence_logits = torch.randn(b, a)
        attn_weights = torch.softmax(torch.randn(b, a, t), dim=-1)
        has_evidence_labels = torch.randint(0, 2, (b, a))
        start_labels = torch.full((b, a), -100, dtype=torch.long)
        end_labels = torch.full((b, a), -100, dtype=torch.long)

        loss, components = faithfulness_loss_fn(
            pred_scores,
            has_evidence_logits,
            attn_weights,
            has_evidence_labels,
            start_labels,
            end_labels,
            low_threshold=0.25,
            high_threshold=0.5,
        )

        assert set(components.keys()) == {"faith_low", "faith_high", "faith_consistency"}
        assert not bool(torch.isnan(loss))

    def test_compute_total_loss_combines_components_and_backprops(self) -> None:
        from project.config import AspectConfig
        from project.losses import compute_total_loss

        b, a, t = 2, 5, 8
        cfg = AspectConfig(
            enabled=True,
            hidden_layer=-1,
            attn_dropout=0.1,
            score_hidden=8,
            lambda_ce=1.0,
            lambda_score=0.5,
            lambda_span=0.5,
            lambda_faith=0.1,
            low_score_threshold=0.25,
            high_score_threshold=0.5,
            prefer_span_evidence=False,
            heads_checkpoint_name="aspect_heads.pt",
        )
        ce_loss = torch.tensor(2.0, requires_grad=True)
        aspect_scores = torch.rand(b, a, requires_grad=True)
        aspect_scores_gt = torch.rand(b, a)
        has_evidence_logits = torch.randn(b, a, requires_grad=True)
        has_evidence_labels = torch.randint(0, 2, (b, a))
        start_logits = torch.randn(b, a, t, requires_grad=True)
        end_logits = torch.randn(b, a, t, requires_grad=True)
        start_labels = torch.full((b, a), -100, dtype=torch.long)
        end_labels = torch.full((b, a), -100, dtype=torch.long)
        attn_weights = torch.softmax(torch.randn(b, a, t), dim=-1)

        total, components = compute_total_loss(
            ce_loss,
            aspect_scores,
            aspect_scores_gt,
            has_evidence_logits,
            has_evidence_labels,
            start_logits,
            end_logits,
            start_labels,
            end_labels,
            attn_weights,
            cfg,
        )

        expected_keys = {
            "ce_loss",
            "score_loss",
            "span_loss",
            "has_evidence_loss",
            "faith_loss",
            "total_loss",
            "faith_low",
            "faith_high",
            "faith_consistency",
        }
        assert set(components.keys()) == expected_keys
        assert components["total_loss"] == pytest.approx(total.item())

        total.backward()
        assert aspect_scores.grad is not None
        assert has_evidence_logits.grad is not None
        assert start_logits.grad is not None
        assert end_logits.grad is not None

    def test_compute_total_loss_weights_are_applied(self) -> None:
        """Zeroing every lambda except lambda_ce should recover plain CE loss."""
        from project.config import AspectConfig
        from project.losses import compute_total_loss

        b, a, t = 2, 5, 8
        cfg = AspectConfig(
            enabled=True,
            hidden_layer=-1,
            attn_dropout=0.1,
            score_hidden=8,
            lambda_ce=1.0,
            lambda_score=0.0,
            lambda_span=0.0,
            lambda_faith=0.0,
            low_score_threshold=0.25,
            high_score_threshold=0.5,
            prefer_span_evidence=False,
            heads_checkpoint_name="aspect_heads.pt",
        )
        ce_loss = torch.tensor(3.14)
        aspect_scores = torch.rand(b, a)
        aspect_scores_gt = torch.rand(b, a)
        has_evidence_logits = torch.randn(b, a)
        has_evidence_labels = torch.randint(0, 2, (b, a))
        start_logits = torch.randn(b, a, t)
        end_logits = torch.randn(b, a, t)
        start_labels = torch.full((b, a), -100, dtype=torch.long)
        end_labels = torch.full((b, a), -100, dtype=torch.long)
        attn_weights = torch.softmax(torch.randn(b, a, t), dim=-1)

        total, _ = compute_total_loss(
            ce_loss,
            aspect_scores,
            aspect_scores_gt,
            has_evidence_logits,
            has_evidence_labels,
            start_logits,
            end_logits,
            start_labels,
            end_labels,
            attn_weights,
            cfg,
        )

        assert total.item() == pytest.approx(ce_loss.item())


class TestAspectGuidedModelSmoke:
    """Lightweight end-to-end wiring check for project.hybrid_model."""

    def _make_fake_base_model(self, d_model: int = 16, vocab: int = 50):
        transformers = pytest.importorskip("transformers")  # noqa: F841
        from types import SimpleNamespace

        from torch import nn

        class FakeConfig:
            hidden_size = d_model

        class FakeBaseModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab, d_model)
                self.lm_head = nn.Linear(d_model, vocab)
                self.config = FakeConfig()

            def forward(
                self,
                input_ids,
                attention_mask=None,
                labels=None,
                output_hidden_states=False,
                return_dict=True,
                **kwargs,
            ):
                h = self.embed(input_ids)
                logits = self.lm_head(h)
                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
                    )
                hidden_states = (h, h) if output_hidden_states else None
                return SimpleNamespace(loss=loss, logits=logits, hidden_states=hidden_states)

            def generate(self, *args, **kwargs):
                return torch.zeros(1, 3, dtype=torch.long)

        return FakeBaseModel()

    def test_forward_produces_all_expected_outputs(self) -> None:
        from project.config import AspectConfig
        from project.hybrid_model import AspectGuidedModel

        cfg = AspectConfig(
            enabled=True,
            hidden_layer=-1,
            attn_dropout=0.1,
            score_hidden=8,
            lambda_ce=1.0,
            lambda_score=0.5,
            lambda_span=0.5,
            lambda_faith=0.1,
            low_score_threshold=0.25,
            high_score_threshold=0.5,
            prefer_span_evidence=False,
            heads_checkpoint_name="aspect_heads.pt",
        )
        base = self._make_fake_base_model()
        model = AspectGuidedModel(base, cfg)

        b, t = 2, 9
        input_ids = torch.randint(0, 50, (b, t))
        attention_mask = torch.ones(b, t, dtype=torch.long)
        labels = input_ids.clone()
        user_token_mask = torch.zeros(b, t, dtype=torch.long)
        user_token_mask[:, 2:6] = 1

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            user_token_mask=user_token_mask,
        )

        assert out.ce_loss is not None
        assert out.aspect_scores.shape == (b, len(ASPECTS))
        assert out.aspect_start_logits.shape == (b, len(ASPECTS), t)
        assert out.aspect_attn_weights.shape == (b, len(ASPECTS), t)
        # Attention must respect the provided user_token_mask.
        outside = out.aspect_attn_weights[:, :, user_token_mask[0] == 0]
        assert torch.allclose(outside, torch.zeros_like(outside), atol=1e-6)

    def test_save_and_load_aspect_heads_roundtrip(self, tmp_path) -> None:
        from project.config import AspectConfig
        from project.hybrid_model import AspectGuidedModel, load_aspect_heads, save_aspect_heads

        cfg = AspectConfig(
            enabled=True,
            hidden_layer=-1,
            attn_dropout=0.1,
            score_hidden=8,
            lambda_ce=1.0,
            lambda_score=0.5,
            lambda_span=0.5,
            lambda_faith=0.1,
            low_score_threshold=0.25,
            high_score_threshold=0.5,
            prefer_span_evidence=False,
            heads_checkpoint_name="aspect_heads.pt",
        )
        model = AspectGuidedModel(self._make_fake_base_model(), cfg)
        path = tmp_path / "aspect_heads.pt"
        save_aspect_heads(model, path)
        assert path.exists()

        model2 = AspectGuidedModel(self._make_fake_base_model(), cfg)
        load_aspect_heads(model2, path)

        for p1, p2 in zip(model.aspect_heads.parameters(), model2.aspect_heads.parameters()):
            assert torch.allclose(p1, p2)
