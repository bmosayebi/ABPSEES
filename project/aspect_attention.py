"""Aspect-guided attention module (Phase 1 + Phase 2).

This module implements the auxiliary hybrid architecture described in
``docs/ASPECT_ATTENTION_GUIDE.md``. It is *auxiliary* to (not a replacement
for) the primary generative JSON pipeline: it attaches a small set of
learnable heads on top of frozen/LoRA-adapted Qwen hidden states to (a)
produce an explicit per-aspect attention distribution over input tokens,
(b) regress a per-aspect relevance score from that distribution, and
(c) (Phase 2) predict a per-aspect evidence span with an explicit
"no-evidence" gate.

Notation
--------
Let ``H in R^{T x D}`` be the hidden states of one sequence (``T`` tokens,
``D`` = model hidden size) taken from a configurable decoder layer, and let
``A`` = ``len(ASPECTS)`` = 5 be the fixed number of aspects. Each aspect
``a`` owns a learnable query vector ``q_a in R^D`` (Phase 1). Attention is
restricted to the *user-text* tokens via a boolean mask ``m in {0,1}^T`` so
that the model cannot "cheat" by attending to the system prompt, the
assistant's own JSON completion, or padding.

    alpha_a = softmax( (q_a W_Q)(H W_K)^T / sqrt(D) + (1 - m) * (-inf) )   in R^T
    c_a     = alpha_a (H W_V)                                              in R^D
    s_a     = sigmoid( MLP(c_a) )                                          in [0, 1]

``alpha_a`` is the aspect-conditioned attention distribution (directly
visualizable as a heatmap over the input text), ``c_a`` is the pooled
aspect context, and ``s_a`` is the Phase 1 score-head prediction for
aspect ``a``.

Phase 2 adds, for each aspect, a start/end token classifier and a binary
"has evidence" gate (analogous to the has-answer head in SQuAD 2.0 /
BiDAF-style extractive QA), computed from ``c_a`` and the full sequence
``H`` without materializing a ``[T, D]`` tensor per aspect (see
:class:`SpanHead` for the algebraic identity used).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from project.constants import ASPECTS

# Large negative value used to mask out non-attendable positions before
# softmax / argmax. Kept well within float16 range (max ~65504) so this is
# safe under autocast/fp16 training, unlike -1e9 or -inf which would
# overflow or produce NaNs after masked_fill + softmax in fp16.
NEG_INF = -1.0e4


class AspectEmbedding(nn.Module):
    """Learnable per-aspect query matrix ``Q_A in R^{A x D}``.

    One row per fixed aspect in :data:`project.constants.ASPECTS`, in that
    exact order, so aspect index ``i`` always refers to ``ASPECTS[i]``.
    """

    def __init__(self, num_aspects: int, d_model: int) -> None:
        super().__init__()
        self.num_aspects = num_aspects
        self.d_model = d_model
        # Small init (scaled by 1/sqrt(D)) keeps initial attention close to
        # uniform, avoiding a peaky/degenerate distribution before training.
        self.embedding = nn.Parameter(torch.randn(num_aspects, d_model) * (d_model**-0.5))

    def forward(self) -> torch.Tensor:
        """Return the ``[num_aspects, d_model]`` query matrix."""
        return self.embedding


class AspectConditionedAttention(nn.Module):
    """Cross-attention from a small fixed "aspect vocabulary" onto tokens.

    This is standard scaled dot-product attention (Vaswani et al., 2017)
    with the query sequence replaced by the ``num_aspects`` learned aspect
    embeddings instead of per-token queries, so it can be understood as a
    single extra attention head whose queries are aspect identities rather
    than positions. Key/value projections are shared across aspects; only
    the query differs, which is what makes the resulting attention maps
    directly comparable across aspects for the same input.
    """

    def __init__(self, d_model: int, num_aspects: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.aspect_embedding = AspectEmbedding(num_aspects, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.scale = d_model**-0.5

    def forward(
        self, hidden_states: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute pooled aspect contexts and attention weights.

        Args:
            hidden_states: ``[B, T, D]`` hidden states from the base model.
            token_mask: ``[B, T]`` with 1 for attendable tokens (user text),
                0 elsewhere (system prompt, assistant text, padding). Every
                row must contain at least one ``1``; callers should fall
                back to full-sequence attention for degenerate rows (see
                :func:`safe_token_mask`).

        Returns:
            Tuple of:
                - ``context``: ``[B, A, D]`` pooled context vectors ``c_a``.
                - ``attn_weights``: ``[B, A, T]`` post-softmax attention
                  weights ``alpha_a``, useful for visualization.
        """
        q = self.q_proj(self.aspect_embedding())  # [A, D]
        k = self.k_proj(hidden_states)  # [B, T, D]
        v = self.v_proj(hidden_states)  # [B, T, D]

        scores = torch.einsum("ad,btd->bat", q, k) * self.scale  # [B, A, T]
        mask = token_mask.unsqueeze(1).to(dtype=scores.dtype)  # [B, 1, T]
        scores = scores.masked_fill(mask == 0, NEG_INF)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        context = torch.einsum("bat,btd->bad", attn_weights, v)  # [B, A, D]
        return context, attn_weights


class ScoreHead(nn.Module):
    """Per-aspect MLP regression head: pooled context -> score in [0, 1].

    A single shared 2-layer MLP is applied independently to each aspect's
    context vector ``c_a`` (batched over the aspect dimension), followed by
    a sigmoid to constrain outputs to the valid ``[0, 1]`` score range
    defined by the task.
    """

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """``context``: ``[B, A, D]`` -> scores: ``[B, A]`` in ``[0, 1]``."""
        logits = self.net(context).squeeze(-1)  # [B, A]
        return torch.sigmoid(logits)


class SpanHead(nn.Module):
    """Per-aspect start/end token classifiers + a binary has-evidence gate.

    Naively fusing every token with every aspect context via broadcasted
    elementwise multiplication would materialize a ``[B, A, T, D]`` tensor,
    which is prohibitively expensive for long sequences. We avoid this using
    the following identity. Define an aspect- and token-independent
    "readout" vector ``w in R^D`` (``start_vec`` or ``end_vec``) and a
    per-aspect sigmoid gate ``g_a = sigmoid(W_g c_a) in R^D``. The gated
    start logit for aspect ``a`` at token ``t`` is defined as

        logit_start(a, t) = <w_start * g_a, h_t> = sum_d (w_start[d] * g_a[d]) * h_t[d]

    which factors as a per-aspect vector ``u_a = w_start * g_a in R^D``
    dotted with every token's hidden state -- i.e. a single
    ``[B, A, D] x [B, T, D] -> [B, A, T]`` einsum, with no ``T x D`` blow-up
    per aspect. This is algebraically equivalent to gated fusion followed by
    a linear read-out, computed in ``O(B*A*D + B*A*T)`` instead of
    ``O(B*A*T*D)``.

    The has-evidence gate is a plain linear read-out of the aspect context,
    analogous to the "has-answer" head used in SQuAD 2.0 style extractive QA
    to allow abstaining when there is no valid span.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_model)
        self.start_vec = nn.Parameter(torch.randn(d_model) * (d_model**-0.5))
        self.end_vec = nn.Parameter(torch.randn(d_model) * (d_model**-0.5))
        self.has_evidence_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute masked start/end logits and has-evidence logits.

        Args:
            hidden_states: ``[B, T, D]``.
            context: ``[B, A, D]`` pooled aspect contexts from
                :class:`AspectConditionedAttention`.
            token_mask: ``[B, T]`` attendable-token mask (same convention as
                the attention module).

        Returns:
            Tuple of ``start_logits``, ``end_logits`` (each ``[B, A, T]``,
            masked to ``-1e4`` outside ``token_mask``) and
            ``has_evidence_logits`` (``[B, A]``, unmasked raw logits).
        """
        gate = torch.sigmoid(self.gate_proj(context))  # [B, A, D]
        u_start = gate * self.start_vec  # [B, A, D]
        u_end = gate * self.end_vec  # [B, A, D]

        start_logits = torch.einsum("bad,btd->bat", u_start, hidden_states)  # [B, A, T]
        end_logits = torch.einsum("bad,btd->bat", u_end, hidden_states)  # [B, A, T]

        mask = token_mask.unsqueeze(1).to(dtype=start_logits.dtype)  # [B, 1, T]
        start_logits = start_logits.masked_fill(mask == 0, NEG_INF)
        end_logits = end_logits.masked_fill(mask == 0, NEG_INF)

        has_evidence_logits = self.has_evidence_proj(context).squeeze(-1)  # [B, A]
        return start_logits, end_logits, has_evidence_logits


@dataclass
class AspectHeadsOutput:
    """Structured output of :class:`AspectHeads`."""

    scores: torch.Tensor  # [B, A] in [0, 1]
    attn_weights: torch.Tensor  # [B, A, T] post-softmax attention
    has_evidence_logits: torch.Tensor | None = None  # [B, A]
    start_logits: torch.Tensor | None = None  # [B, A, T]
    end_logits: torch.Tensor | None = None  # [B, A, T]


class AspectHeads(nn.Module):
    """Top-level auxiliary module combining Phase 1 + Phase 2 heads.

    Args:
        d_model: Hidden size of the base model's hidden states.
        num_aspects: Number of fixed aspects (``len(ASPECTS)`` by default).
        attn_dropout: Dropout applied to the aspect-attention weights.
        score_hidden: Hidden width of the :class:`ScoreHead` MLP.
        enable_span: Whether to build the Phase 2 span/has-evidence heads.
            When ``False``, only Phase 1 (attention + score head) runs.
    """

    def __init__(
        self,
        d_model: int,
        num_aspects: int = len(ASPECTS),
        attn_dropout: float = 0.1,
        score_hidden: int = 256,
        enable_span: bool = True,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_aspects = num_aspects
        self.attention = AspectConditionedAttention(d_model, num_aspects, dropout=attn_dropout)
        self.score_head = ScoreHead(d_model, score_hidden)
        self.enable_span = enable_span
        self.span_head = SpanHead(d_model) if enable_span else None

    def forward(self, hidden_states: torch.Tensor, token_mask: torch.Tensor) -> AspectHeadsOutput:
        """Run Phase 1 (+ Phase 2, if enabled) heads on one batch.

        Args:
            hidden_states: ``[B, T, D]`` hidden states from the base model.
            token_mask: ``[B, T]`` attendable-token mask.

        Returns:
            :class:`AspectHeadsOutput` with scores, attention weights, and
            (if ``enable_span``) span/has-evidence logits.
        """
        safe_mask = safe_token_mask(token_mask)
        context, attn_weights = self.attention(hidden_states, safe_mask)
        scores = self.score_head(context)

        start_logits = end_logits = has_evidence_logits = None
        if self.span_head is not None:
            start_logits, end_logits, has_evidence_logits = self.span_head(
                hidden_states, context, safe_mask
            )

        return AspectHeadsOutput(
            scores=scores,
            attn_weights=attn_weights,
            has_evidence_logits=has_evidence_logits,
            start_logits=start_logits,
            end_logits=end_logits,
        )


def safe_token_mask(token_mask: torch.Tensor) -> torch.Tensor:
    """Fall back to full-sequence attention for rows with an empty mask.

    A completely-zero mask row would make every attention score ``-1e4``,
    producing a (numerically valid but semantically meaningless) uniform
    softmax. This can legitimately happen for malformed/truncated samples.
    Rather than let it silently degrade, we detect such rows and re-enable
    full-sequence attention for them so gradients stay well-behaved; the
    corresponding sample should typically also be excluded from the score
    and span losses via its own validity flags upstream.

    Args:
        token_mask: ``[B, T]`` binary mask.

    Returns:
        ``[B, T]`` mask, with all-zero rows replaced by all-ones.
    """
    row_has_token = token_mask.sum(dim=-1, keepdim=True) > 0
    return torch.where(row_has_token, token_mask, torch.ones_like(token_mask))


def decode_best_span(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    max_answer_len: int = 64,
) -> tuple[int, int, float]:
    """Decode the highest-scoring valid ``(start, end)`` span for one aspect.

    Standard extractive-QA decoding: search over all ``start <= end`` pairs
    with ``end - start < max_answer_len`` maximizing
    ``start_logits[start] + end_logits[end]``.

    Args:
        start_logits: ``[T]`` start logits (already masked to valid tokens).
        end_logits: ``[T]`` end logits (already masked to valid tokens).
        max_answer_len: Maximum allowed span length in tokens.

    Returns:
        ``(start_idx, end_idx, score)`` with ``score`` the summed logit of
        the best valid pair.
    """
    t = start_logits.shape[-1]
    best_score = float("-inf")
    best_span = (0, 0)
    # T is small enough (<= max_seq_length, typically a few hundred tokens
    # for the user-text span) that an O(T * max_answer_len) scan is cheap
    # and numerically transparent; a vectorized banded-matrix variant would
    # only matter at much larger T.
    start_vals = start_logits.tolist()
    end_vals = end_logits.tolist()
    for start in range(t):
        if start_vals[start] <= NEG_INF:
            continue
        end_limit = min(t, start + max_answer_len)
        for end in range(start, end_limit):
            if end_vals[end] <= NEG_INF:
                continue
            score = start_vals[start] + end_vals[end]
            if score > best_score:
                best_score = score
                best_span = (start, end)
    return best_span[0], best_span[1], best_score


def decode_spans_for_sample(
    output: AspectHeadsOutput,
    batch_idx: int,
    has_evidence_threshold: float = 0.5,
    max_answer_len: int = 64,
) -> list[tuple[int, int] | None]:
    """Decode per-aspect predicted spans for one sample in a batch.

    Args:
        output: Output of :meth:`AspectHeads.forward`.
        batch_idx: Index of the sample within the batch.
        has_evidence_threshold: Minimum ``sigmoid(has_evidence_logit)`` to
            emit a span instead of ``None`` (no evidence).
        max_answer_len: Maximum allowed span length in tokens.

    Returns:
        List of length ``num_aspects``, each either an inclusive
        ``(start, end)`` token span or ``None`` when the has-evidence gate
        predicts no evidence for that aspect.
    """
    if output.start_logits is None or output.end_logits is None or output.has_evidence_logits is None:
        raise ValueError("AspectHeadsOutput has no span predictions (enable_span=False).")

    num_aspects = output.scores.shape[1]
    has_evidence_probs = torch.sigmoid(output.has_evidence_logits[batch_idx])  # [A]
    spans: list[tuple[int, int] | None] = []
    for a in range(num_aspects):
        if has_evidence_probs[a].item() < has_evidence_threshold:
            spans.append(None)
            continue
        start, end, _ = decode_best_span(
            output.start_logits[batch_idx, a], output.end_logits[batch_idx, a], max_answer_len
        )
        spans.append((start, end))
    return spans
