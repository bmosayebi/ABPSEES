"""Multi-objective training loss for the Aspect-Guided hybrid model.

The primary generative path still uses the standard HuggingFace causal
language modeling cross-entropy loss on assistant-response tokens only
(unchanged, computed by the base model itself). This module adds the three
*auxiliary* objectives introduced by the aspect-guided attention heads
(:mod:`project.aspect_attention`) and combines all four into the single
scalar loss optimized end-to-end:

    L = lambda_ce   * L_CE     (unchanged generative JSON loss)
      + lambda_score * L_score  (per-aspect score regression)
      + lambda_span  * L_span   (per-aspect start/end span classification)
      + lambda_faith * L_faith  (faithfulness self-consistency regularizer)

All four terms are computed per-batch and averaged; the lambda weights come
from :class:`project.config.AspectConfig` and are applied by
:class:`project.trainer.AspectGuidedTrainer`, which is the sole caller of
:func:`compute_total_loss`. Keeping the weighting logic out of the model
(:mod:`project.hybrid_model`) means the model's forward pass always returns
well-defined, un-weighted quantities that are easy to unit test in
isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from project.constants import ASPECTS

if TYPE_CHECKING:
    import torch

    from project.config import AspectConfig


def check_faithfulness_consistency(
    labels: dict[str, Any],
    score_threshold: float = 0.5,
) -> list[str]:
    """Flag inconsistent score/evidence pairs for analysis (not training).

    This is the original *evaluation-side* (text-based) helper, used by
    :func:`project.inference.predict` to warn about inconsistencies in the
    generated JSON. It is independent of the tensor-based training losses
    below.

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


def score_loss_fn(pred_scores: "torch.Tensor", gt_scores: "torch.Tensor") -> "torch.Tensor":
    """Per-aspect score regression loss.

    Uses Smooth L1 (Huber) rather than plain MSE: scores are bounded in
    ``[0, 1]`` and mostly clustered away from 0/0.5 boundaries, so a loss
    that is quadratic near zero error but linear (robust to outliers) for
    larger errors trains more stably than MSE when early-training score
    predictions are far from target.

    Args:
        pred_scores: ``[B, A]`` predicted scores in ``[0, 1]``.
        gt_scores: ``[B, A]`` ground-truth scores in ``[0, 1]``.

    Returns:
        Scalar loss (batch- and aspect-averaged).
    """
    import torch.nn.functional as F

    return F.smooth_l1_loss(pred_scores, gt_scores)


def has_evidence_loss_fn(
    has_evidence_logits: "torch.Tensor",
    has_evidence_labels: "torch.Tensor",
) -> "torch.Tensor":
    """Binary cross-entropy for the per-aspect has-evidence gate.

    Args:
        has_evidence_logits: ``[B, A]`` raw logits.
        has_evidence_labels: ``[B, A]`` binary ground-truth labels (1 if the
            aspect has a non-empty evidence span, else 0).

    Returns:
        Scalar BCE loss.
    """
    import torch.nn.functional as F

    return F.binary_cross_entropy_with_logits(has_evidence_logits, has_evidence_labels.float())


def span_loss_fn(
    start_logits: "torch.Tensor",
    end_logits: "torch.Tensor",
    start_labels: "torch.Tensor",
    end_labels: "torch.Tensor",
) -> "torch.Tensor":
    """Per-aspect start/end token classification loss.

    Standard extractive-QA style cross-entropy over token positions,
    flattened across the batch and aspect dimensions. Aspects/samples with
    no evidence use the ``-100`` ignore-index sentinel (see
    :data:`project.constants.INVALID_TOKEN_SPAN`) for both ``start_labels``
    and ``end_labels``, so they contribute zero to this loss -- their
    supervision comes entirely from :func:`has_evidence_loss_fn` instead.
    This avoids forcing the model to point at an arbitrary token when there
    is genuinely no evidence to extract.

    Args:
        start_logits: ``[B, A, T]`` masked start logits.
        end_logits: ``[B, A, T]`` masked end logits.
        start_labels: ``[B, A]`` long tensor, ``-100`` where no evidence.
        end_labels: ``[B, A]`` long tensor, ``-100`` where no evidence.

    Returns:
        Scalar loss, or a zero tensor (still connected to the graph) when no
        sample in the batch has any valid span.
    """
    import torch.nn.functional as F

    b, a, t = start_logits.shape
    start_logits_flat = start_logits.reshape(b * a, t)
    end_logits_flat = end_logits.reshape(b * a, t)
    start_labels_flat = start_labels.reshape(b * a)
    end_labels_flat = end_labels.reshape(b * a)

    valid = start_labels_flat != -100
    if not bool(valid.any()):
        # No evidence anywhere in this batch: keep the graph connected via a
        # zero-valued but gradient-carrying expression rather than a bare
        # python 0.0, so mixed loss sums remain well-defined tensors. Touch
        # *both* start_logits and end_logits so callers relying on
        # `.grad` being populated after `.backward()` for either tensor
        # (e.g. custom logging/debugging) don't silently get `None`.
        return (start_logits.sum() + end_logits.sum()) * 0.0

    start_loss = F.cross_entropy(start_logits_flat, start_labels_flat, ignore_index=-100)
    end_loss = F.cross_entropy(end_logits_flat, end_labels_flat, ignore_index=-100)
    return 0.5 * (start_loss + end_loss)


def _build_gold_span_mask(
    start_labels: "torch.Tensor",
    end_labels: "torch.Tensor",
    num_tokens: int,
) -> "torch.Tensor":
    """Build a ``[B, A, T]`` binary mask covering each gold evidence span.

    Args:
        start_labels: ``[B, A]`` long tensor, ``-100`` where no evidence.
        end_labels: ``[B, A]`` long tensor, ``-100`` where no evidence.
        num_tokens: Sequence length ``T``.

    Returns:
        Float mask, 1 for tokens inside ``[start, end]`` (inclusive) when a
        valid span exists, 0 everywhere else (including entirely for
        no-evidence aspects).
    """
    import torch

    valid = (start_labels != -100) & (end_labels != -100)
    idx = torch.arange(num_tokens, device=start_labels.device).view(1, 1, num_tokens)
    start = start_labels.clamp(min=0).unsqueeze(-1)
    end = end_labels.clamp(min=0).unsqueeze(-1)
    span_mask = (idx >= start) & (idx <= end)
    return (span_mask & valid.unsqueeze(-1)).float()


def faithfulness_loss_fn(
    pred_scores: "torch.Tensor",
    has_evidence_logits: "torch.Tensor",
    attn_weights: "torch.Tensor",
    has_evidence_labels: "torch.Tensor",
    start_labels: "torch.Tensor",
    end_labels: "torch.Tensor",
    low_threshold: float,
    high_threshold: float,
) -> tuple["torch.Tensor", dict[str, float]]:
    """Faithfulness self-consistency regularizer between score and evidence.

    Three differentiable terms, averaged with equal internal weight (the
    outer ``lambda_faith`` from :class:`AspectConfig` scales the sum):

    1. **Low-score -> empty-evidence** (``term_low``): for aspects the model
       itself scores below ``low_threshold``, penalize a high predicted
       has-evidence probability ``p_evid = sigmoid(has_evidence_logit)``:

           term_low = mean_{a : score_a < low_threshold} [ p_evid(a)^2 ]

    2. **High-score -> evidence-overlap** (``term_high``): for aspects with
       ground-truth evidence *and* a model score above ``high_threshold``,
       encourage the aspect-attention distribution to concentrate its mass
       inside the gold evidence span (``gold_mass`` = attention weight
       summed over gold tokens, in ``[0, 1]`` since attention is a
       softmax):

           term_high = mean_{a : score_a > high_threshold, has_gt_evidence(a)}
                           [ 1 - gold_mass(a) ]

    3. **Score/evidence-presence consistency** (``term_consistency``): a
       self-supervised BCE between the model's *own* has-evidence logit and
       its *own* score thresholded at 0.5, independent of ground truth --
       this discourages the score head and the evidence head from
       "disagreeing" with each other even when both happen to be wrong
       relative to the label (a common failure mode of multi-head models
       trained with independent per-head losses only):

           term_consistency = BCE( has_evidence_logit, 1[score >= 0.5] )

    All threshold-based masks are computed on ``pred_scores.detach()`` so
    gradients flow only through the continuous terms (``p_evid``,
    ``gold_mass``, the BCE logit), not through the (non-differentiable)
    thresholding used to select which aspects each term applies to.

    Args:
        pred_scores: ``[B, A]`` predicted scores in ``[0, 1]``.
        has_evidence_logits: ``[B, A]`` raw has-evidence logits.
        attn_weights: ``[B, A, T]`` aspect-attention weights (post-softmax).
        has_evidence_labels: ``[B, A]`` binary ground-truth evidence-presence.
        start_labels: ``[B, A]`` long tensor, ``-100`` where no evidence.
        end_labels: ``[B, A]`` long tensor, ``-100`` where no evidence.
        low_threshold: Score threshold below which evidence is expected to
            be empty (``AspectConfig.low_score_threshold``).
        high_threshold: Score threshold above which evidence is expected to
            be present and attention-aligned (``AspectConfig.high_score_threshold``).

    Returns:
        Tuple of ``(scalar_loss, component_dict)`` where ``component_dict``
        has float values for ``faith_low``, ``faith_high``, and
        ``faith_consistency`` (for logging).
    """
    import torch
    import torch.nn.functional as F

    p_evid = torch.sigmoid(has_evidence_logits)
    eps = 1.0

    low_mask = (pred_scores.detach() < low_threshold).float()
    term_low = (low_mask * p_evid.pow(2)).sum() / (low_mask.sum() + eps)

    num_tokens = attn_weights.shape[-1]
    gold_mask = _build_gold_span_mask(start_labels, end_labels, num_tokens)
    gold_mass = (attn_weights * gold_mask).sum(dim=-1)  # [B, A]
    high_mask = (pred_scores.detach() > high_threshold).float() * has_evidence_labels.float()
    term_high = (high_mask * (1.0 - gold_mass)).sum() / (high_mask.sum() + eps)

    consistency_target = (pred_scores.detach() >= 0.5).float()
    term_consistency = F.binary_cross_entropy_with_logits(has_evidence_logits, consistency_target)

    total = (term_low + term_high + term_consistency) / 3.0
    components = {
        "faith_low": float(term_low.detach().item()),
        "faith_high": float(term_high.detach().item()),
        "faith_consistency": float(term_consistency.detach().item()),
    }
    return total, components


def compute_total_loss(
    ce_loss: "torch.Tensor",
    aspect_scores: "torch.Tensor",
    aspect_scores_gt: "torch.Tensor",
    has_evidence_logits: "torch.Tensor",
    has_evidence_labels: "torch.Tensor",
    start_logits: "torch.Tensor",
    end_logits: "torch.Tensor",
    start_labels: "torch.Tensor",
    end_labels: "torch.Tensor",
    attn_weights: "torch.Tensor",
    aspect_config: "AspectConfig",
) -> tuple["torch.Tensor", dict[str, float]]:
    """Combine the generative CE loss with the three auxiliary objectives.

    ``L = lambda_ce * ce_loss + lambda_score * score_loss``
    ``  + lambda_span * span_loss + lambda_faith * faith_loss``

    Args:
        ce_loss: Scalar causal-LM cross-entropy from the base model.
        aspect_scores: ``[B, A]`` predicted scores.
        aspect_scores_gt: ``[B, A]`` ground-truth scores.
        has_evidence_logits: ``[B, A]`` has-evidence logits.
        has_evidence_labels: ``[B, A]`` binary ground-truth evidence presence.
        start_logits: ``[B, A, T]`` start logits.
        end_logits: ``[B, A, T]`` end logits.
        start_labels: ``[B, A]`` gold start indices (``-100`` if none).
        end_labels: ``[B, A]`` gold end indices (``-100`` if none).
        attn_weights: ``[B, A, T]`` aspect-attention weights.
        aspect_config: Hyperparameters incl. the ``lambda_*`` weights.

    Returns:
        Tuple of ``(total_loss, components)`` where ``components`` is a
        flat ``dict[str, float]`` suitable for logging (``ce_loss``,
        ``score_loss``, ``span_loss``, ``faith_loss`` and its three
        sub-terms, and ``total_loss``).
    """
    score_loss = score_loss_fn(aspect_scores, aspect_scores_gt)
    evidence_loss = has_evidence_loss_fn(has_evidence_logits, has_evidence_labels)
    span_loss = span_loss_fn(start_logits, end_logits, start_labels, end_labels)
    faith_loss, faith_components = faithfulness_loss_fn(
        pred_scores=aspect_scores,
        has_evidence_logits=has_evidence_logits,
        attn_weights=attn_weights,
        has_evidence_labels=has_evidence_labels,
        start_labels=start_labels,
        end_labels=end_labels,
        low_threshold=aspect_config.low_score_threshold,
        high_threshold=aspect_config.high_score_threshold,
    )

    # The has-evidence BCE is part of the (Phase 2) span-prediction
    # objective family -- it supervises the same "where is the evidence"
    # question as start/end classification, just for the null case -- so it
    # is folded into lambda_span rather than introducing a fifth weight.
    combined_span_loss = 0.5 * (span_loss + evidence_loss)

    total = (
        aspect_config.lambda_ce * ce_loss
        + aspect_config.lambda_score * score_loss
        + aspect_config.lambda_span * combined_span_loss
        + aspect_config.lambda_faith * faith_loss
    )

    components = {
        "ce_loss": float(ce_loss.detach().item()),
        "score_loss": float(score_loss.detach().item()),
        "span_loss": float(combined_span_loss.detach().item()),
        "has_evidence_loss": float(evidence_loss.detach().item()),
        "faith_loss": float(faith_loss.detach().item()),
        "total_loss": float(total.detach().item()),
        **faith_components,
    }
    return total, components
