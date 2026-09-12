# Aspect-Guided Attention: A Scientific Implementation Guide

> این سند، طبق درخواست کاربر، توضیح کامل، علمی و آکادمیک — گام به گام، از صفر
> تا صد — پیاده‌سازی «Aspect-Guided Attention» (فاز ۱: هدهای امتیازدهی مبتنی
> بر attention؛ فاز ۲: استخراج بازه‌ی evidence + منظم‌سازی faithfulness) را
> ارائه می‌دهد. بخش فنی/کد به انگلیسی نوشته شده تا با مستندات و کد پروژه
> (docstringها، نام کلاس‌ها و فایل‌ها) کاملاً همخوان و قابل ارجاع مستقیم باشد.

This document is the companion to `project/aspect_attention.py`,
`project/hybrid_model.py`, `project/losses.py`, and the collator/evaluation
extensions in `project/preprocessing.py` / `project/evaluation.py`. It
explains *why* the design looks the way it does, not just *what* the code
does — cross-referencing the exact classes/functions so the two stay in
sync.

## Table of Contents

1. [Problem Formulation](#1-problem-formulation)
2. [Background: Self-Attention vs. Aspect-Conditioned Attention](#2-background-self-attention-vs-aspect-conditioned-attention)
3. [Formal Notation: Phase 1 and Phase 2 Objectives](#3-formal-notation-phase-1-and-phase-2-objectives)
4. [Faithfulness Regularization: Justification](#4-faithfulness-regularization-justification)
5. [End-to-End Implementation Walkthrough](#5-end-to-end-implementation-walkthrough)
6. [Hyperparameter Guidance for Colab T4](#6-hyperparameter-guidance-for-colab-t4)
7. [Expected Metrics and Ablation Checklist](#7-expected-metrics-and-ablation-checklist)
8. [References](#8-references)

---

## 1. Problem Formulation

ABPSEES fine-tunes Qwen2.5-3B-Instruct to map a Persian laptop-purchase
description \(x\) to a structured judgment over a **fixed set of five
aspects** \(\mathcal{A} = \{\text{performance}, \text{portability},
\text{design}, \text{durability}, \text{cost\_effectiveness}\}\)
(`project/constants.py::ASPECTS`). For each aspect \(a \in \mathcal{A}\),
the model must produce:

- A **relevance score** \(s_a \in [0, 1]\).
- A **faithful evidence span** \(e_a\), which is either the empty string
  or an *exact substring* of \(x\) supporting \(s_a\).

The original (and still primary) approach treats this entirely as
**sequence generation**: the assistant emits a single JSON object, and the
model is trained with standard causal-LM cross-entropy on the
completion tokens. This is simple and leverages the pretrained LM's
language understanding, but it has two structural weaknesses that this
project's hybrid module (Phase 1 + Phase 2) targets directly:

1. **No explicit locus of "attention" per aspect.** The generative model
   must implicitly decide, token-by-token, what part of \(x\) is relevant
   to *each* of five different aspects while producing one linear sequence
   of JSON tokens. There is no direct signal or supervision that says
   "when producing the `performance` score, look here." Different aspects
   competing for the same underlying self-attention patterns can blur the
   *aspect-specific* evidence a human reviewer would point to.
2. **No explicit faithfulness constraint between score and evidence.**
   Nothing in the plain CE loss prevents the model from generating a high
   score with empty evidence, or a low score with a plausible-looking (but
   spurious) evidence string — both are "fluent" completions from the
   LM's perspective even though they are semantically inconsistent.

The hybrid module's job is to add a **small, interpretable auxiliary
signal** — one attention distribution and one score/span pair per aspect —
that (a) can be visualized directly (§5.6), (b) is trained with losses that
explicitly penalize score/evidence inconsistency (§4), and (c) can
optionally *repair* the generative JSON's evidence field at inference time
(`project/inference.py::predict`). Crucially, this is designed as a
**strict addition**: the generative JSON path, its tokenization, and its
loss are byte-for-byte unchanged when `aspect.enabled: false`.

## 2. Background: Self-Attention vs. Aspect-Conditioned Attention

Standard Transformer self-attention (Vaswani et al., 2017) computes, for
every token \(i\), a query \(q_i\) and attends over all tokens' keys:

\[
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
\]

Every layer/head in Qwen2.5 already does this, and it is *token-query*
driven: the query at each position is a function of that position's own
representation. There is no notion of "the query representing aspect
`durability`" anywhere in the stack — aspect information, if it exists at
all in the hidden states, is entangled and implicit.

**Aspect-conditioned attention** (implemented in
`project/aspect_attention.py::AspectConditionedAttention`) instead
replaces the *query* sequence with a small, fixed, **learned aspect
vocabulary**: five query vectors \(q_1, \dots, q_5\)
(`AspectEmbedding`), one per aspect, that are *the same for every input*
and updated only during training. Keys/values still come from the
sequence's hidden states \(H\), so this is a form of **cross-attention**
from a tiny fixed query set onto the (frozen-at-inference) token
representations — conceptually the same mechanism used for pooling in
extractive/attention-based sentence classification (e.g. attention
pooling in ABSA models, see §8) and in "slot"/"probe" query designs (e.g.
DETR's object queries, Perceiver's latent queries), just applied to a
5-slot aspect vocabulary instead of object slots.

Two properties make this useful here:

- **Interpretability.** Because there are exactly 5 queries and they are
  the same across all inputs, the resulting attention weights
  \(\alpha_a \in \mathbb{R}^T\) form a directly plottable heatmap per
  aspect (`project/visualization.py::plot_aspect_attention_heatmap`) — a
  luxury standard self-attention (with its \(O(\text{layers} \times
  \text{heads})\) maps, none tied to a semantic aspect) does not offer.
- **Restricted scope.** Attention is masked to the user-text tokens only
  (`user_token_mask`), so the aspect cannot "cheat" by attending to the
  fixed Persian system prompt or to the model's own JSON completion —
  every unit of attention mass must be spent on the actual input text.

## 3. Formal Notation: Phase 1 and Phase 2 Objectives

Let \(H \in \mathbb{R}^{T \times D}\) be the hidden states of one training
sequence at a configurable decoder layer (`aspect.hidden_layer`, default
`-1` = last layer) and \(m \in \{0, 1\}^T\) the user-token mask. For each
aspect \(a\):

### Phase 1 — Attention + Score

\[
\alpha_a = \text{softmax}\!\left(\frac{(q_a W_Q)(H W_K)^\top}{\sqrt{D}} + (1-m)\cdot(-\infty)\right) \in \mathbb{R}^T
\]
\[
c_a = \alpha_a (H W_V) \in \mathbb{R}^D
\]
\[
\hat{s}_a = \sigma\big(\text{MLP}(c_a)\big) \in [0, 1]
\]

Code: `AspectConditionedAttention.forward` (attention/context),
`ScoreHead.forward` (\(\hat s_a\)). Loss:

\[
L_{\text{score}} = \frac{1}{|\mathcal{A}|}\sum_a \text{SmoothL1}(\hat{s}_a, s_a)
\]

(`project/losses.py::score_loss_fn`). Smooth-L1 (Huber) is used rather
than MSE because early-training predictions can be far from the target
while scores themselves are bounded and mostly away from the loss's
quadratic region — Huber degrades gracefully to a linear penalty for
large errors, avoiding the gradient explosion MSE can produce here.

### Phase 2 — Span Extraction + Has-Evidence Gate

For each token \(t\), define an aspect-specific gate
\(g_a = \sigma(W_g c_a) \in \mathbb{R}^D\) and two shared read-out vectors
\(w_{\text{start}}, w_{\text{end}} \in \mathbb{R}^D\). The start/end logits
are:

\[
\text{start}_a(t) = \langle g_a \odot w_{\text{start}},\, h_t\rangle,\qquad
\text{end}_a(t) = \langle g_a \odot w_{\text{end}},\, h_t\rangle
\]

masked to \(-10^4\) outside \(m\). This is algebraically identical to
gating every token's hidden state with \(g_a\) and then applying a linear
read-out — but computed as a factored einsum
(`u_a = g_a \odot w \Rightarrow \text{logits} = u_a H^\top`) so it never
materializes a \([\,T \times D\,]\) tensor per aspect (see the docstring of
`SpanHead` for the full derivation). A separate linear read-out of
\(c_a\) produces the **has-evidence logit** \(\hat h_a\) (the null/no-answer
gate, directly analogous to SQuAD 2.0's has-answer head, §8).

Ground-truth labels use the `-100` ignore-index sentinel for
`aspect_start`/`aspect_end` whenever there is no evidence
(`token_span == INVALID_TOKEN_SPAN`, `project/constants.py`), so:

\[
L_{\text{span}} = \tfrac{1}{2}\Big(\text{CE}(\text{start\_logits}, \text{start\_labels}) + \text{CE}(\text{end\_logits}, \text{end\_labels})\Big),\quad
L_{\text{has\_ev}} = \text{BCE}(\hat h_a, \mathbb{1}[\text{evidence non-empty}])
\]

both ignore-index-masked so a genuinely evidence-free aspect never forces
the model to point at an arbitrary token (`project/losses.py::span_loss_fn`,
`has_evidence_loss_fn`). In the combined objective these are folded
together as `combined_span_loss = 0.5 * (L_span + L_has_ev)` under the
single `lambda_span` weight (they answer the same underlying "where/if is
the evidence" question).

### Combined Objective

\[
L = \lambda_{ce} L_{CE}(\text{JSON}) + \lambda_{score} L_{\text{score}} + \lambda_{span}\big(\tfrac12 L_{\text{span}} + \tfrac12 L_{\text{has\_ev}}\big) + \lambda_{faith} L_{\text{faith}}
\]

with defaults \(\lambda_{ce}=1.0,\ \lambda_{score}=0.5,\ \lambda_{span}=0.5,\
\lambda_{faith}=0.1\) (`config/default.yaml`, `config/colab.yaml`, the
`aspect:` section). \(L_{CE}\) is computed by the base Qwen model exactly
as before (unchanged); the other three terms are computed by
`project/losses.py::compute_total_loss` and combined by
`project/trainer.py::AspectGuidedTrainer.compute_loss`.

## 4. Faithfulness Regularization: Justification

Multi-head models trained with independent per-head losses only can learn
heads that are individually accurate on average yet **mutually
inconsistent** on any given example — e.g. the score head predicts 0.9
while the has-evidence head predicts "no evidence," which is a
self-contradiction no downstream consumer of the JSON should trust. The
faithfulness regularizer (`project/losses.py::faithfulness_loss_fn`) adds
three differentiable terms that directly target this, all computed on
`pred_scores.detach()` for the *selection* mask (so gradients only flow
through the continuous quantities, not through a non-differentiable
threshold):

1. **Low-score → empty-evidence** (`term_low`): for aspects the model
   itself scores below `aspect.low_score_threshold` (default `0.25`),
   penalize \(p_{\text{evid}} = \sigma(\hat h_a)^2\) — if the model doesn't
   think an aspect matters, it should not simultaneously claim to have
   found supporting evidence.
2. **High-score → evidence-overlap** (`term_high`): for aspects with
   *ground-truth* evidence and a model score above
   `aspect.high_score_threshold` (default `0.5`), encourage the attention
   distribution to concentrate inside the gold span:
   \(\text{gold\_mass}_a = \sum_{t \in [\text{start}_a,\text{end}_a]} \alpha_a(t)\),
   penalizing \(1 - \text{gold\_mass}_a\). This is what makes the
   attention heatmap trustworthy as an *explanation*, not just a
   byproduct.
3. **Score/evidence self-consistency** (`term_consistency`): a BCE between
   the model's own has-evidence logit and its own score thresholded at
   0.5, independent of ground truth — discouraging the score head and the
   evidence head from disagreeing with each other even when both happen
   to be wrong relative to the label.

All three are averaged with equal internal weight and scaled by the outer
`lambda_faith`. At evaluation time, the *rates* of the two classic failure
modes are reported directly via
`project/metrics.py::compute_faithfulness_rates` (high-score/empty-evidence
and low-score/non-empty-evidence), giving a concrete, monitorable proxy
for "is the model's JSON output trustworthy," independent of raw score
accuracy.

## 5. End-to-End Implementation Walkthrough

### 5.1 Data → char/token spans (unchanged)

`project/preprocessing.py::validate_and_enrich_sample` already computes,
per aspect, `char_span`/`token_span` **relative to the raw input text
alone** (tokenized without the chat template). This step is untouched by
the hybrid module.

### 5.2 Preprocess → full-sequence aspect features (new)

`project/preprocessing.py::build_aspect_sft_example` is the bridge: it
renders the *full* chat-formatted training sequence (system + user +
assistant JSON, exactly as the base pipeline does), tokenizes it with
character offsets, then:

1. Locates the raw user text inside that full string
   (`locate_user_text_offset`) and builds `user_token_mask`
   (`build_user_token_mask`).
2. Re-maps each aspect's raw-text `char_span` onto the *same* full-sequence
   offsets (`map_char_span_to_tokens`) to get `aspect_start`/`aspect_end`
   in the coordinate system of the actual training `input_ids`.
3. Explicitly detects truncation: if the evidence's character range is not
   fully covered by the (possibly truncated) offsets, the aspect is
   treated as **no-evidence** rather than emitting a corrupted partial
   span — this was verified empirically (see the test suite) to prevent a
   subtle silent-truncation bug where a shortened span would otherwise be
   accepted as valid.

`project/preprocessing.py::AspectAwareCollator` extends the existing
`CompletionOnlyCollator` (same dynamic padding + response-template label
masking) with padding for `user_token_mask` and stacking of the
fixed-length (`len(ASPECTS)`) `aspect_scores` / `aspect_has_evidence` /
`aspect_start` / `aspect_end` tensors.

### 5.3 Model → hybrid wrapper (new)

`project/hybrid_model.py::AspectGuidedModel` wraps the existing PEFT/QLoRA
Qwen model (built unchanged by `project/model.py::get_model_and_tokenizer`)
without modifying it: `forward()` runs the base model with
`output_hidden_states=True`, extracts the configured layer, and feeds it
through `project/aspect_attention.py::AspectHeads` (Phase 1 attention +
score head, Phase 2 span/has-evidence heads). `generate()` is a pure
delegate to the base model, so JSON generation is bit-for-bit identical to
the non-aspect path.

### 5.4 Loss → multi-objective Trainer (new)

`project/trainer.py::AspectGuidedTrainer` overrides `Trainer.compute_loss`
to pop the aspect labels out of the batch, run the model, and combine
everything via `project/losses.py::compute_total_loss`, logging each
component (`ce_loss`, `score_loss`, `span_loss`, `faith_loss` and its
three sub-terms) alongside the standard Trainer logs.
`project/trainer.py::build_trainer` branches on `config.aspect.enabled`:
when `true`, it builds the hybrid model + `AspectAwareCollator` +
`AspectGuidedTrainer`; when `false`, the original CE-only `Trainer` path
runs completely unchanged.

### 5.5 Train → checkpointing

Because `AspectGuidedModel` is a plain `nn.Module` (not a
`PreTrainedModel`/`PeftModel`), `project/trainer.py::train_model` saves the
final adapter by calling `model.base_model.save_pretrained(...)` directly
(producing the canonical `adapter_model.safetensors` +
`adapter_config.json` layout) plus
`project/hybrid_model.py::save_aspect_heads` for the auxiliary head
weights (`aspect.heads_checkpoint_name`, default `aspect_heads.pt`) —
rather than relying on generic `Trainer.save_model`, which would otherwise
fall back to dumping one raw combined state dict. **Note:** intermediate
per-step checkpoints (`outputs/checkpoints/checkpoint-N/`) still go through
generic `Trainer` checkpointing and will contain a full raw state dict
(base + LoRA + aspect heads); this is fine for in-run
`resume_from_checkpoint` but only the final `best/` directory should be
used for distribution or downstream inference.

### 5.6 Evaluate + Visualize

- `project/evaluation.py::evaluate_aspect_heads` runs the heads (via
  `project/hybrid_model.py::run_aspect_heads_on_text`) on every test
  sample and reports head-based score metrics
  (`compute_score_metrics_from_arrays`), span Token-F1/EM
  (`compute_head_span_metrics`), and faithfulness rates
  (`compute_faithfulness_rates`) — written to
  `outputs/reports/test_metrics_aspect.json` and appended as a Markdown
  section (`format_aspect_report_markdown`) by `run_evaluation`.
- `project/visualization.py::plot_aspect_attention_heatmap` renders the
  per-aspect attention bars with the gold evidence span overlaid;
  `plot_head_confusion_examples` surfaces head-vs-generative-JSON
  disagreements. Both are wired into
  `generate_all_visualizations(..., aspect_results=...)` and demoed in
  `notebooks/08_aspect_attention.ipynb`.
- `project/inference.py::predict` optionally repairs the generated JSON's
  evidence with the span head's decoded substring whenever the generated
  evidence is invalid, or unconditionally when
  `aspect.prefer_span_evidence: true`, and always attaches
  `_aspect_attention` diagnostics to the result dict.

## 6. Hyperparameter Guidance for Colab T4

The aspect heads add a negligible number of parameters relative to the 3B
backbone (a handful of `[D, D]`/`[D, 1]` matrices per aspect, all kept in
float32 regardless of the backbone's 4-bit/float16 precision for numerical
stability of the score/BCE losses), so **the base QLoRA hyperparameters in
`config/colab.yaml` do not need to change** to accommodate the hybrid
module on a 16GB T4. Practical guidance specific to `aspect.*`:

| Parameter | Guidance |
|---|---|
| `hidden_layer` | `-1` (last layer) is the default and works well since it is closest to the CE objective's own representation; try `-6`/mid-stack only if you observe attention collapsing to a few tokens uniformly across aspects. |
| `attn_dropout` | `0.1` is a safe default; increase toward `0.2` if attention overfits to a few frequent evidence phrases on a small dataset. |
| `score_hidden` | `256` is intentionally small (the score head is a 2-layer MLP on a pooled `[D]` vector, not a sequence model) — increasing it rarely helps given ~200 synthetic training samples. |
| `lambda_score` / `lambda_span` | Keep at `0.5` each; if the generative JSON loss plateaus much higher than the auxiliary losses (check the per-component logs), reduce both toward `0.2`–`0.3` to avoid the auxiliary heads dominating the shared backbone's gradient. |
| `lambda_faith` | Keep small (`0.1`); this is a regularizer, not a primary objective — values above `0.3` risk trading off score/span accuracy for consistency. |
| `low_score_threshold` / `high_score_threshold` | `0.25`/`0.5` mirror `LOW_SCORE_EVIDENCE_THRESHOLD` already used by the synthetic data generator (`project/dataset.py`), keeping the faithfulness regularizer's notion of "low"/"high" consistent with how the training data itself was labeled. |
| `prefer_span_evidence` | `false` during early experimentation (only *repairs* invalid generated evidence); switch to `true` once span-head Token F1 on the validation set clearly exceeds the generative JSON's own evidence-validity rate. |

Memory: `output_hidden_states=True` retains one extra full `[B, T, D]`
tensor briefly per forward pass; on a T4 with the existing
`per_device_train_batch_size: 2`, `max_seq_length: 2048` this is a small
addition relative to the 4-bit backbone + activations already in memory,
but if you hit an `OutOfMemoryError` after enabling `aspect.enabled`,
reduce `max_seq_length` first (as already suggested in `COLAB.md`'s
troubleshooting table) before disabling the aspect module entirely.

## 7. Expected Metrics and Ablation Checklist

Compare, in order, on the same held-out test split
(`outputs/reports/test_metrics.json` for the generative path,
`outputs/reports/test_metrics_aspect.json` for the head-based path):

1. **CE-only baseline** (`aspect.enabled: false`): the pre-existing
   generative pipeline's score MAE/RMSE and evidence Token F1/EM
   (`format_report_markdown`).
2. **+ score head only** (`lambda_span=0, lambda_faith=0`): expect
   head-based score MAE roughly comparable to or better than the parsed
   generative score's MAE, since the score head is trained with a direct
   regression loss rather than indirectly through next-token CE.
3. **+ span head** (`lambda_faith=0`): expect head-based span Token
   F1/EM to emerge (starts at ~0 for an untrained span head, since the
   generative baseline has no head-based span at all to compare against);
   track `high_score_empty_evidence_rate` / `low_score_nonempty_evidence_rate`
   — they may still be non-trivial without the faithfulness term.
4. **Full objective** (`+lambda_faith`): expect both faithfulness rates
   from step 3 to *decrease* measurably, ideally without a corresponding
   regression in score MAE or span F1 — this is the key ablation result
   that justifies including the regularizer.

Also inspect a handful of `plot_aspect_attention_heatmap` outputs
qualitatively: attention mass for `performance` should visibly
concentrate on performance-related clauses (e.g. "بازی‌های جدید را با
گرافیک بالا اجرا کند") rather than being diffuse across the whole input,
especially after step 4.

## 8. References

- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS. — the
  scaled dot-product attention mechanism `AspectConditionedAttention`
  specializes.
- Rajpurkar, P., Jia, R., & Liang, P. (2018). *Know What You Don't Know:
  Unanswerable Questions for SQuAD.* ACL (SQuAD 2.0). — the has-answer /
  null-span gate design directly informing `SpanHead.has_evidence_proj`
  and the `-100` ignore-index convention for absent evidence.
- Wang, Y., Huang, M., Zhu, X., & Zhao, L. (2016). *Attention-based LSTM
  for Aspect-level Sentiment Classification.* EMNLP (ATAE-LSTM). —
  early aspect-conditioned attention pooling for aspect-based sentiment
  analysis (ABSA), the direct conceptual ancestor of using a per-aspect
  query vector instead of per-token queries.
- Tay, Y., Luu, A. T., & Hui, S. C. (2018). *Learning to Attend via
  Word-Aspect Associative Fusion for Aspect-Based Sentiment Analysis.*
  AAAI. — further ABSA attention-fusion designs motivating the gated
  fusion identity used in `SpanHead`.
- Chen, C., Zhang, M., Liu, Y., & Ma, S. (2018). *Neural Attentional
  Rating Regression with Review-level Explanations (NARRE).* WWW. —
  explainable-recommendation architectures that jointly predict a rating
  and an attention-selected supporting review, the closest prior-art
  analogue to jointly predicting `aspect_scores` and evidence spans here.
- Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding.* NAACL. — the standard
  extractive-QA start/end token classification head design that
  `decode_best_span` follows (max-scoring valid `(start, end)` pair with a
  length constraint).
