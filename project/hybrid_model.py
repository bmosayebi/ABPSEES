"""Hybrid Aspect-Guided model: Qwen + LoRA plus auxiliary aspect heads.

:class:`AspectGuidedModel` wraps the existing PEFT/QLoRA-adapted Qwen causal
LM (built by :mod:`project.model`, unchanged) and attaches the
:class:`project.aspect_attention.AspectHeads` module on top of a chosen
decoder layer's hidden states. The generative path (``generate``, and the
causal-LM cross-entropy computed from ``labels``) is left completely
untouched -- the aspect heads are a pure *addition*, computed from
``output_hidden_states=True`` on the same forward pass. See
``docs/ASPECT_ATTENTION_GUIDE.md`` for the full design rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers.utils import ModelOutput

from project.aspect_attention import (
    AspectHeads,
    AspectHeadsOutput,
    decode_best_span,
)
from project.config import AppConfig, AspectConfig
from project.constants import ASPECTS
from project.utils import DeviceConfig, get_device_config

logger = logging.getLogger(__name__)


@dataclass
class AspectGuidedOutput(ModelOutput):
    """Structured output of :meth:`AspectGuidedModel.forward`.

    ``loss``/``ce_loss`` hold the *unweighted* causal-LM cross-entropy from
    the base model (identical to what plain fine-tuning would produce).
    Combining it with the auxiliary score/span/faithfulness losses is the
    responsibility of :class:`project.trainer.AspectGuidedTrainer`, which
    owns the ``lambda_*`` weights from :class:`AspectConfig` -- the model
    itself stays a pure feature/quantity computer.
    """

    loss: torch.FloatTensor | None = None
    ce_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    aspect_scores: torch.FloatTensor | None = None
    aspect_attn_weights: torch.FloatTensor | None = None
    aspect_has_evidence_logits: torch.FloatTensor | None = None
    aspect_start_logits: torch.FloatTensor | None = None
    aspect_end_logits: torch.FloatTensor | None = None


def get_hidden_size(model: Any) -> int:
    """Recursively resolve ``hidden_size`` from a (possibly PEFT-wrapped) model.

    Args:
        model: A HuggingFace/PEFT model exposing ``.config`` and/or
            ``.base_model``.

    Returns:
        The model's hidden size.

    Raises:
        ValueError: If ``hidden_size`` cannot be found.
    """
    cfg = getattr(model, "config", None)
    if cfg is not None and hasattr(cfg, "hidden_size"):
        return int(cfg.hidden_size)
    base = getattr(model, "base_model", None)
    if base is not None and base is not model:
        return get_hidden_size(base)
    raise ValueError("Could not determine hidden_size from model config.")


class AspectGuidedModel(torch.nn.Module):
    """Wraps a PEFT/QLoRA Qwen causal LM with auxiliary aspect heads.

    Args:
        base_model: The LoRA-adapted (or plain) causal LM, as returned by
            :func:`project.model.get_model_and_tokenizer` or
            :func:`project.model.load_model_for_inference`.
        aspect_config: :class:`project.config.AspectConfig` hyperparameters.
    """

    def __init__(self, base_model: Any, aspect_config: AspectConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.aspect_config = aspect_config
        d_model = get_hidden_size(base_model)
        self.aspect_heads = AspectHeads(
            d_model=d_model,
            num_aspects=len(ASPECTS),
            attn_dropout=aspect_config.attn_dropout,
            score_hidden=aspect_config.score_hidden,
            enable_span=True,
        )

    @property
    def config(self) -> Any:
        """Proxy the base model's config (mirrors PEFT/HF conventions)."""
        return self.base_model.config

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate generation to the base model (unchanged JSON pipeline)."""
        return self.base_model.generate(*args, **kwargs)

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        """Delegate gradient checkpointing setup to the base model."""
        self.base_model.gradient_checkpointing_enable(**kwargs)

    @property
    def is_gradient_checkpointing(self) -> bool:
        """Delegate the gradient-checkpointing flag to the base model."""
        return bool(getattr(self.base_model, "is_gradient_checkpointing", False))

    def enable_input_require_grads(self) -> None:
        """Delegate to the base model if available (needed for some PEFT + gradient-checkpointing setups)."""
        if hasattr(self.base_model, "enable_input_require_grads"):
            self.base_model.enable_input_require_grads()

    def get_input_embeddings(self) -> Any:
        """Delegate to the base model (used by some Trainer/PEFT utilities)."""
        return self.base_model.get_input_embeddings()

    def print_trainable_parameters(self) -> None:
        """Delegate to the base (PEFT) model if available."""
        if hasattr(self.base_model, "print_trainable_parameters"):
            self.base_model.print_trainable_parameters()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        user_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> AspectGuidedOutput:
        """Run the base causal LM and the auxiliary aspect heads.

        Args:
            input_ids: ``[B, T]`` token ids.
            attention_mask: ``[B, T]`` padding mask (1 = real token).
            labels: ``[B, T]`` causal-LM labels (``-100`` = ignored), or
                ``None`` at pure-inference time.
            user_token_mask: ``[B, T]`` mask restricting aspect attention to
                the raw user-text tokens. Falls back to ``attention_mask``
                (i.e. attend over all non-padding tokens) with a one-time
                warning if not provided, which is only intended for ad-hoc
                / exploratory use -- trained checkpoints expect the
                properly restricted mask produced by
                :mod:`project.preprocessing`.
            **kwargs: Forwarded to the base model (e.g. extra HF kwargs).

        Returns:
            :class:`AspectGuidedOutput` with the base LM's loss/logits and
            the raw (unweighted) aspect head outputs.
        """
        if user_token_mask is None:
            if attention_mask is not None:
                logger.warning(
                    "user_token_mask not provided; falling back to attention_mask "
                    "(aspect attention will span the whole sequence, including the "
                    "prompt and assistant completion)."
                )
            user_token_mask = attention_mask

        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.hidden_states[self.aspect_config.hidden_layer].float()
        aspect_out: AspectHeadsOutput = self.aspect_heads(hidden_states, user_token_mask)

        return AspectGuidedOutput(
            loss=outputs.loss,
            ce_loss=outputs.loss,
            logits=outputs.logits,
            aspect_scores=aspect_out.scores,
            aspect_attn_weights=aspect_out.attn_weights,
            aspect_has_evidence_logits=aspect_out.has_evidence_logits,
            aspect_start_logits=aspect_out.start_logits,
            aspect_end_logits=aspect_out.end_logits,
        )


def get_hybrid_model_and_tokenizer(
    config: AppConfig,
    device_cfg: DeviceConfig | None = None,
) -> tuple[AspectGuidedModel, Any]:
    """Build a training-ready :class:`AspectGuidedModel` + tokenizer.

    Args:
        config: Application configuration (``config.aspect`` must be set).
        device_cfg: Optional resolved device config; auto-detected (CUDA
            required) otherwise.

    Returns:
        ``(hybrid_model, tokenizer)``.
    """
    from project.model import get_model_and_tokenizer

    device_cfg = device_cfg or get_device_config(require_cuda=True)
    base_model, tokenizer = get_model_and_tokenizer(config, device_cfg)
    hybrid = AspectGuidedModel(base_model, config.aspect)
    hybrid.aspect_heads.to(device=next(base_model.parameters()).device, dtype=torch.float32)
    return hybrid, tokenizer


def save_aspect_heads(model: AspectGuidedModel, path: str | Path) -> Path:
    """Persist the auxiliary aspect head weights to disk.

    LoRA adapter weights are saved separately (via ``trainer.save_model``);
    this saves only the additional ``AspectHeads`` parameters, which are not
    part of the PEFT adapter.

    Args:
        model: Trained :class:`AspectGuidedModel`.
        path: Destination file path (``.pt``).

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.aspect_heads.state_dict(), path)
    logger.info("Saved aspect heads to %s", path)
    return path


def load_aspect_heads(model: AspectGuidedModel, path: str | Path) -> None:
    """Load previously saved aspect head weights in-place.

    Args:
        model: :class:`AspectGuidedModel` whose ``aspect_heads`` are updated.
        path: Path to a checkpoint written by :func:`save_aspect_heads`.
    """
    device = next(model.base_model.parameters()).device
    state = torch.load(Path(path), map_location=device)
    model.aspect_heads.load_state_dict(state)
    logger.info("Loaded aspect heads from %s", path)


def load_hybrid_model_for_inference(
    config: AppConfig,
    adapter_path: str | Path | None = None,
    device_cfg: DeviceConfig | None = None,
) -> tuple[AspectGuidedModel, Any]:
    """Load base model + LoRA adapter + aspect heads for inference.

    Args:
        config: Application configuration.
        adapter_path: Directory containing the LoRA adapter and (optionally)
            the aspect heads checkpoint (``config.aspect.heads_checkpoint_name``).
            Defaults to ``checkpoints/best``.
        device_cfg: Optional resolved device config.

    Returns:
        ``(hybrid_model, tokenizer)`` ready for ``.generate()`` and
        aspect-head inference.
    """
    from project.model import load_model_for_inference

    device_cfg = device_cfg or get_device_config(require_cuda=False)
    adapter_path = Path(adapter_path or config.paths.checkpoint_dir / "best")
    base_model, tokenizer = load_model_for_inference(config, str(adapter_path), device_cfg)

    hybrid = AspectGuidedModel(base_model, config.aspect)
    hybrid.aspect_heads.to(device=next(base_model.parameters()).device, dtype=torch.float32)

    heads_path = adapter_path / config.aspect.heads_checkpoint_name
    if heads_path.exists():
        load_aspect_heads(hybrid, heads_path)
    else:
        logger.warning(
            "No aspect heads checkpoint found at %s; using randomly initialized heads.",
            heads_path,
        )
    hybrid.eval()
    return hybrid, tokenizer


def build_aspect_inference_inputs(
    tokenizer: Any,
    text: str,
    max_seq_length: int,
) -> dict[str, Any]:
    """Tokenize a raw input for aspect-head inference (pre-generation).

    Builds the same generation prompt used by :func:`project.evaluation.generate_prediction`,
    tokenizes it with offsets, and restricts the ``user_token_mask`` to the
    raw ``text`` span, exactly like the training-time feature builder in
    :mod:`project.preprocessing`.

    Args:
        tokenizer: HuggingFace tokenizer.
        text: Raw Persian input text.
        max_seq_length: Maximum sequence length (truncation).

    Returns:
        Dict with ``input_ids``, ``attention_mask``, ``user_token_mask``
        (each ``[1, T]`` tensors), plus the plain-Python ``prompt`` string
        and ``offsets`` list for span-to-text decoding.
    """
    from project.preprocessing import build_user_token_mask, locate_user_text_offset
    from project.prompts import format_prompt

    prompt = format_prompt(tokenizer, text, labels=None)
    encoding = tokenizer(
        prompt,
        truncation=True,
        max_length=max_seq_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = encoding.pop("offset_mapping")[0].tolist()

    try:
        char_start = locate_user_text_offset(prompt, text)
    except ValueError:
        logger.warning("Could not locate user text in prompt; using full-sequence attention.")
        char_start = 0
        char_end = len(prompt)
    else:
        char_end = char_start + len(text)

    mask = build_user_token_mask(offsets, char_start, char_end)
    encoding["user_token_mask"] = torch.tensor([mask], dtype=torch.long)
    encoding["prompt"] = prompt
    encoding["offsets"] = offsets
    return encoding


def run_aspect_heads_on_text(
    hybrid_model: AspectGuidedModel,
    tokenizer: Any,
    text: str,
    config: AppConfig,
    has_evidence_threshold: float = 0.5,
    max_answer_len: int = 64,
) -> dict[str, Any]:
    """Run the auxiliary aspect heads on one raw input text.

    Args:
        hybrid_model: Loaded :class:`AspectGuidedModel`.
        tokenizer: HuggingFace tokenizer.
        text: Raw Persian input text.
        config: Application configuration.
        has_evidence_threshold: Minimum has-evidence probability to decode a
            span instead of reporting empty evidence.
        max_answer_len: Maximum decoded span length in tokens.

    Returns:
        Dict with per-aspect ``score_head``, ``has_evidence_prob``, and
        ``span_evidence`` (decoded substring of ``text``, or ``""``), plus
        ``tokens`` and ``attn_weights`` (``[num_aspects, T]``) for
        visualization.
    """
    device = next(hybrid_model.base_model.parameters()).device
    inputs = build_aspect_inference_inputs(tokenizer, text, config.model.max_seq_length)
    prompt = inputs.pop("prompt")
    offsets = inputs.pop("offsets")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    user_token_mask = inputs["user_token_mask"].to(device)

    hybrid_model.eval()
    with torch.no_grad():
        base_outputs = hybrid_model.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = base_outputs.hidden_states[hybrid_model.aspect_config.hidden_layer].float()
        aspect_out = hybrid_model.aspect_heads(hidden_states, user_token_mask)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    per_aspect: dict[str, Any] = {}
    for i, aspect in enumerate(ASPECTS):
        score = float(aspect_out.scores[0, i].item())
        has_ev_prob = float(torch.sigmoid(aspect_out.has_evidence_logits[0, i]).item())
        span_evidence = ""
        span_token: tuple[int, int] | None = None
        if has_ev_prob >= has_evidence_threshold:
            start, end, _ = decode_best_span(
                aspect_out.start_logits[0, i], aspect_out.end_logits[0, i], max_answer_len
            )
            char_start, char_end = offsets[start][0], offsets[end][1]
            span_evidence = prompt[char_start:char_end]
            span_token = (start, end)
        per_aspect[aspect] = {
            "score_head": score,
            "has_evidence_prob": has_ev_prob,
            "span_evidence": span_evidence,
            "span_token": span_token,
        }

    return {
        "per_aspect": per_aspect,
        "tokens": tokens,
        "attn_weights": aspect_out.attn_weights[0].tolist(),
        "user_token_mask": user_token_mask[0].tolist(),
        "prompt": prompt,
        "offsets": offsets,
    }
