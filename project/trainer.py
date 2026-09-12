"""HuggingFace Trainer setup with callbacks and resume support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from project.config import AppConfig, AspectConfig, is_colab
from project.dataset import load_split
from project.hf_compat import compat_training_kwargs
from project.model import get_model_and_tokenizer
from project.preprocessing import (
    AspectAwareCollator,
    CompletionOnlyCollator,
    build_aspect_sft_dataset,
    build_sft_dataset,
    save_enriched_split,
)
from project.prompts import get_response_template
from project.utils import (
    DeviceConfig,
    find_latest_checkpoint,
    get_device_config,
    set_seed,
    setup_colab_environment,
)

logger = logging.getLogger(__name__)


class AspectGuidedTrainer(Trainer):
    """``Trainer`` subclass combining CE + score + span + faithfulness losses.

    Wraps a :class:`project.hybrid_model.AspectGuidedModel`. Each training
    (and evaluation) step, ``compute_loss``:

    1. Pops the aspect supervision tensors (``aspect_scores``,
       ``aspect_has_evidence``, ``aspect_start``, ``aspect_end``) added by
       :class:`project.preprocessing.AspectAwareCollator` out of ``inputs``,
       since the model's ``forward`` only accepts base-LM + ``user_token_mask``
       arguments.
    2. Runs the model to get the raw (un-weighted) causal-LM loss and
       aspect-head outputs.
    3. Combines everything via :func:`project.losses.compute_total_loss`
       using the ``lambda_*`` weights from ``aspect_config``.

    The last computed loss breakdown is kept on
    ``self.last_loss_components`` for logging/inspection (e.g. from a
    notebook or a custom callback).
    """

    def __init__(self, *args: Any, aspect_config: AspectConfig, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.aspect_config = aspect_config
        self.last_loss_components: dict[str, float] = {}

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Combine CE + score + span + faithfulness losses for one batch."""
        from project.losses import compute_total_loss

        aspect_scores_gt = inputs.pop("aspect_scores")
        has_evidence_labels = inputs.pop("aspect_has_evidence")
        start_labels = inputs.pop("aspect_start")
        end_labels = inputs.pop("aspect_end")

        outputs = model(**inputs)

        total_loss, components = compute_total_loss(
            ce_loss=outputs.ce_loss,
            aspect_scores=outputs.aspect_scores,
            aspect_scores_gt=aspect_scores_gt,
            has_evidence_logits=outputs.aspect_has_evidence_logits,
            has_evidence_labels=has_evidence_labels,
            start_logits=outputs.aspect_start_logits,
            end_logits=outputs.aspect_end_logits,
            start_labels=start_labels,
            end_labels=end_labels,
            attn_weights=outputs.aspect_attn_weights,
            aspect_config=self.aspect_config,
        )
        self.last_loss_components = components
        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        """Append the last aspect loss breakdown to standard Trainer logs."""
        if self.last_loss_components:
            for key, value in self.last_loss_components.items():
                if key != "total_loss":
                    logs.setdefault(key, value)
        super().log(logs, *args, **kwargs)


def build_training_arguments(
    config: AppConfig,
    device_cfg: DeviceConfig,
    output_dir: Path | None = None,
) -> TrainingArguments:
    """Create HuggingFace TrainingArguments from config."""
    train_cfg = config.training
    out = output_dir or config.paths.checkpoint_dir

    use_fp16 = train_cfg.fp16 and device_cfg.device == "cuda"
    use_bf16 = train_cfg.bf16 and device_cfg.device == "cuda"

    kwargs = compat_training_kwargs(
        {
            "output_dir": str(out),
            "num_train_epochs": train_cfg.num_epochs,
            "per_device_train_batch_size": train_cfg.per_device_train_batch_size,
            "per_device_eval_batch_size": train_cfg.per_device_eval_batch_size,
            "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
            "learning_rate": train_cfg.learning_rate,
            "weight_decay": train_cfg.weight_decay,
            "warmup_ratio": train_cfg.warmup_ratio,
            "lr_scheduler_type": train_cfg.lr_scheduler_type,
            "max_grad_norm": train_cfg.max_grad_norm,
            "fp16": use_fp16,
            "bf16": use_bf16,
            "eval_strategy": "steps",
            "eval_steps": train_cfg.eval_steps,
            "save_strategy": "steps",
            "save_steps": train_cfg.save_steps,
            "logging_steps": train_cfg.logging_steps,
            "save_total_limit": train_cfg.save_total_limit,
            "load_best_model_at_end": train_cfg.load_best_model_at_end,
            "metric_for_best_model": train_cfg.metric_for_best_model,
            "greater_is_better": train_cfg.greater_is_better,
            "report_to": ["tensorboard"],
            "logging_dir": str(config.paths.tensorboard_dir),
            "gradient_checkpointing": train_cfg.gradient_checkpointing,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "remove_unused_columns": False,
            "dataloader_pin_memory": device_cfg.device == "cuda",
            "dataloader_num_workers": 0,
        },
        TrainingArguments,
    )
    return TrainingArguments(**kwargs)


def _prepare_enriched_splits(config: AppConfig, tokenizer: Any) -> dict[str, list[dict[str, Any]]]:
    """Load or create enriched train/val splits."""
    splits: dict[str, list[dict[str, Any]]] = {}
    for split_name in ("train", "val"):
        enriched_path = config.paths.processed_dir / f"{split_name}_enriched.json"
        if enriched_path.exists():
            from project.utils import load_json

            splits[split_name] = load_json(enriched_path)
        else:
            raw = load_split(config, split_name)
            splits[split_name] = save_enriched_split(raw, tokenizer, config, split_name)
    return splits


def build_tokenized_dataset(
    samples: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> Dataset:
    """Build a HuggingFace Dataset from enriched samples."""
    sft_records = build_sft_dataset(samples, tokenizer)
    ds = Dataset.from_list(sft_records)

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    tokenized = ds.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.map(
        lambda x: {"labels": x["input_ids"].copy()},
        batched=False,
    )
    return tokenized


def build_aspect_tokenized_dataset(
    samples: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> Dataset:
    """Build a HuggingFace Dataset with aspect-guided-attention features.

    Unlike :func:`build_tokenized_dataset` (plain CE-only path), each row
    already carries ``input_ids``/``attention_mask``/``labels`` *and* the
    aspect supervision fields (see
    :func:`project.preprocessing.build_aspect_sft_example`) -- tokenization
    and label alignment happen together per-sample so that evidence spans
    stay correctly mapped onto the exact same ``input_ids``.
    """
    records = build_aspect_sft_dataset(samples, tokenizer, max_seq_length)
    return Dataset.from_list(records)


def build_trainer(
    config: AppConfig,
    resume_from_checkpoint: str | Path | None = None,
) -> tuple[Trainer, Any, Any]:
    """Build a fully configured HuggingFace Trainer.

    When ``config.aspect.enabled`` is true, builds an
    :class:`project.hybrid_model.AspectGuidedModel` and an
    :class:`AspectGuidedTrainer` combining the generative CE loss with the
    auxiliary aspect score/span/faithfulness objectives. Otherwise, falls
    back to the original CE-only pipeline unchanged.
    """
    if is_colab():
        setup_colab_environment(
            hf_token_env=config.colab.hf_token_env,
            use_colab_secrets=config.colab.use_colab_secrets,
            mount_google_drive=config.colab.mount_google_drive,
        )

    set_seed(config.seed)
    device_cfg = get_device_config(require_cuda=True)

    training_args = build_training_arguments(config, device_cfg)
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.training.early_stopping_patience
        )
    ]

    if config.aspect.enabled:
        from project.hybrid_model import get_hybrid_model_and_tokenizer

        model, tokenizer = get_hybrid_model_and_tokenizer(config, device_cfg)
        splits = _prepare_enriched_splits(config, tokenizer)
        train_ds = build_aspect_tokenized_dataset(
            splits["train"], tokenizer, config.model.max_seq_length
        )
        val_ds = build_aspect_tokenized_dataset(
            splits["val"], tokenizer, config.model.max_seq_length
        )
        response_template = get_response_template(tokenizer)
        data_collator = AspectAwareCollator(tokenizer, response_template)

        trainer = AspectGuidedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            callbacks=callbacks,
            aspect_config=config.aspect,
        )
    else:
        model, tokenizer = get_model_and_tokenizer(config, device_cfg)
        splits = _prepare_enriched_splits(config, tokenizer)
        train_ds = build_tokenized_dataset(
            splits["train"], tokenizer, config.model.max_seq_length
        )
        val_ds = build_tokenized_dataset(
            splits["val"], tokenizer, config.model.max_seq_length
        )
        response_template = get_response_template(tokenizer)
        data_collator = CompletionOnlyCollator(tokenizer, response_template)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            callbacks=callbacks,
        )

    if resume_from_checkpoint is None:
        resume_from_checkpoint = find_latest_checkpoint(config.paths.checkpoint_dir)

    return trainer, model, tokenizer


def train_model(
    config: AppConfig,
    resume_from_checkpoint: str | Path | None = None,
) -> Trainer:
    """Run fine-tuning and save the best adapter (+ aspect heads, if enabled)."""
    trainer, model, _ = build_trainer(config, resume_from_checkpoint)
    ckpt = str(resume_from_checkpoint) if resume_from_checkpoint else None
    if ckpt is None:
        latest = find_latest_checkpoint(config.paths.checkpoint_dir)
        ckpt = str(latest) if latest else None

    train_result = trainer.train(resume_from_checkpoint=ckpt)
    logger.info("Training complete: %s", train_result.metrics)

    best_dir = config.paths.checkpoint_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    if config.aspect.enabled:
        # `model` is an AspectGuidedModel, not a PreTrainedModel/PeftModel,
        # so generic `trainer.save_model()` would fall back to dumping a
        # single raw `state_dict()` (base + LoRA + aspect heads combined,
        # including the frozen 4-bit base weights) instead of the canonical
        # `adapter_model.safetensors` + `adapter_config.json` LoRA layout.
        # Save the inner PEFT model and the aspect heads separately instead,
        # exactly mirroring the non-aspect path's adapter-only output.
        from project.hybrid_model import save_aspect_heads

        model.base_model.save_pretrained(str(best_dir))
        heads_path = best_dir / config.aspect.heads_checkpoint_name
        save_aspect_heads(model, heads_path)
        logger.info("Saved best adapter + aspect heads to %s", best_dir)
    else:
        trainer.save_model(str(best_dir))
        logger.info("Saved best adapter to %s", best_dir)

    # Package the checkpoint into a single portable .zip file so it can be
    # downloaded from Colab (or copied elsewhere) as one artifact instead of
    # a multi-file directory. See project/bundle.py and COLAB.md.
    from project.bundle import save_model_bundle

    try:
        bundle_path = save_model_bundle(config, best_dir)
        logger.info("Saved single-file model bundle to %s", bundle_path)
        if is_colab():
            from project.bundle import download_bundle_on_colab

            download_bundle_on_colab(bundle_path)
    except Exception:
        logger.exception(
            "Failed to create the single-file model bundle; the checkpoint "
            "directory at %s is still valid and usable directly.",
            best_dir,
        )

    return trainer
