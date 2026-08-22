"""HuggingFace Trainer setup with callbacks and resume support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments

from project.config import AppConfig, is_colab
from project.dataset import load_split
from project.model import get_model_and_tokenizer
from project.preprocessing import (
    CompletionOnlyCollator,
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

    return TrainingArguments(
        output_dir=str(out),
        num_train_epochs=train_cfg.num_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        max_grad_norm=train_cfg.max_grad_norm,
        fp16=use_fp16,
        bf16=use_bf16,
        eval_strategy="steps",
        eval_steps=train_cfg.eval_steps,
        save_strategy="steps",
        save_steps=train_cfg.save_steps,
        logging_steps=train_cfg.logging_steps,
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=train_cfg.greater_is_better,
        report_to=["tensorboard"],
        logging_dir=str(config.paths.tensorboard_dir),
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
    )


def build_trainer(
    config: AppConfig,
    resume_from_checkpoint: str | Path | None = None,
) -> tuple[Trainer, Any, Any]:
    """Build a fully configured HuggingFace Trainer."""
    if is_colab():
        setup_colab_environment(
            hf_token_env=config.colab.hf_token_env,
            use_colab_secrets=config.colab.use_colab_secrets,
            mount_google_drive=config.colab.mount_google_drive,
        )

    set_seed(config.seed)
    device_cfg = get_device_config(require_cuda=True)
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

    training_args = build_training_arguments(config, device_cfg)
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.training.early_stopping_patience
        )
    ]

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
    """Run fine-tuning and save the best adapter."""
    trainer, _, _ = build_trainer(config, resume_from_checkpoint)
    ckpt = str(resume_from_checkpoint) if resume_from_checkpoint else None
    if ckpt is None:
        latest = find_latest_checkpoint(config.paths.checkpoint_dir)
        ckpt = str(latest) if latest else None

    train_result = trainer.train(resume_from_checkpoint=ckpt)
    logger.info("Training complete: %s", train_result.metrics)

    best_dir = config.paths.checkpoint_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    logger.info("Saved best adapter to %s", best_dir)
    return trainer
