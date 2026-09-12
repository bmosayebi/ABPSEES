"""Configuration loading and validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from project.constants import ASPECTS


def is_colab() -> bool:
    """Return True when running inside a Google Colab notebook."""
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def get_project_root() -> Path:
    """Return the repository root directory (local or Colab layout)."""
    env_root = os.environ.get("ABPSEES_ROOT")
    if env_root:
        return Path(env_root)

    file_root = Path(__file__).resolve().parent.parent

    if is_colab():
        colab_candidates = [
            Path("/content/ABPSEES"),
            Path.cwd(),
            file_root,
        ]
        for candidate in colab_candidates:
            if (candidate / "project" / "config.py").exists():
                return candidate
        return Path("/content/ABPSEES")

    return file_root


@dataclass
class PathsConfig:
    """Filesystem paths (relative to project root unless absolute)."""

    data_dir: Path
    processed_dir: Path
    output_dir: Path
    checkpoint_dir: Path
    tensorboard_dir: Path
    reports_dir: Path


@dataclass
class ModelConfig:
    """Base model settings."""

    name: str
    max_seq_length: int
    trust_remote_code: bool


@dataclass
class LoRAConfig:
    """LoRA adapter hyperparameters."""

    r: int
    lora_alpha: int
    dropout: float
    bias: str
    target_modules: list[str]


@dataclass
class QuantizationConfig:
    """BitsAndBytes 4-bit settings (CUDA / QLoRA)."""

    load_in_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    use_double_quant: bool


@dataclass
class TrainingConfig:
    """HuggingFace Trainer hyperparameters."""

    num_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    max_grad_norm: float
    gradient_checkpointing: bool
    fp16: bool
    bf16: bool
    eval_steps: int
    save_steps: int
    logging_steps: int
    save_total_limit: int
    early_stopping_patience: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool


@dataclass
class DataConfig:
    """Dataset loading and splitting settings."""

    file_pattern: str
    train_ratio: float
    val_ratio: float
    test_ratio: float
    evidence_on_invalid: str
    normalize_unicode: bool
    synthetic_num_samples: int


@dataclass
class InferenceConfig:
    """Generation settings for inference."""

    max_new_tokens: int
    temperature: float
    do_sample: bool
    faithfulness_warn: bool


@dataclass
class ColabConfig:
    """Google Colab runtime settings."""

    enabled: bool
    project_dir: str
    mount_google_drive: bool
    drive_project_path: str
    hf_token_env: str
    use_colab_secrets: bool


@dataclass
class AspectConfig:
    """Aspect-guided attention (Phase 1 + Phase 2) hyperparameters.

    Controls the auxiliary hybrid module that attaches aspect-conditioned
    attention, score-regression heads, and span-extraction heads on top of
    the Qwen hidden states, in addition to the primary generative JSON path.
    """

    enabled: bool
    hidden_layer: int
    attn_dropout: float
    score_hidden: int
    lambda_ce: float
    lambda_score: float
    lambda_span: float
    lambda_faith: float
    low_score_threshold: float
    high_score_threshold: float
    prefer_span_evidence: bool
    heads_checkpoint_name: str


@dataclass
class AppConfig:
    """Top-level application configuration."""

    project_name: str
    seed: int
    paths: PathsConfig
    model: ModelConfig
    lora: LoRAConfig
    quantization: QuantizationConfig
    training: TrainingConfig
    data: DataConfig
    inference: InferenceConfig
    colab: ColabConfig
    aspect: AspectConfig
    project_root: Path = field(default_factory=get_project_root)

    def resolve_path(self, path: Path) -> Path:
        """Resolve a path relative to the project root."""
        if path.is_absolute():
            return path
        return self.project_root / path


def _resolve_paths(raw: dict[str, Any], root: Path) -> PathsConfig:
    """Build PathsConfig with resolved absolute paths."""
    paths = {k: root / v if not Path(v).is_absolute() else Path(v) for k, v in raw.items()}
    return PathsConfig(**paths)


def _default_config_path(root: Path) -> Path:
    """Pick the default config file for the current runtime."""
    colab_cfg = root / "config" / "colab.yaml"
    if is_colab() and colab_cfg.exists():
        return colab_cfg
    return root / "config" / "default.yaml"


def _parse_colab_config(raw: dict[str, Any] | None) -> ColabConfig:
    """Build ColabConfig with sensible defaults."""
    defaults = {
        "enabled": is_colab(),
        "project_dir": "/content/ABPSEES",
        "mount_google_drive": False,
        "drive_project_path": "/content/drive/MyDrive/ABPSEES",
        "hf_token_env": "HF_TOKEN",
        "use_colab_secrets": True,
    }
    merged = {**defaults, **(raw or {})}
    return ColabConfig(
        enabled=bool(merged["enabled"]),
        project_dir=str(merged["project_dir"]),
        mount_google_drive=bool(merged["mount_google_drive"]),
        drive_project_path=str(merged["drive_project_path"]),
        hf_token_env=str(merged["hf_token_env"]),
        use_colab_secrets=bool(merged["use_colab_secrets"]),
    )


def _default_aspect_config() -> dict[str, Any]:
    """Default values for the aspect-guided attention module."""
    return {
        "enabled": False,
        "hidden_layer": -1,
        "attn_dropout": 0.1,
        "score_hidden": 256,
        "lambda_ce": 1.0,
        "lambda_score": 0.5,
        "lambda_span": 0.5,
        "lambda_faith": 0.1,
        "low_score_threshold": 0.25,
        "high_score_threshold": 0.5,
        "prefer_span_evidence": False,
        "heads_checkpoint_name": "aspect_heads.pt",
    }


def _parse_aspect_config(raw: dict[str, Any] | None) -> AspectConfig:
    """Build AspectConfig with sensible defaults, validating thresholds."""
    merged = {**_default_aspect_config(), **(raw or {})}

    low = float(merged["low_score_threshold"])
    high = float(merged["high_score_threshold"])
    if not 0.0 <= low < high <= 1.0:
        raise ValueError(
            "aspect.low_score_threshold must be < aspect.high_score_threshold, "
            f"both in [0, 1], got low={low}, high={high}"
        )

    return AspectConfig(
        enabled=bool(merged["enabled"]),
        hidden_layer=int(merged["hidden_layer"]),
        attn_dropout=float(merged["attn_dropout"]),
        score_hidden=int(merged["score_hidden"]),
        lambda_ce=float(merged["lambda_ce"]),
        lambda_score=float(merged["lambda_score"]),
        lambda_span=float(merged["lambda_span"]),
        lambda_faith=float(merged["lambda_faith"]),
        low_score_threshold=low,
        high_score_threshold=high,
        prefer_span_evidence=bool(merged["prefer_span_evidence"]),
        heads_checkpoint_name=str(merged["heads_checkpoint_name"]),
    )


def resolve_colab_data_dir(config: AppConfig) -> Path:
    """Resolve dataset directory when Google Drive is mounted on Colab.

    Args:
        config: Application configuration.

    Returns:
        Path to the data directory.
    """
    if not is_colab() or not config.colab.mount_google_drive:
        return config.paths.data_dir

    drive_root = Path(config.colab.drive_project_path)
    candidates = [
        drive_root / "data" / "synthetic",
        drive_root / "data" / "raw",
        drive_root / "data",
        config.paths.data_dir,
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.glob("*.json")):
            return candidate
    return config.paths.data_dir


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Args:
        config_path: Path to YAML config. Auto-selects ``colab.yaml`` on Colab.

    Returns:
        Validated :class:`AppConfig` instance.
    """
    root = get_project_root()
    if config_path is None:
        config_path = _default_config_path(root)
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = root / config_path

    with config_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    data_cfg = raw["data"]
    ratios = data_cfg["train_ratio"] + data_cfg["val_ratio"] + data_cfg["test_ratio"]
    if abs(ratios - 1.0) > 1e-6:
        raise ValueError(f"Data split ratios must sum to 1.0, got {ratios}")

    if data_cfg["evidence_on_invalid"] not in {"discard", "raise"}:
        raise ValueError(
            "data.evidence_on_invalid must be 'discard' or 'raise', "
            f"got {data_cfg['evidence_on_invalid']!r}"
        )

    paths = _resolve_paths(raw["paths"], root)
    for p in (
        paths.processed_dir,
        paths.output_dir,
        paths.checkpoint_dir,
        paths.tensorboard_dir,
        paths.reports_dir,
    ):
        p.mkdir(parents=True, exist_ok=True)

    config = AppConfig(
        project_name=raw["project"]["name"],
        seed=raw["project"]["seed"],
        paths=paths,
        model=ModelConfig(**raw["model"]),
        lora=LoRAConfig(**raw["lora"]),
        quantization=QuantizationConfig(**raw["quantization"]),
        training=TrainingConfig(**raw["training"]),
        data=DataConfig(**data_cfg),
        inference=InferenceConfig(**raw["inference"]),
        colab=_parse_colab_config(raw.get("colab")),
        aspect=_parse_aspect_config(raw.get("aspect")),
        project_root=root,
    )

    if is_colab() and config.colab.mount_google_drive:
        config.paths.data_dir = resolve_colab_data_dir(config)

    return config


def validate_sample_scores(labels: dict[str, Any]) -> None:
    """Validate that all aspects are present and scores are in [0, 1]."""
    for aspect in ASPECTS:
        if aspect not in labels:
            raise ValueError(f"Missing aspect '{aspect}' in labels")
        label = labels[aspect]
        if "score" not in label or "evidence" not in label:
            raise ValueError(f"Aspect '{aspect}' must have 'score' and 'evidence'")
        score = label["score"]
        if not isinstance(score, (int, float)) or not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"Score for '{aspect}' must be in [0, 1], got {score}")
