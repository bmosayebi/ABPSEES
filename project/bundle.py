"""Package a trained checkpoint into a single portable bundle file.

Training (`project/trainer.py::train_model`) already saves a LoRA adapter
(`adapter_model.safetensors` + `adapter_config.json`) and, when
``aspect.enabled``, the auxiliary aspect head weights (`aspect_heads.pt`)
into ``outputs/checkpoints/best/`` -- a *directory* with a handful of
files. This module adds a thin packaging layer on top: a single ``.zip``
archive (`save_model_bundle`) containing that directory plus a small
``bundle_metadata.json`` describing exactly which base model / aspect
hyperparameters produced it, and a loader (`load_model_bundle`) that can
reconstruct a working inference-ready model + tokenizer from that single
file alone -- on Colab, or on any other machine (e.g. a local CPU-only
laptop) that has the `project` package installed and Hugging Face Hub
access to download the (unchanged) base Qwen weights.

The bundle intentionally does **not** include the multi-gigabyte base
model weights -- only the small LoRA delta + aspect heads + metadata are
bundled, exactly mirroring how LoRA adapters are normally distributed. The
base model is re-downloaded from the Hugging Face Hub by name (as it
already is for every training/inference run in this project), then the
bundled adapter and aspect heads are applied on top.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from project.config import (
    AppConfig,
    AspectConfig,
    ColabConfig,
    DataConfig,
    InferenceConfig,
    LoRAConfig,
    ModelConfig,
    PathsConfig,
    QuantizationConfig,
    TrainingConfig,
)

logger = logging.getLogger(__name__)

BUNDLE_METADATA_FILENAME = "bundle_metadata.json"
BUNDLE_FORMAT_VERSION = 1
DEFAULT_BUNDLE_FILENAME = "model_bundle.zip"


def _metadata_from_config(config: AppConfig) -> dict[str, Any]:
    """Extract the inference-relevant subset of ``AppConfig`` as plain JSON.

    Only the fields actually needed to *load* the model for inference are
    kept (base model name/settings, quantization, generation defaults,
    aspect hyperparameters, and the unicode-normalization flag used by
    evidence validation) -- training-only fields (LoRA rank, epochs, data
    split ratios, Colab paths, ...) are deliberately omitted since they are
    meaningless once a checkpoint already exists.

    Args:
        config: Application configuration used to produce the checkpoint.

    Returns:
        JSON-serializable metadata dict.
    """
    return {
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "project_name": config.project_name,
        "seed": config.seed,
        "model": asdict(config.model),
        "quantization": asdict(config.quantization),
        "inference": asdict(config.inference),
        "aspect": asdict(config.aspect),
        "data": asdict(config.data),
    }


def _minimal_config_from_metadata(metadata: dict[str, Any], root: Path) -> AppConfig:
    """Reconstruct a minimal, inference-ready ``AppConfig`` from bundle metadata.

    Training-only sections (``lora``, ``training``, ``colab``) are filled
    with inert placeholder values -- they are never read by
    ``project.model.load_model_for_inference`` /
    ``project.hybrid_model.load_hybrid_model_for_inference``, which only
    need ``model``, ``quantization``, ``inference``, ``aspect``, and
    ``data.normalize_unicode``.

    Args:
        metadata: Output of :func:`_metadata_from_config` (round-tripped
            through JSON).
        root: A local directory to use as the reconstructed config's
            project root (e.g. the bundle's extraction directory).

    Returns:
        A usable :class:`AppConfig` for inference-only code paths.
    """
    paths = PathsConfig(
        data_dir=root / "data",
        processed_dir=root / "data" / "processed",
        output_dir=root / "outputs",
        checkpoint_dir=root / "outputs" / "checkpoints",
        tensorboard_dir=root / "outputs" / "tensorboard",
        reports_dir=root / "outputs" / "reports",
    )
    lora = LoRAConfig(r=8, lora_alpha=16, dropout=0.0, bias="none", target_modules=[])
    training = TrainingConfig(
        num_epochs=0,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=0.0,
        weight_decay=0.0,
        warmup_ratio=0.0,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        fp16=False,
        bf16=False,
        eval_steps=1,
        save_steps=1,
        logging_steps=1,
        save_total_limit=1,
        early_stopping_patience=1,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    colab = ColabConfig(
        enabled=False,
        project_dir=str(root),
        mount_google_drive=False,
        drive_project_path=str(root),
        hf_token_env="HF_TOKEN",
        use_colab_secrets=False,
    )

    return AppConfig(
        project_name=metadata.get("project_name", "abpsees"),
        seed=metadata.get("seed", 42),
        paths=paths,
        model=ModelConfig(**metadata["model"]),
        lora=lora,
        quantization=QuantizationConfig(**metadata["quantization"]),
        training=training,
        data=DataConfig(**metadata["data"]),
        inference=InferenceConfig(**metadata["inference"]),
        colab=colab,
        aspect=AspectConfig(**metadata["aspect"]),
        project_root=root,
    )


def save_model_bundle(
    config: AppConfig,
    adapter_dir: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Package a trained checkpoint directory into a single ``.zip`` bundle.

    Args:
        config: Application configuration used to produce ``adapter_dir``
            (its inference-relevant fields are embedded as metadata).
        adapter_dir: Directory containing the LoRA adapter (and, when
            ``config.aspect.enabled``, ``aspect_heads.pt``) as produced by
            :func:`project.trainer.train_model` (typically
            ``outputs/checkpoints/best/``).
        output_path: Destination ``.zip`` path. Defaults to
            ``config.paths.output_dir / "model_bundle.zip"``.

    Returns:
        Path to the written ``.zip`` bundle.

    Raises:
        FileNotFoundError: If ``adapter_dir`` does not exist.
    """
    adapter_dir = Path(adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    if output_path is None:
        output_path = config.paths.output_dir / DEFAULT_BUNDLE_FILENAME
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # shutil.make_archive appends ".zip" itself; strip it if present so we
    # don't end up with "model_bundle.zip.zip".
    base_name = str(output_path)[: -len(".zip")] if str(output_path).endswith(".zip") else str(output_path)

    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = Path(tmp) / "bundle"
        shutil.copytree(adapter_dir, stage_dir)

        metadata = _metadata_from_config(config)
        (stage_dir / BUNDLE_METADATA_FILENAME).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        archive_path = shutil.make_archive(base_name, "zip", root_dir=stage_dir)

    logger.info("Saved model bundle to %s", archive_path)
    return Path(archive_path)


def load_model_bundle(
    bundle_path: str | Path,
    extract_dir: str | Path | None = None,
    require_cuda: bool = False,
) -> tuple[Any, Any, AppConfig]:
    """Extract a ``.zip`` bundle and load a ready-to-use model + tokenizer.

    Works identically on Colab or on any other machine (e.g. a local
    CPU-only laptop): the base Qwen weights are (re-)downloaded from the
    Hugging Face Hub by name (set ``HF_TOKEN``/``HUGGING_FACE_HUB_TOKEN``
    beforehand and accept the model's license on huggingface.co), then the
    bundled LoRA adapter (and aspect heads, if present) are applied on top.

    Args:
        bundle_path: Path to a ``.zip`` file produced by
            :func:`save_model_bundle`.
        extract_dir: Directory to extract into. Defaults to a fresh
            temporary directory.
        require_cuda: Passed through to device resolution; ``False`` (the
            default) allows CPU-only inference.

    Returns:
        ``(model, tokenizer, config)`` -- ``model`` is an
        ``AspectGuidedModel`` when the bundle was trained with
        ``aspect.enabled: true``, otherwise a plain PEFT model. ``config``
        is the minimal reconstructed :class:`AppConfig`, suitable for
        passing straight into ``project.inference.predict``.

    Raises:
        FileNotFoundError: If ``bundle_path`` does not exist or does not
            contain ``bundle_metadata.json`` (i.e. is not a valid ABPSEES
            bundle).
    """
    from project.utils import get_device_config

    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle file not found: {bundle_path}")

    if extract_dir is None:
        extract_dir = Path(tempfile.mkdtemp(prefix="abpsees_bundle_"))
    else:
        extract_dir = Path(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bundle_path) as zf:
        zf.extractall(extract_dir)

    metadata_path = extract_dir / BUNDLE_METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{BUNDLE_METADATA_FILENAME} not found after extracting {bundle_path}; "
            "is this a valid ABPSEES model bundle (created by save_model_bundle)?"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    config = _minimal_config_from_metadata(metadata, root=extract_dir)
    device_cfg = get_device_config(require_cuda=require_cuda)

    if config.aspect.enabled:
        from project.hybrid_model import load_hybrid_model_for_inference

        model, tokenizer = load_hybrid_model_for_inference(config, str(extract_dir), device_cfg)
    else:
        from project.model import load_model_for_inference

        model, tokenizer = load_model_for_inference(config, str(extract_dir), device_cfg)

    logger.info("Loaded model bundle from %s (extracted to %s)", bundle_path, extract_dir)
    return model, tokenizer, config


def download_bundle_on_colab(bundle_path: str | Path) -> bool:
    """Trigger a browser download of the bundle when running inside Colab.

    No-op (returns ``False``) outside Colab, so it is safe to call
    unconditionally after :func:`save_model_bundle`.

    Args:
        bundle_path: Path to the ``.zip`` written by :func:`save_model_bundle`.

    Returns:
        ``True`` if Colab's ``files.download`` was invoked, ``False`` otherwise.
    """
    from project.config import is_colab

    bundle_path = Path(bundle_path)
    if not is_colab():
        return False
    if not bundle_path.exists():
        logger.warning("Bundle file not found for Colab download: %s", bundle_path)
        return False

    try:
        from google.colab import files
    except ImportError:
        logger.warning("google.colab.files is unavailable; skip browser download.")
        return False

    logger.info("Starting Colab browser download of %s", bundle_path)
    files.download(str(bundle_path))
    return True
