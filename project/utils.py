"""Shared utilities: seeding, device detection, logging, I/O."""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


@dataclass
class DeviceConfig:
    """Resolved compute device and training dtype settings."""

    device: str
    use_qlora: bool
    torch_dtype: "torch.dtype"
    attn_implementation: str


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a standard format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_colab_environment(
    hf_token_env: str = "HF_TOKEN",
    use_colab_secrets: bool = True,
    mount_google_drive: bool = False,
) -> None:
    """Configure environment for Google Colab GPU training.

    Args:
        hf_token_env: Secret / environment variable name for HuggingFace token.
        use_colab_secrets: Load token from Colab ``userdata`` secrets.
        mount_google_drive: Mount Google Drive at ``/content/drive``.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    token: str | None = os.environ.get(hf_token_env)

    if use_colab_secrets and token is None:
        try:
            from google.colab import userdata

            token = userdata.get(hf_token_env)
            logger.info("HuggingFace token loaded from Colab Secrets (%s)", hf_token_env)
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("Could not read Colab secret %s: %s", hf_token_env, exc)

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    else:
        logger.warning(
            "No HuggingFace token found. Add secret '%s' in Colab: "
            "🔑 Secrets -> Add secret",
            hf_token_env,
        )

    if mount_google_drive:
        try:
            from google.colab import drive

            drive.mount("/content/drive", force_remount=False)
            logger.info("Google Drive mounted at /content/drive")
        except ImportError:
            logger.warning("google.colab.drive not available; skipping Drive mount.")
        except Exception as exc:
            logger.warning("Google Drive mount failed: %s", exc)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_config(require_cuda: bool = False) -> DeviceConfig:
    """Resolve CUDA QLoRA settings for Colab GPU training.

    Args:
        require_cuda: If True, raise when CUDA is unavailable.

    Raises:
        RuntimeError: If ``require_cuda`` is True and no GPU is available.
    """
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Using CUDA device: %s", gpu_name)
        return DeviceConfig(
            device="cuda",
            use_qlora=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )

    if require_cuda:
        raise RuntimeError(
            "CUDA GPU is required for this pipeline. "
            "In Colab: Runtime -> Change runtime type -> T4 GPU."
        )

    logger.warning("CUDA not available; falling back to CPU (not recommended for training).")
    return DeviceConfig(
        device="cpu",
        use_qlora=False,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )


def empty_cuda_cache() -> None:
    """Release unused CUDA memory."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_json(data: Any, path: str | Path, indent: int = 2) -> None:
    """Write data to a JSON file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Any:
    """Load JSON from a file."""
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find the most recent checkpoint directory under ``checkpoint_dir``."""
    if not checkpoint_dir.exists():
        return None
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
    )
    return checkpoints[-1] if checkpoints else None
