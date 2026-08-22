"""Model loading with QLoRA for CUDA / Colab GPU."""

from __future__ import annotations

import logging
from typing import Any

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from project.config import AppConfig
from project.utils import DeviceConfig, empty_cuda_cache, get_device_config

logger = logging.getLogger(__name__)


def _dtype_from_string(name: str) -> torch.dtype:
    """Map a dtype string from config to ``torch.dtype``."""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype string: {name}")
    return mapping[name]


def build_bnb_config(config: AppConfig, device_cfg: DeviceConfig) -> BitsAndBytesConfig | None:
    """Build BitsAndBytes 4-bit config for CUDA QLoRA."""
    if device_cfg.device != "cuda" or not config.quantization.load_in_4bit:
        return None

    try:
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "bitsandbytes is required for QLoRA. Install with: pip install bitsandbytes"
        ) from exc

    compute_dtype = _dtype_from_string(config.quantization.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.quantization.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=config.quantization.use_double_quant,
    )


def build_lora_config(config: AppConfig) -> LoraConfig:
    """Create PEFT LoRA configuration."""
    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.dropout,
        bias=config.lora.bias,
        target_modules=config.lora.target_modules,
        task_type="CAUSAL_LM",
    )


def load_tokenizer(config: AppConfig) -> Any:
    """Load and configure the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    config: AppConfig,
    device_cfg: DeviceConfig | None = None,
) -> Any:
    """Load the base causal LM with 4-bit quantization on CUDA."""
    device_cfg = device_cfg or get_device_config(require_cuda=True)
    bnb_config = build_bnb_config(config, device_cfg)

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
        "attn_implementation": device_cfg.attn_implementation,
        "low_cpu_mem_usage": True,
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = device_cfg.torch_dtype
    else:
        model_kwargs["torch_dtype"] = device_cfg.torch_dtype
        model_kwargs["device_map"] = "auto"

    empty_cuda_cache()
    model = AutoModelForCausalLM.from_pretrained(config.model.name, **model_kwargs)

    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model)

    if config.training.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.config.use_cache = False

    return model


def get_model_and_tokenizer(
    config: AppConfig,
    device_cfg: DeviceConfig | None = None,
) -> tuple[Any, Any]:
    """Load tokenizer and attach LoRA adapters to the base model."""
    device_cfg = device_cfg or get_device_config(require_cuda=True)
    tokenizer = load_tokenizer(config)
    base_model = load_base_model(config, device_cfg)
    lora_config = build_lora_config(config)
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    logger.info(
        "Loaded model %s on %s (QLoRA=%s)",
        config.model.name,
        device_cfg.device,
        build_bnb_config(config, device_cfg) is not None,
    )
    return model, tokenizer


def load_model_for_inference(
    config: AppConfig,
    adapter_path: str | None = None,
    device_cfg: DeviceConfig | None = None,
) -> tuple[Any, Any]:
    """Load base model with optional LoRA adapter for inference."""
    device_cfg = device_cfg or get_device_config(require_cuda=False)
    tokenizer = load_tokenizer(config)
    base_model = load_base_model(config, device_cfg)

    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        logger.info("Loaded LoRA adapter from %s", adapter_path)
    else:
        model = base_model
        model.eval()
        logger.warning("No adapter path provided; using base model only.")

    return model, tokenizer
