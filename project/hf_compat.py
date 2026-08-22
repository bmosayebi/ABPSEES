"""HuggingFace TrainingArguments compatibility across library versions."""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def compat_training_kwargs(kwargs: dict[str, Any], training_arguments_cls: type) -> dict[str, Any]:
    """Adapt TrainingArguments kwargs across transformers v4 and v5.

    Args:
        kwargs: Desired training argument fields.
        training_arguments_cls: The ``TrainingArguments`` class to inspect.

    Returns:
        Filtered and remapped kwargs accepted by the installed transformers version.
    """
    params = set(inspect.signature(training_arguments_cls.__init__).parameters.keys())

    if "warmup_ratio" in kwargs and "warmup_ratio" not in params:
        ratio = kwargs.pop("warmup_ratio")
        if "warmup_steps" in params:
            kwargs["warmup_steps"] = ratio
            logger.info("Mapped warmup_ratio=%s to warmup_steps for transformers v5+", ratio)

    if "logging_dir" in kwargs and "logging_dir" not in params:
        logging_dir = kwargs.pop("logging_dir")
        os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(logging_dir))
        logger.info("Set TENSORBOARD_LOGGING_DIR=%s for transformers v5+", logging_dir)

    if "eval_strategy" in kwargs and "eval_strategy" not in params and "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    if "gradient_checkpointing" in kwargs and "gradient_checkpointing" not in params:
        kwargs.pop("gradient_checkpointing")
    if "gradient_checkpointing_kwargs" in kwargs and "gradient_checkpointing_kwargs" not in params:
        kwargs.pop("gradient_checkpointing_kwargs")

    filtered = {key: value for key, value in kwargs.items() if key in params}
    dropped = set(kwargs) - set(filtered)
    if dropped:
        logger.debug("Dropped unsupported TrainingArguments keys: %s", sorted(dropped))
    return filtered
