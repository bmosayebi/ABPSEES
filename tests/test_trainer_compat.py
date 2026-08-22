"""Tests for TrainingArguments compatibility layer."""

from __future__ import annotations

from unittest.mock import MagicMock

from project.hf_compat import compat_training_kwargs


class _FakeTrainingArguments:
    def __init__(
        self,
        output_dir: str,
        warmup_steps: float = 0,
        learning_rate: float = 5e-5,
        evaluation_strategy: str = "no",
    ) -> None:
        self.output_dir = output_dir


class TestTrainingArgsCompat:
    def test_warmup_ratio_mapped_when_unsupported(self) -> None:
        result = compat_training_kwargs(
            {"output_dir": "/tmp", "warmup_ratio": 0.1, "learning_rate": 1e-4},
            _FakeTrainingArguments,
        )
        assert "warmup_ratio" not in result
        assert result["warmup_steps"] == 0.1

    def test_logging_dir_env_fallback(self, monkeypatch) -> None:
        class _V4Args:
            def __init__(self, output_dir: str, warmup_ratio: float = 0.0) -> None:
                pass

        result = compat_training_kwargs(
            {"output_dir": "/tmp", "logging_dir": "/tmp/tb", "warmup_ratio": 0.1},
            _V4Args,
        )
        assert "logging_dir" not in result
        import os

        assert os.environ.get("TENSORBOARD_LOGGING_DIR") == "/tmp/tb"
