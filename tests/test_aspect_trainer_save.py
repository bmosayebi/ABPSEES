"""Tests for AspectGuidedTrainer checkpoint saving (tied-weight safe path)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")


class TestAspectGuidedTrainerSave:
    """`_save` must persist LoRA + aspect heads, never the full state_dict."""

    def test_save_writes_adapter_and_aspect_heads(self, tmp_path: Path) -> None:
        from project.config import AspectConfig
        from project.trainer import AspectGuidedTrainer

        aspect_config = AspectConfig(
            enabled=True,
            hidden_layer=-1,
            attn_dropout=0.1,
            score_hidden=8,
            lambda_ce=1.0,
            lambda_score=0.5,
            lambda_span=0.5,
            lambda_faith=0.1,
            low_score_threshold=0.25,
            high_score_threshold=0.5,
            prefer_span_evidence=False,
            heads_checkpoint_name="aspect_heads.pt",
        )

        fake_base = MagicMock()
        fake_heads = MagicMock()
        fake_model = MagicMock()
        fake_model.base_model = fake_base
        fake_model.aspect_heads = fake_heads

        # Build a minimal Trainer-like object without calling Trainer.__init__
        # (which needs a real TrainingArguments + datasets).
        trainer = AspectGuidedTrainer.__new__(AspectGuidedTrainer)
        trainer.aspect_config = aspect_config
        trainer.model = fake_model
        trainer.args = SimpleNamespace(output_dir=str(tmp_path / "default_out"))
        trainer.tokenizer = None
        trainer.processing_class = None

        out = tmp_path / "checkpoint-25"
        with patch("project.hybrid_model.save_aspect_heads") as save_heads:
            with patch("torch.save") as torch_save:
                trainer._save(str(out))

        fake_base.save_pretrained.assert_called_once_with(str(out))
        save_heads.assert_called_once()
        heads_arg = save_heads.call_args[0][1]
        assert Path(heads_arg).name == "aspect_heads.pt"
        assert Path(heads_arg).parent == out
        torch_save.assert_called_once()

    def test_save_does_not_call_full_state_dict(self, tmp_path: Path) -> None:
        from project.config import AspectConfig
        from project.trainer import AspectGuidedTrainer

        aspect_config = AspectConfig(
            enabled=True,
            hidden_layer=-1,
            attn_dropout=0.1,
            score_hidden=8,
            lambda_ce=1.0,
            lambda_score=0.5,
            lambda_span=0.5,
            lambda_faith=0.1,
            low_score_threshold=0.25,
            high_score_threshold=0.5,
            prefer_span_evidence=False,
            heads_checkpoint_name="aspect_heads.pt",
        )

        fake_model = MagicMock()
        fake_model.base_model = MagicMock()
        fake_model.state_dict.side_effect = AssertionError(
            "full state_dict must not be used for checkpointing"
        )

        trainer = AspectGuidedTrainer.__new__(AspectGuidedTrainer)
        trainer.aspect_config = aspect_config
        trainer.model = fake_model
        trainer.args = SimpleNamespace(output_dir=str(tmp_path))
        trainer.tokenizer = None
        trainer.processing_class = None

        with patch("project.hybrid_model.save_aspect_heads"):
            with patch("torch.save"):
                trainer._save(str(tmp_path / "ckpt"))

        fake_model.state_dict.assert_not_called()
