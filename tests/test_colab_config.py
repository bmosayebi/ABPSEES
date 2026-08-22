"""Tests for Colab configuration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from project.config import AppConfig, ColabConfig, load_config, resolve_colab_data_dir


class TestColabConfig:
    def test_colab_config_load(self) -> None:
        root = Path(__file__).resolve().parent.parent
        config = load_config(root / "config" / "colab.yaml")
        assert config.colab.enabled is True
        assert config.colab.project_dir == "/content/ABPSEES"

    @patch("project.config.is_colab", return_value=True)
    def test_resolve_colab_data_dir_with_drive(self, _mock_colab, tmp_path) -> None:
        data_dir = tmp_path / "data" / "synthetic"
        data_dir.mkdir(parents=True)
        (data_dir / "sample.json").write_text("[]", encoding="utf-8")

        root = Path(__file__).resolve().parent.parent
        config = load_config(root / "config" / "colab.yaml")
        config.colab = ColabConfig(
            enabled=True,
            project_dir="/content/ABPSEES",
            mount_google_drive=True,
            drive_project_path=str(tmp_path),
            hf_token_env="HF_TOKEN",
            use_colab_secrets=True,
        )
        resolved = resolve_colab_data_dir(config)
        assert resolved == data_dir
