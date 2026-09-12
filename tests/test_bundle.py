"""Tests for single-file model bundling (project/bundle.py)."""

from __future__ import annotations

import json
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from project.bundle import (
    BUNDLE_METADATA_FILENAME,
    _metadata_from_config,
    _minimal_config_from_metadata,
    download_bundle_on_colab,
    load_model_bundle,
    save_model_bundle,
)
from project.config import load_config


class TestMetadataRoundTrip:
    """Metadata extracted from AppConfig must reconstruct an equivalent minimal config."""

    def test_round_trip_preserves_inference_relevant_fields(self) -> None:
        config = load_config()
        metadata = _metadata_from_config(config)

        # Simulate JSON serialization exactly as save_model_bundle does.
        metadata = json.loads(json.dumps(metadata))

        rebuilt = _minimal_config_from_metadata(metadata, root=config.project_root)

        assert rebuilt.model.name == config.model.name
        assert rebuilt.model.max_seq_length == config.model.max_seq_length
        assert rebuilt.quantization.load_in_4bit == config.quantization.load_in_4bit
        assert rebuilt.inference.max_new_tokens == config.inference.max_new_tokens
        assert rebuilt.data.normalize_unicode == config.data.normalize_unicode
        assert rebuilt.aspect.enabled == config.aspect.enabled
        assert rebuilt.aspect.lambda_score == config.aspect.lambda_score
        assert rebuilt.aspect.heads_checkpoint_name == config.aspect.heads_checkpoint_name

    def test_metadata_omits_training_only_fields(self) -> None:
        config = load_config()
        metadata = _metadata_from_config(config)

        assert "lora" not in metadata
        assert "training" not in metadata
        assert "colab" not in metadata
        assert "paths" not in metadata


class TestSaveModelBundle:
    """Zip packaging of a checkpoint directory into a single portable file."""

    def _make_fake_adapter_dir(self, tmp_path, with_aspect_heads: bool) -> "object":
        adapter_dir = tmp_path / "best"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake-safetensors-bytes")
        if with_aspect_heads:
            (adapter_dir / "aspect_heads.pt").write_bytes(b"fake-torch-bytes")
        return adapter_dir

    def test_bundle_contains_adapter_files_and_metadata(self, tmp_path) -> None:
        config = load_config()
        adapter_dir = self._make_fake_adapter_dir(tmp_path, with_aspect_heads=config.aspect.enabled)
        output_path = tmp_path / "model_bundle.zip"

        bundle_path = save_model_bundle(config, adapter_dir, output_path)

        assert bundle_path.exists()
        assert bundle_path.suffix == ".zip"

        with zipfile.ZipFile(bundle_path) as zf:
            names = set(zf.namelist())
            assert "adapter_config.json" in names
            assert "adapter_model.safetensors" in names
            assert BUNDLE_METADATA_FILENAME in names
            if config.aspect.enabled:
                assert "aspect_heads.pt" in names

            metadata = json.loads(zf.read(BUNDLE_METADATA_FILENAME))
            assert metadata["model"]["name"] == config.model.name
            assert metadata["aspect"]["enabled"] == config.aspect.enabled

    def test_default_output_path_under_output_dir(self, tmp_path) -> None:
        config = load_config()
        adapter_dir = self._make_fake_adapter_dir(tmp_path, with_aspect_heads=False)

        bundle_path = save_model_bundle(config, adapter_dir, output_path=tmp_path / "custom_name.zip")

        assert bundle_path.name == "custom_name.zip"

    def test_raises_for_missing_adapter_dir(self, tmp_path) -> None:
        config = load_config()
        with pytest.raises(FileNotFoundError):
            save_model_bundle(config, tmp_path / "does_not_exist", tmp_path / "out.zip")


class TestLoadModelBundle:
    """Round-trip load: extract .zip, reconstruct config, dispatch to the right loader."""

    def _make_bundle(self, tmp_path, aspect_enabled: bool):
        config = load_config()
        config.aspect.enabled = aspect_enabled
        adapter_dir = tmp_path / "best"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"fake")
        if aspect_enabled:
            (adapter_dir / "aspect_heads.pt").write_bytes(b"fake")
        return save_model_bundle(config, adapter_dir, tmp_path / "bundle.zip")

    def test_dispatches_to_plain_loader_when_aspect_disabled(self, tmp_path) -> None:
        bundle_path = self._make_bundle(tmp_path, aspect_enabled=False)
        fake_model, fake_tokenizer = MagicMock(), MagicMock()

        with patch("project.model.load_model_for_inference", return_value=(fake_model, fake_tokenizer)) as mocked:
            model, tokenizer, config = load_model_bundle(bundle_path, extract_dir=tmp_path / "extracted")

        assert model is fake_model
        assert tokenizer is fake_tokenizer
        assert config.aspect.enabled is False
        mocked.assert_called_once()
        called_adapter_path = mocked.call_args[0][1]
        assert called_adapter_path == str(tmp_path / "extracted")

    def test_dispatches_to_hybrid_loader_when_aspect_enabled(self, tmp_path) -> None:
        bundle_path = self._make_bundle(tmp_path, aspect_enabled=True)
        fake_model, fake_tokenizer = MagicMock(), MagicMock()

        with patch(
            "project.hybrid_model.load_hybrid_model_for_inference", return_value=(fake_model, fake_tokenizer)
        ) as mocked:
            model, tokenizer, config = load_model_bundle(bundle_path, extract_dir=tmp_path / "extracted")

        assert model is fake_model
        assert tokenizer is fake_tokenizer
        assert config.aspect.enabled is True
        mocked.assert_called_once()

    def test_raises_for_missing_bundle_file(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            load_model_bundle(tmp_path / "does_not_exist.zip")

    def test_colab_download_is_noop_outside_colab(self, tmp_path) -> None:
        missing = tmp_path / "does_not_exist.zip"
        assert download_bundle_on_colab(missing) is False

        existing = tmp_path / "model_bundle.zip"
        existing.write_bytes(b"zip")
        assert download_bundle_on_colab(existing) is False

    def test_raises_for_bundle_without_metadata(self, tmp_path) -> None:
        # A .zip that isn't an ABPSEES bundle (no bundle_metadata.json inside).
        bad_zip = tmp_path / "not_a_bundle.zip"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("random_file.txt", "hello")

        with pytest.raises(FileNotFoundError):
            load_model_bundle(bad_zip, extract_dir=tmp_path / "extracted2")
