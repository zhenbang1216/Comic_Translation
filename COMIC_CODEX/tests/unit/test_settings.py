from pathlib import Path

import pytest
from pydantic import ValidationError

from comic_codex.config.paths import AppPaths
from comic_codex.config.settings import PipelineSettings, load_settings


def test_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        PipelineSettings(detection_confidence=1.1, ocr_confidence=0.7)


def test_load_settings_injects_paths(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        "schema_version: 1\n"
        "pipeline:\n"
        "  detection_confidence: 0.5\n"
        "  ocr_confidence: 0.7\n"
        "resources:\n"
        "  max_resident_heavy_models: 1\n"
        "online_translation:\n"
        "  enabled: false\n",
        encoding="utf-8",
    )
    paths = AppPaths.discover(tmp_path / "project", {"COMIC_CODEX_DATA_ROOT": str(tmp_path)})

    settings = load_settings(config_file, paths)

    assert settings.paths is paths
    assert settings.online_translation.enabled is False


def test_load_settings_rejects_unknown_keys(tmp_path: Path) -> None:
    config_file = tmp_path / "settings.yaml"
    config_file.write_text("schema_version: 1\nunknown: true\n", encoding="utf-8")
    paths = AppPaths.discover(tmp_path / "project", {})

    with pytest.raises(ValidationError):
        load_settings(config_file, paths)
