from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from comic_codex.config.paths import AppPaths


class StrictSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PipelineSettings(StrictSettings):
    detection_confidence: float = Field(ge=0.0, le=1.0)
    ocr_confidence: float = Field(ge=0.0, le=1.0)


class ResourceSettings(StrictSettings):
    max_resident_heavy_models: int = Field(ge=1)


class OnlineTranslationSettings(StrictSettings):
    enabled: bool = False


class AppSettings(StrictSettings):
    schema_version: Literal[1]
    pipeline: PipelineSettings
    resources: ResourceSettings
    online_translation: OnlineTranslationSettings
    paths: AppPaths


def load_settings(config_file: Path, paths: AppPaths) -> AppSettings:
    raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("settings file must contain a YAML mapping")
    return AppSettings.model_validate({**raw, "paths": paths})

