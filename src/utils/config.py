"""漫画翻译系统的配置加载器。"""
import yaml
from pathlib import Path
from typing import Any


class Config:
    """从YAML文件加载的单例配置。"""

    _instance = None
    _data: dict[str, Any] = {}

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, config_path: str | Path) -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value = self._data
        for k in keys:
            if not isinstance(value, dict):
                return default
            value = value.get(k)
            if value is None:
                return default
        return value

    @property
    def models(self) -> dict:
        return self._data.get("models", {})

    @property
    def pipeline(self) -> dict:
        return self._data.get("pipeline", {})

    @property
    def gui(self) -> dict:
        return self._data.get("gui", {})

    @property
    def paths(self) -> dict:
        return self._data.get("paths", {})


config = Config()
