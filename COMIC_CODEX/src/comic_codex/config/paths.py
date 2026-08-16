from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class AppPaths(BaseModel):
    model_config = ConfigDict(frozen=True)

    project_root: Path
    data_root: Path
    models_dir: Path
    cache_dir: Path
    outputs_dir: Path
    datasets_dir: Path

    @classmethod
    def discover(cls, project_root: Path, environ: Mapping[str, str]) -> "AppPaths":
        value = environ.get(
            "COMIC_CODEX_DATA_ROOT", r"E:\Create\Data\COMIC\COMIC_CODEX"
        )
        data_root = Path(value).expanduser().resolve()
        root = project_root.resolve()
        return cls(
            project_root=root,
            data_root=data_root,
            models_dir=data_root / "models",
            cache_dir=data_root / "cache",
            outputs_dir=data_root / "outputs",
            datasets_dir=data_root / "datasets",
        )

