import json
from pathlib import Path
from typing import Protocol

from comic_codex.domain.project import ComicProject
from comic_codex.project_store.atomic_json import write_text_atomically


class UnsupportedSchemaVersionError(ValueError):
    pass


class ProjectStore(Protocol):
    def save(self, project: ComicProject, path: Path) -> None: ...

    def load(self, path: Path) -> ComicProject: ...


class JsonProjectStore:
    def save(self, project: ComicProject, path: Path) -> None:
        content = project.model_dump_json(indent=2) + "\n"
        write_text_atomically(path, content)

    def load(self, path: Path) -> ComicProject:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("project file must contain a JSON object")
        schema_version = raw.get("schema_version")
        if schema_version != 1:
            raise UnsupportedSchemaVersionError(
                f"unsupported project schema version: {schema_version!r}"
            )
        return ComicProject.model_validate(raw)

