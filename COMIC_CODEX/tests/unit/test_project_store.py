import json
from pathlib import Path

import pytest

from comic_codex.domain.project import ComicProject
from comic_codex.project_store.store import (
    JsonProjectStore,
    UnsupportedSchemaVersionError,
)


def test_save_and_load_are_lossless(
    tmp_path: Path, comic_project: ComicProject
) -> None:
    path = tmp_path / "demo.comicproj"
    store = JsonProjectStore()

    store.save(comic_project, path)

    assert store.load(path) == comic_project
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_save_preserves_existing_file_when_replace_fails(
    tmp_path: Path, comic_project: ComicProject, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "demo.comicproj"
    original = b"existing project bytes"
    path.write_bytes(original)

    def fail_replace(source: Path, target: Path) -> None:
        raise OSError(f"cannot replace {source} with {target}")

    monkeypatch.setattr("comic_codex.project_store.atomic_json.os.replace", fail_replace)

    with pytest.raises(OSError, match="cannot replace"):
        JsonProjectStore().save(comic_project, path)

    assert path.read_bytes() == original
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_load_rejects_unknown_schema_before_model_validation(tmp_path: Path) -> None:
    path = tmp_path / "future.comicproj"
    path.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")

    with pytest.raises(UnsupportedSchemaVersionError, match="2"):
        JsonProjectStore().load(path)

