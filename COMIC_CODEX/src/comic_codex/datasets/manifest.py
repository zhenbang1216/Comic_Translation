import hashlib
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from comic_codex.project_store.atomic_json import write_text_atomically


class DatasetAsset(BaseModel):
    relative_path: str
    size_bytes: int
    extension: str
    sha256: str


class DatasetManifest(BaseModel):
    schema_version: Literal[1] = 1
    root: str
    generated_at: datetime
    assets: list[DatasetAsset]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path, extensions: frozenset[str]) -> DatasetManifest:
    resolved_root = root.resolve(strict=True)
    normalized_extensions = frozenset(
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
    )
    candidate_paths: list[Path] = []
    for current_root, directory_names, file_names in os.walk(
        resolved_root, followlinks=False
    ):
        directory_names.sort()
        for file_name in sorted(file_names):
            path = Path(current_root) / file_name
            if path.is_symlink() or path.suffix.lower() not in normalized_extensions:
                continue
            candidate_paths.append(path)

    assets = [
        DatasetAsset(
            relative_path=path.relative_to(resolved_root).as_posix(),
            size_bytes=path.stat().st_size,
            extension=path.suffix.lower(),
            sha256=_sha256(path),
        )
        for path in sorted(
            candidate_paths, key=lambda item: item.relative_to(resolved_root).as_posix()
        )
    ]
    return DatasetManifest(
        root=str(resolved_root), generated_at=datetime.now(UTC), assets=assets
    )


def write_manifest(manifest: DatasetManifest, output: Path) -> None:
    root = Path(manifest.root).resolve()
    resolved_output = output.resolve()
    if resolved_output.is_relative_to(root):
        raise ValueError("manifest output must be outside the scanned root")
    write_text_atomically(output, manifest.model_dump_json(indent=2) + "\n")

