from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from comic_codex.datasets.manifest import DatasetManifest
from comic_codex.project_store.atomic_json import write_text_atomically

BaselineCategory = Literal[
    "clear_horizontal",
    "clear_vertical",
    "complex_background",
    "blurred_low_resolution",
]
LicenseStatus = Literal["verified", "research_only", "unknown"]


class BaselineSpec(BaseModel):
    category: BaselineCategory
    source_relative_path: str
    annotation_relative_path: str
    language: str
    license_status: LicenseStatus
    notes: str


class BaselineRecord(BaselineSpec):
    id: str
    sha256: str


class BaselineSelection(BaseModel):
    schema_version: Literal[1] = 1
    source_manifest_root: str
    generated_at: datetime
    records: list[BaselineRecord]


def build_baseline_selection(
    manifest: DatasetManifest, specs: list[BaselineSpec]
) -> BaselineSelection:
    assets_by_path = {asset.relative_path: asset for asset in manifest.assets}
    source_paths = [spec.source_relative_path for spec in specs]
    if len(source_paths) != len(set(source_paths)):
        raise ValueError("baseline selection contains a duplicate asset")

    category_counts: dict[str, int] = {}
    records: list[BaselineRecord] = []
    for spec in specs:
        asset = assets_by_path.get(spec.source_relative_path)
        if asset is None:
            raise ValueError(f"asset not found in manifest: {spec.source_relative_path}")
        count = category_counts.get(spec.category, 0) + 1
        category_counts[spec.category] = count
        records.append(
            BaselineRecord(
                **spec.model_dump(),
                id=f"{spec.category}-{count:03d}",
                sha256=asset.sha256,
            )
        )
    return BaselineSelection(
        source_manifest_root=manifest.root,
        generated_at=datetime.now(UTC),
        records=records,
    )


def write_baseline_selection(selection: BaselineSelection, output: Path) -> None:
    write_text_atomically(output, selection.model_dump_json(indent=2) + "\n")

