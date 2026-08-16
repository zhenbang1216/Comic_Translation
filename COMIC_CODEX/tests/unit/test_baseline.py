from datetime import UTC, datetime

import pytest

from comic_codex.datasets.baseline import BaselineSpec, build_baseline_selection
from comic_codex.datasets.manifest import DatasetAsset, DatasetManifest


def make_manifest() -> DatasetManifest:
    return DatasetManifest(
        root="C:/raw",
        generated_at=datetime.now(UTC),
        assets=[
            DatasetAsset(
                relative_path="pages/a.jpg",
                size_bytes=1,
                extension=".jpg",
                sha256="a" * 64,
            )
        ],
    )


def test_build_selection_resolves_hash_and_stable_id() -> None:
    selection = build_baseline_selection(
        make_manifest(),
        [
            BaselineSpec(
                category="clear_horizontal",
                source_relative_path="pages/a.jpg",
                annotation_relative_path="annotations.json",
                language="id",
                license_status="unknown",
                review_status="confirmed",
                review_evidence="reviewed",
                notes="high sharpness candidate",
            )
        ],
    )

    assert selection.records[0].id == "clear_horizontal-001"
    assert selection.records[0].sha256 == "a" * 64


def test_build_selection_rejects_missing_asset() -> None:
    with pytest.raises(ValueError, match="not found"):
        build_baseline_selection(
            make_manifest(),
            [
                BaselineSpec(
                    category="blurred_low_resolution",
                    source_relative_path="missing.jpg",
                    annotation_relative_path="annotations.json",
                    language="id",
                    license_status="unknown",
                    review_status="confirmed",
                    review_evidence="reviewed",
                    notes="missing",
                )
            ],
        )


def test_build_selection_rejects_duplicate_asset() -> None:
    spec = BaselineSpec(
        category="clear_horizontal",
        source_relative_path="pages/a.jpg",
        annotation_relative_path="annotations.json",
        language="id",
        license_status="unknown",
        review_status="confirmed",
        review_evidence="reviewed",
        notes="duplicate",
    )

    with pytest.raises(ValueError, match="duplicate"):
        build_baseline_selection(make_manifest(), [spec, spec])
