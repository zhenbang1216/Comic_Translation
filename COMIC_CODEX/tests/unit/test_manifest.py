from pathlib import Path

import pytest

from comic_codex.datasets.manifest import build_manifest, write_manifest


def test_manifest_is_sorted_and_hashes_content(tmp_path: Path) -> None:
    (tmp_path / "b.jpg").write_bytes(b"b")
    (tmp_path / "a.jpg").write_bytes(b"a")

    manifest = build_manifest(tmp_path, frozenset({".jpg"}))

    assert [asset.relative_path for asset in manifest.assets] == ["a.jpg", "b.jpg"]
    assert (
        manifest.assets[0].sha256
        == "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb"
    )


def test_manifest_filters_extensions_case_insensitively(tmp_path: Path) -> None:
    (tmp_path / "page.JPG").write_bytes(b"image")
    (tmp_path / "ignore.bin").write_bytes(b"ignore")

    manifest = build_manifest(tmp_path, frozenset({".jpg"}))

    assert [asset.relative_path for asset in manifest.assets] == ["page.JPG"]
    assert manifest.assets[0].extension == ".jpg"


def test_write_manifest_refuses_output_inside_scanned_root(tmp_path: Path) -> None:
    manifest = build_manifest(tmp_path, frozenset({".jpg"}))

    with pytest.raises(ValueError, match="outside the scanned root"):
        write_manifest(manifest, tmp_path / "manifest.json")


def test_write_manifest_creates_valid_json_outside_root(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "page.jpg").write_bytes(b"image")
    output = tmp_path / "manifests" / "assets.json"

    write_manifest(build_manifest(raw, frozenset({".jpg"})), output)

    assert output.exists()
    assert not output.with_suffix(".json.tmp").exists()
