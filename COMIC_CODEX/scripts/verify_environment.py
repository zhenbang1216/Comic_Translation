import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from uuid import uuid4

from comic_codex.config.paths import AppPaths
from comic_codex.config.settings import load_settings
from comic_codex.datasets.baseline import BaselineSelection
from comic_codex.domain.page import ComicPage
from comic_codex.domain.project import Chapter, ComicProject
from comic_codex.project_store.store import JsonProjectStore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_baseline(path: Path) -> dict[str, object]:
    selection = BaselineSelection.model_validate_json(path.read_text(encoding="utf-8"))
    if not 40 <= len(selection.records) <= 60:
        raise ValueError("baseline must contain 40 to 60 records")
    categories = {record.category for record in selection.records}
    required = {
        "clear_horizontal",
        "clear_vertical",
        "complex_background",
        "blurred_low_resolution",
    }
    if categories != required:
        raise ValueError(f"baseline categories differ from required set: {categories}")
    hashes = [record.sha256 for record in selection.records]
    if len(hashes) != len(set(hashes)):
        raise ValueError("baseline contains duplicate asset hashes")
    root = Path(selection.source_manifest_root).resolve()
    for record in selection.records:
        if record.review_status != "confirmed":
            raise ValueError(f"baseline record is not confirmed: {record.id}")
        source = (root / Path(record.source_relative_path)).resolve()
        if root not in source.parents:
            raise ValueError(f"baseline source escapes manifest root: {record.id}")
        if not source.is_file():
            raise FileNotFoundError(source)
        if _sha256(source) != record.sha256:
            raise ValueError(f"baseline hash mismatch: {record.id}")
        annotation = (root / Path(record.annotation_relative_path)).resolve()
        if root not in annotation.parents:
            raise ValueError(f"baseline annotation escapes manifest root: {record.id}")
        if not annotation.is_file():
            raise FileNotFoundError(annotation)
    return {"status": "pass", "records": len(selection.records)}


def _project_round_trip(cache_dir: Path) -> dict[str, object]:
    page = ComicPage(id=uuid4(), source_image=Path("verification.png"), page_index=0)
    project = ComicProject(
        id=uuid4(),
        title="environment verification",
        source_language="ja",
        target_language="zh",
        chapters=[Chapter(id=uuid4(), title="chapter", pages=[page])],
    )
    verification_dir = cache_dir / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    project_path = verification_dir / "smoke.comicproj"
    store = JsonProjectStore()
    try:
        store.save(project, project_path)
        if store.load(project_path) != project:
            raise ValueError("project round trip changed content")
    finally:
        project_path.unlink(missing_ok=True)
    return {"status": "pass"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the COMIC_CODEX foundation")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--skip-baseline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report: dict[str, object] = {
        "python": {},
        "paths": {},
        "config": {},
        "project_round_trip": {},
        "baseline": {},
        "result": "fail",
    }
    failures: list[str] = []

    version_ok = sys.version_info[:2] == (3, 11)
    report["python"] = {"status": "pass" if version_ok else "fail", "version": sys.version}
    if not version_ok:
        failures.append("python")

    try:
        paths = AppPaths.discover(args.project_root, os.environ)
        for directory in (
            paths.models_dir,
            paths.cache_dir,
            paths.outputs_dir,
            paths.datasets_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        report["paths"] = {"status": "pass", "data_root": str(paths.data_root)}
    except Exception as error:
        failures.append("paths")
        report["paths"] = {"status": "fail", "error": str(error)}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    try:
        config_path = args.config or args.project_root / "configs" / "default.yaml"
        settings = load_settings(config_path, paths)
        report["config"] = {"status": "pass", "schema_version": settings.schema_version}
    except Exception as error:
        failures.append("config")
        report["config"] = {"status": "fail", "error": str(error)}

    try:
        report["project_round_trip"] = _project_round_trip(paths.cache_dir)
    except Exception as error:
        failures.append("project_round_trip")
        report["project_round_trip"] = {"status": "fail", "error": str(error)}

    if args.skip_baseline:
        report["baseline"] = {"status": "skipped"}
        failures.append("baseline")
    else:
        try:
            baseline_path = args.baseline or (
                paths.datasets_dir / "manifests" / "baseline-selection.json"
            )
            report["baseline"] = _verify_baseline(baseline_path)
        except Exception as error:
            failures.append("baseline")
            report["baseline"] = {"status": "fail", "error": str(error)}

    report["result"] = "pass" if not failures else "fail"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

