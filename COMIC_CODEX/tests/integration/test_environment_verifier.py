import json
import os
import subprocess
import sys
from pathlib import Path


def test_environment_verifier_reports_skipped_baseline_as_incomplete(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = tmp_path / "settings.yaml"
    config.write_text(
        "schema_version: 1\n"
        "pipeline:\n"
        "  detection_confidence: 0.5\n"
        "  ocr_confidence: 0.7\n"
        "resources:\n"
        "  max_resident_heavy_models: 1\n"
        "online_translation:\n"
        "  enabled: false\n",
        encoding="utf-8",
    )
    environment = {
        **os.environ,
        "COMIC_CODEX_DATA_ROOT": str(tmp_path / "data"),
        "PYTHONPATH": str(project_root / "src"),
    }

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "verify_environment.py"),
            "--project-root",
            str(project_root),
            "--config",
            str(config),
            "--skip-baseline",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 1, completed.stderr
    report = json.loads(completed.stdout)
    assert set(report) == {
        "python",
        "paths",
        "config",
        "project_round_trip",
        "baseline",
        "result",
    }
    assert report["result"] == "fail"
    assert report["baseline"]["status"] == "skipped"
    assert not (tmp_path / "data" / "cache" / "verification" / "smoke.comicproj").exists()
