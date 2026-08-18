from pathlib import Path

from comic_codex.config.paths import AppPaths


def test_default_data_root_is_external() -> None:
    paths = AppPaths.discover(Path(r"E:\Create\project\COMIC\COMIC_CODEX"), {})

    assert paths.data_root == Path(r"E:\Create\Data\COMIC\COMIC_CODEX")
    assert paths.models_dir == paths.data_root / "models"


def test_environment_overrides_data_root(tmp_path: Path) -> None:
    paths = AppPaths.discover(
        tmp_path / "project", {"COMIC_CODEX_DATA_ROOT": str(tmp_path)}
    )

    assert paths.data_root == tmp_path.resolve()

