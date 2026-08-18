import sys
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def comic_project():  # type: ignore[no-untyped-def]
    from comic_codex.domain.page import ComicPage
    from comic_codex.domain.project import Chapter, ComicProject

    page = ComicPage(id=uuid4(), source_image=Path("page-001.png"), page_index=0)
    return ComicProject(
        schema_version=1,
        id=uuid4(),
        title="fixture",
        source_language="ja",
        target_language="zh",
        chapters=[Chapter(id=uuid4(), title="chapter 1", pages=[page])],
    )
