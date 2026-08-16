from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from comic_codex.domain.page import ComicPage
from comic_codex.domain.project import Chapter, ComicProject


def make_project(page: ComicPage) -> ComicProject:
    return ComicProject(
        schema_version=1,
        id=uuid4(),
        title="demo",
        source_language="ja",
        target_language="zh",
        chapters=[Chapter(id=uuid4(), title="chapter 1", pages=[page])],
    )


def test_duplicate_page_ids_are_rejected() -> None:
    page = ComicPage(id=uuid4(), source_image=Path("a.png"), page_index=0)

    with pytest.raises(ValidationError):
        ComicProject(
            schema_version=1,
            id=uuid4(),
            title="demo",
            source_language="ja",
            target_language="zh",
            chapters=[Chapter(id=uuid4(), title="c1", pages=[page, page])],
        )


def test_duplicate_page_indexes_are_rejected() -> None:
    first = ComicPage(id=uuid4(), source_image=Path("a.png"), page_index=0)
    second = ComicPage(id=uuid4(), source_image=Path("b.png"), page_index=0)

    with pytest.raises(ValidationError):
        Chapter(id=uuid4(), title="c1", pages=[first, second])


def test_find_page_returns_exact_page() -> None:
    page = ComicPage(id=uuid4(), source_image=Path("a.png"), page_index=0)
    project = make_project(page)

    assert project.find_page(page.id) is page


def test_find_page_raises_for_unknown_id() -> None:
    page = ComicPage(id=uuid4(), source_image=Path("a.png"), page_index=0)
    project = make_project(page)
    missing_id = uuid4()

    with pytest.raises(KeyError, match=str(missing_id)):
        project.find_page(missing_id)


def test_project_json_round_trip_is_lossless() -> None:
    page = ComicPage(id=uuid4(), source_image=Path("a.png"), page_index=0)
    project = make_project(page)

    restored = ComicProject.model_validate_json(project.model_dump_json())

    assert restored == project


@pytest.mark.parametrize("language", ["JA", "japanese", "zh_cn", "z"])
def test_language_code_must_use_supported_shape(language: str) -> None:
    page = ComicPage(id=uuid4(), source_image=Path("a.png"), page_index=0)

    with pytest.raises(ValidationError):
        ComicProject(
            schema_version=1,
            id=uuid4(),
            title="demo",
            source_language=language,
            target_language="zh",
            chapters=[Chapter(id=uuid4(), title="c1", pages=[page])],
        )
