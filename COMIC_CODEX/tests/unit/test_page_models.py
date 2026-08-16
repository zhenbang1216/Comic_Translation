from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from comic_codex.domain.geometry import Point, Polygon
from comic_codex.domain.page import ComicPage, OCRCandidate, TextRegion

POLYGON = Polygon(
    points=(Point(x=0, y=0), Point(x=10, y=0), Point(x=10, y=10))
)


def test_selected_text_returns_selected_candidate() -> None:
    region = TextRegion(
        id=uuid4(),
        polygon=POLYGON,
        ocr_candidates=[
            OCRCandidate(
                text="原文",
                confidence=0.91,
                provider="paddleocr",
                source_variant="original",
            )
        ],
        selected_ocr_index=0,
    )

    assert region.selected_text() == "原文"


def test_selected_text_returns_none_without_selection() -> None:
    region = TextRegion(id=uuid4(), polygon=POLYGON)

    assert region.selected_text() is None


def test_candidate_confidence_must_be_probability() -> None:
    with pytest.raises(ValidationError):
        OCRCandidate(
            text="x", confidence=2.0, provider="test", source_variant="original"
        )


def test_selected_index_must_reference_candidate() -> None:
    with pytest.raises(ValidationError):
        TextRegion(id=uuid4(), polygon=POLYGON, selected_ocr_index=0)


def test_page_has_safe_independent_defaults() -> None:
    first = ComicPage(id=uuid4(), source_image=Path("first.png"), page_index=0)
    second = ComicPage(id=uuid4(), source_image=Path("second.png"), page_index=1)

    first.text_regions.append(TextRegion(id=uuid4(), polygon=POLYGON))

    assert second.text_regions == []
