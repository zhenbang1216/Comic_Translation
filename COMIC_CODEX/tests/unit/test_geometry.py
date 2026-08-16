import pytest
from pydantic import ValidationError

from comic_codex.domain.geometry import Point, Polygon


def test_polygon_bounds() -> None:
    polygon = Polygon(
        points=(Point(x=3, y=8), Point(x=9, y=2), Point(x=7, y=11))
    )

    assert polygon.bounds() == (3.0, 2.0, 9.0, 11.0)


def test_polygon_requires_three_distinct_points() -> None:
    with pytest.raises(ValidationError):
        Polygon(
            points=(Point(x=0, y=0), Point(x=1, y=1), Point(x=0, y=0))
        )
