from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator


class Point(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: float
    y: float


class Polygon(BaseModel):
    model_config = ConfigDict(frozen=True)

    points: tuple[Point, ...]

    @model_validator(mode="after")
    def require_three_distinct_points(self) -> Self:
        coordinates = {(point.x, point.y) for point in self.points}
        if len(coordinates) < 3:
            raise ValueError("polygon requires at least three distinct points")
        return self

    def bounds(self) -> tuple[float, float, float, float]:
        xs = [point.x for point in self.points]
        ys = [point.y for point in self.points]
        return min(xs), min(ys), max(xs), max(ys)

