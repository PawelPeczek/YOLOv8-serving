from dataclasses import dataclass
from typing import List, Tuple, Callable, TypeVar

from dataclasses_json import DataClassJsonMixin

T = TypeVar("T")
RawPredictionResponse = dict


@dataclass(frozen=True)
class RelativePointCoordinates(DataClassJsonMixin):
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return self.x, self.y


@dataclass(frozen=True)
class AbsolutePointCoordinates(DataClassJsonMixin):
    x: int
    y: int

    def to_tuple(self) -> Tuple[int, int]:
        return self.x, self.y


@dataclass(frozen=True)
class BoundingBox(DataClassJsonMixin):
    object_category: str
    confidence: float
    left_top_relative: RelativePointCoordinates
    right_bottom_relative: RelativePointCoordinates
    left_top_absolute: AbsolutePointCoordinates
    right_bottom_absolute: AbsolutePointCoordinates


@dataclass(frozen=True)
class KeyPoint(DataClassJsonMixin):
    index: int
    point_class: str
    point_category: str
    confidence: float
    point_coordinates_relative: RelativePointCoordinates
    point_coordinates_absolute: AbsolutePointCoordinates


@dataclass(frozen=True)
class Detection(DataClassJsonMixin):
    image_size: Tuple[int, int]
    bounding_box: BoundingBox
    key_points: List[KeyPoint]

    def transform(self, transformation: Callable[["Detection"], T]) -> T:
        return transformation(self)
