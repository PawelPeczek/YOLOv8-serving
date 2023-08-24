from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from dataclasses_json import DataClassJsonMixin, config


def decode_mapping(dictionary: Dict[str, str]) -> Dict[int, str]:
    return {int(k): v for k, v in dictionary.items()}


@dataclass(frozen=True)
class ModelConfig(DataClassJsonMixin):
    weights_file_name: str
    inference_size: int
    bbox_confidence_threshold: float
    nms_iou_threshold: float
    key_points_mapping: Dict[int, str] = field(
        metadata=config(
            encoder=decode_mapping,
        )
    )
    key_points_categories: Dict[int, str] = field(
        metadata=config(
            encoder=decode_mapping,
        )
    )


InferenceResultSerialised = List[dict]


@dataclass(frozen=True)
class PreProcessingContext:
    original_height: int
    original_width: int
    left_pad_pct: float
    top_pad_pct: float
    horizontal_pad_pct: float
    vertical_pad_pct: float


@dataclass(frozen=True)
class RawInferenceResults:
    boxes: List[List[int]]
    boxes_scores: List[float]
    boxes_key_points: List[torch.Tensor]


@dataclass(frozen=True)
class PointCoordinates(DataClassJsonMixin):
    x: float
    y: float


@dataclass(frozen=True)
class BoundingBox(DataClassJsonMixin):
    object_category: str
    confidence: float
    left_top: PointCoordinates
    right_bottom: PointCoordinates


@dataclass(frozen=True)
class KeyPoint(DataClassJsonMixin):
    index: int
    point_class: str
    point_category: str
    confidence: float
    point_coordinates: PointCoordinates


@dataclass(frozen=True)
class Detection(DataClassJsonMixin):
    bounding_box: BoundingBox
    key_points: List[KeyPoint]
