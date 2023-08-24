from typing import List

import numpy as np
import cv2 as cv

from .entities import Detection, BoundingBox, KeyPoint

BBOX_COLOR = (255, 0, 0)
BBOX_THICKNESS = 3

KEY_POINTS_COLOURS = {
    "head": (0, 0, 255),
    "arms": (220, 0, 0),
    "corpus": (0, 255, 0),
    "legs": (128, 128, 0),
}
KEY_POINT_THICKNESS = 3
KEY_POINTS_LINE_THICKNESS = 2


CONNECTIONS = {
    "head": [(4, 2), (2, 0), (0, 1), (1, 3)],
    "arms": [
        (10, 8),
        (8, 6),
        (9, 7),
        (7, 5),
    ],
    "corpus": [(6, 5), (5, 11), (11, 12), (12, 6)],
    "legs": [(16, 14), (14, 12), (15, 13), (13, 11)],
}


def annotate_image(
    image: np.ndarray,
    image_detection: List[Detection],
    bbox_confidence_threshold: float = 0.3,
    key_point_confidence_threshold: float = 0.3,
) -> np.ndarray:
    image = image.copy()
    bboxes = [e.bounding_box for e in image_detection]
    image = draw_bboxes(
        image=image,
        bboxes=bboxes,
        threshold=bbox_confidence_threshold,
    )
    objects_key_points = [e.key_points for e in image_detection]
    return draw_key_points(
        image=image,
        objects_key_points=objects_key_points,
        threshold=key_point_confidence_threshold,
    )


def draw_bboxes(
    image: np.ndarray,
    bboxes: List[BoundingBox],
    threshold: float,
) -> np.ndarray:
    for bbox in bboxes:
        if bbox.confidence < threshold:
            continue
        left_top = bbox.left_top_absolute.to_tuple()
        right_bottom = bbox.right_bottom_absolute.to_tuple()
        image = cv.rectangle(
            image,
            left_top,
            right_bottom,
            BBOX_COLOR,
            BBOX_THICKNESS,
        )
    return image


def draw_key_points(
    image: np.ndarray,
    objects_key_points: List[List[KeyPoint]],
    threshold: float = 0.3,
) -> np.ndarray:
    image = image.copy()
    for key_points in objects_key_points:
        image = draw_object_key_points(
            image=image, key_points=key_points, threshold=threshold
        )
    return image


def draw_object_key_points(
    image: np.ndarray,
    key_points: List[KeyPoint],
    threshold: float = 0.3,
) -> np.ndarray:
    visible_key_points = {}
    for key_point in key_points:
        if key_point.confidence < threshold:
            continue
        point = key_point.point_coordinates_absolute.to_tuple()
        visible_key_points[key_point.index] = point
        image = cv.circle(
            image,
            point,
            radius=KEY_POINT_THICKNESS,
            color=KEY_POINTS_COLOURS[key_point.point_category],
            thickness=-1,
        )
    neck_visible = 6 in visible_key_points and 5 in visible_key_points
    neck_point = None
    if neck_visible:
        neck_point = (
            (visible_key_points[5][0] + visible_key_points[6][0]) // 2,
            (visible_key_points[5][1] + visible_key_points[6][1]) // 2,
        )
        image = cv.circle(
            image,
            neck_point,
            radius=KEY_POINT_THICKNESS,
            color=KEY_POINTS_COLOURS["corpus"],
            thickness=-1,
        )
    if neck_visible and 0 in visible_key_points:
        image = cv.line(
            image,
            neck_point,
            visible_key_points[0],
            KEY_POINTS_COLOURS["head"],
            KEY_POINTS_LINE_THICKNESS,
        )
    for category, connections in CONNECTIONS.items():
        for start, end in connections:
            if start not in visible_key_points or end not in visible_key_points:
                continue
            image = cv.line(
                image,
                visible_key_points[start],
                visible_key_points[end],
                KEY_POINTS_COLOURS[category],
                KEY_POINTS_LINE_THICKNESS,
            )
    return image
