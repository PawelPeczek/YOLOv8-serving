from typing import Tuple, List

import cv2
import numpy as np
import torch
from torch import nn

from entities import (
    ModelConfig,
    PreProcessingContext,
    RawInferenceResults,
    BoundingBox,
    PointCoordinates,
    KeyPoint,
    Detection,
)


INFERENCE_SCALING = 255
BBOX_CATEGORY = "person"


def infer(
    model: nn.Module,
    images: List[np.ndarray],
    device: torch.device,
    model_config: ModelConfig,
) -> List[List[Detection]]:
    inference_batch, pre_processing_contexts = pre_process_images(
        images=images,
        model_config=model_config,
    )
    inference_batch = inference_batch.to(device)
    with torch.no_grad():
        batch_predictions = model(inference_batch)
    nms_results = run_nms_on_inference_batch(
        batch_predictions=batch_predictions.detach().to("cpu"),
        bbox_confidence_threshold=model_config.bbox_confidence_threshold,
        iou_threshold=model_config.nms_iou_threshold,
    )
    return post_process_batch_predictions(
        batch_inference_results=nms_results,
        pre_processing_contexts=pre_processing_contexts,
        model_config=model_config,
    )


def pre_process_images(
    images: List[np.ndarray],
    model_config: ModelConfig,
) -> Tuple[torch.Tensor, List[PreProcessingContext]]:
    pre_processed_images, contexts = [], []
    for image in images:
        pre_processed_image, context = pre_process_image(
            image=image,
            model_config=model_config,
        )
        pre_processed_images.append(pre_processed_image)
        contexts.append(context)
    return (
        torch.tensor(np.array(pre_processed_images)).permute(0, 3, 1, 2)
        / INFERENCE_SCALING,
        contexts,
    )


def pre_process_image(
    image: np.ndarray,
    model_config: ModelConfig,
) -> Tuple[np.ndarray, PreProcessingContext]:
    max_dim = max(image.shape[0], image.shape[1])
    pad_x = max(image.shape[0] - image.shape[1], 0)
    pad_y = max(image.shape[1] - image.shape[0], 0)
    pad_top = pad_y // 2
    pad_bottom = pad_y - pad_top
    pad_left = pad_x // 2
    pad_right = pad_x - pad_left
    pad_left_arr = np.zeros((image.shape[0], pad_left, 3), dtype=np.uint8)
    pad_right_arr = np.zeros((image.shape[0], pad_right, 3), dtype=np.uint8)
    pad_top_arr = np.zeros((pad_top, image.shape[1], 3), dtype=np.uint8)
    pad_bottom_arr = np.zeros((pad_bottom, image.shape[1], 3), dtype=np.uint8)
    padded_image = np.concatenate(
        [e for e in (pad_left_arr, image, pad_right_arr) if e.shape[1] != 0],
        axis=1,
    )
    padded_image = np.concatenate(
        [e for e in (pad_top_arr, padded_image, pad_bottom_arr) if e.shape[0] != 0],
        axis=0,
    )
    resized_image = cv2.resize(
        padded_image, (model_config.inference_size, model_config.inference_size)
    )
    context = PreProcessingContext(
        original_height=image.shape[0],
        original_width=image.shape[1],
        left_pad_pct=pad_left / max_dim,
        top_pad_pct=pad_top / max_dim,
        horizontal_pad_pct=pad_x / max_dim,
        vertical_pad_pct=pad_y / max_dim,
    )
    return resized_image, context


def run_nms_on_inference_batch(
    batch_predictions: torch.Tensor,
    bbox_confidence_threshold: float,
    iou_threshold: float,
) -> List[RawInferenceResults]:
    return [
        run_nms_on_image_predictions(
            image_predictions=image_predictions,
            bbox_confidence_threshold=bbox_confidence_threshold,
            iou_threshold=iou_threshold,
        )
        for image_predictions in batch_predictions
    ]


def run_nms_on_image_predictions(
    image_predictions: torch.Tensor,
    bbox_confidence_threshold: float,
    iou_threshold: float,
) -> RawInferenceResults:
    boxes = []
    key_points = []
    scores = []
    for i in range(image_predictions.shape[1]):
        box_score = image_predictions[4, i]
        if box_score < bbox_confidence_threshold:
            continue
        x, y, w, h = image_predictions[0:4, i].round().to(int).tolist()
        left_x = int(round((x - w / 2), 0))
        left_y = int(round((y - h / 2), 0))
        right_x = int(round((x + w / 2), 0))
        right_y = int(round((x + h / 2), 0))
        scores.append(box_score.item())
        boxes.append((left_x, left_y, right_x, right_y))
        key_points.append(image_predictions[5:, i])
    indices = cv2.dnn.NMSBoxes(boxes, scores, bbox_confidence_threshold, iou_threshold)
    selected_boxes = [boxes[i] for i in indices]
    selected_scores = [scores[i] for i in indices]
    selected_key_points = [key_points[i] for i in indices]
    return RawInferenceResults(
        boxes=selected_boxes,
        boxes_scores=selected_scores,
        boxes_key_points=selected_key_points,
    )


def post_process_batch_predictions(
    batch_inference_results: List[RawInferenceResults],
    pre_processing_contexts: List[PreProcessingContext],
    model_config: ModelConfig,
) -> List[List[Detection]]:
    return [
        post_process_image_predictions(
            inference_results=inference_results,
            pre_processing_context=pre_processing_context,
            model_config=model_config,
        )
        for inference_results, pre_processing_context in zip(
            batch_inference_results, pre_processing_contexts
        )
    ]


def post_process_image_predictions(
    inference_results: RawInferenceResults,
    pre_processing_context: PreProcessingContext,
    model_config: ModelConfig,
) -> List[Detection]:
    results = []
    ox_rescale = 1.0 - pre_processing_context.horizontal_pad_pct
    oy_rescale = 1.0 - pre_processing_context.vertical_pad_pct
    for bbox, bbox_score, bbox_key_points in zip(
        inference_results.boxes,
        inference_results.boxes_scores,
        inference_results.boxes_key_points,
    ):
        left_top = PointCoordinates(
            x=(
                bbox[0] / model_config.inference_size
                - pre_processing_context.left_pad_pct
            )
            / ox_rescale,
            y=(
                bbox[1] / model_config.inference_size
                - pre_processing_context.top_pad_pct
            )
            / oy_rescale,
        )
        right_bottom = PointCoordinates(
            x=(
                bbox[2] / model_config.inference_size
                - pre_processing_context.left_pad_pct
            )
            / ox_rescale,
            y=(
                bbox[3] / model_config.inference_size
                - pre_processing_context.top_pad_pct
            )
            / oy_rescale,
        )
        bounding_box = BoundingBox(
            object_category=BBOX_CATEGORY,
            confidence=bbox_score,
            left_top=left_top,
            right_bottom=right_bottom,
        )
        key_points = []
        for i in range(17):
            x, y = bbox_key_points[3 * i : 3 * i + 2].round().to(int).tolist()
            confidence = bbox_key_points[3 * i + 2].item()
            point_coordinates = PointCoordinates(
                x=(
                    x / model_config.inference_size
                    - pre_processing_context.left_pad_pct
                )
                / ox_rescale,
                y=(y / model_config.inference_size - pre_processing_context.top_pad_pct)
                / oy_rescale,
            )
            key_point = KeyPoint(
                index=i,
                point_class=model_config.key_points_mapping[i],
                point_category=model_config.key_points_categories[i],
                confidence=confidence,
                point_coordinates=point_coordinates,
            )
            key_points.append(key_point)
        detection = Detection(
            bounding_box=bounding_box,
            key_points=key_points,
        )
        results.append(detection)
    return results
