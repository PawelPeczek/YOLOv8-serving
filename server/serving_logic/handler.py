import os
from typing import Optional, List, Dict, Any

import cv2
import numpy as np
import torch
from torch.jit import RecursiveScriptModule
from ts.context import Context

from constants import REQUEST_IMAGE_FIELD
from entities import ModelConfig, InferenceResultSerialised
from inference import infer
from serving_utils import extract_device_from_context, load_model_config, load_model


class Handler:
    def __init__(self):
        self.__model_config: Optional[ModelConfig] = None
        self.__model: Optional[RecursiveScriptModule] = None
        self.__device: Optional[torch.device] = None

    def initialize(self, context: Context) -> None:
        self.__model_config = load_model_config(context=context)
        self.__device = extract_device_from_context(context=context)
        self.__model = load_model(
            model_config=self.__model_config,
            context=context,
        )

    def handle(
        self, requests_body: List[Dict[str, Any]], context: Context
    ) -> List[InferenceResultSerialised]:
        self.__ensure_handler_initialised()
        self.__ensure_batch_size_equals_one(requests_body=requests_body)
        request_images = [
            decode_request(request_body=request_body) for request_body in requests_body
        ]
        inference_results = infer(
            model=self.__model,
            images=request_images,
            device=self.__device,
            model_config=self.__model_config,
        )
        return [
            [detection.to_dict() for detection in inference_result]
            for inference_result in inference_results
        ]

    def __ensure_handler_initialised(self) -> None:
        to_verify_not_empty = [self.__device, self.__model, self.__model_config]
        if any(e is None for e in to_verify_not_empty):
            raise RuntimeError("Handler not initialised correctly.")

    def __ensure_batch_size_equals_one(
        self, requests_body: List[Dict[str, Any]]
    ) -> None:
        if len(requests_body) > 1:
            raise NotImplementedError(
                "To fully correctly handle torchserve requests batching one must properly dispatch "
                "potential errors for specific responses - such that one faulty input does not destroy "
                "whole batch of inference by causing Exception. This is however something not tackled in "
                "official torchserve handlers (they also handle only batch_size=1 to my knowledge) - at the "
                "same time doable easily (just original author of this code is to lazy). To support that in the "
                "future - InferenceFunction accepts list of decoded images and returns list of inference results."
            )


def decode_request(request_body: Dict[str, Any]) -> np.ndarray:
    raw_image = request_body[REQUEST_IMAGE_FIELD]
    bytes_array = np.asarray(raw_image, dtype=np.uint8)
    return (cv2.imdecode(bytes_array, cv2.IMREAD_COLOR)[:, :, ::-1]).copy()
