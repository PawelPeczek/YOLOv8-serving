from functools import partial
from multiprocessing.pool import ThreadPool
from typing import List, Tuple, Iterable, Generator, Union, Optional, Callable

import numpy as np
import cv2 as cv
import requests

from .entities import Detection, RawPredictionResponse, T

DEFAULT_SERVICE_URL = "http://127.0.0.1:8080"
PREDICTIONS_ENDPOINT = "/predictions/yolov8_pose"
PING_ENDPOINT = "/ping"
IMAGE_FIELD = "image"
DEFAULT_ENCODING_QUALITY = 90
MAX_WORKERS = 16


class YoloClient:
    def __init__(
        self,
        service_url: str = DEFAULT_SERVICE_URL,
        encoding_quality: int = DEFAULT_ENCODING_QUALITY,
        max_workers_for_parallel_calls: int = MAX_WORKERS,
    ):
        self.__service_url = service_url
        self.__encoding_quality = encoding_quality
        self.__max_workers_for_parallel_calls = max_workers_for_parallel_calls

    def predict_on_item(
        self,
        source: Union[str, np.ndarray],
        prediction_transform: Optional[Callable[[Detection], T]] = None,
    ) -> Tuple[np.ndarray, List[Union[Detection, T]]]:
        if issubclass(type(source), str):
            if source.split(".")[-1].lower() in {"jpg", "jpeg"}:
                return self.predict_on_image_file(
                    file_path=source,
                    prediction_transform=prediction_transform,
                )
            else:
                raise ValueError("Only JPEG files allowed")
        return self.predict_on_image(
            image=source,
            prediction_transform=prediction_transform,
        )

    def predict_on_stream(
        self,
        source: Union[str, Iterable[np.ndarray]],
        prediction_transform: Optional[Callable[[Detection], T]] = None,
    ) -> Generator[List[Union[Detection, T]], None, None]:
        if issubclass(type(source), str):
            yield from self.predict_on_video_file(
                file_path=source,
                prediction_transform=prediction_transform,
            )
        else:
            yield from self.predict_on_images(
                stream=source,
                prediction_transform=prediction_transform,
            )

    def predict_on_image_file(
        self,
        file_path: str,
        prediction_transform: Optional[Callable[[Detection], T]] = None,
    ) -> Tuple[np.ndarray, List[Union[Detection, T]]]:
        image = cv.imread(file_path)
        return self.predict_on_image(
            image=image,
            prediction_transform=prediction_transform,
        )

    def predict_on_video_file(
        self,
        file_path: str,
        prediction_transform: Optional[Callable[[Detection], T]] = None,
    ) -> Generator[Tuple[np.ndarray, List[Union[Detection, T]]], None, None]:
        decoded_stream = decode_video(file_path=file_path)
        yield from self.predict_on_images(
            stream=decoded_stream,
            prediction_transform=prediction_transform,
        )

    def predict_on_image(
        self,
        image: np.ndarray,
        prediction_transform: Optional[Callable[[Detection], T]] = None,
    ) -> Tuple[np.ndarray, List[Union[Detection, T]]]:
        prediction = get_prediction(
            image=image,
            service_url=self.__service_url,
            encoding_quality=self.__encoding_quality,
        )
        if prediction_transform is None:
            return prediction
        return prediction[0], [
            e.transform(transformation=prediction_transform) for e in prediction[1]
        ]

    def predict_on_images(
        self,
        stream: Iterable[np.ndarray],
        prediction_transform: Optional[Callable[[Detection], T]] = None,
    ) -> Generator[Tuple[np.ndarray, List[Union[Detection, T]]], None, None]:
        for prediction in get_predictions(
            images=stream,
            service_url=self.__service_url,
            encoding_quality=self.__encoding_quality,
            max_workers=self.__max_workers_for_parallel_calls,
        ):
            if prediction_transform is None:
                yield prediction
            else:
                yield prediction[0], [
                    e.transform(transformation=prediction_transform)
                    for e in prediction[1]
                ]


def check_service_health(service_url: str) -> str:
    response = requests.get(f"{service_url}{PING_ENDPOINT}")
    response.raise_for_status()
    response_content = response.json()
    return response_content["status"]


def get_predictions(
    images: Iterable[np.ndarray],
    service_url: str,
    encoding_quality: int = DEFAULT_ENCODING_QUALITY,
    max_workers: int = MAX_WORKERS,
) -> Generator[Tuple[np.ndarray, List[Detection]], None, None]:
    with ThreadPool(processes=max_workers) as pool:
        yield from pool.imap(
            func=partial(
                get_prediction,
                service_url=service_url,
                encoding_quality=encoding_quality,
            ),
            iterable=images,
        )


def get_prediction(
    image: np.ndarray,
    service_url: str,
    encoding_quality: int = 90,
) -> Tuple[np.ndarray, List[Detection]]:
    encoded_image = encode_image_to_jpg(
        image=image,
        quality=encoding_quality,
    )
    response = requests.post(
        f"{service_url}{PREDICTIONS_ENDPOINT}", files={IMAGE_FIELD: encoded_image}
    )
    response.raise_for_status()
    raw_prediction = response.json()
    return image, parse_response(
        raw_response=raw_prediction, image_size=image.shape[:2]
    )


def encode_image_to_jpg(image: np.ndarray, quality: int) -> bytes:
    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
    result, encoded_image = cv.imencode(".jpg", image, encode_param)
    if not result:
        raise ValueError(f"Could not encode image to JPEG with quality {quality}")
    return encoded_image.tobytes()


def parse_response(
    raw_response: List[RawPredictionResponse],
    image_size: Tuple[int, int],
) -> List[Detection]:
    results = []
    for response_detection in raw_response:
        response_detection["image_size"] = image_size
        response_detection["bounding_box"]["left_top_relative"] = response_detection[
            "bounding_box"
        ]["left_top"]
        response_detection["bounding_box"][
            "right_bottom_relative"
        ] = response_detection["bounding_box"]["right_bottom"]
        del response_detection["bounding_box"]["left_top"]
        del response_detection["bounding_box"]["right_bottom"]
        response_detection["bounding_box"][
            "left_top_absolute"
        ] = relative_coordinates_to_absolute(
            relative_coordinates_raw=response_detection["bounding_box"][
                "left_top_relative"
            ],
            image_size=image_size,
        )
        response_detection["bounding_box"][
            "right_bottom_absolute"
        ] = relative_coordinates_to_absolute(
            relative_coordinates_raw=response_detection["bounding_box"][
                "right_bottom_relative"
            ],
            image_size=image_size,
        )
        for keypoint in response_detection["key_points"]:
            keypoint["point_coordinates_relative"] = keypoint["point_coordinates"]
            del keypoint["point_coordinates"]
            keypoint["point_coordinates_absolute"] = relative_coordinates_to_absolute(
                relative_coordinates_raw=keypoint["point_coordinates_relative"],
                image_size=image_size,
            )
        results.append(Detection.from_dict(response_detection))
    return results


def relative_coordinates_to_absolute(
    relative_coordinates_raw: dict,
    image_size: Tuple[int, int],
) -> dict:
    return {
        "x": round_to_int(relative_coordinates_raw["x"] * image_size[1]),
        "y": round_to_int(relative_coordinates_raw["y"] * image_size[0]),
    }


def round_to_int(value: float) -> int:
    return int(round(value, 0))


def decode_video(file_path: str) -> Generator[np.ndarray, None, None]:
    stream = cv.VideoCapture(file_path)
    while stream.isOpened():
        status, frame = stream.read()
        if not status:
            break
        yield frame
    stream.release()
