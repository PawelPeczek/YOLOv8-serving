# YOLOv8 Pose Estimation server

## Overview
Server is created based on two components
* YOLOv8 model (pose-estimation) trained under [ultralytics](https://github.com/ultralytics/ultralytics) 
repository and **exported to TorchScript** (see [export script](./cicd/prepare_model_package.py))
* [TorchServe](https://pytorch.org/serve/) server for exposing model predictions via HTTP

## Interface

### `POST /predictions/yolov8_pose`

#### Accept
Multipart body parameter: `image` - JPEG encoded image bytes

#### Returns
JSON document
```json
[
  {
      "bounding_box": {"object_category": "person",
      "confidence": 0.9118603467941284,
      "left_top": {"x": 0.03125, "y": 0.12999999999999998},
      "right_bottom": {"x": 0.19375, "y": 0.255}},
     "key_points": [
       {
            "index": 0,
            "point_class": "nose",
            "point_category": "head",
            "confidence": 0.9863465428352356,
            "point_coordinates": {"x": 0.1421875, "y": 0.195}
       },
       ...
      {
            "index": 16,
            "point_class": "right-ankle",
            "point_category": "legs",
            "confidence": 0.986329972743988,
            "point_coordinates": {"x": 0.0703125, "y": 0.7324999999999999}
      }
     ]
  }
]
```

Where:
* `bounding_box` describes BBox with detected object
* `key_points` describes detected key-points within `bounding_box` object

Important notes:
* Low score bounding boxes are filtered out according to [configuration](./model_packages/yolov8/config.json)
* NMS is applied with IOU threshold according to [configuration](./model_packages/yolov8/config.json)
* key-points index -> class resolution is denoted in according to [configuration](./model_packages/yolov8/config.json)
* each point coordinate is returned as value in range [0.0, 1.0] - representing position in % of input image resolution

#### Minimal client code in Python
```python
import cv2 as cv
import requests

SERVICE_URL = "http://127.0.0.1:8080"

def make_request(image):
    encode_param = [int(cv.IMWRITE_JPEG_QUALITY), 90]
    result, encoded_image = cv.imencode('.jpg', image, encode_param)
    response = requests.post(
        f"{SERVICE_URL}/predictions/yolov8_pose",
        files={"image": encoded_image}
    )
    return response.json()
```

### `GET /ping`
Endpoint to check service health.

## Hitting the ground running
In `server` subdirectory run:
```bash
repository_root/server$ ./cicd/build.sh

# FOR GPU INFERENCE
repository_root/server$ ./cicd/run_gpu.sh

# FOR CPU INFERENCE
repository_root/server$ ./cicd/run_cpu.sh
```

## How to start development?
In order to start development, it is advised to create conda environment:
```bash
repository_root/server$ conda create -n YOLOv8-server python=3.9
```

Once this is done, dependencies should be installed.
```bash
repository_root/server$ conda actiavte YOLOv8-server
(YOLOv8-server) repository_root/server$ pip install \
  -r requirements/requirements-dev.txt \
  -r requirements/requirements.txt \
  -r requirements/requirements-torchserve.txt
```

## How to navigate in repository structure?

### `cicd` directory
In this directory, there are set of scripts to prepare, build and run the service. There is 
a [script](./cicd/prepare_model_package.py)) to download and export selected YOLO model (manipulate
version of YOLO model to deploy different variants if you wish).

### `docker` directory
Directory with dockerfile of service.

### `model_packages` directory
Directory where model weights and configs for inference are stored. The content is automatically
prepared by build script (to allow `*.mar` TorchServe package creation while build).

### `requirements` directory
Directory with requirements files - decoupled by the type of dependencies.

### `serving_config` directory
Contains configs determining TorchServe behaviour and script to prepare serving `*.mar` package
required by TorchServe.

### `serving_logic` directory
Python package with TorchServe handler and inference logic. This code and model package are wrapped 
together into `*.mar` package, embedded into docker image and loaded by TorchServe HTTP server to
expose predictions.

## FAQ

### Changes in my code are not visible in service! How to fix that?
TorchServe requires statically compiled `*.mar` package containing model and inference handler.
Once you make changes into code - you need to run:
```bash
repository_root/server$ ./cicd/build.sh
```
for changes to be reflected in docker image.

## Limitations
* Batching on the server side is not enabled - handler is only adjusted to accept `batch_size=1`. It does not
mean, however, that clients cannot make batch requests on their side (by sending requests in parallel). The ultimate
solution would be to batch on server side (requires careful error handling) and on client side.
* Inference endpoint only accepts JPEG images
