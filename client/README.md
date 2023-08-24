# YOLOv8 Pose Estimation client

## Overview
The library is a simple, yet functional client for [YOLOv8 Pose Estimation inference server](../server).
It allows for easy integration via HTTP to make inference on image files, `np.arrays`, streams of 
`np.arrays` and video files.

## Hitting the ground running

### Installation
```bash
repository_root/client$ pip install . 
```

### Minimal usage examples

```python
import numpy as np
import cv2 as cv

imprt
numpy as np
from yolov8_client.entities import Detection
from yolov8_client.client import YoloClient
from yolov8_client.visualisation import annotate_image

client = YoloClient()

# Checking server status
client.check_service_status()
>> "Healthy"

# Prediction on image file
image, prediction = client.predict_on_item(
    "path/to/image.jpg"
)
annotated_image = annotate_image(image=image, image_detection=prediction)

# Prediction on image
image = cv.imread("path/to/image.jpg")
_, prediction = client.predict_on_item(image)
annotated_image = annotate_image(image=image, image_detection=prediction)

# Prediction on images stream
stream = (image for _ in range(1000))
for image, prediction in client.predict_on_stream(stream):
    ...

# Prediction on video file
for image, prediction in client.predict_on_stream("path/to/video.mp4"):
    ...


# transforming of predictions (works in each mode)

def my_transform(detection: Detection) -> tuple:
    # returns only BBoxes coordinates
    return detection.bounding_box.left_top_absolute.to_tuple() + \
        detection.bounding_box.right_bottom_absolute.to_tuple()


image, prediction = client.predict_on_item(
    "path/to/image.jpg",
    prediction_transform=my_transform,
)
np.array(prediction)
>> np.ndarray([
    [0, 10, 20, 30],
    [0, 10, 20, 30],
    [0, 10, 20, 30],
    [0, 10, 20, 30]
])
```

More usage examples [here](../examples)

## How to start development?
In order to start development, it is advised to create conda environment:
```bash
repository_root/client$ conda create -n YOLOv8-client python=3.9
```

Once this is done, dependencies should be installed.
```bash
repository_root/client$ conda actiavte YOLOv8-server
(YOLOv8-client) repository_root/client$ pip install \
  -r requirements-dev.txt \
  -r requirements.txt
```