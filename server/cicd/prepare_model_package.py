import os
import shutil

from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n-pose.pt")
    model_path = model.export(format="torchscript")
    shutil.move(
        model_path,
        os.path.join("model_packages", "yolov8", os.path.basename(model_path)),
    )
    os.remove("yolov8n-pose.pt")
