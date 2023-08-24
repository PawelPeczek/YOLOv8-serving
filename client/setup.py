import os.path
from distutils.core import setup
from typing import List

VERSION = "0.1.0"
README_PATH = os.path.join(os.path.dirname(__file__), "README.md")
REQUIREMENTS_PATH = os.path.join(os.path.dirname(__file__), "requirements.txt")
LICENSE_PATH = os.path.join(os.path.dirname(__file__), "..", "LICENSE")


def load_readme() -> str:
    with open(README_PATH) as f:
        return f.read()


def load_requirements() -> List[str]:
    with open(REQUIREMENTS_PATH) as f:
        return [line.strip() for line in f.readlines() if len(line.strip()) > 0]


def load_license() -> str:
    with open(LICENSE_PATH) as f:
        return f.read()


setup(
    name="yolov8_client",
    version=VERSION,
    description="Client for YOLOv8 Pose Estimation serving.",
    long_description=load_readme(),
    author="Paweł Pęczek",
    author_email="pawel.m.peczek@gmail.com",
    url="https://github.com/PawelPeczek/YOLOv8-serving",
    packages=["yolov8_client"],
    install_requires=load_requirements(),
    license=load_license(),
)
