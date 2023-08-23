python cicd/prepare_model_package.py
docker build -f ./docker/Dockerfile -t yolov8-serving .
