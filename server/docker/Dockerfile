FROM docker.io/nvidia/cuda:12.0.0-base-ubuntu20.04
ARG MODEL_FAMILY=yolov8
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y software-properties-common
RUN add-apt-repository -y ppa:deadsnakes/ppa && \
    apt update -y && \
    apt install -y python3.9 python3-pip python3.9-distutils openjdk-11-jre-headless python3-opencv
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

RUN mkdir build_dependencies
COPY requirements/requirements.txt build_dependencies/requirements.txt
COPY requirements/requirements-torchserve.txt build_dependencies/requirements-torchserve.txt
RUN python -m pip install --upgrade pip setuptools distlib
RUN python -m pip install -r build_dependencies/requirements-torchserve.txt
RUN python -m pip install -r build_dependencies/requirements.txt
RUN rm -r build_dependencies

ENV BASE_DIR=/opt/ml/torchserve
ENV SERVING_LOGIC_DIR=$BASE_DIR/serving_logic
ENV SERVING_CONFIG_DIR=$BASE_DIR/serving_config
ENV LOG_LOCATION=$BASE_DIR/logs
ENV MODEL_DIR=/opt/ml/torchserve/model
RUN mkdir -p $MODEL_DIR $LOG_LOCATION $SERVING_CONFIG_DIR $SERVING_LOGIC_DIR
COPY model_packages/${MODEL_FAMILY} $MODEL_DIR
COPY serving_config $SERVING_CONFIG_DIR
COPY serving_logic $SERVING_LOGIC_DIR
RUN chmod ugo+x $SERVING_CONFIG_DIR/prepare_serving.sh

WORKDIR $SERVING_CONFIG_DIR
RUN ./prepare_serving.sh

WORKDIR $BASE_DIR

CMD ["torchserve", "--start", "--ts-config", "/opt/ml/torchserve/serving_config/config.properties", "--foreground"]
