# This file was created by ]init[ AG 2023.
#
# see https://hub.docker.com/r/nvidia/cuda/tags?page=1&name=11.8.0-cudnn8-devel-ubuntu22.0
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS base
VOLUME /tmp
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
  apt-get install -y git python3-pip
# see https://pytorch.org/get-started/locally/
RUN --mount=type=cache,id=custom-pip,target=/root/.cache/pip pip3 install \
  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
WORKDIR /opt/qa_service
COPY requirements.txt requirements.txt
RUN --mount=type=cache,id=custom-pip,target=/root/.cache/pip pip3 install -r requirements.txt

FROM base AS dev
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,id=custom-pip,target=/root/.cache/pip pip3 install -r requirements-dev.txt

FROM base AS local
WORKDIR /opt/qa_service/qa_service
CMD ["uvicorn", "qa_service:app", "--host", "0.0.0.0", "--port", "8200"]
