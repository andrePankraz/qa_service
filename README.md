<!---
This file was created by ]init[ AG 2022.
-->

# QA Service

Solve Question Answering (QA) problems with Natural Language Processing (NLP) from AI.

The tests are performed in a Docker container that also works in the Windows Subsystem for Linux (WSL).
An NVIDIA graphics card with at least 4 GB VRAM is recommended, depending on the models used.
CUDA is part of the Docker image, only the NVIDIA graphics driver needs to be installed.

Docker must have CUDA enabled (e.g. for WSL see https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

## Start as local service with Test-UI

- Clone https://github.com/andrePankraz/qa_service

      $ cd qa_service
      $ docker compose up

  - Will take some time at first start (images & packages are downloaded, >10 GB)
  - Wait & check if up and running
- Go to URL: http://localhost:8200/
  - Will take some time at first start (models are downloaded, several GB)

## Start for Development

- Clone https://github.com/andrePankraz/qa_service

      $ cd qa_service
      $ docker compose --env-file docker/.envs/dev.env up

  - Will take some time at first start (images & packages are downloaded, >10 GB)
  - Wait & check if up and running
- Install [VS Code](https://code.visualstudio.com/)
  - Install Extension
    - Dev Containers
    - Docker
    - Markdown All in One
- Attach VS Code to Docker Container
  - Attach to running containers... (Lower left edge in VS Code)
    - select qa_service-python-1
  - Explorer Open folder -> /opt/qa_service
  - Run / Start Debug
    - VS Code Extension Python will be installed the first time (Wait and another Start Debug)
    - Select Python Interpreter
- Go to URL: http://localhost:8200/
  - Will take some time at first start (models are downloaded, several GB)
