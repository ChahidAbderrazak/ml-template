#!/bin/bash
clear
# --------------- get teh .env variables ---------------
echo " ==> [env] Importing the variables..." && echo ""
. .env
DOCKER_IMG="${IMG_BUILDER}:${VERSION}"
# ------------------- run dockers ---------------------
echo " ==> [Docker] running the ${PROJECT_NAME} ..." && echo ""
docker run --rm -it   \
    -p 8080:8080 \
    -v ./config:/app/config \
    -v ./data/download:/app/data/download \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    "${DOCKER_IMG}"

