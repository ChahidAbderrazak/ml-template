#!/bin/bash
# --------------- get teh .env variables ---------------
echo " ==> [env] Importing the variables..." && echo ""
. .env
DOCKER_IMG="${IMG_BUILDER}:${VERSION}"
# ------------------- run dockers ---------------------
echo "root=$(pwd)"
echo " ==> [Dev][Docker][${PROJECT_NAME}] running the debugging codes  ..." && echo ""
# docker run --rm -it   \
#     -p 8080:8080 \
#     -v $(pwd)/config:/app/config \
# 		-v $(pwd)/src:/app/src \
# 		-v $(pwd)/src/static/json:/app/src/static/json \
#     -v $(pwd)/data/download:/app/data/download \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     -e DISPLAY=$DISPLAY \
#     "${DOCKER_IMG}"


docker run -it --rm "${DOCKER_IMG}" sh
