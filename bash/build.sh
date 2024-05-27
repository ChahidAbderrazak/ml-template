 #!/bin/bash
# --------------- get teh .env variables ---------------
echo " ==> [env] Importing the variables..." && echo ""
. .env
DOCKER_IMG="${IMG_BUILDER}:${VERSION}"
# ------------------- build dockers ---------------------
docker system prune

echo " ==> [Docker][${PROJECT_NAME}] Building the docker image ${DOCKER_IMG} ..." && echo ""
docker build -t "${DOCKER_IMG}" .

