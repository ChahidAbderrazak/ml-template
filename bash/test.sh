#!/bin/bash
# --------------- get teh .env variables ---------------
echo " ==> [env] Importing the variables..." && echo ""
. .env
DOCKER_IMG="${IMG_BUILDER}:${VERSION}"
# ------------------- run dockers ---------------------
echo "root=$(pwd)"
echo " ==> [Testing][Docker][${PROJECT_NAME}] running the code Unit-tests   ..." && echo ""

docker run -it --rm \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/tests:/app/tests \
  "${DOCKER_IMG}" sh -c " cd src && python -m pytest"

  #  "${DOCKER_IMG}" sh -c "python -m pytest > /app/logs/pytest.out"
