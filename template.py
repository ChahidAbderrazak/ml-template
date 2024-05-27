import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# list of files to be created in the root directory of the project.
list_of_files_dirs = [
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/utils/__init__.py",
    f"src/utils/common.py",
    f"src/config/__init__.py",
    f"src/config/configuration.py",
    f"src/pipeline/__init__.py",
    f"src/entity/__init__.py",
    f"src/entity/config_entity.py",
    f"src/constants/__init__.py",
    f"src/templates/index.html",
    f"src/static/files/",
    f"src/main.py",
    f"src/app.py",
    f"dev/trials.ipynb",
    f"tests/test_app.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
]

# create the directories and files in the root directory of the project.
for filepath in list_of_files_dirs:
    filepath = Path(filepath)
    directory, filename = os.path.split(filepath)

    if directory != "":
        os.makedirs(directory, exist_ok=True)

        logging.info(
            f"Creating directory; {directory} for the file: {filename}")

    if filename != '':
        if ((not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0)):

            with open(filepath, "w") as f:
                pass
                logging.info(
                    f"Creating empty file: {filepath} [size={os.path.getsize(filepath)}]")

        else:
            logging.info(f"{filename} is already exists")
