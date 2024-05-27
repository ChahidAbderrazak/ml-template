import os
import setuptools

# open README.md file and read the content into a variable
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ =  "0.0.0"
SRC_REPO = "inspect-road"
GITHUB_REPO_NAME = "Road-conditions-inspection"
GITHUB_USER_NAME = "ChahidAbderrazak"
SHORT_DESCRIPTION = " Classification model for Xray CT data defect detection."
AUTHOR_EMAIL = "abderrazak.chahid@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=GITHUB_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description=SHORT_DESCRIPTION,
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{GITHUB_USER_NAME}/{GITHUB_REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{GITHUB_USER_NAME}/{GITHUB_REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
