from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="soubhagya264",
    description="A package for creating a Caption_Bot project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Image_cap_visually_impared",
    author_email="nsoubhagya264@gmail.com",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        "dvc",
        "tensorflow",
        "matplotlib",
        "numpy",
        "tqdm",
        "PyYAML",
        "opencv-python",
    ]
)