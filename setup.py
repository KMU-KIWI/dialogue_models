from setuptools import setup
from setuptools import find_packages

setup(
    name="train",
    package_dir={"": "src"},
    packages=find_packages("src"),
    version="0.0.1",
    description="models for emotion recognition and generation",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
