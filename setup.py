# setup.py
from setuptools import setup, find_packages

setup(
    name="segmetrics",
    version="0.1.0",
    author="Maximilian Kotz",
    description="A library for computing metrics on image segmentations",
    packages=find_packages(),  # Automatically finds packages in the project
    install_requires=[
        "numpy",
        "numba",
    ],
    python_requires=">=3.9",
)
