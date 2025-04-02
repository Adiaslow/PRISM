#!/usr/bin/env python3
# setup.py

from setuptools import find_packages, setup

setup(
    name="prism-molecular",
    version="0.1.0",
    description="Parallel Recursive Isomorphism Search for Molecules",
    author="PRISM Team",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "networkx>=2.6.0",
        "rdkit>=2021.03.1",
        "tqdm>=4.61.0",
        "numba>=0.53.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
