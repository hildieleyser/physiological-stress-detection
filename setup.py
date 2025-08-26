#!/usr/bin/env python3
"""
Setup script for Physiological Stress Detection package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="physiological-stress-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced machine learning pipeline for physiological stress detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/physiological-stress-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "notebook>=6.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "stress-detection=src.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    keywords=[
        "machine-learning",
        "physiological-signals",
        "stress-detection",
        "time-series",
        "wearable-sensors",
        "shap",
        "interpretability",
        "ensemble-learning",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/physiological-stress-detection/issues",
        "Source": "https://github.com/yourusername/physiological-stress-detection",
        "Documentation": "https://github.com/yourusername/physiological-stress-detection#readme",
    },
) 