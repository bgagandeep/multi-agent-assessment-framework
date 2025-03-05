#!/usr/bin/env python3
"""
Setup script for the Multi-Agent Assessment Framework.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="multi-agent-assessment-framework",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A framework for evaluating and comparing multi-agent frameworks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multi-agent-assessment-framework",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "maaf=src.main:main",
        ],
    },
) 