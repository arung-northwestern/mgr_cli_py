"""
Setup script for MCLI - MOF Analysis CLI Tool
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README if it exists
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="mcli",
    version="0.1.0",
    author="MOF Analysis Team",
    author_email="",
    description="A command-line interface for analyzing Metal-Organic Frameworks (MOFs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    keywords="mof metal-organic-frameworks materials-analysis cli",
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "mcli=mcli.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
