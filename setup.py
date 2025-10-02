#!/usr/bin/env python3
"""
Setup script for MCLI - MOF Analysis CLI Tool with Rust extension
"""

from setuptools import setup, find_packages
import os

# Import setuptools_rust conditionally
try:
    from setuptools_rust import RustExtension, Binding
except ImportError:
    RustExtension = None
    Binding = None

# Get the absolute path to the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))

# Read README if it exists
readme_path = os.path.join(current_dir, "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="mcli",
    version="0.1.0",
    author="MOF Analysis Team",
    description="A command-line interface for analyzing Metal-Organic Frameworks (MOFs)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="python_cli"),
    package_dir={"": "python_cli"},
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
    setup_requires=["setuptools-rust"],
    # Conditionally compile the Rust extension
    rust_extensions=[
        RustExtension(
            "mcli_rust",
            path="Cargo.toml",
            binding=Binding.PyO3,
            debug=False,
            # Enable additional optimizations for better performance
            rustc_flags=["-C", "target-cpu=native", "-C", "opt-level=3"],
        )
    ] if RustExtension else [],
    entry_points={
        "console_scripts": [
            "mcli=mcli.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)