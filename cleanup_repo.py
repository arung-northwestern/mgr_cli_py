#!/usr/bin/env python3
"""
Repository cleanup script for MCLI
Removes build artifacts, cache files, and temporary test data
"""

import os
import shutil
import glob
from pathlib import Path

def remove_if_exists(path, description):
    """Remove file or directory if it exists"""
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
            print(f"🗂️  Removed directory: {description}")
        else:
            os.remove(path)
            print(f"📄 Removed file: {description}")
    else:
        print(f"⚪ Already clean: {description}")

def main():
    print("🧹 Cleaning up MCLI repository...")

    # Build artifacts and cache
    print("\n📦 Removing build artifacts and cache:")
    remove_if_exists("dist", "PyInstaller executables")
    remove_if_exists("build", "Python build artifacts")
    remove_if_exists("target", "Rust build artifacts")
    remove_if_exists("mcli.egg-info", "Python egg-info")
    remove_if_exists("python_cli/mcli.egg-info", "Nested Python egg-info")
    remove_if_exists("*.so", "Compiled Rust extensions")
    remove_if_exists("python_cli/*.so", "Nested compiled extensions")

    # Python cache
    print("\n🐍 Removing Python cache:")
    for pycache in glob.glob("**/__pycache__", recursive=True):
        remove_if_exists(pycache, f"Python cache: {pycache}")

    # Test files and temporary data
    print("\n🧪 Removing test files and temporary data:")
    remove_if_exists("test_standalone_npz.py", "Standalone numpy test file")
    remove_if_exists("test_cli.py", "CLI test file")
    remove_if_exists("build_and_test.py", "Build and test script")

    # Test configuration files
    test_configs = ["test_config.json", "test_config2.json", "test_config_h.json"]
    for config in test_configs:
        remove_if_exists(config, f"Test config: {config}")

    # Test output directories and result files
    print("\n📊 Removing test output and results:")
    test_dirs = ["test_h_param", "test_output2", "advanced_test", "batch_test", "custom_output"]
    for test_dir in test_dirs:
        remove_if_exists(test_dir, f"Test output directory: {test_dir}")

    # Individual result files
    result_files = glob.glob("*.npz") + glob.glob("**/results.npz")
    for result_file in result_files:
        remove_if_exists(result_file, f"Result file: {result_file}")

    # Generated documentation
    print("\n📚 Removing generated documentation:")
    remove_if_exists("README.html", "Generated HTML documentation")
    remove_if_exists("README_files", "Documentation assets")

    print("\n✅ Repository cleanup complete!")
    print("\n📋 Summary of what was kept:")
    print("  • Core source code (src/, python_cli/, setup.py, etc.)")
    print("  • Configuration files (requirements.txt, pyproject.toml)")
    print("  • Documentation (README.md)")
    print("  • Example CIF files (cifs/)")
    print("  • Essential config files")

if __name__ == "__main__":
    main()
