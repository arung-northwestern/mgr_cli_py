#!/usr/bin/env python3
"""
MCLI Wrapper for PyInstaller builds (numpy version)
"""
import sys
import os

# Add the current directory to Python path for PyInstaller
if getattr(sys, '_MEIPASS', False):
    # Running in PyInstaller bundle
    bundle_dir = sys._MEIPASS
    sys.path.insert(0, bundle_dir)
    sys.path.insert(0, os.path.join(bundle_dir, 'mcli'))

# Import and run the main CLI
from mcli.cli import main

if __name__ == '__main__':
    main()
