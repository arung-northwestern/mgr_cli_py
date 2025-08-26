#!/usr/bin/env python3
"""
Standalone MCLI entry point script
"""

import sys
import os

# Add the python_cli directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
python_cli_dir = os.path.join(current_dir, 'python_cli')
sys.path.insert(0, python_cli_dir)

# Import and run the CLI
from mcli.cli import main

if __name__ == '__main__':
    main()
