#!/usr/bin/env python3
"""
Septum Alignment Board Entry Point.
Delegates server execution and management to the modular `septum_alignment_board` package.
"""

import sys
from pathlib import Path

# Add package directory to python path
sys.path.insert(0, str(Path(__file__).parent))

from septum_alignment_board.__main__ import main

if __name__ == "__main__":
    main()
