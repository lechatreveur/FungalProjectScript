#!/usr/bin/env python3
"""
Fungal Cell Tracking Correction Tool Entry Point.
Delegates server execution and management to the modular `tracking_corrector` package.
"""

import sys
from pathlib import Path

# Add package directory to python path
sys.path.insert(0, str(Path(__file__).parent))

from tracking_corrector.__main__ import main

if __name__ == "__main__":
    main()
