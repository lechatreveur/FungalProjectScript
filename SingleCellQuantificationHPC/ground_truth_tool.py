#!/usr/bin/env python3
"""
Ground-Truth Cell Tracking & Correction Tool Entry Point.
3-Keyframe Curator & Direct Cellpose Training Data Exporter.
"""

import sys
from pathlib import Path

# Add package directory to python path
sys.path.insert(0, str(Path(__file__).parent))

from ground_truth_corrector.__main__ import main

if __name__ == "__main__":
    main()
