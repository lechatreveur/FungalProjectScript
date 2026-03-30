#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 18:37:16 2026

@author: user
"""

import cv2
import numpy as np
from pathlib import Path
import re

T_RE = re.compile(r"_t_(\d{3})_c_0", re.IGNORECASE)
def t_index(p: Path) -> int:
    m = T_RE.search(p.stem)
    return int(m.group(1)) if m else 10**9

film = "A14-YES-1t-FBFBF_F0"
working_dir = Path("/Volumes/Movies/2025_12_31_M92")
frames_dir = working_dir / film / f"Frames_{film}"

out_mp4 = working_dir / film / f"{film}_stabilized_preview.mp4"
fps = 10  # change as you like

files = sorted(frames_dir.glob("*.tif"), key=t_index)
if not files:
    raise FileNotFoundError(f"No .tif files in {frames_dir}")

# read first to get size
first = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
if first is None:
    raise RuntimeError(f"Could not read {files[0]}")
if first.ndim == 2:
    h, w = first.shape
else:
    h, w = first.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(str(out_mp4), fourcc, fps, (w, h), isColor=True)
if not vw.isOpened():
    raise RuntimeError("Could not open VideoWriter (codec issue). Try changing fourcc to 'avc1' or use ffmpeg.")

for p in files:
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Could not read {p}")

    # convert to grayscale if needed
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # normalize 16-bit/float to 8-bit for video viewing
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, (1, 99.8))  # robust contrast
    g = np.clip((g - lo) / (hi - lo + 1e-6), 0, 1)
    g8 = (g * 255).astype(np.uint8)

    # MP4 typically expects 3-channel BGR
    bgr = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
    vw.write(bgr)

vw.release()
print("Wrote:", out_mp4)