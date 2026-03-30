#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 17:08:03 2025

@author: user
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from SingleCellDataAnalysis.config import WORKING_DIR

INPUT_CSV = os.path.join(WORKING_DIR, "gp_summary_features.csv")

# ==== Step 1: Load summary data ====
print(f"📥 Loading GP summary features from {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# ==== Step 2: Define feature pairs ====
summary_pairs = [
    ("pol1_mean", "pol2_mean"),
    ("pol1_slope", "pol2_slope"),
    ("pol1_p1", "pol2_p1"),
    ("pol1_p2", "pol2_p2"),
    ("pol1_p3", "pol2_p3"),
    ("pol1_a1", "pol2_a1"),
    ("pol1_a2", "pol2_a2"),
    ("pol1_a3", "pol2_a3"),
]

titles = [
    "Mean",
    "Slope",
    "Period 1",
    "Period 2",
    "Period 3",
    "Amplitude 1",
    "Amplitude 2",
    "Amplitude 3"
]

# ==== Step 3: Create 2x4 grid plot ====
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.flatten()

for i, ((f1, f2), title) in enumerate(zip(summary_pairs, titles)):
    ax = axs[i]
    ax.scatter(df[f1], df[f2], alpha=0.6)
    ax.set_xlabel(f"{f1}")
    ax.set_ylabel(f"{f2}")
    ax.set_title(title)
    ax.grid(True)

plt.tight_layout()
plt.suptitle("pol1 vs pol2: GP Summary Comparisons", fontsize=16, y=1.03)
plt.show()
