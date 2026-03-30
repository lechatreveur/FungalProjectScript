#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 18:01:47 2025

@author: user
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from SingleCellDataAnalysis.config import WORKING_DIR

# === Load data ===
INPUT_CSV = os.path.join(WORKING_DIR, "gp_summary_features.csv")
print(f"📥 Loading GP summary features from {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# === Parameters to pool ===
param_keys = ['mean', 'slope', 'p1', 'p2', 'p3', 'a1', 'a2', 'a3']
AMP_THRESHOLD = 20.0  # ⬅️ threshold for filtering periods based on amplitude

# === Create 2x4 subplots ===
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.flatten()

for i, key in enumerate(param_keys):
    pol1_col = f"pol1_{key}"
    pol2_col = f"pol2_{key}"

    if pol1_col in df.columns and pol2_col in df.columns:
        if key.startswith('p'):  # apply amplitude filter to period columns
            amp1_col = f"pol1_a{key[-1]}"
            amp2_col = f"pol2_a{key[-1]}"
            # Ensure amplitude columns exist
            if amp1_col in df.columns and amp2_col in df.columns:
                pol1_vals = df.loc[df[amp1_col] >= AMP_THRESHOLD, pol1_col]
                pol2_vals = df.loc[df[amp2_col] >= AMP_THRESHOLD, pol2_col]
                pooled_values = pd.concat([pol1_vals, pol2_vals])
            else:
                axs[i].set_visible(False)
                print(f"⚠️ Missing amplitude columns: {amp1_col} or {amp2_col}")
                continue
        else:
            # No filtering for non-period parameters
            pooled_values = pd.concat([df[pol1_col], df[pol2_col]])

        sns.histplot(pooled_values, bins=40, kde=True, ax=axs[i], color="steelblue")
        axs[i].set_title(f"Distribution of pooled {key}")
        axs[i].set_xlabel(key)
    else:
        axs[i].set_visible(False)
        print(f"⚠️ Missing columns: {pol1_col} or {pol2_col}")

plt.suptitle("Distribution of Pooled Parameters (pol1 + pol2, filtered periods)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
