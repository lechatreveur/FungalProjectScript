#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_summary_analysis.py - Redefine pol1 by mean, filter by amplitude, then run pooled histograms and cluster heatmap
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import argparse
from SingleCellDataAnalysis.config import WORKING_DIR

# === Argument parsing ===
parser = argparse.ArgumentParser(description="Run summary feature analysis with amplitude threshold")
parser.add_argument("amp_threshold", type=float, nargs="?", default=1.0, help="Amplitude threshold (default=1.0)")
args = parser.parse_args()
AMP_THRESHOLD = args.amp_threshold

# === Load data ===
INPUT_CSV = os.path.join(WORKING_DIR, "gp_summary_features.csv")
print(f"📥 Loading GP summary features from {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# === Redefine pol1 as pole with higher mean ===
print("🔄 Redefining pol1 as pole with higher mean...")
swap_mask = df["pol2_mean"] > df["pol1_mean"]
for suffix in ['mean', 'slope', 'p1', 'p2', 'p3', 'a1', 'a2', 'a3']:
    col1 = f"pol1_{suffix}"
    col2 = f"pol2_{suffix}"
    df.loc[swap_mask, [col1, col2]] = df.loc[swap_mask, [col2, col1]].values

# === Pooled histograms (pol1 + pol2) ===
print("📊 Plotting pooled histograms...")
param_keys = ['mean', 'slope', 'p1', 'p2', 'p3', 'a1', 'a2', 'a3']
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
axs = axs.flatten()

for i, key in enumerate(param_keys):
    pol1_col = f"pol1_{key}"
    pol2_col = f"pol2_{key}"

    if key.startswith("p"):
        amp1_col = f"pol1_a{key[-1]}"
        amp2_col = f"pol2_a{key[-1]}"
        vals1 = df.loc[df[amp1_col] >= AMP_THRESHOLD, pol1_col]
        vals2 = df.loc[df[amp2_col] >= AMP_THRESHOLD, pol2_col]
        pooled = pd.concat([vals1, vals2])
    else:
        pooled = pd.concat([df[pol1_col], df[pol2_col]])

    sns.histplot(pooled, bins=40, kde=True, ax=axs[i], color="steelblue")
    axs[i].set_title(f"Distribution of {key}")
    axs[i].set_xlabel(key)

plt.suptitle(f"Distribution of Pooled Parameters (AMP ≥ {AMP_THRESHOLD})", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# === Clustered Heatmap ===
print("📡 Preparing clustered heatmap...")
def bin_weight(a): return (a >= AMP_THRESHOLD).astype(float)

feature_data = pd.DataFrame({
    "pol1_wp1": df["pol1_p1"] * bin_weight(df["pol1_a1"]),
    "pol2_wp1": df["pol2_p1"] * bin_weight(df["pol2_a1"]),
    "pol1_wp2": df["pol1_p2"] * bin_weight(df["pol1_a2"]),
    "pol2_wp2": df["pol2_p2"] * bin_weight(df["pol2_a2"]),
    "pol1_wp3": df["pol1_p3"] * bin_weight(df["pol1_a3"]),
    "pol2_wp3": df["pol2_p3"] * bin_weight(df["pol2_a3"]),
    "pol1_slope": df["pol1_slope"],
    "pol2_slope": df["pol2_slope"],
    "pol1_mean": df["pol1_mean"],
    "pol2_mean": df["pol2_mean"]
})


# === Z-score standardization across pol1/pol2 pairs ===
print("📏 Jointly standardizing pol1/pol2 pairs...")
zscored_feature_data = pd.DataFrame(index=feature_data.index)

for base in ["wp1", "wp2", "wp3", "slope", "mean"]:
    col1 = f"pol1_{base}"
    col2 = f"pol2_{base}"
    
    joint = pd.concat([feature_data[col1], feature_data[col2]])
    mean = joint.mean()
    std = joint.std()
    
    zscored_feature_data[col1] = (feature_data[col1] - mean) / std
    zscored_feature_data[col2] = (feature_data[col2] - mean) / std


# === KMeans clustering ===
print("🤖 Running KMeans...")
X_scaled = zscored_feature_data.values
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

# === Plot heatmap ===
print("🎨 Plotting heatmap...")
sns.clustermap(
    zscored_feature_data.assign(cluster=clusters).sort_values("cluster").drop(columns="cluster"),
    row_cluster=True, col_cluster=False,
    #standard_scale=1,
    cmap="vlag"mkzxkdj, vmin=-3, vmax=3,
    figsize=(12, 10),
    yticklabels=False
)
plt.suptitle(f"Clustered Heatmap (AMP ≥ {AMP_THRESHOLD})", fontsize=16)
plt.tight_layout()
plt.show()

print("✅ All analysis completed.")

