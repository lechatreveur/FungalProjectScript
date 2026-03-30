#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_summary_heatmap.py - Z-scored heatmap of 16 summary features across cells, clustered
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
from SingleCellDataAnalysis.config import WORKING_DIR

# === Load data ===
INPUT_CSV = os.path.join(WORKING_DIR, "gp_summary_features.csv")
print(f"📥 Loading GP summary features from {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# === Reassign pol1 to the higher-mean polarity ===
# List of suffixes shared between pol1 and pol2
suffixes = ['mean', 'slope', 'p1', 'p2', 'p3', 'a1', 'a2', 'a3']

# Identify rows where pol2_mean > pol1_mean
swap_mask = df["pol2_mean"] > df["pol1_mean"]

# Swap the values where necessary
for suffix in suffixes:
    pol1_col = f"pol1_{suffix}"
    pol2_col = f"pol2_{suffix}"
    
    # Temporarily store pol1 values
    tmp = df.loc[swap_mask, pol1_col].copy()
    
    # Swap
    df.loc[swap_mask, pol1_col] = df.loc[swap_mask, pol2_col]
    df.loc[swap_mask, pol2_col] = tmp

print(f"🔁 Swapped polarities in {swap_mask.sum()} cells where pol2 had a higher mean.")

# Define threshold
AMP_THRESHOLD = 20.0

# === Construct binary-weighted period features ===
feature_data = pd.DataFrame({
    "pol1_wp1": df["pol1_p1"] * (df["pol1_a1"] >= AMP_THRESHOLD),
    "pol2_wp1": df["pol2_p1"] * (df["pol2_a1"] >= AMP_THRESHOLD),
    "pol1_wp2": df["pol1_p2"] * (df["pol1_a2"] >= AMP_THRESHOLD),
    "pol2_wp2": df["pol2_p2"] * (df["pol2_a2"] >= AMP_THRESHOLD),
    "pol1_wp3": df["pol1_p3"] * (df["pol1_a3"] >= AMP_THRESHOLD),
    "pol2_wp3": df["pol2_p3"] * (df["pol2_a3"] >= AMP_THRESHOLD),
    "pol1_slope": df["pol1_slope"],
    "pol2_slope": df["pol2_slope"],
    "pol1_mean": df["pol1_mean"],
    "pol2_mean": df["pol2_mean"]
})



# === Standardize pol1 and pol2 pairs together ===
feature_data_scaled = feature_data.copy()

# List of variable bases to normalize jointly
paired_bases = ["wp1", "wp2", "wp3", "slope", "mean"]

for base in paired_bases:
    col1 = f"pol1_{base}"
    col2 = f"pol2_{base}"

    # Stack both columns together to compute global mean and std
    combined = pd.concat([feature_data[col1], feature_data[col2]])
    mean = combined.mean()
    std = combined.std()

    # Apply same normalization to both columns
    feature_data_scaled[col1] = (feature_data[col1] - mean) / std
    feature_data_scaled[col2] = (feature_data[col2] - mean) / std

# Clustering

X_scaled = feature_data_scaled.values  # Already normalized
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df["cluster"] = clusters

# Clustered heatmap
sns.clustermap(
    feature_data_scaled.assign(cluster=clusters).sort_values("cluster").drop("cluster", axis=1),
    row_cluster=True, col_cluster=False,
    cmap="vlag", vmin=-3, vmax=3,
    figsize=(12, 10), yticklabels=False,
    cbar_kws={"label": "Z-score"}
)
plt.suptitle("Clustered Heatmap of Standardized Period Features (Paired Normalization)")
plt.tight_layout()
plt.show()

#%%
