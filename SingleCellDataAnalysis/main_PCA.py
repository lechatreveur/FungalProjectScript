#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 15:58:21 2025

@author: user
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_PCA.py - Perform PCA and UMAP on GP summary features
Created on Jun 20, 2025
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # Avoid BLAS over-threading issues

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from umap.umap_ import UMAP  # Avoid TensorFlow by importing only the light version
from SingleCellDataAnalysis.config import WORKING_DIR as WORK_DIR

# ==== Configuration ====
INPUT_CSV = os.path.join(WORK_DIR, "gp_summary_features.csv")
OUTPUT_PCA_CSV = os.path.join(WORK_DIR, "pca_transformed_features.csv")
N_COMPONENTS = 16
N_CLUSTERS = 4

# ==== Step 1: Load Data ====
print(f"📥 Loading data from {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

if "cell_id" in df.columns:
    cell_ids = df["cell_id"]
    X = df.drop(columns=["cell_id"])
else:
    cell_ids = None
    X = df.copy()

# ==== Step 2: Standardize Features ====
print("📊 Standardizing features...")
X_scaled = StandardScaler().fit_transform(X)

# ==== Step 3: PCA ====
print("🔍 Running PCA...")
pca = PCA(n_components=N_COMPONENTS)
X_pca = pca.fit_transform(X_scaled)

# ---- Scree Plot ----
explained_var = pca.explained_variance_ratio_
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(explained_var) + 1), explained_var, 'o-', linewidth=2, color='green')
plt.title("Scree Plot: Variance Explained by Each Principal Component")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.xticks(np.arange(1, len(explained_var) + 1))
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Cumulative Variance ----
cumulative_var = np.cumsum(explained_var)
print("📊 Cumulative variance explained:")
for i, var in enumerate(cumulative_var, start=1):
    print(f"  PC{i}: {var:.4f}")

# ==== Step 4: Save PCA Results ====
print(f"💾 Saving PCA results to {OUTPUT_PCA_CSV}")
df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(N_COMPONENTS)])
if cell_ids is not None:
    df_pca.insert(0, "cell_id", cell_ids)
df_pca.to_csv(OUTPUT_PCA_CSV, index=False)

# ==== Step 5: PCA Plot ====
print("📈 Plotting PCA scatter plot...")
plt.figure(figsize=(8, 6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], s=20, alpha=0.7)
plt.xlabel(f"PC1 ({explained_var[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({explained_var[1]*100:.1f}% variance)")
plt.title("PCA of GP Summary Features")
plt.grid(True)
plt.tight_layout()
plt.show()

print("✅ PCA completed.")

# ==== Step 6: K-Means Clustering ====
print(f"🤖 Running K-means with {N_CLUSTERS} clusters...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42)
clusters = kmeans.fit_predict(X_pca)
df_pca["cluster"] = clusters

# ---- K-Means Plot ----
plt.figure(figsize=(8, 6))
plt.scatter(df_pca["PC1"], df_pca["PC2"], c=clusters, cmap='tab10', s=30)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA + K-means Clustering")
plt.grid(True)
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.show()

# ==== Step 7: UMAP Projection ====
print("🧬 Running UMAP...")
umap_model = UMAP(random_state=42)
X_umap = umap_model.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], c=clusters, cmap='tab10', s=30)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP Projection Colored by K-means Clusters")
plt.grid(True)
plt.tight_layout()
plt.show()
