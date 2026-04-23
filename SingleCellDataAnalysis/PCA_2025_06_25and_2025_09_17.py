#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA analysis of raw trajectories for 2025-06-25 and 2025-09-17 experiments.
Focuses on capturing temporal dynamics directly from corrected pole intensities.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from SingleCellDataAnalysis.PCA_utils import get_trajectories_from_dir, run_pca_workflow

#%% Define paths
EXP_DIR_JUNE = "/Volumes/X10 Pro/Movies/2025_06_25/"
EXP_DIR_SEPT = "/Volumes/X10 Pro/Movies/2025_09_17/"

TARGET_LEN = 100 # 100 points per pole

#%% Load June 25 Data
print(f"Loading data from June 25: {EXP_DIR_JUNE}")
X_june, metadata_june = get_trajectories_from_dir(EXP_DIR_JUNE, target_len=TARGET_LEN)
metadata_june['experiment'] = '2025_06_25'
print(f"Loaded {X_june.shape[0]} cells from June 25.")

#%% Load September 17 Data
print(f"Loading data from Sept 17: {EXP_DIR_SEPT}")
X_sept, metadata_sept = get_trajectories_from_dir(EXP_DIR_SEPT, target_len=TARGET_LEN)
metadata_sept['experiment'] = '2025_09_17'
print(f"Loaded {X_sept.shape[0]} cells from Sept 17.")

#%% Combine Datasets
X_combined = np.vstack([X_june, X_sept])
metadata_combined = pd.concat([metadata_june, metadata_sept], ignore_index=True)
print(f"Combined dataset shape: {X_combined.shape}")

#%% Run PCA
print("Running PCA on raw trajectories (shape: %s)..." % str(X_combined.shape))
X_pca, pca, scaler = run_pca_workflow(X_combined, n_components=10)

# Add PCA components to metadata for plotting
for i in range(X_pca.shape[1]):
    metadata_combined[f'PC{i+1}'] = X_pca[:, i]

#%% Visualize Explained Variance
plt.figure(figsize=(8, 5))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance - Raw Trajectories')
plt.grid(True)
plt.show()

#%% Plot PC1 vs PC2
plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=metadata_combined, 
    x='PC1', y='PC2', 
    hue='experiment', 
    palette='viridis',
    alpha=0.7
)
plt.title('PCA: PC1 vs PC2 (Raw Trajectories)')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
plt.show()

#%% Visualize PC Loadings (Temporal Modes)
# Loadings tell us which time points contribute most to the variance
time_axis = np.arange(TARGET_LEN)
plt.figure(figsize=(12, 6))

for i in range(3):
    loadings = pca.components_[i]
    p1_loadings = loadings[:TARGET_LEN]
    p2_loadings = loadings[TARGET_LEN:]
    
    plt.subplot(1, 3, i+1)
    plt.plot(time_axis, p1_loadings, label='Pole 1', color='red', alpha=0.8)
    plt.plot(time_axis, p2_loadings, label='Pole 2', color='blue', alpha=0.8)
    plt.title(f'PC{i+1} Loadings')
    plt.xlabel('Time (frames)')
    if i == 0: plt.ylabel('Weight')
    plt.legend()

plt.tight_layout()
plt.show()

#%% Visualize Actual Trajectories for PC Extremes
# This helps you "see" what high vs low scores mean biologically
from SingleCellDataAnalysis.PCA_utils import plot_top_cells_for_pc

# Change pc_idx to 1, 2, 3... to explore different components
plot_top_cells_for_pc(metadata_combined, X_combined, pc_idx=1, n_cells=3, target_len=TARGET_LEN)

#%% Export Results
output_combined = "/Users/user/Documents/Python_Scripts/FungalProjectScript/data/pca_raw_trajectories_results.csv"
metadata_combined.to_csv(output_combined, index=False)
print(f"Results saved to {output_combined}")
