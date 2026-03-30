#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:58:37 2025

@author: user
"""
import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

# main.py
from SingleCellDataAnalysis.config import WORKING_DIR, FILE_NAMES, FRAME_NUMBER, ROLLING_WINDOW, N_SIGMA
from SingleCellDataAnalysis.load_data import load_and_merge_csv, offset_cell_ids_globally
from SingleCellDataAnalysis.preprocessing import filter_valid_cells, compute_derivatives
from SingleCellDataAnalysis.feature_extraction import extract_features
from SingleCellDataAnalysis.gumm import plot_gumm
from SingleCellDataAnalysis.filter_extremes import get_all_extreme_cells
from SingleCellDataAnalysis.alignment import prepare_signals, run_mcmc
from SingleCellDataAnalysis.export_aligned import generate_aligned_time_column, export_aligned_dataframe
from SingleCellDataAnalysis.visualization import plot_aligned_signals, plot_aligned_heatmaps
from SingleCellDataAnalysis.spectral_analysis import fft_dominant_frequencies, plot_fft_dominant_frequencies
from SingleCellDataAnalysis.spectral_analysis import gp_infer_periods, plot_gp_periods
from SingleCellDataAnalysis.spectral_analysis import plot_multi_periodic_gp_grid

import os
import pandas as pd
import numpy as np

# ---- Step 1: Load Data ----
print("📥 Loading data...")
#df_all = load_and_merge_csv(FILE_NAMES, WORKING_DIR)
merged_csv = os.path.join(WORKING_DIR, "complete_cells_time_series.csv")
df_all = pd.read_csv(merged_csv)
#df_all = offset_cell_ids_globally(df_all)




# ---- Step 2: Filter and Preprocess ----
print("🧹 Filtering valid cells...")
df_all = filter_valid_cells(df_all, 101)#FRAME_NUMBER)
print(f"✅ {df_all['cell_id'].nunique()} cells retained.")

print("⚙️ Computing derivatives...")
df_all = compute_derivatives(df_all, ROLLING_WINDOW)

# ---- Step 3: Feature Extraction ----
print("📊 Extracting features...")
growth_matrix = extract_features(df_all)

# ---- Step 4: Visualize and Fit GUMM ----
plot_gumm(growth_matrix["avg_d_cell_area"], "GUMM for avg_d_cell_area", "avg_d_cell_area")
plot_gumm(growth_matrix["max_d_nu_dis"], "GUMM for max_d_nu_dis", "max_d_nu_dis")
plot_gumm(growth_matrix["max_d_cell_area"], "GUMM for max_d_cell_area", "max_d_cell_area")
plot_gumm(growth_matrix["max_cell_area"], "GUMM for max_cell_area", "max_cell_area")
plot_gumm(growth_matrix["min_cell_area"], "GUMM for min_cell_area", "min_cell_area")
plot_gumm(growth_matrix["std_d_cell_area"], "GUMM for std_d_cell_area", "std_d_cell_area")

# ---- Step 5: Identify and Remove Extreme Cells ----
print("🚫 Detecting extreme cells...")
extreme_ids = get_all_extreme_cells(growth_matrix, n_sigma=N_SIGMA)
print(f"📌 Removing {len(extreme_ids)} extreme cells:", extreme_ids.tolist())

df_all = df_all[~df_all['cell_id'].isin(extreme_ids)]



# ---- Step 6: Alignment Preparation ----
features_xcorr = ['nu_dis', 'weighted_area', 'septum_int_corr']
time_points = sorted(df_all['time_point'].unique())
padding = 16 * FRAME_NUMBER
global_time = np.arange(0, FRAME_NUMBER + padding)

print("➕ Computing weighted and corrected intensity features...")
df_all['septum_int_corr'] = df_all['septum_int'] - df_all['cyt_int']
df_all['pol1_int_corr'] = df_all['pol1_int'] - df_all['cyt_int']
df_all['pol2_int_corr'] = df_all['pol2_int'] - df_all['cyt_int']
df_all['pol1_minus_pol2'] = df_all['pol1_int_corr'] - df_all['pol2_int_corr']
df_all['weighted_area'] = df_all['cell_area'] / 500

print("⏱ Preparing signals for alignment...")
cell_signals, cell_ids = prepare_signals(df_all, features_xcorr, time_points)

# ---- Step 7: Run MCMC Alignment ----
print("🔁 Running MCMC alignment...")
best_shifts, best_mean, mse_trace = run_mcmc(cell_signals, global_time, time_points, lambda_reg=0.0, n_iter=25000)

# ---- Step 8: Add Aligned Time & Export ----
print("💾 Adding aligned time column and exporting...")
df_all = generate_aligned_time_column(df_all, best_shifts, time_points)
export_aligned_dataframe(df_all, WORKING_DIR)

# ---- Step 9: Visualization ----
print("📈 Plotting aligned signals...")
plot_aligned_signals(df_all, cell_ids, best_shifts, global_time, time_points, features_xcorr, mean_trace=best_mean)
#
print("🧯 Creating heatmaps for polar features...")
heatmap_features = ['pol1_int_corr', 'pol2_int_corr', 'pol1_minus_pol2']
plot_aligned_heatmaps(df_all, cell_ids, best_shifts, global_time, time_points, heatmap_features)

#%% ---- Step 10: Frequency Analysis ----
print("🔬 Running FFT analysis...")
fft_results = fft_dominant_frequencies(df_all, cell_ids, best_shifts, global_time, time_points, features=['pol1_int_corr', 'pol2_int_corr'])
plot_fft_dominant_frequencies(fft_results)

print("🌊 Running GP period inference...")
gp_results = gp_infer_periods(df_all, cell_ids, best_shifts, global_time, time_points, features=['pol1_int_corr', 'pol2_int_corr'])
plot_gp_periods(gp_results)


plot_multi_periodic_gp_grid(df_all, cell_ids, time_points)

print("✅ Pipeline completed.")
