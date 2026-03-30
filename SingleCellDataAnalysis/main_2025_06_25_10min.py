#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 10:18:32 2025

@author: user
"""


import os

import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.config import WORKING_DIR, FRAME_NUMBER
from SingleCellDataAnalysis.visualization import plot_aligned_signals, plot_aligned_heatmaps
from SingleCellDataAnalysis.spectral_analysis import (
    fft_dominant_frequencies, plot_fft_dominant_frequencies,
    gp_infer_periods, plot_gp_periods, plot_multi_periodic_gp_grid
)
from SingleCellDataAnalysis.load_data import load_preprocessed_data
from SingleCellDataAnalysis.spectral_analysis import quantify_gp_features
from SingleCellDataAnalysis.simple_shape_analysis import (
    plot_simple_model_grid, simulate_stepwise_cells, quantify_all_cells,
    summarize_model_distribution, build_model_category_heatmap_df, 
    pivot_heatmap_matrix,plot_model_heatmap,extract_oscillation_data,
    plot_oscillation_scatter,get_median_global_times,extract_frequency_with_time,
    plot_frequency_timeline,plot_amplitude_timeline, extract_slope_with_time, plot_slope_timeline,filter_slope_data,
    fit_slope_vs_time,plot_slope_with_regression, prepare_pol1_pol2_slope_with_time,plot_pol1_vs_pol2_with_lines,
    plot_first_time_distribution,compute_correlation_corrected,plot_correlation_timeline,
    cluster_cells_from_model_params,plot_amplitude_distributions,cluster_cells_by_amplitude_and_delay,
    annotate_tree_with_aligned_time,plot_tree_with_annotations,quantify_all_cells_xcor,plot_aligned_time_by_dendrogram_order
)
#%% preprocessus
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

# ---- Step 1: Load Data ----
print("📥 Loading data...")
#df_all = load_and_merge_csv(FILE_NAMES, WORKING_DIR)
merged_csv = os.path.join(WORKING_DIR, "complete_cells_time_series.csv")
df_all = pd.read_csv(merged_csv)
#df_all = offset_cell_ids_globally(df_all)




# ---- Step 2: Filter and Preprocess ----
print("🧹 Filtering valid cells...")
df_all = filter_valid_cells(df_all, 51)#FRAME_NUMBER)
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



#%% ---- Step 6: Alignment Preparation ----
features_xcorr = ['nu_dis', 'weighted_area', 'septum_int_corr']
padding = 16 * FRAME_NUMBER
global_time = np.arange(0, FRAME_NUMBER + padding)

# Load 20 min aligned dataset
df_all_20, time_points_20, global_time_20, _, best_shifts_20 = load_preprocessed_data(
    WORKING_DIR, filename="combined_all_cells_with_aligned_time_20min.csv"
)

print("📊 Reconstructing aligned reference signals using best_shifts_20...")

# Step 1: Extract signals
ref_signals_20, ref_cell_ids_20 = prepare_signals(df_all_20, features_xcorr, time_points_20)

# Step 2: Align each signal based on its shift
aligned_ref_signals = []
for cell_id in ref_cell_ids_20:
    if cell_id not in best_shifts_20:
        continue

    signal = ref_signals_20[cell_id]  # shape: (n_features, time_points)
    shift = best_shifts_20[cell_id]
    aligned_signal = np.full((len(features_xcorr), len(global_time_20)), np.nan)

    start_idx = shift
    end_idx = shift + signal.shape[1]

    # Boundary conditions
    if start_idx < 0:
        slice_start = -start_idx
        aligned_signal[:, 0:signal.shape[1]-slice_start] = signal[:, slice_start:]
    elif end_idx > len(global_time_20):
        slice_len = len(global_time_20) - start_idx
        aligned_signal[:, start_idx:] = signal[:, :slice_len]
    else:
        aligned_signal[:, start_idx:end_idx] = signal

    aligned_ref_signals.append(aligned_signal)

# Step 3: Stack and average to build reference mean
aligned_ref_signals = np.stack(aligned_ref_signals)  # (n_cells, n_features, n_time)
ref_mean_signal = np.nanmean(aligned_ref_signals, axis=0)  # (n_features, n_time)

#%% ---- Step 7: Align New Data to Reference ----

# Ensure you already loaded df_all before this cell
features_xcorr = ['nu_dis', 'weighted_area', 'septum_int_corr']
time_points = sorted(df_all['time_point'].unique())
padding = 16 * FRAME_NUMBER
global_time = np.arange(0, FRAME_NUMBER + padding)

print("➕ Computing corrected intensity features...")
df_all['septum_int_corr'] = df_all['septum_int'] - df_all['cyt_int']
df_all['pol1_int_corr'] = df_all['pol1_int'] - df_all['cyt_int']
df_all['pol2_int_corr'] = df_all['pol2_int'] - df_all['cyt_int']
df_all['pol1_minus_pol2'] = df_all['pol1_int_corr'] - df_all['pol2_int_corr']
df_all['weighted_area'] = df_all['cell_area'] / 500

print("⏱ Preparing new signals for alignment to 20min mean...")
cell_signals, cell_ids = prepare_signals(df_all, features_xcorr, time_points)

print("🔁 Brute-force aligning to 20 min reference mean using global MSE...")

best_shifts = []
ref_trace = np.sum(ref_mean_signal, axis=0)  # shape: (global_time,)

for cid in cell_ids:
    signal = cell_signals[cid]  # shape: (n_features, timepoints)
    query_trace = np.sum(signal, axis=0)     # shape: (FRAME_NUMBER,)
    signal_len = len(query_trace)

    best_score = np.inf
    best_shift = 0

    # Loop over all valid shifts across global_time
    for shift in range(len(global_time) - signal_len + 1):
        ref_window = ref_trace[shift:shift + signal_len]
        if np.any(np.isnan(ref_window)):
            continue

        mse = np.mean((query_trace - ref_window)**2)
        if mse < best_score:
            best_score = mse
            best_shift = shift

    best_shifts.append(best_shift)

best_shifts = np.array(best_shifts)
best_shifts_dict = {cid: shift for cid, shift in zip(cell_ids, best_shifts)}



best_mean = ref_mean_signal  # reuse 20-min reference mean

# ---- Add Aligned Time & Export ----
print("💾 Adding aligned time column and exporting...")
df_all = generate_aligned_time_column(df_all, best_shifts_dict, time_points)
export_aligned_dataframe(df_all, WORKING_DIR)

# ---- Plotting ----
print("📈 Plotting aligned signals...")
best_shifts_dict = {cid: shift for cid, shift in zip(cell_ids, best_shifts)}
plot_aligned_signals(df_all, cell_ids, best_shifts_dict, global_time_20, time_points, features_xcorr, mean_trace=best_mean)


print("🧯 Creating heatmaps for polar features...")
heatmap_features = ['pol1_int_corr', 'pol2_int_corr', 'pol1_minus_pol2']
best_shifts_dict = {cid: shift for cid, shift in zip(cell_ids, best_shifts)}
plot_aligned_heatmaps(df_all, cell_ids, best_shifts_dict, global_time_20, time_points, heatmap_features)



#%%

plot_simple_model_grid(df_all, cell_ids, time_points, model_type='linear', start_idx=0)
#plot_simple_model_grid(df_all, cell_ids, time_points, model_type='constant', start_idx=26)
#plot_simple_model_grid(df_all, cell_ids, time_points, model_type='step', start_idx=51)
#plot_simple_model_grid(df_all, cell_ids, time_points, model_type='step', start_idx=76)


# df_sim = simulate_stepwise_cells()
# cell_ids = df_sim['cell_id'].unique()

# plot_simple_model_grid(df_sim, cell_ids=cell_ids, time_points=np.arange(51), model_type='step')
#%% Quantify pole trajectories by fitting sine waves
df_results = quantify_all_cells(df_all, cell_ids,delta_threshold=10)
summary_table = summarize_model_distribution(df_results)
print(summary_table)
#%% Quantify cross corelation between detrended pole trajectories
df_results_xcor = quantify_all_cells_xcor(df_all, cell_ids,delta_threshold=10)
#%% Combine results of quantification
df_combined = pd.merge(df_results, df_results_xcor, on='cell_id', how='left')

#%% Clustering cells with every raw parameter values of harmonic wave
#df_results = pd.read_csv("model_fits_by_cell.csv")
clustered_df = cluster_cells_from_model_params(df_combined, n_harmonics=10)
plot_amplitude_distributions(clustered_df)#, n_harmonics=10)
#%% clustering cells with designed parameters
df_norm,ordered_cell_ids, row_linkage = cluster_cells_by_amplitude_and_delay(df_combined)
#%%
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=0)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=25)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=50)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=75)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=100)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=125)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=150)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=175)
# %%

# Group by cell and get min aligned time
aligned_times = df_all.groupby('cell_id')['aligned_time'].min()

# Ensure index types are string for compatibility
aligned_times.index = aligned_times.index.astype(str)
df_norm.index = df_norm.index.astype(str)

# Filter aligned times to only those cells used in clustering
aligned_times = aligned_times[df_norm.index]

# Now you can pass to other functions
tree_root, node_list= annotate_tree_with_aligned_time(df_norm, row_linkage, aligned_times)
plot_tree_with_annotations(tree_root)
plot_aligned_time_by_dendrogram_order(row_linkage, df_norm, aligned_times)
#%% get clades
def extract_leaf_labels(node):
    if node.is_leaf():
        return [node.get_id()]
    else:
        return extract_leaf_labels(node.left) + extract_leaf_labels(node.right)

clade_ids = [347,351,313,346,338,340,333,349,329,336,350]  # replace with your chosen node IDs
clade_cell_ids = {}

for node_id in clade_ids:
    clade_node = node_list[node_id]
    leaf_ids = extract_leaf_labels(clade_node)
    cell_ids = [df_norm.index[i] for i in leaf_ids]
    clade_cell_ids[node_id] = cell_ids

#%%
# Group by cell and get mean cell area
cell_areas = df_all.groupby('cell_id')['cell_area'].mean()

# Ensure index types are string for compatibility
cell_areas.index = cell_areas.index.astype(str)
df_norm.index = df_norm.index.astype(str)

# Filter cell_areas to only those cells used in clustering
cell_areas = cell_areas[df_norm.index]

#%% Lasso regression for each clade
import os
import pandas as pd
from sklearn.linear_model import LassoCV

# 1. Define individual (unsummed) features in interleaved order
# Features that exist as both pol1_ and pol2_ versions
paired_features = ['a', 'mid', 'A_1', 'A_2', 'A_3', 'A_mid', 'A_short',
                   'delay1', 'delay2', 'delay3']

# Features that are global (not duplicated for pol1/pol2)
global_features = ['xcor_max', 'xcor_lag', 'xcor_zero_lag', 'a1a2', 'd', 'dd']

# Create interleaved pol1_ / pol2_ feature list
interleaved_features = [f for pair in zip([f'pol1_' + f for f in paired_features],
                                          [f'pol2_' + f for f in paired_features])
                        for f in pair]

# Combine all features
#all_features = interleaved_features + global_features
interleaved_features += global_features
#%% 2. Run LassoCV and collect results
# rows = []

# for clade_id, cell_ids in clade_cell_ids.items():
#     X = df_norm.loc[cell_ids, interleaved_features]
#     #y = aligned_times.loc[cell_ids]
#     y = cell_areas.loc[cell_ids]

#     # Fit LassoCV with 5-fold CV
#     model = LassoCV(cv=5, random_state=42).fit(X, y)

#     row = {
#         'clade_id': clade_id,
#         'intercept': model.intercept_,
#         'r_squared': model.score(X, y),
#         'alpha': model.alpha_  # best regularization strength
#     }
#     for feat, coef in zip(interleaved_features, model.coef_):
#         row[feat] = coef
#     rows.append(row)

# # 3. Create DataFrame
# results_df = pd.DataFrame(rows)
# results_df = results_df[['clade_id', 'r_squared', 'alpha', 'intercept'] + interleaved_features]

# # 4. Save to WORKING_DIR
# output_path = os.path.join(WORKING_DIR, "clade_lasso_results_pol1_pol2.csv")
# results_df.to_csv(output_path, index=False)

# print(f"Saved Lasso regression results to:\n{output_path}")

#%% Lasso for all cells
# Compute average cell area
cell_areas = df_all.groupby('cell_id')['cell_area'].mean()

# Ensure index consistency
cell_areas.index = cell_areas.index.astype(str)
df_norm.index = df_norm.index.astype(str)

# Align cell IDs across df_norm and cell_areas
common_cells = df_norm.index.intersection(cell_areas.index)

X = df_norm.loc[common_cells, interleaved_features]
#y = cell_areas.loc[common_cells]
y = aligned_times.loc[common_cells]


model = LassoCV(cv=5, random_state=42)
model.fit(X, y)
row = {
    'intercept': model.intercept_,
    'r_squared': model.score(X, y),
    'alpha': model.alpha_
}
for feat, coef in zip(interleaved_features, model.coef_):
    row[feat] = coef

results_df = pd.DataFrame([row])
results_df = results_df[['r_squared', 'alpha', 'intercept'] + interleaved_features]

# Save to CSV
output_path = os.path.join(WORKING_DIR, "global_lasso_cell_area.csv")
results_df.to_csv(output_path, index=False)

print(f"Saved global Lasso regression result to:\n{output_path}")
#%% Logistic regression on cell area
# Compute average cell area per cell
cell_areas = df_all.groupby('cell_id')['cell_area'].mean()
cell_areas.index = cell_areas.index.astype(str)
df_norm.index = df_norm.index.astype(str)

# Create binary labels: 1 if cell area above median, 0 otherwise
labels = (cell_areas > cell_areas.median()).astype(int)

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
import os

# Features that exist as both pol1_ and pol2_ versions
paired_features = ['a', 'mid', 'A_1', 'A_2', 'A_3', 'A_mid', 'A_short',
                   'delay1', 'delay2', 'delay3']

# Features that are global (not duplicated for pol1/pol2)
global_features = ['xcor_max', 'xcor_lag', 'xcor_zero_lag', 'a1a2', 'd', 'dd']

# Create interleaved pol1_ / pol2_ feature list
interleaved_features = [f for pair in zip([f'pol1_' + f for f in paired_features],
                                          [f'pol2_' + f for f in paired_features])
                        for f in pair]

# Combine all features
#all_features = interleaved_features + global_features
interleaved_features += global_features
#%%

# Filter to common cells
common_cells = df_norm.index.intersection(labels.index)
X = df_norm.loc[common_cells, interleaved_features]
y = labels.loc[common_cells]

# Fit logistic regression with L1 penalty (sparse features)
model = LogisticRegressionCV(
    cv=5,
    penalty='l1',
    solver='saga',
    scoring='accuracy',
    random_state=42,
    max_iter=10000
)
model.fit(X, y)

# Save coefficients
coefs = pd.Series(model.coef_[0], index=interleaved_features)
results_df = pd.DataFrame([{
    'intercept': model.intercept_[0],
    'accuracy': model.score(X, y),
    **coefs.to_dict()
}])

# Save
output_path = os.path.join(WORKING_DIR, "logistic_l1_cell_area_binary.csv")
results_df.to_csv(output_path, index=False)

print(f"Saved logistic regression results to:\n{output_path}")

# Optional: Evaluate performance
print("\nClassification Report:")
print(classification_report(y, model.predict(X)))

#%% plot heatmap
# Reuse helper
heatmap_df = build_model_category_heatmap_df(df_all, df_results, best_shifts, time_bin_width=20)
heatmap_matrix = pivot_heatmap_matrix(heatmap_df)
plot_model_heatmap(heatmap_matrix)


#%% plot osciltion par

df_osc = extract_oscillation_data(df_results)
plot_oscillation_scatter(df_osc)

# Step 1: build global time reference per cell
median_global_time_dict = get_median_global_times(df_all, best_shifts)

# Step 2: extract freq + time per pol
df_freq_time = extract_frequency_with_time(df_results, median_global_time_dict)

# Step 3: plot
plot_frequency_timeline(df_freq_time)

plot_amplitude_timeline(df_freq_time)
#%% plot slop

df_slope_time = extract_slope_with_time(df_results, df_all, best_shifts)
plot_slope_timeline(df_slope_time)



df_filtered = filter_slope_data(df_slope_time, time_min=50, time_max=225)
reg = fit_slope_vs_time(df_filtered)
print(f"Regression Coef: {reg.coef_[0]:.4f}, Intercept: {reg.intercept_:.4f}, R^2: {reg.score(df_filtered['global_time'].values.reshape(-1, 1), df_filtered['slope'].values):.4f}")
plot_slope_with_regression(df_filtered, reg)


plot_first_time_distribution(df_slope_time, pol1="pol1", pol2="pol2")

#%%
df_pivoted = prepare_pol1_pol2_slope_with_time(df_slope_time, df_all, best_shifts)
plot_pol1_vs_pol2_with_lines(df_pivoted)


#%% Covariance

df_corr = compute_correlation_corrected(df_all)
plot_correlation_timeline(df_corr)

print("✅ Spectral analysis complete.")
