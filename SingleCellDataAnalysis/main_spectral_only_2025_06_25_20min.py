#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 14:05:32 2025
main_spectral_only.py - Runs visualization and spectral analysis using preprocessed data
Created on Fri Jun 21 2025
@author: user
"""
import os
import pandas as pd
import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.config import WORKING_DIR
from SingleCellDataAnalysis.load_data import load_preprocessed_data
from SingleCellDataAnalysis.signal_analysis import (
    plot_simple_model_grid,  
    quantify_all_cells,
    summarize_model_distribution,
    )
from SingleCellDataAnalysis.signal_cor import (
    quantify_all_cells_xcor,
    quantify_all_cells_acor,
    plot_perodicity_result,
    plot_perodicity_vs_NC
    )
from SingleCellDataAnalysis.clustering import (
    cluster_cells_from_model_params,
    plot_amplitude_distributions,
    cluster_cells_by_amplitude_and_delay,
    annotate_tree_with_aligned_time,
    plot_tree_with_annotations,
    plot_aligned_time_by_dendrogram_order
)
from SingleCellDataAnalysis.regression_helpers import (
    run_lasso_on_clades,
    run_global_lasso,
    run_logistic_l1
)

df_all, time_points, global_time, cell_ids, best_shifts = load_preprocessed_data(WORKING_DIR,  filename="combined_all_cells_with_aligned_time_20min.csv")
#%%

plot_simple_model_grid(df_all, cell_ids, time_points, model_type='linear', start_idx=0)

#%% Quantify pole trajectories by fitting sine waves
df_results = quantify_all_cells(df_all, cell_ids,delta_threshold=10)
summary_table = summarize_model_distribution(df_results)
print(summary_table)
#%% Quantify cross and auto corelation between detrended pole trajectories
df_results_xcor = quantify_all_cells_xcor(df_all, cell_ids,delta_threshold=10)

df_results_acor = quantify_all_cells_acor(df_all, cell_ids,delta_threshold=10)
#%%
cell_id = [251]#[cell_ids[2]]
#_ = quantify_all_cells_xcor(df_all, cell_id, delta_threshold=10, visualize=True)
periodicity_result = quantify_all_cells_acor(df_all, cell_ids, delta_threshold=10, visualize=True)
#%%
plot_perodicity_vs_NC(periodicity_result)
#plot_perodicity_result(periodicity_result)
#%% Combine results of quantification
df_combined = pd.merge(df_results, periodicity_result, on='cell_id', how='left')


#%% Clustering cells with every raw parameter values of harmonic wave
#df_results = pd.read_csv("model_fits_by_cell.csv")
# clustered_df = cluster_cells_from_model_params(df_combined, n_harmonics=10)
# plot_amplitude_distributions(clustered_df)#, n_harmonics=10)
#%% clustering cells with designed parameters
df_norm,ordered_cell_ids, row_linkage = cluster_cells_by_amplitude_and_delay(df_combined)
#%%
show_fit=False
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=0, show_fit=show_fit)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=25, show_fit=show_fit)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=50, show_fit=show_fit)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=75, show_fit=show_fit)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=100, show_fit=show_fit)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=125, show_fit=show_fit)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=150, show_fit=show_fit)
plot_simple_model_grid(df_all, ordered_cell_ids, time_points, model_type='linear', start_idx=175, show_fit=show_fit)
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


#%%
# 1. Define individual (unsummed) features in interleaved order
base_features = ['mid', 'A_1', 'A_2', 'A_3', 'A_mid', 'A_short']
interleaved_features = [f for pair in zip([f'pol1_' + f for f in base_features],
                                          ['pol2_' + f for f in base_features])
                        for f in pair]
# Lasso on clades
run_lasso_on_clades(df_norm, cell_areas, clade_cell_ids, interleaved_features,
                    os.path.join(WORKING_DIR, "clade_lasso_results_pol1_pol2_20min.csv"))

# Global Lasso
run_global_lasso(df_norm, aligned_times, interleaved_features,
                 os.path.join(WORKING_DIR, "global_lasso_cell_area_20min.csv"))

# Logistic regression
binary_labels = (cell_areas > cell_areas.median()).astype(int)
run_logistic_l1(df_norm, binary_labels, interleaved_features,
                os.path.join(WORKING_DIR, "logistic_l1_cell_area_binary_20min.csv"))

#%%

# Sort the IDs by their shift values (i.e., by the dictionary values)
sorted_ids = sorted(best_shifts, key=best_shifts.get)
plot_simple_model_grid(df_all, sorted_ids, time_points, model_type='linear', start_idx=0)
plot_simple_model_grid(df_all, sorted_ids, time_points, model_type='linear', start_idx=25)
plot_simple_model_grid(df_all, sorted_ids, time_points, model_type='linear', start_idx=50)
plot_simple_model_grid(df_all, sorted_ids, time_points, model_type='linear', start_idx=75)
plot_simple_model_grid(df_all, sorted_ids, time_points, model_type='linear', start_idx=100)
plot_simple_model_grid(df_all, sorted_ids, time_points, model_type='linear', start_idx=125)
plot_simple_model_grid(df_all, sorted_ids, time_points, model_type='linear', start_idx=150)
plot_simple_model_grid(df_all, sorted_ids, time_points, model_type='linear', start_idx=175)

# #%% Lasso regression for each clade
# import os
# import pandas as pd
# from sklearn.linear_model import LassoCV

# # 1. Define individual (unsummed) features in interleaved order
# base_features = ['mid', 'A_1', 'A_2', 'A_3', 'A_mid', 'A_short']
# interleaved_features = [f for pair in zip([f'pol1_' + f for f in base_features],
#                                           ['pol2_' + f for f in base_features])
#                         for f in pair]

# # 2. Run LassoCV and collect results
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

# #%% Lasso for all cells
# # Compute average cell area
# cell_areas = df_all.groupby('cell_id')['cell_area'].mean()

# # Ensure index consistency
# cell_areas.index = cell_areas.index.astype(str)
# df_norm.index = df_norm.index.astype(str)

# # Align cell IDs across df_norm and cell_areas
# common_cells = df_norm.index.intersection(cell_areas.index)

# X = df_norm.loc[common_cells, interleaved_features]
# #y = cell_areas.loc[common_cells]
# y = aligned_times.loc[common_cells]


# model = LassoCV(cv=5, random_state=42)
# model.fit(X, y)
# row = {
#     'intercept': model.intercept_,
#     'r_squared': model.score(X, y),
#     'alpha': model.alpha_
# }
# for feat, coef in zip(interleaved_features, model.coef_):
#     row[feat] = coef

# results_df = pd.DataFrame([row])
# results_df = results_df[['r_squared', 'alpha', 'intercept'] + interleaved_features]

# # Save to CSV
# output_path = os.path.join(WORKING_DIR, "global_lasso_cell_area.csv")
# results_df.to_csv(output_path, index=False)

# print(f"Saved global Lasso regression result to:\n{output_path}")
# #%% Logistic regression on cell area
# # Compute average cell area per cell
# cell_areas = df_all.groupby('cell_id')['cell_area'].mean()
# cell_areas.index = cell_areas.index.astype(str)
# df_norm.index = df_norm.index.astype(str)

# # Create binary labels: 1 if cell area above median, 0 otherwise
# labels = (cell_areas > cell_areas.median()).astype(int)

# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.metrics import classification_report
# import os

# # Define features
# base_features = ['mid', 'A_1', 'A_2', 'A_3', 'A_mid', 'A_short']
# interleaved_features = [f for pair in zip([f'pol1_' + f for f in base_features],
#                                           ['pol2_' + f for f in base_features])
#                         for f in pair]

# # Filter to common cells
# common_cells = df_norm.index.intersection(labels.index)
# X = df_norm.loc[common_cells, interleaved_features]
# y = labels.loc[common_cells]

# # Fit logistic regression with L1 penalty (sparse features)
# model = LogisticRegressionCV(
#     cv=5,
#     penalty='l1',
#     solver='saga',
#     scoring='accuracy',
#     random_state=42,
#     max_iter=10000
# )
# model.fit(X, y)

# # Save coefficients
# coefs = pd.Series(model.coef_[0], index=interleaved_features)
# results_df = pd.DataFrame([{
#     'intercept': model.intercept_[0],
#     'accuracy': model.score(X, y),
#     **coefs.to_dict()
# }])

# # Save
# output_path = os.path.join(WORKING_DIR, "logistic_l1_cell_area_binary.csv")
# results_df.to_csv(output_path, index=False)

# print(f"Saved logistic regression results to:\n{output_path}")

# # Optional: Evaluate performance
# print("\nClassification Report:")
# print(classification_report(y, model.predict(X)))

# #%% plot heatmap
# # Reuse helper
# heatmap_df = build_model_category_heatmap_df(df_all, df_results, best_shifts, time_bin_width=20)
# heatmap_matrix = pivot_heatmap_matrix(heatmap_df)
# plot_model_heatmap(heatmap_matrix)


# #%% plot osciltion par

# df_osc = extract_oscillation_data(df_results)
# plot_oscillation_scatter(df_osc)

# # Step 1: build global time reference per cell
# median_global_time_dict = get_median_global_times(df_all, best_shifts)

# # Step 2: extract freq + time per pol
# df_freq_time = extract_frequency_with_time(df_results, median_global_time_dict)

# # Step 3: plot
# plot_frequency_timeline(df_freq_time)

# plot_amplitude_timeline(df_freq_time)
# #%% plot slop

# df_slope_time = extract_slope_with_time(df_results, df_all, best_shifts)
# plot_slope_timeline(df_slope_time)



# df_filtered = filter_slope_data(df_slope_time, time_min=50, time_max=225)
# reg = fit_slope_vs_time(df_filtered)
# print(f"Regression Coef: {reg.coef_[0]:.4f}, Intercept: {reg.intercept_:.4f}, R^2: {reg.score(df_filtered['global_time'].values.reshape(-1, 1), df_filtered['slope'].values):.4f}")
# plot_slope_with_regression(df_filtered, reg)


# plot_first_time_distribution(df_slope_time, pol1="pol1", pol2="pol2")

# #%%
# df_pivoted = prepare_pol1_pol2_slope_with_time(df_slope_time, df_all, best_shifts)
# plot_pol1_vs_pol2_with_lines(df_pivoted)


# #%% Covariance

# df_corr = compute_correlation_corrected(df_all)
# plot_correlation_timeline(df_corr)

# print("✅ Spectral analysis complete.")
