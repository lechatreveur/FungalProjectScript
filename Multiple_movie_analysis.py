#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 16:44:10 2025

@author: user
"""

#%% Imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Paths
working_dir = "/Volumes/Movies/2025_06_04_M68/"
file_names = [ "A14_1", "A14_2", "A14_3" ,"A14_4","A14_5","A14_6","A14_7", "A14_8","A14_9" ]

# Master dataframe
df_all = pd.DataFrame()
max_cell_id = 0  # Track running max to offset IDs

# Loop through datasets
for fname in file_names:
    # Define paths
    tracked_cells_folder = os.path.join(working_dir, f"{fname}/TrackedCells_{fname}")
    csv_path = os.path.join(tracked_cells_folder, "all_cells_time_series.csv")

    # Load
    df = pd.read_csv(csv_path)

    # Store original cell ID
    df['original_cell_id'] = df['cell_id']

    # Offset to avoid ID collision across datasets
    df['cell_id'] += max_cell_id

    # Add source tracking
    df['source_file'] = fname

    # Update max for next loop
    max_cell_id = df['cell_id'].max() + 1

    # Append to master
    df_all = pd.concat([df_all, df], ignore_index=True)

# Save combined result
combined_csv_path = os.path.join(working_dir, "combined_all_cells_time_series.csv")
df_all.to_csv(combined_csv_path, index=False)

#%% Filtering: keep only cells with 41 time points
frame_number = 51
valid_cells = df_all.groupby("cell_id")["time_point"].count()
valid_cells = valid_cells[valid_cells == frame_number].index
df_all = df_all[df_all['cell_id'].isin(valid_cells)]
# Count how many cells have exactly 41 time points
num_cells = df_all['cell_id'].nunique()
print(f"✅ Number of cells with exactly {frame_number} time points: {num_cells}")

#%% Preprocessing: sort and compute derivative
df_all = df_all.sort_values(by=['cell_id', 'time_point'])

# First-order derivatives
df_all['d_cell_length'] = df_all.groupby('cell_id')['cell_length'].diff()
df_all['d_cell_area'] = df_all.groupby('cell_id')['cell_area'].diff()
df_all['d_nu_dis'] = df_all.groupby('cell_id')['nu_dis'].diff()

# Smoothed derivatives with rolling mean (window=20)
df_all['d_cell_length_avg5'] = (
    df_all.groupby('cell_id')['d_cell_length']
    .rolling(window=20, center=True, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

df_all['d_cell_area_avg5'] = (
    df_all.groupby('cell_id')['d_cell_area']
    .rolling(window=20, center=True, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

df_all['d_nu_dis_avg5'] = (
    df_all.groupby('cell_id')['d_nu_dis']
    .rolling(window=20, center=True, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)


#%% Compute average d_cell_length and d_cell_area per cell
avg_d_length = (
    df_all
    .groupby("cell_id")["d_cell_length"]
    .mean()
    .rename("avg_d_cell_length")
    .reset_index()
)

avg_d_area = (
    df_all
    .groupby("cell_id")["d_cell_area"]
    .mean()
    .rename("avg_d_cell_area")
    .reset_index()
)

max_area = (
    df_all
    .groupby("cell_id")["cell_area"]
    .max()
    .rename("max_cell_area")
    .reset_index()
)

min_area = (
    df_all
    .groupby("cell_id")["cell_area"]
    .min()
    .rename("min_cell_area")
    .reset_index()
)

# Compute std and max(abs) of d_cell_area per cell
std_d_area = (
    df_all
    .groupby("cell_id")["d_cell_area"]
    .std()
    .rename("std_d_cell_area")
    .reset_index()
)

max_abs_d_area = (
    df_all
    .groupby("cell_id")["d_cell_area"]
    .apply(lambda x: np.max(np.abs(x)))
    .rename("max_abs_d_cell_area")
    .reset_index()
)

max_abs_d_nu_dis = (
    df_all
    .groupby("cell_id")["d_nu_dis"]
    .apply(lambda x: np.max(np.abs(x)))
    .rename("max_abs_d_nu_dis")
    .reset_index()
)



# Merge avg_d_length and avg_d_area
growth_matrix_per_cell = pd.merge(avg_d_length, avg_d_area, on="cell_id")

# Merge with max_area
growth_matrix_per_cell = pd.merge(growth_matrix_per_cell, max_area, on="cell_id")

# Merge with min_area
growth_matrix_per_cell = pd.merge(growth_matrix_per_cell, min_area, on="cell_id")

# Merge into the main matrix
growth_matrix_per_cell = growth_matrix_per_cell.merge(std_d_area, on="cell_id")
growth_matrix_per_cell = growth_matrix_per_cell.merge(max_abs_d_area, on="cell_id")
growth_matrix_per_cell = growth_matrix_per_cell.merge(max_abs_d_nu_dis, on="cell_id")

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from scipy.optimize import minimize

#===============================
# Mixture Model Fitting Function
#===============================
def gaussian_uniform_mixture_log_likelihood(params, data):
    mu, sigma, pi = params
    a, b = np.min(data), np.max(data)
    if sigma <= 0 or not (0 < pi < 1):
        return np.inf
    norm_pdf = norm.pdf(data, mu, sigma)
    unif_pdf = uniform.pdf(data, loc=a, scale=b - a)
    total_likelihood = pi * norm_pdf + (1 - pi) * unif_pdf
    return -np.sum(np.log(total_likelihood))

def fit_gaussian_uniform_mixture(data):
    mu0 = np.mean(data)
    sigma0 = np.std(data)
    pi0 = 0.9
    result = minimize(
        gaussian_uniform_mixture_log_likelihood,
        x0=[mu0, sigma0, pi0],
        args=(data,),
        bounds=[(None, None), (1e-6, None), (1e-3, 1 - 1e-3)],
        method='L-BFGS-B'
    )
    return result.x  # mu, sigma, pi

#===================
# Fit & Plot Section
#===================
def plot_gumm(data, title, xlabel):
    mu, sigma, pi = fit_gaussian_uniform_mixture(data)
    a, b = np.min(data), np.max(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, _ = ax.hist(data, bins=120, density=True, alpha=0.6, edgecolor='black', label="Histogram")

    x = np.linspace(a, b, 1000)
    y_mix = pi * norm.pdf(x, mu, sigma) + (1 - pi) * uniform.pdf(x, loc=a, scale=b - a)
    y_norm = pi * norm.pdf(x, mu, sigma)
    y_unif = (1 - pi) * uniform.pdf(x, loc=a, scale=b - a)

    ax.plot(x, y_mix, 'k-', label='Mixture')
    ax.plot(x, y_norm, 'r--', label='Normal component')
    ax.plot(x, y_unif, 'g--', label='Uniform component')
    ax.axvline(0, color='blue', linestyle='--', label='Zero Growth')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"📊 Fitted parameters for {title}:\nμ = {mu:.3f}, σ = {sigma:.3f}, π = {pi:.3f}\n")
#===============
# Apply to Data
#===============
data_length = growth_matrix_per_cell["avg_d_cell_length"].values
data_d_area = growth_matrix_per_cell["avg_d_cell_area"].values
data_max_area = growth_matrix_per_cell["max_cell_area"].values
data_min_area = growth_matrix_per_cell["min_cell_area"].values
data_std_d_area = growth_matrix_per_cell["std_d_cell_area"].values
data_max_abs_d_area = growth_matrix_per_cell["max_abs_d_cell_area"].values
data_max_abs_d_nu_dis = growth_matrix_per_cell["max_abs_d_nu_dis"].values

plot_gumm(data_length,
          title="Gaussian-Uniform Mixture Fit for Δ Cell Length",
          xlabel="Average Smoothed d(cell_length) (px)")

plot_gumm(data_d_area,
          title="Gaussian-Uniform Mixture Fit for Δ Cell Area",
          xlabel="Average Smoothed d(cell_area) (px²)")

plot_gumm(data_max_area,
          title="Gaussian-Uniform Mixture Fit for Max Cell Area",
          xlabel="Area (px²)")

plot_gumm(data_min_area,
          title="Gaussian-Uniform Mixture Fit for Min Cell Area",
          xlabel="Area (px²)")

# Optional: visualize new metrics
plot_gumm(data_std_d_area,
          title="Gaussian-Uniform Mixture Fit for Std Dev of d(Cell Area)",
          xlabel="Std Dev of Smoothed d(cell_area) (px²)")

plot_gumm(data_max_abs_d_area,
          title="Gaussian-Uniform Mixture Fit for Max Abs d(Cell Area)",
          xlabel="Max Abs Value of Smoothed d(cell_area) (px²)")

plot_gumm(data_max_abs_d_nu_dis,
          title="Gaussian-Uniform Mixture Fit for Max Abs d(Nu Dis)",
          xlabel="Max Abs Value of Smoothed d(nu_dis) (px²)")

#%%
# Step 1: Fit the GUMM model to avg_d_cell_area
area_data = growth_matrix_per_cell["avg_d_cell_area"].values
mu_area, sigma_area, pi_area = fit_gaussian_uniform_mixture(area_data)

# Step 2: Define extreme area thresholds (e.g., mean ± 1.96σ)
n_sigma = 1.96
upper_thresh = mu_area + n_sigma * sigma_area
lower_thresh = mu_area - n_sigma * sigma_area

# Step 3: Filter cells with avg_d_cell_area outside the normal range
extreme_avg_area_cells = growth_matrix_per_cell[
    (growth_matrix_per_cell['avg_d_cell_area'] > upper_thresh) |
    (growth_matrix_per_cell['avg_d_cell_area'] < lower_thresh)
]

# Step 4: Fit GUMM and threshold for max_cell_area
max_area_data = growth_matrix_per_cell["max_cell_area"].values
mu_max, sigma_max, _ = fit_gaussian_uniform_mixture(max_area_data)
upper_max_thresh = mu_max + n_sigma * sigma_max
extreme_max_area_cells = growth_matrix_per_cell[
    growth_matrix_per_cell["max_cell_area"] > upper_max_thresh
]

# Step 5: Fit GUMM and threshold for min_cell_area
min_area_data = growth_matrix_per_cell["min_cell_area"].values
mu_min, sigma_min, _ = fit_gaussian_uniform_mixture(min_area_data)
lower_min_thresh = mu_min - n_sigma * sigma_min
extreme_min_area_cells = growth_matrix_per_cell[
    growth_matrix_per_cell["min_cell_area"] < lower_min_thresh
]

# GUMM fit on std_d_cell_area
std_data = growth_matrix_per_cell["std_d_cell_area"].values
mu_std, sigma_std, _ = fit_gaussian_uniform_mixture(std_data)
upper_std_thresh = mu_std + n_sigma * sigma_std
# Identify extreme std and max_abs
extreme_std_cells = growth_matrix_per_cell[
    growth_matrix_per_cell["std_d_cell_area"] > upper_std_thresh
]

# GUMM fit on max_abs_d_cell_area
max_abs_data = growth_matrix_per_cell["max_abs_d_cell_area"].values
mu_abs, sigma_abs, _ = fit_gaussian_uniform_mixture(max_abs_data)
upper_abs_thresh = mu_abs + n_sigma * sigma_abs
extreme_max_abs_cells = growth_matrix_per_cell[
    growth_matrix_per_cell["max_abs_d_cell_area"] > upper_abs_thresh
]

# GUMM fit on max_abs_d_nu_dis
max_nu_data = growth_matrix_per_cell["max_abs_d_nu_dis"].values
mu_nu, sigma_nu, _ = fit_gaussian_uniform_mixture(max_nu_data)
upper_nu_thresh = 10 #mu_nu + n_sigma * sigma_nu
extreme_max_nu_cells = growth_matrix_per_cell[
    growth_matrix_per_cell["max_abs_d_nu_dis"] > upper_nu_thresh
]

# Combine all extreme cell_ids
extreme_cell_ids = pd.concat([
    extreme_avg_area_cells['cell_id'],
    extreme_max_area_cells['cell_id'],
    extreme_min_area_cells['cell_id'],
    extreme_std_cells['cell_id'],
    extreme_max_abs_cells['cell_id'],
    extreme_max_nu_cells['cell_id'],
    pd.Series([788, 899, 1061])
]).drop_duplicates()

# Final removal
print(f"📌 Removing {len(extreme_cell_ids)} extreme cells based on multiple criteria:")
print(extreme_cell_ids.tolist())

df_all = df_all[~df_all['cell_id'].isin(extreme_cell_ids)]



# #%% Clean NaNs
# df_clean = df_all[['time_point','cell_id', 'cell_length','cell_area','nu_dis','d_cell_length_avg5',
#                    'nu_int', 'septum_int', 'pol1_int', 'pol2_int', 'cyt_int','avg_d_cell_length']].dropna()
#%% check extreme cells id
#extreme_ids = extreme_max_nu_cells['cell_id'].unique()
extreme_ids = [788, 899, 1061]

# Trace info from df_all
extreme_info = df_all[df_all['cell_id'].isin(extreme_ids)][
    ['cell_id', 'original_cell_id', 'source_file']
].drop_duplicates()

print(extreme_info)



#%% Feature time course plots
features = ['cell_area', 'nu_dis', 'nu_int', 'cyt_int', 'septum_int', 'pol1_int', 'pol2_int']
#cell_ids = df_all['cell_id'].unique()
#cell_ids = extreme_info['cell_id'].unique()
cell_ids = [788, 899, 1061]


for feature in features:
    plt.figure(figsize=(8, 5))
    for cell_id in cell_ids:
        df_cell = df_all[df_all['cell_id'] == cell_id]
        plt.plot(df_cell['time_point'], df_cell[feature], marker='o', label=f"Cell {cell_id}")
    plt.title(f"{feature} Over Time")
    plt.xlabel("Time Point")
    plt.ylabel(feature)
    plt.legend()
    plt.tight_layout()
    plt.show()

#%% Global time warping alignment with MCMC for MSE

import seaborn as sns
import matplotlib.pyplot as plt

# Ensure seaborn styles
sns.set(style="whitegrid")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

df_all['septum_int_corr'] = df_all['septum_int'] - df_all['cyt_int']
df_all['pol1_int_corr'] = df_all['pol1_int'] - df_all['cyt_int']
df_all['pol2_int_corr'] = df_all['pol2_int'] - df_all['cyt_int']
df_all['pol1_minus_pol2'] = df_all['pol1_int_corr']  - df_all['pol2_int_corr']
df_all['weighted_area'] = df_all['cell_area']/100
# ---- Setup ----
features_xcorr = ['nu_dis', 'weighted_area', 'septum_int_corr']
lambda_reg = 0.1  # Tune this value as needed

cell_ids = df_all['cell_id'].unique()
time_points = sorted(df_all['time_point'].unique())
T = len(time_points)
n_features = len(features_xcorr)
padding = 9.5 * T
global_time = np.arange(0, T + padding)
shift_range = len(global_time) - T

# Create signal dict (features × timepoints)
cell_signals = {}
for cell_id in cell_ids:
    df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
    matrix = np.array([
        df_cell.set_index("time_point").reindex(time_points)[f].values
        for f in features_xcorr
    ])
    cell_signals[cell_id] = matrix

# ---- MSE computation function ----
def compute_total_mse(shifts, lambda_reg=0.0):
    acc = np.zeros((n_features, len(global_time)))
    weights = np.zeros_like(acc)

    #cyt_accumulator = [[] for _ in range(len(global_time))]

    for cell_id in cell_ids:
        signal = cell_signals[cell_id]
        shift = shifts[cell_id]
        valid = ~np.isnan(signal)
        acc[:, shift:shift+T] += np.where(valid, signal, 0)
        weights[:, shift:shift+T] += valid

        # NEW: track cyt_int values per global timepoint
        # df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
        # cyt_values = df_cell.set_index("time_point").reindex(time_points)["cyt_int"].values
        # aligned = np.full_like(global_time, np.nan, dtype=np.float64)
        # aligned[shift:shift+T] = cyt_values

        # for t, val in enumerate(aligned):
        #     if not np.isnan(val):
        #         cyt_accumulator[t].append(val)

    avg = np.divide(acc, weights, where=weights != 0)

    total_mse = 0
    for cell_id in cell_ids:
        signal = cell_signals[cell_id]
        shift = shifts[cell_id]
        valid = ~np.isnan(signal) & (weights[:, shift:shift+T] > 0)
        mse = np.mean((signal[valid] - avg[:, shift:shift+T][valid])**2)
        total_mse += mse

    # # NEW: Regularization - variance of the cyt_int variances
    # variances = [np.var(vals) for vals in cyt_accumulator if len(vals) > 1]
    # if len(variances) > 1:
    #     var_of_var = np.var(variances)
    # else:
    #     var_of_var = 0.0

    total_cost = total_mse #+ lambda_reg * var_of_var

    return total_cost, avg


# ---- MCMC Optimization ----
# Step 1: Compute mean cell length per cell
mean_cell_lengths = {
    cell_id: np.nanmean(cell_signals[cell_id][features_xcorr.index('weighted_area')])
    for cell_id in cell_ids
}

# Step 2: Sort cells by increasing mean cell length
sorted_cells = sorted(mean_cell_lengths, key=mean_cell_lengths.get)

# Step 3: Evenly space shifts across allowed range
num_cells = len(sorted_cells)
available_shift_range = shift_range  # already = len(global_time) - T

# Evenly space shifts, e.g. 0, gap, 2*gap, ..., up to shift_range
shift_values = np.linspace(0, available_shift_range, num_cells).astype(int)
initial_shifts = {
    cell_id: shift for cell_id, shift in zip(sorted_cells, shift_values)
}

current_shifts = initial_shifts.copy()
best_shifts = current_shifts.copy()
best_score, best_mean = compute_total_mse(best_shifts, lambda_reg)


# ---- Plot Initial Alignment ----
fig, axs = plt.subplots(nrows=n_features, figsize=(10, 2.5 * n_features), sharex=True)

for i, feature in enumerate(features_xcorr):
    ax = axs[i]

    for cell_id in sorted_cells:
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
        values = df_cell.set_index("time_point").reindex(time_points)[feature].values

        shift = initial_shifts[cell_id]
        aligned = np.full_like(global_time, np.nan, dtype=np.float64)
        aligned[shift:shift+T] = values

        ax.plot(global_time, aligned, alpha=0.3)

    ax.set_title(f"Initial Aligned: {feature}")
    ax.set_ylabel("Value")
    ax.grid(True)

axs[-1].set_xlabel("Global Master Timeline (Initial Alignment)")
plt.suptitle("Initial Condition: Aligned Cell Tracks Before MCMC", fontsize=16)
plt.tight_layout()
plt.show()
#
n_iter = 100000
#temperature = 0.1
mse_trace = [best_score]
initial_temp = 1.0
for i in range(n_iter):
    temperature = initial_temp * (0.99 ** (i/1))

    proposal = best_shifts.copy()
    cell = random.choice(list(cell_ids))
    direction = random.choice([ -10, -1, 1, 10])

    new_shift = np.clip(proposal[cell] + direction, 0, shift_range)
    proposal[cell] = new_shift

    proposed_score, proposed_mean = compute_total_mse(proposal, lambda_reg)
    delta = proposed_score - best_score

    if delta < 0 or np.exp(-delta / temperature) > np.random.rand():
        best_shifts = proposal
        best_score = proposed_score
        best_mean = proposed_mean  


    mse_trace.append(best_score)

    if i % 100 == 0:
        print(f"Step {i}: total MSE = {best_score:.4f}")

# ---- Plot MSE trace ----
plt.figure(figsize=(8, 4))
plt.plot(mse_trace)
plt.xlabel("Iteration")
plt.ylabel("Total MSE")
plt.title("MCMC Optimization Trace")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Plot aligned signals ----
fig, axs = plt.subplots(nrows=n_features, figsize=(10, 2.5 * n_features), sharex=True)

for i, feature in enumerate(features_xcorr):
    ax = axs[i]

    for cell_id in cell_ids:
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
        values = df_cell.set_index("time_point").reindex(time_points)[feature].values

        shift = best_shifts[cell_id]
        aligned = np.full_like(global_time, np.nan, dtype=np.float64)
        aligned[shift:shift+T] = values

        ax.plot(global_time, aligned, alpha=0.3)

    ax.plot(global_time, best_mean[i], color='black', linewidth=2, label='Mean')
    ax.set_title(f"{feature}")
    ax.set_ylabel("Value")
    ax.grid(True)

axs[-1].set_xlabel("Global Master Timeline (Aligned Time Points)")
plt.suptitle("Aligned Cell Tracks on Master Timeline (MCMC)", fontsize=16)
plt.tight_layout()
plt.show()

#%

# Define all other features to visualize
additional_features = [
    'pol1_int_corr', 'pol2_int_corr', 'pol1_minus_pol2'
]

n_additional = len(additional_features)

fig, axs = plt.subplots(nrows=n_additional, figsize=(10, 2.5 * n_additional), sharex=True)

for i, feature in enumerate(additional_features):
    ax = axs[i]

    # Initialize accumulator and weights
    acc = np.zeros(len(global_time))
    weight = np.zeros(len(global_time))

    for cell_id in cell_ids:
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
        values = df_cell.set_index("time_point").reindex(time_points)[feature].values

        shift = best_shifts[cell_id]
        aligned = np.full_like(global_time, np.nan, dtype=np.float64)
        aligned[shift:shift+T] = values

        ax.plot(global_time, aligned, alpha=0.3)

        # Update mean accumulator
        valid = ~np.isnan(aligned)
        acc[valid] += aligned[valid]
        weight[valid] += 1

    # Compute and plot mean
    mean_trace = np.divide(acc, weight, out=np.zeros_like(acc), where=weight != 0)
    ax.plot(global_time, mean_trace, color='black', linewidth=2, label='Mean')

    ax.set_title(f"{feature}")
    ax.set_ylabel("Value")
    ax.grid(True)

axs[-1].set_xlabel("Global Master Timeline (Aligned Time Points)")
plt.suptitle("Additional Aligned Cell Tracks with Mean (MCMC)", fontsize=16)
plt.tight_layout()
plt.show()
#%% See IDs of a specific time point

# Specify the global time index you're interested in
t_query = 300  # change this to your time point of interest

# Find contributing cell_ids
contributing_cells = [
    cell_id for cell_id in cell_ids
    if best_shifts[cell_id] <= t_query < best_shifts[cell_id] + T
]

print(f"🔍 At global time {t_query}, {len(contributing_cells)} cells contribute.")
print("Cell IDs:", contributing_cells)

#%% Create a list to collect aligned time info
aligned_time_records = []

for cell_id in cell_ids:
    shift = best_shifts[cell_id]

    # Extract the original time points for this cell
    df_cell = df_all[df_all["cell_id"] == cell_id][["cell_id", "time_point"]].copy()

    # Compute global aligned time for each time_point
    df_cell["aligned_time"] = df_cell["time_point"] + shift

    # Append to list
    aligned_time_records.append(df_cell)

# Concatenate all results
df_aligned_time = pd.concat(aligned_time_records, ignore_index=True)

# Merge aligned_time back into df_all
df_all = df_all.merge(df_aligned_time, on=["cell_id", "time_point"])

# Define the output file path
output_path = os.path.join(working_dir, "combined_all_cells_with_aligned_time.csv")

# Save to CSV
df_all.to_csv(output_path, index=False)

print(f"✅ Saved aligned data to: {output_path}")

#%% Heatmap for aligned time points
import seaborn as sns

# Sort cells by first appearance on the master timeline
ordered_cells_by_shift = sorted(cell_ids, key=lambda cid: best_shifts[cid])

# Create heatmap layout
fig, axs = plt.subplots(nrows=n_additional, figsize=(10, 2.5 * n_additional), sharex=True)

for i, feature in enumerate(additional_features):
    heatmap_data = []

    for cell_id in ordered_cells_by_shift:
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
        values = df_cell.set_index("time_point").reindex(time_points)[feature].values

        shift = best_shifts[cell_id]
        aligned = np.full_like(global_time, np.nan, dtype=np.float64)
        aligned[shift:shift+T] = values

        # For "pol1_minus_pol2", flip if mean is negative
        if i == 2:  # third plot
            mean_val = np.nanmean(values)
            if mean_val < 0:
                aligned = -aligned

        heatmap_data.append(aligned)

    heatmap_array = np.array(heatmap_data)

    # Apply fixed color scale only to the third feature
    if i == 2:
        sns.heatmap(
            heatmap_array,
            ax=axs[i],
            cmap="RdBu",
            cbar=True,
            vmin=-20,
            vmax=20,
            xticklabels=False,
            yticklabels=False
        )
    else:
        sns.heatmap(
            heatmap_array,
            ax=axs[i],
            cmap="viridis",
            cbar=True,
            #vmin=0,
            #vmax=5,
            xticklabels=False,
            yticklabels=False
        )

    axs[i].set_title(f"{feature}")
    axs[i].set_ylabel("Cells (Ordered by Appearance)")

axs[-1].set_xlabel("Global Master Timeline (Aligned Time Points)")
plt.suptitle("Aligned Feature Heatmaps (MCMC)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
#%% analize the master timeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Configuration ----
features = ['cell_length', 'cell_area', 'nu_dis', 'nu_int', 'cyt_int',
            'd_cell_length', 'd_cell_area', 'septum_int_corr',
            'pol1_int_corr', 'pol2_int_corr', 'pol1_minus_pol2']
num_bins = 100
timeline_col = 'aligned_time'

# ---- Prepare Bins ----
bins = {feature: np.linspace(df_all[feature].min(), df_all[feature].max(), num_bins + 1)
        for feature in features}

# ---- Compute Probability Distributions ----
heatmap_data = {feature: [] for feature in features}
time_points = sorted(df_all[timeline_col].unique())

for t in time_points:
    df_t = df_all[df_all[timeline_col] == t]
    for feature in features:
        counts, _ = np.histogram(df_t[feature], bins=bins[feature])
        prob_density = counts / counts.sum() if counts.sum() > 0 else np.zeros_like(counts)
        heatmap_data[feature].append(prob_density)

# ---- Convert to Heatmap DataFrames ----
heatmap_dfs = {
    feature: pd.DataFrame(
        data=np.array(heatmap_data[feature]).T,
        index=(bins[feature][:-1] + bins[feature][1:]) / 2,
        columns=time_points
    )
    for feature in features
}

#%% ---- Plot All Heatmaps with Flipped Y-Axis and Rounded Tick Labels ----
fig, axs = plt.subplots(len(features), 1, figsize=(14, len(features)*2.5), constrained_layout=True)

for ax, feature in zip(axs, features):
    data_to_plot = heatmap_dfs[feature]

    # Apply log10 transform for 'pol1_minus_pol2'
    if feature == 'pol1_minus_pol2':
        epsilon = 1e-3
        data_to_plot = np.log10(data_to_plot + epsilon)
        color_map = 'magma'
        value_min = np.log10(epsilon)
        value_max = 0
        color_label = 'log10(Empirical Probability)'
    else:
        color_map = 'magma'
        value_min = 0
        value_max = 0.05
        color_label = 'Empirical Probability'

    heatmap = sns.heatmap(
        data_to_plot,
        ax=ax,
        cmap=color_map,
        cbar_kws={'label': color_label},
        xticklabels=50,
        vmin=value_min,
        vmax=value_max,
        yticklabels=False
    )
    ax.set_title(f'{feature} over Aligned Time')
    ax.set_ylabel(feature)
    ax.set_xlabel('Aligned Time')
    ax.invert_yaxis()

    # Round y-tick labels to 2 decimal places
    y_ticks = ax.get_yticks()
    ax.set_yticklabels([f"{val:.2f}" for val in heatmap.get_yticks()])




#%% Fit FFT
import numpy as np
import matplotlib.pyplot as plt

# Define features of interest
features_fft = ['pol1_int_corr', 'pol2_int_corr']#, 'pol1_minus_pol2']
colors = ['tab:blue', 'tab:orange']#, 'tab:green']

# Sampling frequency assumptions (1 unit per time point)
fs = 1.0
freqs = np.fft.rfftfreq(len(time_points), d=1/fs)

# Create a plot
plt.figure(figsize=(10, 5))

for idx, feature in enumerate(features_fft):
    aligned_positions = []
    dominant_freqs = []

    for cell_id in cell_ids:
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
        values = df_cell.set_index("time_point").reindex(time_points)[feature].values

        if np.sum(~np.isnan(values)) >= 3:
            values_detrended = values - np.nanmean(values)
            values_filled = np.nan_to_num(values_detrended)

            fft_values = np.fft.rfft(values_filled)
            power = np.abs(fft_values)

            dominant_freq = freqs[np.argmax(power[1:]) + 1]  # skip zero-frequency
            shift = best_shifts[cell_id]
            aligned_positions.append(global_time[shift])
            dominant_freqs.append(dominant_freq)

    # Plot
    plt.scatter(aligned_positions, dominant_freqs, label=feature, alpha=0.7, s=30, color=colors[idx])

# Final plot adjustments
plt.xlabel("Global Master Timeline (Aligned Start Time)")
plt.ylabel("Dominant Frequency")
plt.title("Most Significant Frequency per Cell (FFT)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define features
features_fft = ['pol1_int_corr', 'pol2_int_corr']
colors = ['tab:blue', 'tab:orange']

# Sampling frequency
fs = 1.0
freqs = np.fft.rfftfreq(len(time_points), d=1/fs)

# Define 3 time periods
period_edges = np.array_split(global_time, 3)
period_bounds = [(p[0], p[-1]) for p in period_edges]

# ---- Precompute dominant frequencies per cell and feature ----
dominant_freqs_all = {f: [[] for _ in range(3)] for f in features_fft}
all_dominant_freqs = []

for cell_id in cell_ids:
    df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
    shift = best_shifts[cell_id]
    start_time = global_time[shift]

    for f_idx, feature in enumerate(features_fft):
        values = df_cell.set_index("time_point").reindex(time_points)[feature].values
        if np.sum(~np.isnan(values)) >= 3:
            detrended = values - np.nanmean(values)
            filled = np.nan_to_num(detrended)
            fft_values = np.fft.rfft(filled)
            power = np.abs(fft_values)
            dom_freq = freqs[np.argmax(power[1:]) + 1]  # skip zero

            for p_idx, (start, end) in enumerate(period_bounds):
                if start <= start_time <= end:
                    dominant_freqs_all[feature][p_idx].append(dom_freq)
                    all_dominant_freqs.append(dom_freq)
                    break

# ---- Determine shared bins ----
shared_bins = np.histogram_bin_edges(all_dominant_freqs, bins=15)

# ---- Plot ----
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

for p_idx in range(3):
    for f_idx, feature in enumerate(features_fft):
        sns.histplot(
            dominant_freqs_all[feature][p_idx],
            ax=axs[p_idx],
            bins=shared_bins,
            kde=True,
            stat='probability',
            color=colors[f_idx],
            label=feature,
            alpha=0.5
        )

    axs[p_idx].set_title(f"Period {p_idx+1} ({period_bounds[p_idx][0]}–{period_bounds[p_idx][1]})")
    axs[p_idx].set_xlabel("Dominant Frequency")
    axs[p_idx].set_xlim(shared_bins[0], shared_bins[-1])
    axs[p_idx].grid(True)
    if p_idx == 0:
        axs[p_idx].set_ylabel("Probability")
    axs[p_idx].legend()

plt.suptitle("Probability Distribution of Dominant Frequencies Across Periods", fontsize=16)
plt.tight_layout()
plt.show()
#%%
import GPy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

features_gp = ['pol1_int_corr', 'pol2_int_corr']
colors = ['tab:blue', 'tab:orange']

# Define 5 global time periods
period_edges = np.array_split(global_time, 5)
period_bounds = [(p[0], p[-1]) for p in period_edges]

# Store results
learned_periods = {f: [[] for _ in range(5)] for f in features_gp}
mean_values = {f: [[] for _ in range(5)] for f in features_gp}
all_periods = []
all_means = []

for cell_id in cell_ids:
    df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
    shift = best_shifts[cell_id]
    start_time = global_time[shift]
    t_obs = np.array(time_points).reshape(-1, 1)

    for f_idx, feature in enumerate(features_gp):
        y = df_cell.set_index("time_point").reindex(time_points)[feature].values.reshape(-1, 1)

        # Skip if too many NaNs
        if np.sum(~np.isnan(y)) < 5:
            continue

        y_flat = y.flatten()
        y_detrended = np.nan_to_num(y_flat - np.nanmean(y_flat))  # Detrend for GP
        mean_val = np.nanmean(y_flat)

        # Fit GP
        kernel = GPy.kern.StdPeriodic(input_dim=1)
        kernel.lengthscale.set_prior(GPy.priors.Gamma(5.0, 1.0))
        kernel.period.constrain_bounded(2.0, 50.0)

        model = GPy.models.GPRegression(t_obs, y_detrended.reshape(-1, 1), kernel)
        model.optimize(messages=False, max_iters=500)

        period = float(model.kern.period.values[0])

        # Assign to one of the 5 periods
        for p_idx, (start, end) in enumerate(period_bounds):
            if start <= start_time <= end:
                learned_periods[feature][p_idx].append(period)
                mean_values[feature][p_idx].append(mean_val)
                all_periods.append(period)
                all_means.append(mean_val)
                break

#%% ---- Plot Histograms ----

# Define bin edges for shared axes
period_bins = np.histogram_bin_edges(all_periods, bins=30)
mean_bins = np.histogram_bin_edges(all_means, bins=30)
period_centers = 0.5 * (period_bins[:-1] + period_bins[1:])
mean_centers = 0.5 * (mean_bins[:-1] + mean_bins[1:])
n_bins = len(period_centers)

fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20, 8), sharey='row')

for p_idx in range(5):
    # ----- Top row: GP Periods -----
    ax_top = axs[0, p_idx]
    bottoms_top = np.zeros(n_bins)

    for f_idx, feature in enumerate(features_gp):
        data = np.array(learned_periods[feature][p_idx])
        hist, _ = np.histogram(data, bins=period_bins)
        prob = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist)

        ax_top.bar(
            period_centers,
            prob,
            width=np.diff(period_bins),
            bottom=bottoms_top,
            color=colors[f_idx],
            alpha=0.7,
            label=feature if p_idx == 0 else None,
            edgecolor='black'
        )
        bottoms_top += prob

    ax_top.set_title(f"GP Periods (T{p_idx + 1}: {period_bounds[p_idx][0]}–{period_bounds[p_idx][1]})")
    ax_top.set_xlabel("Inferred GP Period")
    if p_idx == 0:
        ax_top.set_ylabel("Probability")
    ax_top.set_xlim(period_bins[0], period_bins[-1])
    ax_top.grid(True)

    # ----- Bottom row: Mean Values -----
    ax_bot = axs[1, p_idx]
    bottoms_bot = np.zeros(len(mean_centers))

    for f_idx, feature in enumerate(features_gp):
        data = np.array(mean_values[feature][p_idx])
        hist, _ = np.histogram(data, bins=mean_bins)
        prob = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist)

        ax_bot.bar(
            mean_centers,
            prob,
            width=np.diff(mean_bins),
            bottom=bottoms_bot,
            color=colors[f_idx],
            alpha=0.7,
            label=feature if p_idx == 0 else None,
            edgecolor='black'
        )
        bottoms_bot += prob

    ax_bot.set_title(f"Mean Feature Values (T{p_idx + 1})")
    ax_bot.set_xlabel("Mean Value")
    if p_idx == 0:
        ax_bot.set_ylabel("Probability")
    ax_bot.set_xlim(mean_bins[0], mean_bins[-1])
    ax_bot.grid(True)

# Final formatting
axs[0, 0].legend(loc='upper right')
axs[1, 0].legend(loc='upper right')
fig.suptitle("Top: GP-Inferred Periods | Bottom: Mean Feature Values Across Time Points", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#%%
import numpy as np
import matplotlib.pyplot as plt

# Setup bin edges for heatmap
period_bins = np.histogram_bin_edges(all_periods, bins=30)
mean_bins = np.histogram_bin_edges(all_means, bins=30)

fig, axs = plt.subplots(nrows=len(features_gp), ncols=5, figsize=(22, 8), sharex=True, sharey=True)

for f_idx, feature in enumerate(features_gp):
    for p_idx in range(5):
        ax = axs[f_idx, p_idx] if len(features_gp) > 1 else axs[p_idx]

        periods = np.array(learned_periods[feature][p_idx])
        means = np.array(mean_values[feature][p_idx])

        if len(periods) == 0 or len(means) == 0:
            ax.set_visible(False)
            continue

        # Plot 2D histogram (heatmap)
        h = ax.hist2d(periods, means, bins=[period_bins, mean_bins], cmap="viridis", density=True)

        ax.set_xlim(period_bins[0], period_bins[-1])
        ax.set_ylim(mean_bins[0], mean_bins[-1])

        if f_idx == len(features_gp) - 1:
            ax.set_xlabel("GP-Inferred Period")

        if p_idx == 0:
            ax.set_ylabel(f"{feature}\nMean Value")

        ax.set_title(f"T{p_idx+1}: {period_bounds[p_idx][0]}–{period_bounds[p_idx][1]}")
        ax.grid(False)

# Add colorbar
fig.subplots_adjust(right=0.87)
cbar_ax = fig.add_axes([0.9, 0.15, 0.015, 0.7])
fig.colorbar(h[3], cax=cbar_ax, label="Density")

fig.suptitle("2D Heatmaps: GP-Inferred Period vs Mean Feature Value", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 0.88, 0.95])
plt

#%% ---- Plot Period vs Aligned Start Time ----
plt.figure(figsize=(10, 5))

for f_idx, feature in enumerate(features_gp):
    x_vals = []  # global time at first alignment
    y_vals = []  # learned period

    for cell_id in cell_ids:
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
        shift = best_shifts[cell_id]
        start_time = global_time[shift]

        # Check if we have a learned period
        t_obs = np.array(time_points).reshape(-1, 1)
        y = df_cell.set_index("time_point").reindex(time_points)[feature].values.reshape(-1, 1)
        if np.sum(~np.isnan(y)) < 5:
            continue

        y = np.nan_to_num(y - np.nanmean(y))

        kernel = GPy.kern.StdPeriodic(input_dim=1)
        kernel.lengthscale.set_prior(GPy.priors.Gamma(5.0, 1.0))
        kernel.period.constrain_bounded(2.0, 1000.0)

        model = GPy.models.GPRegression(t_obs, y, kernel)
        model.optimize(messages=False, max_iters=500)

        period = float(model.kern.period.values[0])
        x_vals.append(start_time)
        y_vals.append(period)

    # Scatter plot for current feature
    plt.scatter(x_vals, y_vals, color=colors[f_idx], label=feature, alpha=0.7, s=30)

plt.xlabel("Aligned Start Time on Global Timeline")
plt.ylabel("Inferred GP Period")
plt.title("GP Periodicity at First Aligned Time Point")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%% ---- Plot Mean Feature Value vs Aligned Start Time ----
features_gp = ['pol1_int_corr', 'pol2_int_corr']#, 'pol1_minus_pol2']
colors = ['tab:blue', 'tab:orange']#, 'tab:green']
plt.figure(figsize=(10, 5))

for f_idx, feature in enumerate(features_gp):
    x_vals = []  # aligned global start time for each cell
    y_vals = []  # mean feature value across all 51 time points

    for cell_id in cell_ids:
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
        shift = best_shifts[cell_id]
        start_time = global_time[shift]  # aligned start time on global timeline

        # Get full feature time series
        y = df_cell.set_index("time_point").reindex(time_points)[feature].values

        # Skip if not enough data
        if np.sum(~np.isnan(y)) < 5:
            continue

        mean_val = np.nanmean(y)  # average across all 51 time points

        x_vals.append(start_time)
        y_vals.append(mean_val)

    # Plot for this feature
    plt.scatter(x_vals, y_vals, color=colors[f_idx], label=feature, alpha=0.7, s=30)

plt.xlabel("Aligned Start Time on Global Timeline")
plt.ylabel("Mean Feature Value (across time)")
plt.title("Mean Feature Value vs Aligned Start Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

#%%
import numpy as np
import matplotlib.pyplot as plt

# Store mean values by period
period_edges = np.array_split(np.sort(global_time), 5)
period_bounds = [(p[0], p[-1]) for p in period_edges]

# Collect mean values per cell, grouped by time period
period_means = [[] for _ in range(5)]  # each entry is a list of (mean_pol1, mean_pol2)

for cell_id in cell_ids:
    df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
    shift = best_shifts[cell_id]
    start_time = global_time[shift]

    y1 = df_cell.set_index("time_point").reindex(time_points)["pol1_int_corr"].values
    y2 = df_cell.set_index("time_point").reindex(time_points)["pol2_int_corr"].values

    if np.sum(~np.isnan(y1)) < 5 or np.sum(~np.isnan(y2)) < 5:
        continue

    mean1 = np.nanmean(y1)
    mean2 = np.nanmean(y2)

    for p_idx, (start, end) in enumerate(period_bounds):
        if start <= start_time <= end:
            period_means[p_idx].append((mean1, mean2))
            break

#%% ---- Plot Scatter ----
fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharex=True, sharey=True)

for p_idx in range(5):
    ax = axs[p_idx]
    points = np.array(period_means[p_idx])
    if len(points) == 0:
        continue
    x_vals = points[:, 0]  # mean of pol1
    y_vals = points[:, 1]  # mean of pol2

    ax.scatter(x_vals, y_vals, color='purple', alpha=0.6, s=30, edgecolor='k')
    ax.set_title(f"T{p_idx+1}: {period_bounds[p_idx][0]}–{period_bounds[p_idx][1]}")
    ax.set_xlabel("Mean pol1_int_corr")
    if p_idx == 0:
        ax.set_ylabel("Mean pol2_int_corr")
    ax.grid(True)

fig.suptitle("Covariation Between Mean Feature Values Across 5 Aligned Periods", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#%%
import GPy
import numpy as np
import matplotlib.pyplot as plt

# ==== Helper Functions ====

def fit_exponential_gp(t, y, lengthscale_prior=(2.0, 3.0)):
    """Fit an Exponential GP model to the data."""
    kernel = GPy.kern.Exponential(input_dim=1)
    kernel.lengthscale.set_prior(GPy.priors.Gamma(*lengthscale_prior))
    model = GPy.models.GPRegression(t, y, kernel)
    model.optimize(messages=False)
    mu, _ = model.predict(t)
    return model, mu

def fit_periodic_gp(t, y, period_bounds=(2.0, 50.0)):
    """Fit a Periodic GP model to the detrended data."""
    kernel = GPy.kern.StdPeriodic(input_dim=1)
    kernel.lengthscale.set_prior(GPy.priors.Gamma(1.0, 1.0))
    kernel.period.constrain_bounded(*period_bounds)
    model = GPy.models.GPRegression(t, y, kernel)
    model.optimize(messages=False, max_iters=500)
    return model

def fit_linear_plus_periodic_gp(t, y, period_bounds=(2.0, 60.0), fix_noise=False):
    """Fit a GP with Linear + Periodic kernel and controlled noise."""
    k_lin = GPy.kern.Linear(input_dim=1)
    k_per = GPy.kern.StdPeriodic(input_dim=1)
    k_per.lengthscale.set_prior(GPy.priors.Gamma(1.0, 1.0))
    k_per.period.constrain_bounded(*period_bounds)

    kernel = k_lin + k_per
    model = GPy.models.GPRegression(t, y, kernel)

    if fix_noise:
        model.Gaussian_noise.constrain_fixed(1e-2)
    else:
        model.Gaussian_noise.set_prior(GPy.priors.Gamma(2.0, 0.1))

    model.optimize(messages=False, max_iters=500)
    return model

from functools import reduce

def fit_multi_periodic_gp(t, y, period_bounds_list, fix_noise=False):
    """
    Fit GP with Linear + multiple periodic components.
    period_bounds_list: list of (min, max) tuples for each periodic kernel
    """
    k_lin = GPy.kern.Linear(input_dim=1)
    periodic_kernels = []

    for bounds in period_bounds_list:
        k_per = GPy.kern.StdPeriodic(input_dim=1)
        k_per.lengthscale.set_prior(GPy.priors.Gamma(1.0, 1.0))
        k_per.period.constrain_bounded(*bounds)
        periodic_kernels.append(k_per)

    if not periodic_kernels:
        kernel = k_lin
    else:
        kernel = reduce(lambda x, y: x + y, [k_lin] + periodic_kernels)

    model = GPy.models.GPRegression(t, y, kernel)

    if fix_noise:
        model.Gaussian_noise.constrain_fixed(1e-2)
    else:
        model.Gaussian_noise.set_prior(GPy.priors.Gamma(2.0, 0.1))

    model.optimize(messages=False, max_iters=1000)
    return model


# ==== Plot Setup ====

fig, axs = plt.subplots(5, 5, figsize=(20, 20), sharey=True)
axs = axs.flatten()

# Loop over 25 cells
for i, cell_id in enumerate(cell_ids[:25]):
    ax = axs[i]
    df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")

    t = df_cell["time_point"].values.reshape(-1, 1)
    y1 = df_cell["pol1_int_corr"].values.reshape(-1, 1)
    y2 = df_cell["pol2_int_corr"].values.reshape(-1, 1)

    # Filter valid rows
    valid = ~np.isnan(y1[:, 0]) & ~np.isnan(y2[:, 0])
    t = t[valid]
    y1 = y1[valid]
    y2 = y2[valid]

    if len(t) < 5:
        ax.set_visible(False)
        continue

    # Mean before detrending
    y1_mean = np.mean(y1)
    y2_mean = np.mean(y2)

 
    try:
        #model1 = fit_linear_plus_periodic_gp(t, y1)#, fix_noise=True)
        #model2 = fit_linear_plus_periodic_gp(t, y2)#, fix_noise=True)
        period_bounds_list=[(2.0, 20.0), (20.0, 40.0), (40.0, 60.0)]
        model1 = fit_multi_periodic_gp(t, y1-y1_mean, period_bounds_list)#, fix_noise=True)
        model2 = fit_multi_periodic_gp(t, y2-y2_mean, period_bounds_list)#, fix_noise=True)
    except Exception as e:
        print(f"Combined GP failed on Cell {cell_id}: {e}")
        ax.set_visible(False)
        continue

    t_pred = np.linspace(t.min(), t.max(), 200)[:, None]
    mu1, var1 = model1.predict(t_pred)
    mu2, var2 = model2.predict(t_pred)
    
    # Plot pol1
    ax.plot(t, y1, 'o', markersize=3, color='blue', label='pol1')
    ax.plot(t_pred, mu1+y1_mean, 'b-', label='GP pol1')
    ax.fill_between(t_pred[:, 0], mu1[:, 0] + y1_mean- 2*np.sqrt(var1[:, 0]), mu1[:, 0] +y1_mean + 2*np.sqrt(var1[:, 0]), color='blue', alpha=0.2)
    
    # Plot pol2
    ax.plot(t, y2, 'x', markersize=3, color='darkorange', label='pol2')
    ax.plot(t_pred, mu2+y2_mean, 'orange', label='GP pol2')
    ax.fill_between(t_pred[:, 0], mu2[:, 0] +y2_mean- 2*np.sqrt(var2[:, 0]), mu2[:, 0] +y2_mean+ 2*np.sqrt(var2[:, 0]), color='orange', alpha=0.2)

    # Formatting
    ax.set_ylim(-1, 40)
    ax.grid(True)
    if i % 5 == 0:
        ax.set_ylabel("Signal")
    if i >= 20:
        ax.set_xlabel("Time")

    # p1 = model1.kern.parts[1].period.values[0]
    # a1 = model1.kern.parts[1].variance.values[0]
    # slope1 = model1.kern.parts[0].variances[0]
    
    # p2 = model2.kern.parts[1].period.values[0]
    # a2 = model2.kern.parts[1].variance.values[0]
    # slope2 = model2.kern.parts[0].variances[0]
    
    # ax.set_title(
    #     f"Cell {cell_id}\n"
    #     f"pol1: P={p1:.1f}, A={a1:.1f}, slope={slope1:.2f} | "
    #     f"pol2: P={p2:.1f}, A={a2:.1f}, slope={slope2:.2f}",
    #     fontsize=7
    # )
    # --- Extract periodic components for pol1 ---
    periods1 = []
    amps1 = []
    for part in model1.kern.parts[1:]:  # skip linear at index 0
        periods1.append(part.period.values[0])
        amps1.append(part.variance.values[0])
    sorted1 = sorted(zip(periods1, amps1), key=lambda x: -x[1])
    
    # --- Extract periodic components for pol2 ---
    periods2 = []
    amps2 = []
    for part in model2.kern.parts[1:]:
        periods2.append(part.period.values[0])
        amps2.append(part.variance.values[0])
    sorted2 = sorted(zip(periods2, amps2), key=lambda x: -x[1])
    
    # --- Prepare string summary of top periodic components ---
    def fmt_top_components(sorted_list, top=3):
        return " / ".join([f"P={p:.1f}, A={a:.1f}" for p, a in sorted_list[:top]])
    
    slope1 = model1.kern.parts[0].variances[0]
    slope2 = model2.kern.parts[0].variances[0]
    
    ax.set_title(
        f"Cell {cell_id}\n"
        f"pol1: {fmt_top_components(sorted1)} | slope={slope1:.2f}\n"
        f"pol2: {fmt_top_components(sorted2)} | slope={slope2:.2f}",
        fontsize=7
    )


# Hide unused axes
for j in range(len(cell_ids[26:51]), 25):
    axs[j].set_visible(False)

# Shared legend and title
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4)
fig.suptitle("GP Fit (Exponential Detrending + Periodic Modeling) for pol1 and pol2", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


