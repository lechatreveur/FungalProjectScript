#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:56:46 2025

@author: user
"""

# spectral_analysis.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SingleCellDataAnalysis.gp_helpers import (
    fit_exponential_gp,
    fit_periodic_gp,
    fit_linear_plus_periodic_gp,
    fit_multi_periodic_gp
)


def fft_dominant_frequencies(df_all, cell_ids, best_shifts, global_time, time_points, features, fs=1.0):
    """
    Extract dominant frequency for each cell and feature using FFT.
    Returns a dict of {feature: list of (start_time, freq)}.
    """
    freqs = np.fft.rfftfreq(len(time_points), d=1/fs)
    results = {f: [] for f in features}

    for cell_id in cell_ids:
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
        shift = best_shifts[cell_id]
        start_time = global_time[shift]

        for feature in features:
            values = df_cell.set_index("time_point").reindex(time_points)[feature].values
            if np.sum(~np.isnan(values)) < 3:
                continue
            detrended = values - np.nanmean(values)
            filled = np.nan_to_num(detrended)
            fft_vals = np.fft.rfft(filled)
            power = np.abs(fft_vals)
            dom_freq = freqs[np.argmax(power[1:]) + 1]  # skip DC
            results[feature].append((start_time, dom_freq))

    return results


def plot_fft_dominant_frequencies(results):
    """
    Plot scatter of dominant frequencies vs aligned start times.
    """
    plt.figure(figsize=(10, 5))

    for feature, data in results.items():
        if not data:
            continue
        x_vals, y_vals = zip(*data)
        plt.scatter(x_vals, y_vals, label=feature, alpha=0.7, s=30)

    plt.xlabel("Aligned Start Time")
    plt.ylabel("Dominant Frequency")
    plt.title("FFT-Derived Dominant Frequencies")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

import GPy

def gp_infer_periods(df_all, cell_ids, best_shifts, global_time, time_points, features, period_bounds=(2.0, 60.0)):
    """
    Fit GP with periodic kernel and return inferred periods.
    """
    results = {f: [] for f in features}

    t_obs = np.array(time_points).reshape(-1, 1)

    for cell_id in cell_ids:
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
        shift = best_shifts[cell_id]
        start_time = global_time[shift]

        for feature in features:
            y = df_cell.set_index("time_point").reindex(time_points)[feature].values.reshape(-1, 1)
            if np.sum(~np.isnan(y)) < 5:
                continue
            y = y - np.nanmean(y)
            y = np.nan_to_num(y)

            kernel = GPy.kern.StdPeriodic(input_dim=1)
            kernel.period.constrain_bounded(*period_bounds)
            model = GPy.models.GPRegression(t_obs, y, kernel)
            model.optimize(messages=False, max_iters=500)

            period = float(model.kern.period.values[0])
            results[feature].append((start_time, period))

    return results


def plot_gp_periods(results):
    """
    Plot GP-inferred period vs aligned start time.
    """
    plt.figure(figsize=(10, 5))

    for feature, data in results.items():
        if not data:
            continue
        x_vals, y_vals = zip(*data)
        plt.scatter(x_vals, y_vals, label=feature, alpha=0.7, s=30)

    plt.xlabel("Aligned Start Time")
    plt.ylabel("GP-Inferred Period")
    plt.title("GP Period Inference from Periodic Kernel")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
# spectral_analysis.py (continued)

def plot_multi_periodic_gp_grid(df_all, cell_ids, time_points, feature1='pol1_int_corr', feature2='pol2_int_corr',
                                 period_bounds_list=[(2.0, 20.0), (20.0, 40.0), (40.0, 60.0)],
                                 n_cells=25):
    """
    Fits multi-periodic GPs to feature1 and feature2 for selected cells and plots them in a grid.
    """
    fig, axs = plt.subplots(5, 5, figsize=(20, 20), sharey=True)
    axs = axs.flatten()

    for i, cell_id in enumerate(cell_ids[:n_cells]):
        ax = axs[i]
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")

        t = df_cell["time_point"].values.reshape(-1, 1)
        y1 = df_cell[feature1].values.reshape(-1, 1)
        y2 = df_cell[feature2].values.reshape(-1, 1)

        valid = ~np.isnan(y1[:, 0]) & ~np.isnan(y2[:, 0])
        t, y1, y2 = t[valid], y1[valid], y2[valid]

        if len(t) < 5:
            ax.set_visible(False)
            continue

        y1_mean = np.mean(y1)
        y2_mean = np.mean(y2)

        try:
            model1 = fit_multi_periodic_gp(t, y1 - y1_mean, period_bounds_list)
            model2 = fit_multi_periodic_gp(t, y2 - y2_mean, period_bounds_list)
        except Exception as e:
            print(f"❌ GP failed on Cell {cell_id}: {e}")
            ax.set_visible(False)
            continue

        t_pred = np.linspace(t.min(), t.max(), 200)[:, None]
        mu1, var1 = model1.predict(t_pred)
        mu2, var2 = model2.predict(t_pred)

        ax.plot(t, y1, 'o', markersize=3, color='blue', label='pol1')
        ax.plot(t_pred, mu1 + y1_mean, 'b-', label='GP pol1')
        ax.fill_between(t_pred[:, 0], mu1[:, 0] + y1_mean - 2*np.sqrt(var1[:, 0]),
                        mu1[:, 0] + y1_mean + 2*np.sqrt(var1[:, 0]), color='blue', alpha=0.2)

        ax.plot(t, y2, 'x', markersize=3, color='darkorange', label='pol2')
        ax.plot(t_pred, mu2 + y2_mean, 'orange', label='GP pol2')
        ax.fill_between(t_pred[:, 0], mu2[:, 0] + y2_mean - 2*np.sqrt(var2[:, 0]),
                        mu2[:, 0] + y2_mean + 2*np.sqrt(var2[:, 0]), color='orange', alpha=0.2)

        ax.set_ylim(-10, 60)
        ax.grid(True)
        if i % 5 == 0:
            ax.set_ylabel("Signal")
        if i >= 20:
            ax.set_xlabel("Time")

        def fmt_top_components(sorted_list, top=3):
            return " / ".join([f"P={p:.1f}, A={a:.1f}" for p, a in sorted_list[:top]])

        sorted1 = sorted([(p.period.values[0], p.variance.values[0]) for p in model1.kern.parts[1:]], key=lambda x: -x[1])
        sorted2 = sorted([(p.period.values[0], p.variance.values[0]) for p in model2.kern.parts[1:]], key=lambda x: -x[1])
        slope1 = model1.kern.parts[0].variances[0]
        slope2 = model2.kern.parts[0].variances[0]

        ax.set_title(
            f"Cell {cell_id}\n"
            f"pol1: {fmt_top_components(sorted1)} | slope={slope1:.2f}\n"
            f"pol2: {fmt_top_components(sorted2)} | slope={slope2:.2f}",
            fontsize=7
        )

    # Hide unused axes
    for j in range(n_cells, 25):
        axs[j].set_visible(False)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4)
    fig.suptitle("GP Fit (Exponential Detrending + Periodic Modeling) for pol1 and pol2", fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def quantify_gp_features(df_all, cell_ids, time_points,
                         feature1='pol1_int_corr', feature2='pol2_int_corr',
                         period_bounds_list=[(2.0, 20.0), (20.0, 40.0), (40.0, 60.0)]):
    """
    Fits multi-periodic GPs and extracts 8 summary features per signal per cell:
    - 3 periods
    - 3 amplitudes
    - linear slope
    - signal mean

    Returns:
        pd.DataFrame with one row per cell, columns:
            ['cell_id', 'pol1_mean', 'pol1_slope', 'pol1_p1', 'pol1_a1', ..., 'pol2_a3']
    """
    records = []

    for cell_id in cell_ids:
        df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")

        t = df_cell["time_point"].values.reshape(-1, 1)
        y1 = df_cell[feature1].values.reshape(-1, 1)
        y2 = df_cell[feature2].values.reshape(-1, 1)

        valid = ~np.isnan(y1[:, 0]) & ~np.isnan(y2[:, 0])
        t, y1, y2 = t[valid], y1[valid], y2[valid]

        if len(t) < 5:
            continue

        y1_mean = float(np.mean(y1))
        y2_mean = float(np.mean(y2))

        try:
            model1 = fit_multi_periodic_gp(t, y1 - y1_mean, period_bounds_list)
            model2 = fit_multi_periodic_gp(t, y2 - y2_mean, period_bounds_list)
        except Exception as e:
            print(f"❌ GP fit failed on Cell {cell_id}: {e}")
            continue

        # Extract parameters
        def extract_gp_summary(model, signal_mean):
            slope = float(model.kern.parts[0].variances[0])
            periods = []
            amplitudes = []
            for k in model.kern.parts[1:]:  # skip linear part
                periods.append(float(k.period.values[0]))
                amplitudes.append(float(k.variance.values[0]))
            # Pad if fewer than 3 components
            while len(periods) < 3:
                periods.append(np.nan)
                amplitudes.append(np.nan)
            return [signal_mean, slope] + periods + amplitudes

        row = {'cell_id': int(cell_id)}
        row.update({f'pol1_{k}': v for k, v in zip(
            ['mean', 'slope', 'p1', 'p2', 'p3', 'a1', 'a2', 'a3'],
            extract_gp_summary(model1, y1_mean)
        )})
        row.update({f'pol2_{k}': v for k, v in zip(
            ['mean', 'slope', 'p1', 'p2', 'p3', 'a1', 'a2', 'a3'],
            extract_gp_summary(model2, y2_mean)
        )})

        records.append(row)

    result_df = pd.DataFrame.from_records(records)
    return result_df
