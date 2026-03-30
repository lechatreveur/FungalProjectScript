#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:47:22 2025

@author: user
"""

# preprocessing.py

import numpy as np
import pandas as pd

def filter_valid_cells(df_all, frame_number):
    valid_cells = df_all.groupby("cell_id")["time_point"].count()
    valid_cells = valid_cells[valid_cells == frame_number].index
    return df_all[df_all['cell_id'].isin(valid_cells)]

def compute_derivatives(df_all, rolling_window=20):
    df_all = df_all.sort_values(by=['cell_id', 'time_point'])

    df_all['d_cell_length'] = df_all.groupby('cell_id')['cell_length'].diff()
    df_all['d_cell_area'] = df_all.groupby('cell_id')['cell_area'].diff()
    df_all['d_nu_dis'] = df_all.groupby('cell_id')['nu_dis'].diff()
    df_all['d_pattern_score'] = df_all.groupby('cell_id')['pattern_score_norm'].diff()

    df_all['d_cell_length_avg'] = (
        df_all.groupby('cell_id')['d_cell_length']
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean().reset_index(level=0, drop=True)
    )
    df_all['d_cell_area_avg'] = (
        df_all.groupby('cell_id')['d_cell_area']
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean().reset_index(level=0, drop=True)
    )
    df_all['d_nu_dis_avg'] = (
        df_all.groupby('cell_id')['d_nu_dis']
        .rolling(window=rolling_window, center=True, min_periods=1)
        .mean().reset_index(level=0, drop=True)
    )
    

    return df_all


def add_first_derivative(
    df,
    feature="pattern_score_norm",
    time_col="time_point",
    group_col="cell_id",
    out_col=None,
    dt=None,
    enforce_monotonic_time=True,
):
    """
    Add a per-cell first derivative d(feature)/d(time) to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must include columns [group_col, time_col, feature].
    feature : str
        Column to differentiate.
    time_col : str
        Time/index column (e.g. frame number).
    group_col : str
        Grouping column (e.g. cell id).
    out_col : str or None
        Name of the derivative column. If None -> f"{feature}_d1".
    dt : float or None
        If not None, assumes uniform spacing and uses this spacing for all rows
        (e.g., seconds per frame). If None, uses the actual (possibly uneven)
        spacing in `time_col` via np.gradient.
    enforce_monotonic_time : bool
        If True, sorts within each group by time and drops duplicate times
        (keeping the first occurrence).

    Returns
    -------
    df_out : pd.DataFrame
        Original df with a new derivative column.
    """
    if out_col is None:
        out_col = f"{feature}_d1"

    # Work on a copy to avoid mutating caller's df
    df_out = df.copy()

    def _deriv_one_group(g):
        # sort & de-duplicate (optional)
        if enforce_monotonic_time:
            g = g.sort_values(time_col, kind="mergesort")  # stable
            g = g[~g[time_col].duplicated(keep="first")]

        t = g[time_col].to_numpy(dtype=float, copy=False)
        y = g[feature].to_numpy(dtype=float, copy=False)

        # If all-NaN or <2 points, derivative is undefined
        if len(g) < 2 or np.all(~np.isfinite(y)) or np.all(~np.isfinite(t)):
            g[out_col] = np.nan
            return g

        # Mask invalid y but keep time spacing; fill NaNs locally to avoid breaking gradient
        # Strategy: simple forward/backward fill within the group, then gradient, then
        # re-null positions that were originally NaN to avoid over-trusting imputed values.
        y_valid_mask = np.isfinite(y)
        if not np.all(y_valid_mask):
            y_filled = y.copy()
            # forward fill
            for i in range(1, len(y_filled)):
                if not np.isfinite(y_filled[i]) and np.isfinite(y_filled[i-1]):
                    y_filled[i] = y_filled[i-1]
            # backward fill
            for i in range(len(y_filled)-2, -1, -1):
                if not np.isfinite(y_filled[i]) and np.isfinite(y_filled[i+1]):
                    y_filled[i] = y_filled[i+1]
        else:
            y_filled = y

        # Choose spacing: uniform dt or actual t
        if dt is None:
            # Use actual (possibly uneven) time grid
            # np.gradient handles non-uniform spacing if we pass `t`
            dy_dt = np.gradient(y_filled, t)
        else:
            # Uniform spacing
            dy_dt = np.gradient(y_filled, dt)

        # Re-mask derivative where original y was NaN
        dy_dt = np.asarray(dy_dt, dtype=float)
        dy_dt[~y_valid_mask] = np.nan

        g[out_col] = dy_dt
        return g

    df_out = df_out.groupby(group_col, group_keys=False).apply(_deriv_one_group)
    return df_out
