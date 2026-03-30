#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:51:58 2025

@author: user
"""

# alignment.py

import numpy as np
import random
import pandas as pd

def prepare_signals(df_all, features, time_points):
    """
    Returns a dictionary {cell_id: signal matrix} with features × time for each cell.
    """
    cell_ids = df_all['cell_id'].unique()
    signal_dict = {}

    for cid in cell_ids:
        df_cell = df_all[df_all["cell_id"] == cid].sort_values("time_point")
        matrix = np.array([
            df_cell.set_index("time_point").reindex(time_points)[f].values
            for f in features
        ])
        signal_dict[cid] = matrix

    return signal_dict, cell_ids


def compute_mse(shifts, cell_signals, global_time, T, lambda_reg=0.0):
    """
    Compute total MSE after aligning all cells according to 'shifts'.
    Returns total cost and the global average signal.
    """
    n_features = next(iter(cell_signals.values())).shape[0]
    acc = np.zeros((n_features, len(global_time)))
    weights = np.zeros_like(acc)

    for cid, signal in cell_signals.items():
        shift = shifts[cid]
        valid = ~np.isnan(signal)
        acc[:, shift:shift+T] += np.where(valid, signal, 0)
        weights[:, shift:shift+T] += valid

    avg = np.divide(acc, weights, where=weights != 0)

    total_mse = 0
    for cid, signal in cell_signals.items():
        shift = shifts[cid]
        valid = ~np.isnan(signal) & (weights[:, shift:shift+T] > 0)
        diff = signal - avg[:, shift:shift+T]
        mse = np.mean((diff[valid]) ** 2)
        total_mse += mse

    return total_mse, avg


def initialize_shifts(cell_signals, time_points, global_time, init_span_frac=0.7):
    """
    Assign even-spaced initial shifts within a *shorter* window than the full MCMC range.
    init_span_frac: fraction (0,1] of the full shift range to use for initialization, centered.
    """
    T = len(time_points)
    shift_range_full = len(global_time) - T  # max allowed shift during MCMC

    # Safety for tiny ranges or edge cases
    shift_range_full = max(0, int(shift_range_full))

    # Span used just for initialization (shorter than full)
    span = max(1, int(round(shift_range_full * float(init_span_frac))))
    # Center the shorter span inside the full range
    min_shift_init = (shift_range_full - span) // 2
    max_shift_init = min_shift_init + span

    # Rank cells by the average of the first feature (e.g., weighted_area)
    means = {cid: np.nanmean(signal[1]) for cid, signal in cell_signals.items()}
    sorted_cells = sorted(means, key=means.get)

    # Evenly space initial shifts within the *shorter* init window
    if len(sorted_cells) == 1:
        shift_values = np.array([(min_shift_init + max_shift_init) // 2], dtype=int)
    else:
        shift_values = np.linspace(min_shift_init, max_shift_init, len(sorted_cells)).astype(int)

    shifts = {cid: shift for cid, shift in zip(sorted_cells, shift_values)}
    return shifts, shift_range_full


def run_mcmc(cell_signals, global_time, time_points, lambda_reg=0.0,
             n_iter=10000, initial_temp=1.0, init_span_frac=0.7, rng=None):
    """
    Run MCMC optimization to minimize alignment MSE.
    - init_span_frac: fraction of the full shift range used ONLY for initialization.
    - rng: optional np.random.Generator for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    T = len(time_points)
    # Initialize within a shorter, centered window
    shifts, shift_range_full = initialize_shifts(
        cell_signals, time_points, global_time, init_span_frac=init_span_frac
    )

    best_shifts = shifts.copy()
    best_score, best_mean = compute_mse(best_shifts, cell_signals, global_time, T, lambda_reg)
    mse_trace = [best_score]

    # Proposal step sizes; you can tweak or make this schedule adaptive
    step_choices = np.array([-10, -3, -1, 1, 3, 10])
    #step_choices = np.array([ -3, -1, 1, 3])

    for i in range(n_iter):
        temperature = initial_temp * (0.99 ** i)

        proposal = best_shifts.copy()
        cid = rng.choice(list(proposal.keys()))
        delta_shift = int(rng.choice(step_choices))
        new_shift = int(np.clip(proposal[cid] + delta_shift, 0, shift_range_full))
        proposal[cid] = new_shift

        new_score, new_mean = compute_mse(proposal, cell_signals, global_time, T, lambda_reg)
        delta = new_score - best_score

        # Metropolis acceptance
        if delta < 0 or np.exp(-delta / max(1e-12, temperature)) > rng.random():
            best_shifts = proposal
            best_score = new_score
            best_mean = new_mean

        mse_trace.append(best_score)

        if i % 100 == 0:
            print(f"Step {i}: MSE = {best_score:.4f}")

    return best_shifts, best_mean, mse_trace



def compute_aligned_mean_std(cell_signals, shifts, global_time, time_points, min_count=3):
    """
    Compute aligned mean and std (per feature × global_time) given final shifts.
    min_count: minimum #cells contributing to a time bin to compute mean/std; else NaN.
    Returns: mean (F x L), std (F x L), count (F x L)
    """
    F = next(iter(cell_signals.values())).shape[0]
    L = len(global_time)
    T = len(time_points)

    sum_  = np.zeros((F, L), dtype=float)
    sum2  = np.zeros((F, L), dtype=float)
    count = np.zeros((F, L), dtype=float)

    for cid, sig in cell_signals.items():
        s = shifts[cid]
        seg = slice(s, s + T)
        valid = ~np.isnan(sig)
        vals = np.where(valid, sig, 0.0)
        sum_[:, seg]  += vals
        sum2[:, seg]  += vals**2
        count[:, seg] += valid

    mean = np.divide(sum_, count, out=np.zeros_like(sum_), where=count > 0)
    var  = np.divide(sum2, count, out=np.zeros_like(sum2), where=count > 0) - mean**2
    var  = np.maximum(var, 0.0)
    std  = np.sqrt(var)

    # mask low support
    mean[count < min_count] = np.nan
    std[count  < min_count] = np.nan

    return mean, std, count


def find_outlier_cells_by_sigma(cell_signals, shifts, mean, std, global_time, time_points,
                                feature_names, threshold_sigma=1.96, min_support_frac=0.05):
    """
    Flag cells whose aligned signals exceed |z| > threshold_sigma for at least
    (min_support_frac) of valid frames in ANY feature.

    Returns:
      - per_feature_df: long table of exceed stats per (cell, feature)
      - cell_summary: per-cell summary
      - outlier_cells: sorted list of flagged cell_ids
    """
    T = len(time_points)
    need_frames = max(1, int(np.ceil(T * float(min_support_frac))))

    records = []
    cell_rows = []

    for cid, sig in cell_signals.items():
        s = shifts[cid]
        seg = slice(s, s + T)

        mu  = mean[:, seg]
        sd  = std[:, seg]
        valid = (~np.isnan(sig)) & (sd > 0)

        z = np.zeros_like(sig, dtype=float)
        z[valid] = (sig[valid] - mu[valid]) / sd[valid]
        exceed = np.abs(z) > float(threshold_sigma)

        n_valid_by_feat  = valid.sum(axis=1).astype(int)
        n_exceed_by_feat = exceed.sum(axis=1).astype(int)
        frac_by_feat     = np.divide(n_exceed_by_feat, np.maximum(1, n_valid_by_feat))

        # per-feature rows
        for f_idx, fname in enumerate(feature_names):
            n_valid  = int(n_valid_by_feat[f_idx])
            n_exceed = int(n_exceed_by_feat[f_idx])
            frac     = float(frac_by_feat[f_idx]) if n_valid > 0 else 0.0
            max_absz = (float(np.nanmax(np.where(valid[f_idx], np.abs(z[f_idx]), np.nan)))
                        if n_valid > 0 else np.nan)
            first_idx = int(np.argmax(exceed[f_idx])) if n_exceed > 0 else -1
            first_align_t = (s + first_idx) if n_exceed > 0 else None

            records.append({
                "cell_id": cid,
                "feature": fname,
                "n_valid": n_valid,
                "n_exceed": n_exceed,
                "frac_exceed": frac,
                "max_abs_z": max_absz,
                "first_exceed_aligned_time": first_align_t,
            })

        # per-cell summary (ANY feature)
        any_exceed_frames = int(n_exceed_by_feat.max())
        flagged = any_exceed_frames >= need_frames
        cell_rows.append({
            "cell_id": cid,
            "max_exceed_frames_any_feature": any_exceed_frames,
            "required_frames": need_frames,
            "flagged_outlier": bool(flagged),
            "max_abs_z_any_feature": float(np.nanmax(np.where(valid, np.abs(z), np.nan)))
        })

    per_feature_df = pd.DataFrame.from_records(records)
    cell_summary   = pd.DataFrame.from_records(cell_rows)
    outlier_cells  = sorted(cell_summary.loc[cell_summary["flagged_outlier"], "cell_id"].unique().tolist())
    return per_feature_df, cell_summary, outlier_cells


import numpy as np

# ---------- Core: align ONE series to a reference (no init, no MCMC) ----------
def align_single_signal(signal, global_time, time_points, reference,
                        lambda_reg=0.0):
    """
    Brute-force shift search to align a single 1D signal to a fixed reference.

    Parameters
    ----------
    signal : array-like, shape (T,)
        The signal to align (time dimension T).
    global_time : array-like, shape (G,)
        Global timeline (only length G is used).
    time_points : array-like, shape (T,)
        Local indices/time points (only length T is used).
    reference : array-like, shape (G,)
        Fixed template on the global grid to align to.
    lambda_reg : float, default 0.0
        Optional L2 penalty favoring smaller shifts:
        objective = MSE + lambda_reg * (s^2) / (max_shift^2 + 1e-12)

    Returns
    -------
    best_shift : int
    aligned_on_global : np.ndarray, shape (G,)
        The signal embedded on the global grid at best_shift (NaNs elsewhere).
    best_score : float
    scores_trace : np.ndarray, shape (max_shift + 1,)
    """
    signal = np.asarray(signal, dtype=float)
    reference = np.asarray(reference, dtype=float)
    G = len(global_time)
    T = len(time_points)

    if G < T:
        raise ValueError(f"global_time length ({G}) must be >= signal length ({T}).")

    max_shift = G - T
    scores = np.empty(max_shift + 1, dtype=float)
    denom = (max_shift ** 2) + 1e-12

    def nan_mse(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        if not np.any(m):
            return np.inf
        d = a[m] - b[m]
        return float(np.mean(d * d))

    for s in range(max_shift + 1):
        ref_win = reference[s:s + T]
        mse = nan_mse(signal, ref_win)
        reg = lambda_reg * (s ** 2) / denom
        scores[s] = mse + reg

    best_shift = int(np.argmin(scores))
    best_score = float(scores[best_shift])

    aligned = np.full(G, np.nan, dtype=float)
    aligned[best_shift:best_shift + T] = signal

    return best_shift, aligned, best_score, scores


# ---------- Pipeline wrapper: build/use template, align ALL cells on one feature ----------
def run_single_signal_alignment(cell_signals,
                                global_time,
                                time_points,
                                features_order,
                                feature_name='pattern_score_norm',
                                reference=None,
                                lambda_reg=0.0,
                                center_method='median'):
    """
    Pipeline-friendly wrapper to align ONE chosen feature for ALL cells.

    Parameters
    ----------
    cell_signals : dict[cid -> np.ndarray]
        Output of your `prepare_signals`. Each value is (F, T) or (T,) if only one feature.
    global_time : array-like, shape (G,)
    time_points : array-like, shape (T,)
    features_order : list[str]
        The feature names in the same order used by `prepare_signals`.
    feature_name : str
        Which feature to align (must exist in `features_order`).
    reference : None or np.ndarray, shape (G,)
        If None: build a template by centering all cells, embedding on the global grid,
        then taking the per-timepoint median/mean (NaN-robust).
    lambda_reg : float
        Regularization passed to `align_single_signal`.
    center_method : {'median','mean'}
        Aggregation for template building if `reference` is None.

    Returns
    -------
    shifts : dict[cid -> int]
    aligned_matrix : np.ndarray, shape (N_cells, G)
        Row i corresponds to `cell_ids[i]` below.
    scores : dict[cid -> float]
        Best objective per cell.
    scores_traces : dict[cid -> np.ndarray]
        Full objective curve per tested shift (useful for diagnostics).
    reference_used : np.ndarray, shape (G,)
    cell_ids : list
        Order of cells matching `aligned_matrix`.
    """
    # Resolve feature index
    if isinstance(features_order, (list, tuple)):
        try:
            fidx = features_order.index(feature_name)
        except ValueError:
            raise ValueError(f"feature_name='{feature_name}' not found in features_order.")
    else:
        raise ValueError("features_order must be a sequence of feature names.")

    G = len(global_time)
    T = len(time_points)
    max_shift = G - T
    center_shift = max(0, max_shift // 2)

    # Helper: extract 1D series for a cell
    def extract_feature_1d(sig):
        sig = np.asarray(sig)
        if sig.ndim == 1:
            if sig.shape[0] != T:
                raise ValueError("1D signal length must equal len(time_points).")
            return sig.astype(float)
        elif sig.ndim == 2:
            if not (0 <= fidx < sig.shape[0]):
                raise ValueError("feature index out of range for a cell signal.")
            if sig.shape[1] != T:
                raise ValueError("2D signal time dimension must equal len(time_points).")
            return sig[fidx].astype(float)
        else:
            raise ValueError("Each cell signal must be 1D (T,) or 2D (F, T).")

    # If no reference provided, build a centered template from all cells (NaN-robust).
    if reference is None:
        emb = []  # (N_cells, G)
        for cid, sig in cell_signals.items():
            s1d = extract_feature_1d(sig)
            row = np.full(G, np.nan, dtype=float)
            row[center_shift:center_shift + T] = s1d
            emb.append(row)
        emb = np.asarray(emb) if len(emb) > 0 else np.empty((0, G))

        if center_method == 'median':
            reference_used = np.nanmedian(emb, axis=0)
        elif center_method == 'mean':
            reference_used = np.nanmean(emb, axis=0)
        else:
            raise ValueError("center_method must be 'median' or 'mean'.")
    else:
        reference_used = np.asarray(reference, dtype=float)
        if reference_used.shape[0] != G:
            raise ValueError("Provided reference length must equal len(global_time).")

    # Align all cells
    cell_ids = list(cell_signals.keys())
    aligned_matrix = np.full((len(cell_ids), G), np.nan, dtype=float)
    shifts, scores, scores_traces = {}, {}, {}

    for i, cid in enumerate(cell_ids):
        s1d = extract_feature_1d(cell_signals[cid])
        best_shift, aligned, best_score, trace = align_single_signal(
            signal=s1d,
            global_time=global_time,
            time_points=time_points,
            reference=reference_used,
            lambda_reg=lambda_reg
        )
        shifts[cid] = best_shift
        scores[cid] = best_score
        scores_traces[cid] = trace
        aligned_matrix[i] = aligned

    return shifts, aligned_matrix, scores, scores_traces, reference_used, cell_ids

