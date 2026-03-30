#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:53:23 2025

@author: user
"""

# visualization.py

import matplotlib.pyplot as plt
import numpy as np


def plot_aligned_signals(
    df_all,
    cell_ids,
    best_shifts,
    global_time,
    time_points,
    features,
    mean_trace=None,
    std_trace=None,
    title_prefix="Aligned",
    ci_sigma=1.96,
):
    """
    Plots aligned single-cell traces per feature over the global timeline.
    Optionally overlays the mean trace and a ±(ci_sigma)*std band.

    Args
    ----
    df_all : pd.DataFrame
        Must contain columns: ["cell_id", "time_point"] + features
    cell_ids : iterable
    best_shifts : dict {cell_id: shift}
    global_time : 1D array-like
    time_points : 1D array-like (original per-cell sampling grid)
    features : list[str]
    mean_trace : np.ndarray or None, shape (n_features, len(global_time))
    std_trace  : np.ndarray or None, shape (n_features, len(global_time))
    title_prefix : str
    ci_sigma : float
    """
    import numpy as np
    import matplotlib.pyplot as plt

    n_features = len(features)
    T = len(time_points)
    L = len(global_time)

    # Shape checks for mean/std if provided
    if mean_trace is not None:
        if mean_trace.shape != (n_features, L):
            raise ValueError(
                f"mean_trace shape {mean_trace.shape} must be (n_features={n_features}, len(global_time)={L})"
            )
    if std_trace is not None:
        if std_trace.shape != (n_features, L):
            raise ValueError(
                f"std_trace shape {std_trace.shape} must be (n_features={n_features}, len(global_time)={L})"
            )

    fig, axs = plt.subplots(nrows=n_features, figsize=(12, 2.8 * n_features), sharex=True)
    if n_features == 1:
        axs = [axs]

    for i, feature in enumerate(features):
        ax = axs[i]

        # Plot all aligned single-cell traces for this feature
        for cid in cell_ids:
            df_cell = df_all[df_all["cell_id"] == cid].sort_values("time_point")
            values = df_cell.set_index("time_point").reindex(time_points)[feature].values

            aligned = np.full(L, np.nan, dtype=float)
            shift = int(best_shifts[cid])

            start = shift
            end = shift + T

            aligned_start = max(start, 0)
            aligned_end = min(end, L)
            value_start = aligned_start - start
            value_end = value_start + (aligned_end - aligned_start)

            # Validate slicing bounds
            if value_start < 0 or value_end > values.size or aligned_start >= aligned_end:
                continue

            aligned[aligned_start:aligned_end] = values[value_start:value_end]
            ax.plot(global_time, aligned, alpha=0.25, linewidth=0.7)

        # Overlay mean trace
        legend_needed = False
        if mean_trace is not None:
            ax.plot(global_time, mean_trace[i], linewidth=2.0, label="Mean")
            legend_needed = True

        # Overlay ±ci_sigma * std band (only where std is finite)
        if mean_trace is not None and std_trace is not None:
            lo = mean_trace[i] - ci_sigma * std_trace[i]
            hi = mean_trace[i] + ci_sigma * std_trace[i]
            valid = np.isfinite(lo) & np.isfinite(hi)
            if np.any(valid):
                ax.fill_between(global_time[valid], lo[valid], hi[valid], alpha=0.2, label=f"±{ci_sigma}σ")
                legend_needed = True

        if legend_needed:
            ax.legend(loc="upper right", frameon=True)

        ax.set_title(f"{title_prefix}: {feature}", fontsize=12)
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    axs[-1].set_xlabel("Global Master Timeline", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_aligned_heatmaps(
    df_all,
    cell_ids,
    best_shifts,
    global_time,
    time_points,
    features,
    cmap_list=None,
    title="Aligned Heatmaps",
):
    """
    Matplotlib-only heatmaps (no seaborn/scipy dependency).
    Each feature gets its own heatmap: rows=cells, cols=global_time.
    """
    nF = len(features)
    fig, axs = plt.subplots(nF, 1, figsize=(12, max(2.5, nF * 2.5)), sharex=True)
    if nF == 1:
        axs = [axs]

    ordered_cells = sorted(cell_ids, key=lambda cid: best_shifts.get(cid, 0))

    L = len(global_time)
    T = len(time_points)

    for i, feature in enumerate(features):
        heatmap_data = []

        for cid in ordered_cells:
            df_cell = df_all[df_all["cell_id"] == cid].sort_values("time_point")
            values = df_cell.set_index("time_point").reindex(time_points)[feature].values

            aligned = np.full(L, np.nan, dtype=float)
            shift = int(round(best_shifts.get(cid, 0)))  # ensure integer

            start = shift
            end = shift + T

            aligned_start = max(start, 0)
            aligned_end = min(end, L)
            value_start = aligned_start - start
            value_end = value_start + (aligned_end - aligned_start)

            # skip invalid slices / empty
            if aligned_start >= aligned_end:
                continue
            if value_start < 0 or value_end > len(values):
                continue
            if (aligned_end - aligned_start) <= 0:
                continue

            aligned[aligned_start:aligned_end] = values[value_start:value_end]
            heatmap_data.append(aligned)

        if not heatmap_data:
            axs[i].set_title(f"{feature} (no data)")
            continue

        mat = np.asarray(heatmap_data, dtype=float)

        cmap = cmap_list[i] if cmap_list else "viridis"
        im = axs[i].imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
        )
        axs[i].set_title(f"{feature}")
        axs[i].set_ylabel("Cells")
        fig.colorbar(im, ax=axs[i], fraction=0.02, pad=0.01)

    axs[-1].set_xlabel("Global Master Timeline (index)")
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
import numpy as np
import matplotlib.pyplot as plt

def plot_aligned_single_feature(
    aligned_matrix,
    global_time,
    cell_ids=None,
    title="Aligned: single feature",
    mean_trace=None,
    std_trace=None,
    ci_sigma=1.96,
    highlight=None,
    highlight_kwargs=None,
    line_alpha=0.25,
    line_width=0.7,
):
    """
    Plot aligned single-cell traces (one feature) over the global timeline.
    Mirrors the style of your multi-feature plot: all traces faint, optional mean,
    and an optional ±(ci_sigma)*std band.

    Parameters
    ----------
    aligned_matrix : np.ndarray, shape (N_cells, L)
        Each row is a cell's aligned series on the global grid (NaNs allowed).
    global_time : 1D array-like of length L
        Global timeline (x-axis).
    cell_ids : list or None
        Optional list of cell IDs in the same row order as aligned_matrix.
        Used only for legend labels when highlighting.
    title : str
        Plot title.
    mean_trace : 1D array-like or None (length L)
        If provided, plotted as the bold mean curve. If None, computed as nanmean(aligned_matrix, axis=0).
    std_trace : 1D array-like or None (length L)
        If provided with mean_trace, draws a ±ci_sigma*std band. If None, computed as nanstd(...).
    ci_sigma : float
        Width of the confidence band in units of std (e.g., 1.96 ≈ 95% if normal).
    highlight : list or None
        Optional list of indices (row indices in aligned_matrix) or cell IDs to highlight.
    highlight_kwargs : dict or None
        Matplotlib kwargs for highlighted traces (e.g., {'linewidth':2.0, 'alpha':0.9}).
    line_alpha : float
        Alpha for non-highlighted traces.
    line_width : float
        Line width for non-highlighted traces.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    aligned_matrix = np.asarray(aligned_matrix, dtype=float)
    L = aligned_matrix.shape[1]
    global_time = np.asarray(global_time)
    if global_time.shape[0] != L:
        raise ValueError("len(global_time) must match aligned_matrix.shape[1].")

    # Compute mean/std if not provided
    computed_mean = False
    if mean_trace is None:
        mean_trace = np.nanmean(aligned_matrix, axis=0)
        computed_mean = True
    if std_trace is None:
        std_trace = np.nanstd(aligned_matrix, axis=0)

    # Normalize highlight inputs -> indices
    highlight_idxs = set()
    if highlight is not None:
        if cell_ids is not None and any(isinstance(h, (str, int)) for h in highlight):
            # If highlight items look like IDs, map to indices when possible
            id_to_idx = {cid: i for i, cid in enumerate(cell_ids)}
            for h in highlight:
                if isinstance(h, (str, int)) and h in id_to_idx:
                    highlight_idxs.add(id_to_idx[h])
                elif isinstance(h, int) and 0 <= h < aligned_matrix.shape[0]:
                    highlight_idxs.add(h)
        else:
            # Treat as indices
            for h in highlight:
                if isinstance(h, int) and 0 <= h < aligned_matrix.shape[0]:
                    highlight_idxs.add(h)

    if highlight_kwargs is None:
        highlight_kwargs = {"linewidth": 2.0, "alpha": 0.9}

    fig, ax = plt.subplots(figsize=(12, 3.0))

    # Plot all aligned traces
    for i in range(aligned_matrix.shape[0]):
        y = aligned_matrix[i]
        if np.all(np.isnan(y)):
            continue  # skip empty
        if i in highlight_idxs:
            lbl = None
            if cell_ids is not None:
                # Only label the first time each ID appears
                lbl = f"cell {cell_ids[i]}" if list(highlight_idxs).index(i) == 0 else None
            ax.plot(global_time, y, **highlight_kwargs, label=lbl)
        else:
            ax.plot(global_time, y, alpha=line_alpha, linewidth=line_width)

    # Overlay mean trace
    if mean_trace is not None:
        ax.plot(global_time, mean_trace, linewidth=2.0, label="Mean")

    # Overlay ±ci_sigma * std band (only where finite)
    if mean_trace is not None and std_trace is not None:
        lo = mean_trace - ci_sigma * std_trace
        hi = mean_trace + ci_sigma * std_trace
        valid = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(global_time)
        if np.any(valid):
            ax.fill_between(global_time[valid], lo[valid], hi[valid], alpha=0.2, label=f"±{ci_sigma}σ")

    # Legend only if something labeled is present
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(loc="upper right", frameon=True)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Value")
    ax.set_xlabel("Global Master Timeline", fontsize=11)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    plt.show()

    return fig, ax


def keep_base_only(df, id_col="cell_id"):
    s = df[id_col].astype(str)
    is_base = ~s.str.contains("_")   # keeps "10", drops "10_1","10_2"
    return df[is_base].copy()
def load_good_gfp1_gfp2_for_field(field, FILM_NAMES, WORKING_DIR, all_global_maps, all_res, only_has_septum_bf1=False):
    from SingleCellDataAnalysis.multi_field import  make_field_sequence
    from SingleCellDataAnalysis.multi_field_data_analysis import (
        load_good_global_ids,
        load_good_aligned_gfp,
        get_aligned_csv,
    )
    from SingleCellDataAnalysis.manifest_utils import load_manifest
    """
    Returns (df_gfp1, df_gfp2) for this field.
    If only_has_septum_bf1=True:
        keep only global IDs whose BF1 local cell has manifest.has==1.
    """
    field_seq = make_field_sequence(field, FILM_NAMES)
    gfp1 = field_seq[0][1]
    bf1  = field_seq[1][1]
    gfp2 = field_seq[2][1]

    # QC-good global ids
    good_gids = set(load_good_global_ids(WORKING_DIR, field))

    global_maps = all_global_maps[field]   # film -> {local_id -> global_id}
    pair_maps   = all_res[field]           # run_field_sequence outputs

    # ---- OPTIONAL: restrict to BF1 cells with has==1 in manifest ----
    if only_has_septum_bf1:
        manifest = load_manifest(WORKING_DIR)  # your helper: reads training_dataset/manifest.csv
        bf1_has_local = (
            manifest.loc[(manifest["film_name"] == bf1) & (manifest["has"] == 1), "cell_id"]
            .dropna()
            .astype(int)
            .tolist()
        )

        bf1_l2g = global_maps.get(bf1, {})  # local -> global for BF1
        bf1_has_global = {bf1_l2g[c] for c in bf1_has_local if c in bf1_l2g}

        # intersect with QC-good
        good_gids = good_gids.intersection(bf1_has_global)

    good_gids = sorted(good_gids)

    aligned_gfp1_csv = get_aligned_csv(pair_maps, gfp1, bf1)
    aligned_gfp2_csv = get_aligned_csv(pair_maps, gfp2, bf1)

    df1 = load_good_aligned_gfp(global_maps, good_gids, aligned_gfp1_csv, gfp1)
    df2 = load_good_aligned_gfp(global_maps, good_gids, aligned_gfp2_csv, gfp2)

    # tag metadata + pooled unique id
    for df, which in ((df1, "GFP1"), (df2, "GFP2")):
        df["field"] = field
        df["which"] = which
        df["global_id_in_field"] = df["cell_id"]
        df["cell_id"] = df["field"].astype(str) + ":" + df["global_id_in_field"].astype(str)

    df1 = keep_base_only(df1)
    df2 = keep_base_only(df2)
    return df1, df2

import numpy as np
import pandas as pd

def build_bf1_end_anchor_for_pool(
    df_pool: pd.DataFrame,
    WORKING_DIR: str,
    FILM_NAMES,
    all_global_maps: dict,
    only_has_septum: bool = True,
) -> pd.DataFrame:
    """
    Returns a table with columns:
      cell_id (pooled string like 'F0:123'), field, global_id_in_field,
      bf1_local_id, bf1_has, bf1_end_frame

    Uses training_dataset/manifest.csv:
      - expects columns: film_name, cell_id, has, start_idx, end_idx
    """
    from SingleCellDataAnalysis.manifest_utils import load_manifest
    from SingleCellDataAnalysis.multi_field import make_field_sequence
    manifest = load_manifest(WORKING_DIR)

    # We only need these columns
    need = {"film_name", "cell_id", "has", "start_idx", "end_idx"}
    missing = need - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")

    out_rows = []

    # df_pool must contain these
    need_pool = {"cell_id", "field", "global_id_in_field"}
    missing_pool = need_pool - set(df_pool.columns)
    if missing_pool:
        raise ValueError(f"df_pool missing columns: {sorted(missing_pool)}")

    for field, sub in df_pool[["cell_id", "field", "global_id_in_field"]].drop_duplicates().groupby("field"):
        # BF1 film name for this field (your naming scheme)
        #bf1 = f"A14-YES-1t-FBFBF-2_{field}"
        field_seq = make_field_sequence(field, FILM_NAMES)
        bf1 = field_seq[1][1]   # BF1, always
        # local->global map for BF1
        l2g = all_global_maps[field].get(bf1, {})
        if not l2g:
            # no mapping for this field/film; skip anchors
            for _, r in sub.iterrows():
                out_rows.append({
                    "cell_id": r["cell_id"],
                    "field": field,
                    "global_id_in_field": r["global_id_in_field"],
                    "bf1_local_id": np.nan,
                    "bf1_has": np.nan,
                    "bf1_end_frame": np.nan,
                })
            continue

        # invert to global->local (keep first)
        g2l = {}
        for local_id, gid in l2g.items():
            if gid not in g2l:
                g2l[gid] = local_id

        # manifest rows for this BF1
        m = manifest.loc[manifest["film_name"] == bf1].copy()
        if m.empty:
            # no manifest entries -> anchors unknown
            for _, r in sub.iterrows():
                out_rows.append({
                    "cell_id": r["cell_id"],
                    "field": field,
                    "global_id_in_field": r["global_id_in_field"],
                    "bf1_local_id": np.nan,
                    "bf1_has": np.nan,
                    "bf1_end_frame": np.nan,
                })
            continue

        # numeric sanitize
        for c in ["cell_id", "has", "start_idx", "end_idx"]:
            m[c] = pd.to_numeric(m[c], errors="coerce")

        # compute an "end frame" per BF1 local cell
        # If end_idx present (>=0): use it.
        # Else if start_idx present: impute end = start + median_duration among rows with both.
        # Else: impute end = median(end_idx) among valid end_idx.
        m["end_frame"] = m["end_idx"].where(m["end_idx"] >= 0, np.nan)

        valid_dur = m.loc[(m["start_idx"] >= 0) & (m["end_idx"] >= 0), "end_idx"] - m.loc[(m["start_idx"] >= 0) & (m["end_idx"] >= 0), "start_idx"]
        med_dur = float(np.nanmedian(valid_dur.values)) if len(valid_dur) else np.nan

        med_end = float(np.nanmedian(m["end_frame"].values)) if np.isfinite(m["end_frame"]).any() else np.nan

        # fill end_frame where missing but has septum and start exists
        can_fill = m["end_frame"].isna() & (m["has"] == 1) & (m["start_idx"] >= 0) & np.isfinite(med_dur)
        m.loc[can_fill, "end_frame"] = m.loc[can_fill, "start_idx"] + med_dur

        # last-resort fill (still missing) with med_end for has==1
        can_fill2 = m["end_frame"].isna() & (m["has"] == 1) & np.isfinite(med_end)
        m.loc[can_fill2, "end_frame"] = med_end

        # build dict local_id -> (has, end_frame)
        m_small = m.dropna(subset=["cell_id"]).copy()
        m_small["cell_id"] = m_small["cell_id"].astype(int)
        has_map = dict(zip(m_small["cell_id"], m_small["has"]))
        end_map = dict(zip(m_small["cell_id"], m_small["end_frame"]))

        for _, r in sub.iterrows():
            gid_field = r["global_id_in_field"]
            # global_id_in_field might be str; normalize to int if possible
            try:
                gid_int = int(str(gid_field).split("_", 1)[0])
            except Exception:
                gid_int = None

            local = g2l.get(gid_int, None) if gid_int is not None else None
            bf_has = has_map.get(local, np.nan) if local is not None else np.nan
            bf_end = end_map.get(local, np.nan) if local is not None else np.nan

            if only_has_septum and not (bf_has == 1):
                bf_end = np.nan

            out_rows.append({
                "cell_id": r["cell_id"],
                "field": field,
                "global_id_in_field": r["global_id_in_field"],
                "bf1_local_id": local if local is not None else np.nan,
                "bf1_has": bf_has,
                "bf1_end_frame": bf_end,
            })

    return pd.DataFrame(out_rows)

import os
import pandas as pd
import numpy as np

def bf_time_series_length_frames(WORKING_DIR: str, bf1_film: str) -> int:
    """
    BF length = number of timepoints in BF1.
    Uses: WORKING_DIR/<bf1_film>/TrackedCells_<bf1_film>/all_cells_time_series.csv
    """
    csv_path = os.path.join(
        WORKING_DIR, bf1_film, f"TrackedCells_{bf1_film}", "all_cells_time_series.csv"
    )
    df = pd.read_csv(csv_path, usecols=["time_point"])
    tp = pd.to_numeric(df["time_point"], errors="coerce").dropna().astype(int)

    # robust: if frames are 0..T-1, this equals max+1; if missing, fall back to nunique
    if len(tp) == 0:
        raise ValueError(f"No time_point values found in {csv_path}")

    tmax = int(tp.max())
    # If it looks contiguous from 0, use max+1; else use nunique.
    if int(tp.min()) == 0 and tp.nunique() >= (tmax + 1) * 0.9:
        return tmax + 1
    return int(tp.nunique())

def movie_length_from_df(df: pd.DataFrame) -> int:
    tp = pd.to_numeric(df["time_point"], errors="coerce").dropna().astype(int)
    if len(tp) == 0:
        return 0
    tmax = int(tp.max())
    if int(tp.min()) == 0 and tp.nunique() >= (tmax + 1) * 0.9:
        return tmax + 1
    return int(tp.nunique())

def build_lengths_per_field(WORKING_DIR: str, FILM_NAMES, fields=("F0","F1","F2")):
    """
    Returns dicts:
      Lbf[field], Lg1[field], Lg2[field]
    using BF1 all_cells_time_series.csv for Lbf,
    and FILM_NAMES (index 0=GFP1, 1=BF1, 2=GFP2) for film naming.
    """
    Lbf, Lg1, Lg2 = {}, {}, {}

    for field in fields:
        gfp1_film = f"{FILM_NAMES[0]}_{field}"
        bf1_film  = f"{FILM_NAMES[1]}_{field}"
        gfp2_film = f"{FILM_NAMES[2]}_{field}"

        # BF length from BF1 time series (your definition)
        Lbf[field] = bf_time_series_length_frames(WORKING_DIR, bf1_film)

        # GFP lengths from their own time series (same definition; consistent)
        Lg1[field] = bf_time_series_length_frames(WORKING_DIR, gfp1_film)
        Lg2[field] = bf_time_series_length_frames(WORKING_DIR, gfp2_film)

    return Lbf, Lg1, Lg2
def attach_with_bf_axis(df, anchors, which, Lbf, Lg1, Lg2, only_has_septum):
    df2 = df.merge(anchors[["cell_id","bf1_end_frame"]], on="cell_id", how="left")
    df2["time_point"] = pd.to_numeric(df2["time_point"], errors="coerce")
    df2["bf1_end_frame"] = pd.to_numeric(df2["bf1_end_frame"], errors="coerce")

    if only_has_septum:
        df2 = df2.dropna(subset=["bf1_end_frame"])

    # compute per-row L values from field
    df2["Lbf"] = df2["field"].map(Lbf)
    df2["Lg1"] = df2["field"].map(Lg1)
    df2["Lg2"] = df2["field"].map(Lg2)

    if which == "GFP1":
        # end at -E (last frame lands exactly at -E)
        df2["x_frame"] = df2["time_point"] - (df2["Lg1"] - 1) - df2["bf1_end_frame"]
    else:
        # start at (Lbf - E)
        df2["x_frame"] = (df2["Lbf"] - df2["bf1_end_frame"]) + df2["time_point"]

    df2 = df2.dropna(subset=["time_point","x_frame"])
    return df2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple

def plot_movie_gfp1_gfp2(
    df_gfp1_pool: pd.DataFrame,
    df_gfp2_pool: pd.DataFrame,
    *,
    WORKING_DIR: str,
    FILM_NAMES,
    all_global_maps: dict,
    features=("pol1_int_corr", "pol2_int_corr", "pol1_minus_pol2"),
    title_prefix="Pooled GFP1+GFP2 on BF-axis (septum-end at x=0)",
    only_has_septum_bf1: bool = True,
    clip_x_minutes: Optional[Tuple[float, float]] = None,
    # time resolution inputs (seconds per frame)
    bf_seconds_per_frame: float = 60.0,
    gfp_seconds_per_frame: float = 12.0,
    # NEW: x-grid bin size (minutes)
    bin_minutes: float = 1.0,
):
    """
    X-axis is in MINUTES.
    x=0 is the BF1 septum-end time (per cell).
    GFP1 occupies negative x; GFP2 occupies positive x (shifted by BF length).
    """

    # anchors (bf1_end_frame is in BF frames)
    anchors = build_bf1_end_anchor_for_pool(
        df_pool=df_gfp1_pool[["cell_id", "field", "global_id_in_field"]].drop_duplicates(),
        WORKING_DIR=WORKING_DIR,
        FILM_NAMES=FILM_NAMES,
        all_global_maps=all_global_maps,
        only_has_septum=only_has_septum_bf1,
    )

    # lengths per field in native frames
    fields_present = tuple(sorted(set(df_gfp1_pool["field"].astype(str).unique()) |
                                  set(df_gfp2_pool["field"].astype(str).unique())))
    Lbf, Lg1, Lg2 = build_lengths_per_field(WORKING_DIR, FILM_NAMES, fields=fields_present)

    bf_min_per_frame  = float(bf_seconds_per_frame)  / 60.0
    gfp_min_per_frame = float(gfp_seconds_per_frame) / 60.0

    def attach_minutes_axis(df: pd.DataFrame, which: str):
        df2 = df.merge(anchors[["cell_id", "bf1_end_frame"]], on="cell_id", how="left")
        df2["time_point"] = pd.to_numeric(df2["time_point"], errors="coerce")
        df2["bf1_end_frame"] = pd.to_numeric(df2["bf1_end_frame"], errors="coerce")

        if only_has_septum_bf1:
            df2 = df2.dropna(subset=["bf1_end_frame"])
        df2 = df2.dropna(subset=["time_point"])

        # map per-row lengths (frames)
        df2["Lbf_frames"] = df2["field"].map(Lbf).astype(float)
        df2["Lg1_frames"] = df2["field"].map(Lg1).astype(float)
        df2["Lg2_frames"] = df2["field"].map(Lg2).astype(float)

        # convert to minutes
        E_min   = df2["bf1_end_frame"] * bf_min_per_frame
        Lbf_min = df2["Lbf_frames"]    * bf_min_per_frame
        Lg1_min = df2["Lg1_frames"]    * gfp_min_per_frame
        Lg2_min = df2["Lg2_frames"]    * gfp_min_per_frame
        t_min   = df2["time_point"]    * gfp_min_per_frame

        if which == "GFP1":
            # make last GFP1 frame land at x = -E_min
            df2["x_min"] = t_min - (Lg1_min - gfp_min_per_frame) - E_min
        else:
            # start GFP2 at x = Lbf_min - E_min
            df2["x_min"] = (Lbf_min - E_min) + t_min

        df2 = df2.dropna(subset=["x_min"])
        return df2

    d1 = attach_minutes_axis(df_gfp1_pool, "GFP1")
    d2 = attach_minutes_axis(df_gfp2_pool, "GFP2")

    # cell order by BF1 end frame (earliest -> latest)
    order_tbl = anchors.copy()
    order_tbl["bf1_end_frame"] = pd.to_numeric(order_tbl["bf1_end_frame"], errors="coerce")
    order_tbl = order_tbl.dropna(subset=["bf1_end_frame"])
    cell_order = order_tbl.sort_values("bf1_end_frame")["cell_id"].astype(str).tolist()

    cells1 = set(d1["cell_id"].astype(str).unique())
    cells2 = set(d2["cell_id"].astype(str).unique())
    common = [c for c in cell_order if (c in cells1 and c in cells2)]
    if not common:
        raise ValueError("No common cells between GFP1 and GFP2 after anchoring/filtering.")

    # build x grid in minutes
    xmin = float(np.nanmin([d1["x_min"].min(), d2["x_min"].min()]))
    xmax = float(np.nanmax([d1["x_min"].max(), d2["x_min"].max()]))

    step = float(bin_minutes)
    if step <= 0:
        raise ValueError("bin_minutes must be > 0")

    xgrid = np.arange(np.floor(xmin / step) * step, np.ceil(xmax / step) * step + step, step)

    if clip_x_minutes is not None:
        lo, hi = float(clip_x_minutes[0]), float(clip_x_minutes[1])
        xgrid = xgrid[(xgrid >= lo) & (xgrid <= hi)]
        if len(xgrid) == 0:
            raise ValueError("clip_x_minutes removed all x points.")

    x0 = xgrid[0]

    def fill_matrix(M: np.ndarray, df: pd.DataFrame, feature: str):
        df = df.copy()
        df["cell_id"] = df["cell_id"].astype(str)
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df["x_min"] = pd.to_numeric(df["x_min"], errors="coerce")

        for i, cid in enumerate(common):
            sub = df[df["cell_id"] == cid][["x_min", feature]].dropna()
            if sub.empty:
                continue

            # bin to nearest xgrid index (minutes)
            xi = np.rint((sub["x_min"].values - x0) / step).astype(int)
            vals = sub[feature].values.astype(float)

            mask = (xi >= 0) & (xi < M.shape[1])
            xi = xi[mask]
            vals = vals[mask]
            if len(xi) == 0:
                continue

            for c, v in zip(xi, vals):
                M[i, c] = v

    nfeat = len(features)
    fig, axs = plt.subplots(
        nrows=nfeat, ncols=1,
        figsize=(14, 3.2 * nfeat),
        sharex=True, sharey=True
    )
    if nfeat == 1:
        axs = np.array([axs])

    extent = [xgrid[0], xgrid[-1], len(common), 0]

    for r, feat in enumerate(features):
        M = np.full((len(common), len(xgrid)), np.nan, dtype=float)
        fill_matrix(M, d1, feat)
        fill_matrix(M, d2, feat)

        ax = axs[r]
        im = ax.imshow(M, aspect="auto", interpolation="nearest", extent=extent)
        ax.set_title(f"{feat}  (GFP1 then GFP2 on minute axis)")
        ax.set_ylabel("cells (ordered by BF1 septum-end)")
        ax.axvline(0.0, linewidth=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axs[-1].set_xlabel(
        f"time (minutes), x=0 is BF1 septum-end | BF={bf_seconds_per_frame:.0f}s/frame, GFP={gfp_seconds_per_frame:.0f}s/frame, bin={bin_minutes:g} min"
    )
    fig.suptitle(title_prefix, y=1.02, fontsize=14)
    fig.tight_layout()
    plt.show()

    return {
        "cells": common,
        "anchors": anchors,
        "xgrid_minutes": xgrid,
        "bin_minutes": bin_minutes,
        "bf_min_per_frame": bf_min_per_frame,
        "gfp_min_per_frame": gfp_min_per_frame,
    }