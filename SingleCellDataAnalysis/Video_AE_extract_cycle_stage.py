#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_extract_cycle_stage.py  — Strategy C+, Step 1b

Extracts cell-cycle-stage signals from each cell's raw cell_ID_data.csv
for all 419 cells in the video cache.

Signals extracted (averaged over the 101 frames):
  - cell_length     : Physical cell length (grows through cell cycle)
  - cell_area       : Cell area (proxy for cell size / cycle stage)
  - nu_dis          : Nuclear displacement from center (rises during mitosis)
  - septum_int_norm : Septum GFP intensity (relative to cytoplasm)
  - pattern_score   : Hourglass / split-rectangles score (high only for dividing cells)

These are used to:
  1. Order cells along the cell cycle (cell_length, nu_dis, septum → progression)
  2. Identify dividing cells (pattern_score_norm > threshold)
  3. Create "cycle-stitching" positive pairs:
       dividing cells ↔ newborn cells (same point in cell cycle across the division boundary)

Outputs:
  cycle_stage_features.npy   — (N, 5) feature matrix, one row per cell
  cycle_stage_labels.npy     — (N,) composite cycle stage score [0,1]
  dividing_indices.npy       — indices of dividing cells (pattern_score > threshold)
  newborn_indices.npy        — indices of newborn cells (small, low pattern score)
  cycle_transition_pairs.npy — (N_div × K, 2) pairs [dividing_idx, newborn_idx]
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.Video_AE_data_loader import (
    EXPERIMENT_BASES, STACKED_CSV_PATHS, ID_MAP_CSV_PATHS,
    FILM_FOLDER_MAP, resolve_cell_info_sept17, resolve_cell_info_generic
)

# ==============================================================================
BASE_DIR     = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
GIDS_PATH    = os.path.join(BASE_DIR, "video_gids.txt")
OUT_FEATURES = os.path.join(BASE_DIR, "cycle_stage_features.npy")
OUT_SCORES   = os.path.join(BASE_DIR, "cycle_stage_scores.npy")
OUT_DIVIDING = os.path.join(BASE_DIR, "dividing_indices.npy")
OUT_NEWBORN  = os.path.join(BASE_DIR, "newborn_indices.npy")
OUT_TRANS    = os.path.join(BASE_DIR, "cycle_transition_pairs.npy")

DIVIDING_THRESHOLD = 0.5   # pattern_score_norm above this → dividing cell (fallback only)
NEWBORN_FRAC       = 0.25  # bottom fraction of cell_length = "newborn"
DIVIDING_LENGTH_FRAC = 0.75  # top fraction of cell_length = potentially dividing
DIVIDING_NUDIS_FRAC  = 0.75  # top fraction of nu_dis = nuclear separation occurring
K_PAIRS            = 10    # number of newborn partners per dividing cell
# ==============================================================================


FEATURE_COLS = ['cell_length', 'cell_area', 'nu_dis', 'septum_int', 'pattern_score_norm']
FEATURE_NAMES = FEATURE_COLS  # same


def find_cell_data_csv(gid: str, stacked_dfs: dict, id_map_dfs: dict) -> str | None:
    """
    Given a global cell ID (e.g. 'June25_20m_100'), return the path to
    its cell_ID_data.csv file, or None if it cannot be found.
    """
    known_labels = sorted(EXPERIMENT_BASES.keys(), key=lambda x: -len(x))

    exp_label = None
    suffix = None
    for lbl in known_labels:
        prefix = lbl + '_'
        if gid.startswith(prefix):
            exp_label = lbl
            suffix = gid[len(prefix):]
            break

    if exp_label is None:
        return None

    base_dir = EXPERIMENT_BASES[exp_label]

    # Resolve film name and orig cell ID
    if exp_label == 'Sept17':
        parts = suffix.split('_')
        if len(parts) >= 2 and parts[-1] in ['GFP1', 'GFP2']:
            source = parts[-1]
            gcid = "_".join(parts[:-1])
            parts_gcid = gcid.split('_')
            field = parts_gcid[0]
            orig_id = int(parts_gcid[-1])
            film_name = f"A14_1TP1_{field}" if source == 'GFP1' else f"A14_1TP2_{field}"
        else:
            local_cell_id = int(suffix)
            df_s = stacked_dfs.get('Sept17')
            if df_s is None:
                return None
            film_name, orig_id = resolve_cell_info_sept17(local_cell_id, df_s)
    elif exp_label == 'June25_20m':
        local_cell_id = int(suffix)
        film_name = FILM_FOLDER_MAP.get(('June25_20m', 'GFP1', 'F0'))
        orig_id = local_cell_id
    else:
        local_cell_id = int(suffix)
        df_m = id_map_dfs.get(exp_label)
        if df_m is None:
            return None
        film_name, orig_id = resolve_cell_info_generic(local_cell_id, df_m, exp_label)

    if film_name is None or orig_id is None:
        return None

    csv_path = os.path.join(
        base_dir, film_name,
        f"TrackedCells_{film_name}",
        f"cell_{orig_id}_data.csv"
    )
    return csv_path if os.path.exists(csv_path) else None


def extract_features_from_csv(csv_path: str) -> np.ndarray | None:
    """
    Load a cell_ID_data.csv, compute the mean of each feature column over all
    time points, and return a (5,) feature vector.

    Returns None if the file cannot be read or required columns are missing.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"   [warn] Cannot read {csv_path}: {e}")
        return None

    result = []
    for col in FEATURE_COLS:
        if col not in df.columns:
            print(f"   [warn] Missing column '{col}' in {csv_path}")
            result.append(np.nan)
        else:
            vals = pd.to_numeric(df[col], errors='coerce').dropna().values
            result.append(float(vals.mean()) if len(vals) > 0 else np.nan)

    return np.array(result, dtype=np.float32)


def main():
    # --- Load GIDs ---
    with open(GIDS_PATH) as f:
        gids = [l.strip() for l in f if l.strip()]
    n_cells = len(gids)
    print(f"📋 {n_cells} cells in video cache.")

    # --- Load ID maps ---
    print("📥 Loading ID maps...")
    stacked_dfs, id_map_dfs = {}, {}
    for exp, path in STACKED_CSV_PATHS.items():
        if path and os.path.exists(path):
            stacked_dfs[exp] = pd.read_csv(path)
    for exp, path in ID_MAP_CSV_PATHS.items():
        if path and os.path.exists(path):
            id_map_dfs[exp] = pd.read_csv(path)

    # --- Extract features for each cell ---
    print("🔬 Extracting cycle-stage features from cell_ID_data.csv files...")
    features = np.full((n_cells, len(FEATURE_COLS)), np.nan, dtype=np.float32)
    has_pattern = np.zeros(n_cells, dtype=bool)  # track which cells have real pattern_score
    n_found = 0
    n_missing = 0

    for i, gid in enumerate(gids):
        csv_path = find_cell_data_csv(gid, stacked_dfs, id_map_dfs)
        if csv_path is None:
            n_missing += 1
            if i < 5 or n_missing <= 5:
                print(f"   [miss] {gid}")
            continue
        feat = extract_features_from_csv(csv_path)
        if feat is not None:
            features[i] = feat
            n_found += 1
            if not np.isnan(feat[4]):   # pattern_score_norm is index 4
                has_pattern[i] = True


    print(f"   Found: {n_found}/{n_cells}  |  Missing: {n_missing}")
    print(f"   Cells with genuine pattern_score_norm: {has_pattern.sum()}")

    # --- Report NaN columns ---
    for j, name in enumerate(FEATURE_NAMES):
        n_nan = np.isnan(features[:, j]).sum()
        print(f"   {name:25s}: {n_cells - n_nan}/{n_cells} valid")

    # --- Fill NaN with column median for columns OTHER than pattern_score_norm ---
    # For pattern_score_norm: fill missing with 0.0 (not dividing) rather than median
    for j in range(features.shape[1]):
        col = features[:, j]
        if j == 4:   # pattern_score_norm: missing = not dividing
            features[np.isnan(col), j] = 0.0
        else:
            med = np.nanmedian(col)
            features[np.isnan(col), j] = med

    np.save(OUT_FEATURES, features)
    print(f"\n✅ Saved features {features.shape} → {OUT_FEATURES}")

    # --- Compute sequential piecewise cycle-stage score ---
    # Biological Sequence:
    # 1. Growing: cell_length increases.
    # 2. Nuclear Division: nu_dis increases.
    # 3. Septation: septum_int increases.
    # 4. Division: pattern_score_norm (hourglass) increases.
    
    length_raw = features[:, 0]
    nu_dis_raw = features[:, 2]
    sept_raw   = features[:, 3]
    pat_raw    = features[:, 4]

    # Thresholds (Stricter and Gated)
    pat_thresh  = 0.90 # Only very clear hourglasses
    sept_thresh = np.percentile(sept_raw, 80) # Top 20%
    nu_thresh   = 18.0 # Clear nuclear separation
    med_length  = np.median(length_raw)
    len_new_thresh = np.percentile(length_raw, 25)

    # Classify backwards with LENGTH GATING
    # A cell can only be in Stage 2, 3, or 4 if it is larger than median length
    is_large = length_raw >= med_length

    mask4 = is_large & (pat_raw >= pat_thresh)
    mask3 = is_large & (~mask4) & (sept_raw >= sept_thresh)
    mask2 = is_large & (~mask4) & (~mask3) & (nu_dis_raw >= nu_thresh)
    mask1 = (~mask4) & (~mask3) & (~mask2) # All small cells and non-dividing large cells

    def minmax(x, mask):
        if mask.sum() == 0: return np.zeros_like(x)
        lo, hi = x[mask].min(), x[mask].max()
        if hi == lo: return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    cycle_score = np.zeros(n_cells, dtype=np.float32)
    cycle_score[mask1] = 0.0 + 0.4 * minmax(length_raw, mask1)[mask1]
    cycle_score[mask2] = 0.4 + 0.3 * minmax(nu_dis_raw, mask2)[mask2]
    cycle_score[mask3] = 0.7 + 0.2 * minmax(sept_raw, mask3)[mask3]
    cycle_score[mask4] = 0.9 + 0.1 * minmax(pat_raw, mask4)[mask4]

    np.save(OUT_SCORES, cycle_score)
    print(f"   Cycle stage scores: min={cycle_score.min():.3f}, "
          f"mean={cycle_score.mean():.3f}, max={cycle_score.max():.3f}")
    
    print(f"   Stage 1 (Growing): {mask1.sum()} cells")
    print(f"   Stage 2 (Nuclear Div): {mask2.sum()} cells")
    print(f"   Stage 3 (Septation): {mask3.sum()} cells")
    print(f"   Stage 4 (Hourglass): {mask4.sum()} cells")

    # --- Identify dividing and newborn cells for transition pairs ---
    # Dividing cells are explicitly those in Stage 4
    dividing_idx = np.where(mask4)[0]
    print(f"   Dividing cells (Stage 4): {len(dividing_idx)}")

    # Newborn cells are those in Stage 1 with length <= 25th percentile
    newborn_mask = mask1 & (length_raw <= len_new_thresh)
    newborn_idx  = np.where(newborn_mask)[0]
    print(f"   Newborn cells (Stage 1 AND length≤P25): {len(newborn_idx)}")

    # --- Build cycle transition pairs ---
    # For each dividing cell, find K nearest newborn cells by cycle score
    # (smallest |score_div - score_newborn|, since they should be adjacent in the cycle)
    print(f"\n🔗 Building cycle-transition pairs (K={K_PAIRS} per dividing cell)...")
    pairs = []
    newborn_scores = cycle_score[newborn_idx]

    for div_i in dividing_idx:
        div_score = cycle_score[div_i]
        # Distance in cycle score: the "wrap-around" distance (since it's a cycle)
        # Dividing cells have score ~1, newborns have score ~0 → they're adjacent in the ring
        raw_dist = np.abs(newborn_scores - div_score)
        # Wrap: if raw_dist > 0.5, the cyclic distance is 1 - raw_dist
        cyclic_dist = np.minimum(raw_dist, 1.0 - raw_dist)
        top_k = np.argsort(cyclic_dist)[:K_PAIRS]
        for nb_i in newborn_idx[top_k]:
            pairs.append([int(div_i), int(nb_i)])

    pairs_arr = np.array(pairs, dtype=np.int32)
    np.save(OUT_TRANS, pairs_arr)
    print(f"   Transition pairs: {len(pairs_arr)}  (avg {len(pairs_arr)/max(len(dividing_idx),1):.1f} per dividing cell)")
    print(f"   Saved → {OUT_TRANS}")

    # --- Summary visualisation ---
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].hist(pat_raw, bins=40, color='steelblue', edgecolor='k', linewidth=0.3)
        axes[0].axvline(DIVIDING_THRESHOLD, color='red', linestyle='--', label=f'threshold={DIVIDING_THRESHOLD}')
        axes[0].set_title('pattern_score_norm distribution')
        axes[0].set_xlabel('pattern_score_norm (mean over 101 frames)')
        axes[0].set_ylabel('# cells')
        axes[0].legend()

        length_raw = features[:, 0]
        len_threshold = np.percentile(length_raw, NEWBORN_FRAC * 100)
        axes[1].hist(length_raw, bins=40, color='darkorange', edgecolor='k', linewidth=0.3)
        axes[1].axvline(len_threshold, color='red', linestyle='--', label=f'newborn threshold')
        axes[1].set_title('cell_length distribution')
        axes[1].set_xlabel('mean cell_length (px)')
        axes[1].legend()

        axes[2].scatter(length_raw, pat_raw, c=cycle_score, cmap='viridis', s=12, alpha=0.7)
        axes[2].scatter(length_raw[dividing_idx], pat_raw[dividing_idx],
                        c='red', s=40, marker='*', label='dividing', zorder=5)
        axes[2].scatter(length_raw[newborn_idx], pat_raw[newborn_idx],
                        c='cyan', s=20, marker='o', label='newborn', zorder=4)
        axes[2].set_xlabel('mean cell_length')
        axes[2].set_ylabel('mean pattern_score_norm')
        axes[2].set_title('Cycle stage (color=composite score)')
        axes[2].legend(fontsize=8)

        plt.tight_layout()
        out_fig = os.path.join(BASE_DIR, "cycle_stage_summary.png")
        plt.savefig(out_fig, dpi=150)
        print(f"\n📊 Summary figure saved → {out_fig}")
    except Exception as e:
        print(f"[warn] Could not save figure: {e}")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
