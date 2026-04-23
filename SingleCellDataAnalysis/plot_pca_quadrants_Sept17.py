#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pca_quadrants_Sept17.py

The PCA uses sequential IDs 1-157 assigned to stacked GFP1+GFP2 traces
from paired cells across two movies. The mapping is stored in:
  unaligned_pairs_quant/bf1_with_gfp1_gfp2_new_ids.csv

Columns: new_cell_id, bf1_id, source (GFP1 or GFP2), orig_gfp_id

So to plot a PCA cell_id N:
  1. Look it up in the ID map -> source movie and orig_gfp_id
  2. Load the correct all_cells_time_series.csv (GFP1 = A14_1TP1_F1,
     GFP2 = A14_1TP2_F1)
  3. Filter by orig_gfp_id (base variant only)
  4. Plot pol1_int_corr and pol2_int_corr
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==== 1. Configuration ====
EXP_DIR = "/Volumes/X10 Pro/Movies/2025_09_17/"

GFP1_CSV = os.path.join(EXP_DIR,
    "A14_1TP1_F1/TrackedCells_A14_1TP1_F1/all_cells_time_series.csv")
GFP2_CSV = os.path.join(EXP_DIR,
    "A14_1TP2_F1/TrackedCells_A14_1TP2_F1/all_cells_time_series.csv")
ID_MAP_CSV  = os.path.join(EXP_DIR,
    "unaligned_pairs_quant/bf1_with_gfp1_gfp2_new_ids.csv")
PCA_CSV     = os.path.join(EXP_DIR, "pca_latest_feature_scores_Sept17.csv")
OUTPUT_DIR  = os.path.join(EXP_DIR, "PCA_Quadrant_Examples")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== 2. Load the ID mapping table ====
print(f"📥 Loading ID map from {ID_MAP_CSV}...")
df_id_map = pd.read_csv(ID_MAP_CSV)
# Ensure types
df_id_map["new_cell_id"] = df_id_map["new_cell_id"].astype(int)
df_id_map["orig_gfp_id"] = df_id_map["orig_gfp_id"].astype(int)
print(f"✅ {len(df_id_map)} entries; "
      f"{df_id_map['new_cell_id'].nunique()} unique PCA cells.")

# ==== 3. Load and split GFP1 and GFP2 time series (base variant only) ====
def _norm_int(x):
    m = re.match(r'^\d+', str(x).strip())
    return int(m.group(0)) if m else None

def load_base_variant(csv_path):
    """Load all_cells_time_series.csv and return only the parent (base) rows."""
    df = pd.read_csv(csv_path, dtype={"cell_id": str})
    m = df["cell_id"].str.extract(r'^(?P<canonical>\d+)(?:_(?P<variant>\d+))?$')
    df["canonical_id"] = m["canonical"].astype(int)
    df["variant"]      = m["variant"].fillna("0").astype(int)
    df_base = df[df["variant"] == 0].copy()
    df_base["cell_id"] = df_base["canonical_id"]
    df_base.drop(columns=["canonical_id", "variant"], inplace=True)
    df_base["pol1_int_corr"] = df_base["pol1_int"] - df_base["cyt_int"]
    df_base["pol2_int_corr"] = df_base["pol2_int"] - df_base["cyt_int"]
    return df_base

print(f"📥 Loading GFP1 data...")
df_gfp1 = load_base_variant(GFP1_CSV)
print(f"   {df_gfp1['cell_id'].nunique()} parent cells")

print(f"📥 Loading GFP2 data...")
df_gfp2 = load_base_variant(GFP2_CSV)
print(f"   {df_gfp2['cell_id'].nunique()} parent cells")

# ==== 4. Build lookup: PCA cell_id → (df_source, orig_gfp_id) ====
source_map = {}
for _, row in df_id_map.iterrows():
    pca_id = row["new_cell_id"]
    df_src = df_gfp1 if row["source"] == "GFP1" else df_gfp2
    source_map[pca_id] = (df_src, row["orig_gfp_id"], row["source"])

# ==== 5. Load PCA scores ====
print(f"📥 Loading PCA scores from {PCA_CSV}...")
df_pca = pd.read_csv(PCA_CSV).set_index("cell_id")
print(f"✅ {len(df_pca)} cells in PCA.")

# ==== 6. Pre-compute per-cell pol1/pol2 swap flag from model fits ====
# Rule (from clustering.py line 57):
#   mid = slope * 50 + intercept  (midpoint of the 100-frame window)
#   if pol2_mid > pol1_mid  ->  swap pol1 and pol2 before plotting
MODEL_FITS_CSV = os.path.join(EXP_DIR, "model_fits_by_cell.csv")
STACKED_CSV    = os.path.join(EXP_DIR,
    "unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv")
print(f"📥 Loading model fits from {MODEL_FITS_CSV}...")
df_fits    = pd.read_csv(MODEL_FITS_CSV)
print(f"📥 Loading stacked data for fallback means from {STACKED_CSV}...")
df_stacked = pd.read_csv(STACKED_CSV)

# Pre-compute raw trajectory means per cell (for fallback when trend_params are NaN)
raw_means = (df_stacked.groupby("cell_id")[["pol1_int_corr", "pol2_int_corr"]]
             .mean()
             .rename(columns={"pol1_int_corr": "pol1_raw_mean",
                               "pol2_int_corr": "pol2_raw_mean"}))

swap_set = set()   # PCA cell IDs that need pol1↔pol2 swapped
for cell_id, grp in df_fits.groupby("cell_id"):
    mids = {}
    for feat in ["pol1", "pol2"]:
        row = grp[grp["feature"] == feat]
        if row.empty:
            continue
        a = row["trend_params.a"].values[0]   # NaN for constant/linear models
        b = row["trend_params.b"].values[0]
        if pd.notna(a) and pd.notna(b):
            mids[feat] = a * 50 + b           # mid-point of 100-frame window
        else:
            # Fallback: use the raw trajectory mean from the stacked data
            if cell_id in raw_means.index:
                mids[feat] = raw_means.loc[cell_id, f"{feat}_raw_mean"]
            else:
                mids[feat] = 0.0              # unknown — leave as-is

    if mids.get("pol2", 0) > mids.get("pol1", 0):
        swap_set.add(int(cell_id))

print(f"✅ {len(swap_set)} / {df_fits['cell_id'].nunique()} cells will have pol1↔pol2 swapped.")

# ==== Colour constants — change once here to propagate everywhere ====
COL_POL1 = "#D62728"   # red
COL_POL2 = "#1F77B4"   # blue


low_q  = 0.25
high_q = 0.75

def get_quadrant_cells(df, pc1_high, pc2_high, n=10):
    pc1_thresh = df["PC1"].quantile(high_q if pc1_high else low_q)
    pc2_thresh = df["PC2"].quantile(high_q if pc2_high else low_q)
    mask1 = df["PC1"] >= pc1_thresh if pc1_high else df["PC1"] <= pc1_thresh
    mask2 = df["PC2"] >= pc2_thresh if pc2_high else df["PC2"] <= pc2_thresh
    candidates = df[mask1 & mask2]
    label = (f"PC1_{'High' if pc1_high else 'Low'}, "
             f"PC2_{'High' if pc2_high else 'Low'}")
    print(f"  {len(candidates)} candidates for {label}")
    return candidates.sample(min(n, len(candidates)), random_state=42).index.tolist()

quadrants = {
    "HighPC1_HighPC2": (True,  True),
    "HighPC1_LowPC2":  (True,  False),
    "LowPC1_HighPC2":  (False, True),
    "LowPC1_LowPC2":   (False, False),
}

# ==== 7. Plotting function ====
def plot_cell_group(cell_ids, title, save_name):
    fig, axes = plt.subplots(2, 5, figsize=(18, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    cells_plotted = 0
    for pca_id in cell_ids:
        if cells_plotted >= 10:
            break

        if pca_id not in source_map:
            print(f"  ⚠️  PCA cell {pca_id} not in ID map — skipping")
            continue

        df_src, orig_id, src_label = source_map[pca_id]
        sub = df_src[df_src["cell_id"] == orig_id].sort_values("time_point")

        if sub.empty:
            print(f"  ⚠️  No data for PCA cell {pca_id} "
                  f"({src_label} orig_id={orig_id})")
            continue

        # Apply the same pol1/pol2 swap used during feature extraction:
        # if pol2_mid > pol1_mid at extraction time, pol labels were swapped.
        if pca_id in swap_set:
            pol1_col, pol2_col = "pol2_int_corr", "pol1_int_corr"
            swap_note = " [swapped]"
        else:
            pol1_col, pol2_col = "pol1_int_corr", "pol2_int_corr"
            swap_note = ""

        ax = axes[cells_plotted]
        ax.plot(sub["time_point"], sub[pol1_col],
                color=COL_POL1, label="Pol1", alpha=0.85, linewidth=1.0)
        ax.plot(sub["time_point"], sub[pol2_col],
                color=COL_POL2, label="Pol2", alpha=0.85, linewidth=1.0)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        pc1_val = df_pca.loc[pca_id, "PC1"]
        pc2_val = df_pca.loc[pca_id, "PC2"]
        ax.set_title(
            f"PCA#{pca_id} | {src_label} cell {orig_id}{swap_note}\n"
            f"PC1={pc1_val:.2f}  PC2={pc2_val:.2f}",
            fontsize=7
        )

        if cells_plotted % 5 == 0:
            ax.set_ylabel("Corrected Intensity")
        if cells_plotted >= 5:
            ax.set_xlabel("Time (frames)")

        cells_plotted += 1

    for j in range(cells_plotted, 10):
        axes[j].axis("off")

    # Shared figure-level legend (colour patch per pole)
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color=COL_POL1, label="Pol1"),
        mpatches.Patch(color=COL_POL2, label="Pol2"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 0.0))

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Saved → {out_path}")

# ==== 8. Execute ====
for name, (p1h, p2h) in quadrants.items():
    print(f"\nProcessing {name}...")
    cids = get_quadrant_cells(df_pca, p1h, p2h)
    if cids:
        plot_cell_group(cids, f"PCA Quadrant: {name}", f"{name}_examples.png")

print(f"\n🚀 Done. Figures are in {OUTPUT_DIR}")
