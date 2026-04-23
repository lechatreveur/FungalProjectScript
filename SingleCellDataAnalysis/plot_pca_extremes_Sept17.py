#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pca_extremes_Sept17.py

For each of PC1, PC2, PC3 — show the 5 highest- and 5 lowest-scoring cells
as a 2 × 5 grid (top row = High end, bottom row = Low end).

Uses the same ID map, pol1/pol2 swap, and colour conventions
as plot_pca_quadrants_Sept17.py.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==== 1. Configuration ====
EXP_DIR = "/Volumes/X10 Pro/Movies/2025_09_17/"

GFP1_CSV     = os.path.join(EXP_DIR, "A14_1TP1_F1/TrackedCells_A14_1TP1_F1/all_cells_time_series.csv")
GFP2_CSV     = os.path.join(EXP_DIR, "A14_1TP2_F1/TrackedCells_A14_1TP2_F1/all_cells_time_series.csv")
ID_MAP_CSV   = os.path.join(EXP_DIR, "unaligned_pairs_quant/bf1_with_gfp1_gfp2_new_ids.csv")
STACKED_CSV  = os.path.join(EXP_DIR, "unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv")
FITS_CSV     = os.path.join(EXP_DIR, "model_fits_by_cell.csv")
PCA_CSV      = os.path.join(EXP_DIR, "pca_latest_feature_scores_Sept17.csv")
OUTPUT_DIR   = os.path.join(EXP_DIR, "PCA_Extremes")
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_EXTREMES = 5   # cells per end
PCS        = ["PC1", "PC2", "PC3"]

# ==== Colour constants ====
COL_POL1 = "#D62728"   # red  — dominant pole
COL_POL2 = "#1F77B4"   # blue — weaker pole

# ==== 2. Load data ====
print("📥 Loading ID map...")
df_id_map = pd.read_csv(ID_MAP_CSV)
df_id_map["new_cell_id"] = df_id_map["new_cell_id"].astype(int)
df_id_map["orig_gfp_id"] = df_id_map["orig_gfp_id"].astype(int)

def load_base_variant(csv_path):
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

print("📥 Loading GFP1 & GFP2 data...")
df_gfp1 = load_base_variant(GFP1_CSV)
df_gfp2 = load_base_variant(GFP2_CSV)

# Build lookup: PCA cell_id → (source_df, orig_gfp_id, label)
source_map = {}
for _, row in df_id_map.iterrows():
    df_src = df_gfp1 if row["source"] == "GFP1" else df_gfp2
    source_map[row["new_cell_id"]] = (df_src, row["orig_gfp_id"], row["source"])

print("📥 Loading PCA scores...")
df_pca = pd.read_csv(PCA_CSV).set_index("cell_id")

# ==== 3. Build pol1/pol2 swap set ====
# Rule: if pol2_mid > pol1_mid -> swap (pol2 is actually the dominant pole)
# mid = a*50 + b for harmonic-fit cells; raw trajectory mean as fallback.
print("📥 Computing swap set...")
df_fits    = pd.read_csv(FITS_CSV)
df_stacked = pd.read_csv(STACKED_CSV)
raw_means  = (df_stacked.groupby("cell_id")[["pol1_int_corr", "pol2_int_corr"]]
              .mean()
              .rename(columns={"pol1_int_corr": "pol1_raw_mean",
                               "pol2_int_corr": "pol2_raw_mean"}))

swap_set = set()
for cell_id, grp in df_fits.groupby("cell_id"):
    mids = {}
    for feat in ["pol1", "pol2"]:
        row = grp[grp["feature"] == feat]
        if row.empty:
            continue
        a = row["trend_params.a"].values[0]
        b = row["trend_params.b"].values[0]
        if pd.notna(a) and pd.notna(b):
            mids[feat] = a * 50 + b
        elif cell_id in raw_means.index:
            mids[feat] = raw_means.loc[cell_id, f"{feat}_raw_mean"]
        else:
            mids[feat] = 0.0
    if mids.get("pol2", 0) > mids.get("pol1", 0):
        swap_set.add(int(cell_id))

print(f"   {len(swap_set)} / {df_fits['cell_id'].nunique()} cells will swap pol1↔pol2.")

# ==== 4. Plotting function ====
def plot_extremes(pc, n=5):
    sorted_cells = df_pca[pc].sort_values()
    bottom_ids = sorted_cells.index[:n].tolist()     # lowest N
    top_ids    = sorted_cells.index[-n:][::-1].tolist()  # highest N

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), sharey=False)

    for row_idx, (group_ids, row_label) in enumerate([
            (top_ids,    f"High {pc}"),
            (bottom_ids, f"Low {pc}")]):

        for col_idx, pca_id in enumerate(group_ids):
            ax = axes[row_idx, col_idx]

            if pca_id not in source_map:
                ax.text(0.5, 0.5, "No mapping", ha="center", va="center",
                        transform=ax.transAxes)
                ax.axis("off")
                continue

            df_src, orig_id, src_label = source_map[pca_id]
            sub = df_src[df_src["cell_id"] == orig_id].sort_values("time_point")

            if sub.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="red")
                ax.axis("off")
                continue

            # Apply same pol1/pol2 swap as used in feature extraction
            if pca_id in swap_set:
                pol1_col, pol2_col = "pol2_int_corr", "pol1_int_corr"
                swap_note = " [sw]"
            else:
                pol1_col, pol2_col = "pol1_int_corr", "pol2_int_corr"
                swap_note = ""

            ax.plot(sub["time_point"], sub[pol1_col],
                    color=COL_POL1, linewidth=1.0, alpha=0.9)
            ax.plot(sub["time_point"], sub[pol2_col],
                    color=COL_POL2, linewidth=1.0, alpha=0.9)
            ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")

            pc_val = df_pca.loc[pca_id, pc]
            ax.set_title(
                f"#{pca_id} {src_label} cell {orig_id}{swap_note}\n"
                f"{pc}={pc_val:.2f}",
                fontsize=8
            )
            ax.set_xlabel("Time (frames)", fontsize=7)
            if col_idx == 0:
                ax.set_ylabel("Corrected Intensity", fontsize=7)
            ax.tick_params(labelsize=7)

        # Row label on the left margin
        axes[row_idx, 0].annotate(
            row_label,
            xy=(-0.35, 0.5), xycoords="axes fraction",
            fontsize=11, fontweight="bold", va="center",
            rotation=90, color="#333333"
        )

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=COL_POL1, label="Pol1 (dominant)"),
        mpatches.Patch(color=COL_POL2, label="Pol2"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=2, fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(f"Top & Bottom {n} cells — {pc}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0.04, 0.05, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, f"{pc}_top_bottom_{n}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved → {out_path}")

# ==== 5. Generate figures ====
for pc in PCS:
    print(f"\nPlotting {pc}...")
    plot_extremes(pc, N_EXTREMES)

print(f"\n🚀 Done. Figures saved to {OUTPUT_DIR}")
