#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 15:43:06 2025

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:58:37 2025

@author: user
"""
import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

# main.py
from SingleCellDataAnalysis.config import WORKING_DIR, FILE_NAMES, FRAME_NUMBER, ROLLING_WINDOW, N_SIGMA
from SingleCellDataAnalysis.load_data import load_and_merge_csv, offset_cell_ids_globally
from SingleCellDataAnalysis.preprocessing import filter_valid_cells, compute_derivatives, add_first_derivative
from SingleCellDataAnalysis.feature_extraction import extract_features
from SingleCellDataAnalysis.gumm import plot_gumm
from SingleCellDataAnalysis.filter_extremes import get_all_extreme_cells
from SingleCellDataAnalysis.alignment import prepare_signals, run_mcmc, run_single_signal_alignment
from SingleCellDataAnalysis.export_aligned import generate_aligned_time_column, export_aligned_dataframe
from SingleCellDataAnalysis.visualization import plot_aligned_signals, plot_aligned_heatmaps, plot_aligned_single_feature
from SingleCellDataAnalysis.spectral_analysis import fft_dominant_frequencies, plot_fft_dominant_frequencies
from SingleCellDataAnalysis.spectral_analysis import gp_infer_periods, plot_gp_periods
from SingleCellDataAnalysis.spectral_analysis import plot_multi_periodic_gp_grid
from SingleCellDataAnalysis.plotting_cells import plot_cells_grid


import os
import pandas as pd
import numpy as np

# ---- Step 1: Load Data ----

print("📥 Loading data...")
#df_all = load_and_merge_csv(FILE_NAMES, WORKING_DIR)
merged_csv = os.path.join(WORKING_DIR, "A14_1TP1_BF_F1/TrackedCells_A14_1TP1_BF_F1/all_cells_time_series.csv")
df_all = pd.read_csv(merged_csv)
#df_all = offset_cell_ids_globally(df_all)



#%%




csv_path = merged_csv

# Example 1: plot all cells (bf only), 10x10 per page, save PNGs
plot_cells_grid(
    csv_path,
    channel="bf",                # omit or change if you want all channels
    nrows=10,
    ncols=10,
    paginate=True,
    save_dir="/Users/user/Desktop/cell_plots",
    filename_prefix="pattern_norm_bf",
    dpi=150,
)

#%%  plot just specific cells in a single 10x10 page (or fewer)
df = pd.read_csv(csv_path)
#some_cells = sorted(df["cell_id"].unique())[:100]
some_cells = np.unique([434,433,430,429,419,426,414,181,167,144,
              407,386,374,357,320,333,261,154,170,177,
              295,291,327,325,342,336,354,388,107,101,
              301,27,43,39,30,51,52,45,53,71,
              75,146,160,59,87,79,120,230,247,286,
              294,311,307,216,218,235,85,64,293,277,
              304,328,286,303,237,199,201,175,168,67,
              9,390,18,96,255,228,262,294,30
              ]).tolist()
plot_cells_grid(
    df,
    cell_ids=some_cells,
    channel="bf",
    paginate=False,   # single page; if >100, extra cells are ignored
)

df_all = df[df['cell_id'].isin(some_cells)]

#%%
from SingleCellDataAnalysis.increasing_period_fit import (
    scan_cells_summary,      # convenience for your var name
    plot_top_k_by_ssr,
    plot_cell_with_increasing_fit,
    plot_cell_with_increasing_fit_gui,
    span_select_gui,
    review_cells_gui,
    overlay_aligned_at_fit_end
)
#%%
# Open GUI for manual selection on a specific cell and update the same summary in-place
# Enforce start≈0.1 and end≈0.4 with ±0.05 tolerance
summary = scan_cells_summary(
    df_all, some_cells,
    enforce_targets=True,
    start_target=0.0, end_target=0.9,
    start_tol=0.1, end_tol=0.5,
)
#%%
res = review_cells_gui(
    df_all, some_cells, summary_df=summary,
    enforce_targets=True,
    start_target=0.0, end_target=1,
    start_tol=5, end_tol=5,
    min_len=21,
    block=False,   # set True if you want it modal
)

fig = res["fig"]  # keep a handle alive

# After you close the GUI, your 'summary' DataFrame is updated (if you clicked Save)
# and 'res' holds the last selection details:
# res -> {'fit': LinearWindowFit(...), 'start': i, 'end': j, 'updated_summary': summary}
#%%
if isinstance(res, dict) and "summary" in res and res["summary"] is not None:
    summary_n = res["summary"]

#%%

# Just overlay all raw traces aligned to their fit end (no truncation)
out = overlay_aligned_at_fit_end(
    df_all,
    summary,
    cell_ids=some_cells,            # or None to use all valid rows in summary
    value_col="pattern_score_norm",
    time_col="time_point",
    show_fit_windows=False,          # set False if you only want raw traces
    individual_alpha=0.25,
    linewidth=1.0,
    title="Aligned to fit end (raw only)",
)

#%% load A14_1TP1_F1 data
print("📥 Loading data...")
#df_all = load_and_merge_csv(FILE_NAMES, WORKING_DIR)
merged_csv_gfp_1 = os.path.join(WORKING_DIR, "A14_1TP1_F1/TrackedCells_A14_1TP1_F1/all_cells_time_series.csv")
df_all_gfp_tp1 = pd.read_csv(merged_csv_gfp_1)

#%% seperate _1 and _2


df = df_all_gfp_tp1.copy()

# Extract canonical id and variant (0 = no suffix)
m = df["cell_id"].str.extract(r'^(?P<canonical>\d+)(?:_(?P<variant>\d+))?$')
df["canonical_cell_id"] = m["canonical"]
df["variant"] = m["variant"].fillna("0").astype(int)

# Split into three DataFrames
df_all_gfp_tp1    = df[df["variant"] == 0].copy()
df_all_gfp_tp1_1  = df[df["variant"] == 1].copy()
df_all_gfp_tp1_2  = df[df["variant"] == 2].copy()

# Drop the suffix: overwrite cell_id with the canonical id in each DataFrame
for d in (df_all_gfp_tp1, df_all_gfp_tp1_1, df_all_gfp_tp1_2):
    d["cell_id"] = d["canonical_cell_id"]
    d.drop(columns=["canonical_cell_id", "variant"], inplace=True)

# Quick counts
print(f"base: {len(df_all_gfp_tp1)}, _1: {len(df_all_gfp_tp1_1)}, _2: {len(df_all_gfp_tp1_2)}")

#%% Map cell ID
from SingleCellDataAnalysis.map_gfp_bf_id import (
    map_gfp_to_bf_ids,
    find_gfp_counterparts,
    align_gfp_to_bf_end_TIMEPOINT,
)
#%%
# Typical junction: last GFP frame vs first BF frame
mapping, pairs_df, files = map_gfp_to_bf_ids(
    WORKING_DIR,
    gfp_timepoint="last",
    bf_timepoint="first",
    gfp_rle_col="rle_gfp",
    bf_rle_col="rle_bf",
    iou_min=0.01,
)

print("Num GFP cells:", pairs_df["gfp_id"].nunique(), 
      "Num BF cells:", pairs_df["bf_id"].nunique())
print("Mapping (GFP → BF):", mapping)

# Save the pairwise IoUs to inspect thresholds / edge cases:
pairs_df.to_csv(os.path.join(WORKING_DIR, "gfp_TP1_bf_TP1_overlap_iou.csv"), index=False)

#%% Identify the correspinding cells from the GFP movie

# Invert mapping (GFP → BF  ==>  BF → GFP)
map_bf_to_gfp = {bf: gfp for gfp, bf in mapping.items()}
# --- Use it ---
gfp_dfs = {
    "base": df_all_gfp_tp1,   # variant==0
    "v1":   df_all_gfp_tp1_1, # variant==1
    "v2":   df_all_gfp_tp1_2  # variant==2
}

matched_table, df_gfp_selected, gfp_some_cells = find_gfp_counterparts(
    some_cells,                  # BF ids you selected earlier
    map_bf_to_gfp,
    gfp_dfs,
    save_csv_path=os.path.join(WORKING_DIR, "bf_to_gfp_matches.csv")
)

print("Mapped BF→GFP counts:", (matched_table["status"]=="mapped").sum(), "/", len(matched_table))
print("Unmapped BF ids:", matched_table.loc[matched_table["status"]=="unmapped","bf_id"].tolist())


#%% Correct intensity features
print("➕ Computing weighted and corrected intensity features...")
df_gfp_selected['septum_int_corr'] = df_gfp_selected['septum_int'] - df_gfp_selected['cyt_int']
df_gfp_selected['pol1_int_corr'] = df_gfp_selected['pol1_int'] - df_gfp_selected['cyt_int']
df_gfp_selected['pol2_int_corr'] = df_gfp_selected['pol2_int'] - df_gfp_selected['cyt_int']
df_gfp_selected['pol1_minus_pol2'] = df_gfp_selected['pol1_int_corr'] - df_gfp_selected['pol2_int_corr']
# df_gfp_selected['weighted_area'] = df_gfp_selected['cell_area'] / 10000
# df_gfp_selected['weighted_pattern_score'] = df_gfp_selected['pattern_score_norm']*150


#%%
df_gfp_aligned = align_gfp_to_bf_end_TIMEPOINT(
    df_gfp_selected=df_gfp_selected,
    matched_table=matched_table,
    summary=summary,
    gfp_id_col="cell_id",
    gfp_time_col="time_point",   # GFP frames
    bf_id_col="cell_id",
    gfp_frames_per_min=5
)



#%%
import numpy as np

# -- keep only the base variant --
df_base = df_gfp_aligned[df_gfp_aligned["gfp_variant"] == "base"].copy()

# use aligned frames (integers) as the plotting time axis
df_plot = df_base.dropna(subset=["aligned_frame_rounded"]).copy()
df_plot["time_point"] = df_plot["aligned_frame_rounded"].astype(int)

# make sure plotting uses the normalized GFP ids as integers
df_plot["cell_id"] = df_plot["gfp_id_norm"].astype(int)

# build a global, unique, sorted timeline
global_time = np.sort(df_plot["time_point"].unique())
time_points = global_time


GFP_FPM = 5  # 5 frames per minute

# Build per-cell shift from the anchor (use one value per cell)
shift_tbl = (
    df_gfp_aligned
      .dropna(subset=["gfp_id_norm", "anchor_bf_end_tp"])
      .groupby("gfp_id_norm", as_index=False)["anchor_bf_end_tp"]
      .first()
)

# Frames shift: negative means moved left on the aligned axis
shift_tbl["shift_frames"] = -shift_tbl["anchor_bf_end_tp"].astype(float) * GFP_FPM
shift_tbl["shift_minutes"] = -shift_tbl["anchor_bf_end_tp"].astype(float)

# Map: {cell_id(int) -> shift_frames}
shift_map = {int(r.gfp_id_norm): float(r.shift_frames) for _, r in shift_tbl.iterrows()}

# Sort the cells you’re about to plot by their shift
# (earliest BF-end first; use reverse=True for latest first)
cell_ids_sorted = sorted(
    df_plot["cell_id"].unique().tolist(),
    key=lambda cid: shift_map.get(int(cid), np.inf)  # cells missing anchors go to the end
)

# If you prefer minutes instead of frames:
# cell_ids_sorted = sorted(
#     df_plot["cell_id"].unique().tolist(),
#     key=lambda cid: (-shift_map_minutes.get(int(cid), np.inf))  # or swap sign/order as you like
# )

# Use this sorted order in your plot call
cell_ids = cell_ids_sorted
best_shifts_aligned = {cid: 0 for cid in cell_ids}  # still zero; we’re only ordering



# choose features to plot (same as you had)
features_xcorr = [
    #"pattern_score_norm",
    #"septum_int_corr",
    "pol1_int_corr",
    "pol2_int_corr",
    "pol1_minus_pol2",
    "cell_area",
    #"weighted_pattern_score",
]

# plot!
plot_aligned_signals(
    df_plot,
    cell_ids,
    best_shifts_aligned,
    global_time,
    time_points,
    features_xcorr,
    mean_trace=None,
    std_trace=None,
    title_prefix="GFP (base) aligned to BF end (frames)"
)
#
heatmap_features = ['pol1_int_corr', 'pol2_int_corr', 'pol1_minus_pol2']
plot_aligned_heatmaps(df_plot, cell_ids, best_shifts_aligned, global_time, time_points, heatmap_features)
#%% load A14_1TP2_F1 data
print("📥 Loading data...")
#df_all = load_and_merge_csv(FILE_NAMES, WORKING_DIR)
merged_csv_gfp_2 = os.path.join(WORKING_DIR, "A14_1TP2_F1/TrackedCells_A14_1TP2_F1/all_cells_time_series.csv")
df_all_gfp_tp2 = pd.read_csv(merged_csv_gfp_2)

# seperate _1 and _2


df = df_all_gfp_tp2.copy()

# Extract canonical id and variant (0 = no suffix)
m = df["cell_id"].str.extract(r'^(?P<canonical>\d+)(?:_(?P<variant>\d+))?$')
df["canonical_cell_id"] = m["canonical"]
df["variant"] = m["variant"].fillna("0").astype(int)

# Split into three DataFrames
df_all_gfp_tp2    = df[df["variant"] == 0].copy()
df_all_gfp_tp2_1  = df[df["variant"] == 1].copy()
df_all_gfp_tp2_2  = df[df["variant"] == 2].copy()

# Drop the suffix: overwrite cell_id with the canonical id in each DataFrame
for d in (df_all_gfp_tp2, df_all_gfp_tp2_1, df_all_gfp_tp2_2):
    d["cell_id"] = d["canonical_cell_id"]
    d.drop(columns=["canonical_cell_id", "variant"], inplace=True)

# Quick counts
print(f"base: {len(df_all_gfp_tp2)}, _1: {len(df_all_gfp_tp2_1)}, _2: {len(df_all_gfp_tp2_2)}")
#%% Map BF IDs to GFP TP2 IDs and print the 1-to-2 mapping
mapping2, bf_to_gfps2, pairs_df2, files2 = map_gfp_to_bf_ids(
    WORKING_DIR,
    gfp_rel="A14_1TP2_F1/TrackedCells_A14_1TP2_F1/",
    bf_rel="A14_1TP1_BF_F1/TrackedCells_A14_1TP1_BF_F1/",
    gfp_timepoint="first",   # GFP starts after BF
    bf_timepoint="last",
    gfp_rle_col="rle_gfp",
    bf_rle_col="rle_bf",
    iou_min=0.01,
    assignment="bf_to_topk_gfp",   # <-- allow one BF to map to multiple GFPs
    k_per_bf=2                     # <-- at most 2 daughters per BF
)

print("GFP cells (TP2):", pairs_df2["gfp_id"].nunique(), "BF cells:", pairs_df2["bf_id"].nunique())
print("Example BF→GFP (daughters) for first few BF ids:")
for bf in list(bf_to_gfps2.keys())[:20]:
    print(f"  BF {bf} → GFP {bf_to_gfps2[bf]}")
# Save the IoUs for inspection
pairs_df2.to_csv(os.path.join(WORKING_DIR, "bf_TP1_gfp_TP2_overlap_iou.csv"), index=False)
#
# import re
# import pandas as pd

# def _norm_int(x):
#     s = str(x).strip().split("_", 1)[0]
#     m = re.match(r"^\d+", s)
#     return int(m.group(0)) if m else None

# # 1) Normalize BF ids from some_cells
# bf_target = sorted({_norm_int(x) for x in some_cells if _norm_int(x) is not None})

# # 2) Normalize keys in bf_to_gfps2 (already ints if built as shown, but be safe)
# bf_to_gfps2_norm = { _norm_int(k): [ _norm_int(g) for g in v ] 
#                      for k, v in bf_to_gfps2.items() if _norm_int(k) is not None }

# # 3) Print only the BF ids from some_cells
# print(f"BF ids requested (some_cells): {len(bf_target)}")
# n_printed = 0
# for bf in bf_target:
#     daughters = bf_to_gfps2_norm.get(bf, [])
#     if daughters:
#         print(f"BF {bf} → GFP {sorted([g for g in daughters if g is not None])}")
#         n_printed += 1

# # 4) Show BF ids from some_cells with NO daughters (no TP2 match above threshold)
# missing = [bf for bf in bf_target if not bf_to_gfps2_norm.get(bf)]
# if missing:
#     print(f"\nNo TP2 GFP mapped for {len(missing)} BF ids (IOU < threshold or absent):")
#     print(missing)

# print(f"\nPrinted {n_printed} BF ids with TP2 daughters out of {len(bf_target)} in some_cells.")
#%% ===== Repeat the GFP TP1 analysis on GFP TP2 =====
import re
import numpy as np
import pandas as pd

def _norm_int(x):
    s = str(x).strip().split("_", 1)[0]
    m = re.match(r"^\d+", s)
    return int(m.group(0)) if m else None

# --- 1) Build a BF→GFP (one-to-many) table restricted to some_cells ---
bf_target = sorted({_norm_int(x) for x in some_cells if _norm_int(x) is not None})

# bf_to_gfps2 may already be ints; normalize to be safe
bf_to_gfps2_norm = { _norm_int(bf): sorted({_norm_int(g) for g in g_list if _norm_int(g) is not None})
                     for bf, g_list in bf_to_gfps2.items() if _norm_int(bf) is not None }

rows = []
for bf in bf_target:
    for g in bf_to_gfps2_norm.get(bf, []):
        rows.append({"bf_id": bf, "gfp_id": g, "status": "mapped"})
matched_table2 = pd.DataFrame(rows)
if matched_table2.empty:
    print("[TP2] No TP2 GFP mapped for the provided some_cells under current IoU threshold.")
else:
    print(f"[TP2] Mapped BF→GFP pairs (TP2) for your some_cells: {len(matched_table2)} pairs; "
          f"{matched_table2['bf_id'].nunique()} BF -> {matched_table2['gfp_id'].nunique()} GFP")

# --- 2) Gather TP2 GFP frames (variants) and keep base-only for plotting ---
gfp_dfs_tp2 = {
    "base": df_all_gfp_tp2,
    "v1":   df_all_gfp_tp2_1,
    "v2":   df_all_gfp_tp2_2,
}

# stack all variants, tag variant, normalize GFP id for joins
tp2_frames = []
for vname, d in gfp_dfs_tp2.items():
    if d is None or len(d) == 0: 
        continue
    tmp = d.copy()
    tmp["gfp_variant"] = vname
    tmp["gfp_id_norm"] = tmp["cell_id"].map(_norm_int)
    tp2_frames.append(tmp)
df_gfp_tp2_all = pd.concat(tp2_frames, ignore_index=True) if tp2_frames else pd.DataFrame()

# Filter to mapped TP2 GFP ids
tp2_gfp_ids = sorted(set(matched_table2["gfp_id"].unique().tolist())) if not matched_table2.empty else []
df_gfp_selected_tp2 = df_gfp_tp2_all[df_gfp_tp2_all["gfp_id_norm"].isin(tp2_gfp_ids)].copy()

# --- 3) Correct intensity features (same as TP1) ---
if not df_gfp_selected_tp2.empty:
    df_gfp_selected_tp2["septum_int_corr"] = df_gfp_selected_tp2["septum_int"] - df_gfp_selected_tp2["cyt_int"]
    df_gfp_selected_tp2["pol1_int_corr"]   = df_gfp_selected_tp2["pol1_int"]   - df_gfp_selected_tp2["cyt_int"]
    df_gfp_selected_tp2["pol2_int_corr"]   = df_gfp_selected_tp2["pol2_int"]   - df_gfp_selected_tp2["cyt_int"]
    df_gfp_selected_tp2["pol1_minus_pol2"] = df_gfp_selected_tp2["pol1_int_corr"] - df_gfp_selected_tp2["pol2_int_corr"]

# --- 4) Align GFP TP2 to BF end TIMEPOINT (1:5) using the same summary of BF fits ---
df_gfp_aligned_tp2 = align_gfp_to_bf_end_TIMEPOINT(
    df_gfp_selected=df_gfp_selected_tp2,
    matched_table=matched_table2,     # one-to-many pairs are fine; helper merges by GFP id
    summary=summary,                  # BF summary with 'end' (timepoint)
    gfp_id_col="cell_id",
    gfp_time_col="time_point",        # GFP frames
    bf_id_col="cell_id",
    gfp_frames_per_min=5
)

# --- 5) Keep base variant ONLY and build plotting grid (aligned frames) ---
df_base_tp2 = df_gfp_aligned_tp2[df_gfp_aligned_tp2["gfp_variant"] == "base"].copy()
df_plot_tp2 = df_base_tp2.dropna(subset=["aligned_frame"]).copy()
df_plot_tp2["aligned_frame_rounded"] = df_plot_tp2["aligned_frame"].round().astype(pd.Int64Dtype())
df_plot_tp2 = df_plot_tp2.dropna(subset=["aligned_frame_rounded"]).copy()

df_plot_tp2["time_point"] = df_plot_tp2["aligned_frame_rounded"].astype(int)
df_plot_tp2["cell_id"]    = df_plot_tp2["gfp_id_norm"].astype(int)

# Unique, sorted timeline
global_time_tp2 = np.sort(df_plot_tp2["time_point"].unique())
time_points_tp2 = global_time_tp2

# Order cells by shift (earliest BF end first)
GFP_FPM = 5
shift_tbl_tp2 = (
    df_gfp_aligned_tp2
      .dropna(subset=["gfp_id_norm", "anchor_bf_end_tp"])
      .groupby("gfp_id_norm", as_index=False)["anchor_bf_end_tp"]
      .first()
)
shift_tbl_tp2["shift_frames"] = -shift_tbl_tp2["anchor_bf_end_tp"].astype(float) * GFP_FPM
shift_map_tp2 = {int(r.gfp_id_norm): float(r.shift_frames) for _, r in shift_tbl_tp2.iterrows()}

cell_ids_tp2 = sorted(
    df_plot_tp2["cell_id"].unique().tolist(),
    key=lambda cid: shift_map_tp2.get(int(cid), np.inf)
)
best_shifts_aligned_tp2 = {cid: 0 for cid in cell_ids_tp2}

print(f"[TP2] Plotting {len(cell_ids_tp2)} GFP TP2 cells (base-only).")

#%% --- 6) Plot signals & heatmaps with the same feature set you used for TP1 ---
features_xcorr_tp2 = [
    "pol1_int_corr",
    "pol2_int_corr",
    "pol1_minus_pol2",
    "cell_area",
]

plot_aligned_signals(
    df_plot_tp2,
    cell_ids_tp2,
    best_shifts_aligned_tp2,
    global_time_tp2,
    time_points_tp2,
    features_xcorr_tp2,
    mean_trace=None,
    std_trace=None,
    title_prefix="GFP TP2 (base) aligned to BF end (frames)"
)

# choose features to show as heatmaps (3)
heatmap_features_tp2 = ["pol1_int_corr", "pol2_int_corr", "pol1_minus_pol2"]

# optional colormaps (same length as features)
cmap_list_tp2 = ["viridis", "viridis", "viridis"]

plot_aligned_heatmaps(
    df_plot_tp2,
    cell_ids_tp2,
    best_shifts_aligned_tp2,
    global_time_tp2,
    time_points_tp2,
    heatmap_features_tp2,      # <-- features go here (6th arg)
    cmap_list=cmap_list_tp2    # <-- colormaps go here (named arg)
)

#%% Load BF_TP2 


print("📥 Loading data...")
#df_all = load_and_merge_csv(FILE_NAMES, WORKING_DIR)
merged_csv2 = os.path.join(WORKING_DIR, "A14_1TP2_BF_F1/TrackedCells_A14_1TP2_BF_F1/all_cells_time_series.csv")
df_all2 = pd.read_csv(merged_csv2)
#df_all = offset_cell_ids_globally(df_all)

#%%  plot just specific cells in a single 10x10 page (or fewer)
df2 = pd.read_csv(merged_csv2)
#some_cells = sorted(df["cell_id"].unique())[:100]
some_cells_2 = np.unique([511, 541, 426, 577, 570, 294, 302, 516, 378, 425, 
                          322, 268, 294, 302, 178, 169, 62, 402, 528, 526, 
                          510, 356, 17, 378, 340, 215, 439, 401, 415, 174, 
                          74, 66, 143, 184, 454, 435, 14, 62, 368, 316, 
                          135, 297, 531
              ]).tolist()
plot_cells_grid(
    df2,
    cell_ids=some_cells_2,
    channel="bf",
    paginate=False,   # single page; if >100, extra cells are ignored
)

df_all_2 = df2[df2['cell_id'].isin(some_cells_2)]
#%%
# Open GUI for manual selection on a specific cell and update the same summary in-place

summary_2 = scan_cells_summary(
    df_all_2, some_cells_2,
    enforce_targets=True,
    start_target=0.0, end_target=0.9,
    start_tol=0.1, end_tol=0.5,
)

res_2 = review_cells_gui(
    df_all_2, some_cells_2, summary_df=summary_2,
    enforce_targets=True,
    start_target=0.0, end_target=1,
    start_tol=5, end_tol=5,
    min_len=11,
    block=False,   # set True if you want it modal
)

fig = res["fig"]  # keep a handle alive
#%%
import os
import pandas as pd
from datetime import datetime

outdir = os.path.join(WORKING_DIR, "manual_fits")
os.makedirs(outdir, exist_ok=True)

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Helpful: ensure numeric types for timepoints
def _sanitize_summary(df):
    df = df.copy()
    for col in ["start", "end", "length", "slope", "intercept", "ssr", "valid"]:
        if col in df.columns:
            if col in ["start", "end", "length"]:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")  # BF timepoints / counts
            elif col == "valid":
                df[col] = df[col].astype(bool)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

summary_s   = _sanitize_summary(summary)
summary2_s  = _sanitize_summary(summary_2)

# Filenames (adjust names if you prefer)
tp1_csv = os.path.join(outdir, f"A14_1TP1_BF_F1_summary_{stamp}.csv")
tp2_csv = os.path.join(outdir, f"A14_1TP2_BF_F1_summary_{stamp}.csv")

summary_s.to_csv(tp1_csv, index=False)
summary2_s.to_csv(tp2_csv, index=False)

print(f"[saved] TP1 summary -> {tp1_csv}")
print(f"[saved] TP2 summary -> {tp2_csv}")

#%% Map  GFP TP2 IDs to BF TP2 IDs
mapping3, bf2_to_gfps2, pairs_df3, files3 = map_gfp_to_bf_ids(
    WORKING_DIR,
    gfp_rel="A14_1TP2_F1/TrackedCells_A14_1TP2_F1/",
    bf_rel="A14_1TP2_BF_F1/TrackedCells_A14_1TP2_BF_F1/",
    gfp_timepoint="last",
    bf_timepoint="first",
    gfp_rle_col="rle_gfp",
    bf_rle_col="rle_bf",
    iou_min=0.01,
    
    
)

print("Num GFP cells:", pairs_df3["gfp_id"].nunique(), 
      "Num BF cells:", pairs_df3["bf_id"].nunique())
print("Mapping (GFP → BF):", mapping3)

# Save the pairwise IoUs to inspect thresholds / edge cases:
pairs_df3.to_csv(os.path.join(WORKING_DIR, "gfp_TP2_bf_TP2_overlap_iou.csv"), index=False)

#%% ===== Align GFP-TP2 to BF-TP2 START anchor =====
import re, numpy as np, pandas as pd

def _norm_int(x):
    s = str(x).strip().split("_", 1)[0]
    m = re.match(r"^\d+", s)
    return int(m.group(0)) if m else None

# --- 1) Build matched table from mapping3 (GFP-TP2 -> BF-TP2) ---
rows_tp2bf = []
for gfp, bf in mapping3.items():  # mapping3: GFP -> BF
    gi = _norm_int(gfp); bi = _norm_int(bf)
    if gi is not None and bi is not None:
        rows_tp2bf.append({"bf_id": int(bi), "gfp_id": int(gi), "status": "mapped"})
matched_table_tp2bf = pd.DataFrame(rows_tp2bf)
print(f"[TP2->BF2 START] pairs: {len(matched_table_tp2bf)} "
      f"({matched_table_tp2bf['bf_id'].nunique()} BF, {matched_table_tp2bf['gfp_id'].nunique()} GFP)")

# --- 2) Stack GFP-TP2 variants; keep only mapped GFP ids ---
gfp_dfs_tp2 = {"base": df_all_gfp_tp2, "v1": df_all_gfp_tp2_1, "v2": df_all_gfp_tp2_2}
tp2_frames = []
for vname, d in gfp_dfs_tp2.items():
    if d is None or len(d) == 0: 
        continue
    tmp = d.copy()
    tmp["gfp_variant"] = vname
    tmp["gfp_id_norm"] = tmp["cell_id"].map(_norm_int)
    tp2_frames.append(tmp)
df_gfp_tp2_all = pd.concat(tp2_frames, ignore_index=True) if tp2_frames else pd.DataFrame()

tp2_mapped_gfps = sorted(set(matched_table_tp2bf["gfp_id"].unique().tolist()))
df_gfp_selected_tp2_alignBF = df_gfp_tp2_all[df_gfp_tp2_all["gfp_id_norm"].isin(tp2_mapped_gfps)].copy()

# --- 3) Corrected GFP features + outlier removal on pol2_int_corr (>100) ---
df_gfp_selected_tp2_alignBF["septum_int_corr"] = df_gfp_selected_tp2_alignBF["septum_int"] - df_gfp_selected_tp2_alignBF["cyt_int"]
df_gfp_selected_tp2_alignBF["pol1_int_corr"]   = df_gfp_selected_tp2_alignBF["pol1_int"]   - df_gfp_selected_tp2_alignBF["cyt_int"]
df_gfp_selected_tp2_alignBF["pol2_int_corr"]   = df_gfp_selected_tp2_alignBF["pol2_int"]   - df_gfp_selected_tp2_alignBF["cyt_int"]
_out = df_gfp_selected_tp2_alignBF["pol2_int_corr"] > 100
if int(_out.sum()):
    print(f"[clean] Dropping {int(_out.sum())} row(s) with pol2_int_corr > 100")
    df_gfp_selected_tp2_alignBF = df_gfp_selected_tp2_alignBF.loc[~_out].copy()
df_gfp_selected_tp2_alignBF["pol1_minus_pol2"] = (
    df_gfp_selected_tp2_alignBF["pol1_int_corr"] - df_gfp_selected_tp2_alignBF["pol2_int_corr"]
)

# --- 4) Build a CLEAN BF-TP2 summary that uses START as the anchor ---
# Keep only the needed columns and ensure there is a SINGLE 'end' column (holding the original 'start')
need_cols = ["cell_id", "start"]
need_cols += [c for c in ("length","slope","intercept","ssr","valid") if c in summary_2.columns]
summary_2_start = summary_2[need_cols].copy()
summary_2_start = summary_2_start.rename(columns={"start": "end"})  # <- now 'end' actually means the START anchor
print("[debug] summary_2_start columns:", list(summary_2_start.columns))

# --- 5) Align GFP-TP2 to BF-TP2 START (timepoint-based) ---
df_gfp_aligned_tp2_toBF2_START = align_gfp_to_bf_end_TIMEPOINT(
    df_gfp_selected=df_gfp_selected_tp2_alignBF,
    matched_table=matched_table_tp2bf,   # expects columns bf_id, gfp_id
    summary=summary_2_start,             # single 'end' col holds START anchor
    gfp_id_col="cell_id",
    gfp_time_col="time_point",
    bf_id_col="cell_id",
    gfp_frames_per_min=5
).rename(columns={"anchor_bf_end_tp": "anchor_bf_start_tp"})

# --- 6) Plotting table (base-only) ---
df_base_tp2_toBF2 = df_gfp_aligned_tp2_toBF2_START[df_gfp_aligned_tp2_toBF2_START["gfp_variant"] == "base"].copy()
df_plot_tp2_toBF2 = df_base_tp2_toBF2.dropna(subset=["aligned_frame"]).copy()
df_plot_tp2_toBF2["aligned_frame_rounded"] = pd.to_numeric(df_plot_tp2_toBF2["aligned_frame"].round(), errors="coerce").astype(pd.Int64Dtype())
df_plot_tp2_toBF2 = df_plot_tp2_toBF2.dropna(subset=["aligned_frame_rounded"]).copy()

df_plot_tp2_toBF2["time_point"] = df_plot_tp2_toBF2["aligned_frame_rounded"].astype(int)
df_plot_tp2_toBF2["cell_id"]    = df_plot_tp2_toBF2["gfp_id_norm"].astype(int)

global_time_tp2_toBF2 = np.sort(df_plot_tp2_toBF2["time_point"].unique())
time_points_tp2_toBF2 = global_time_tp2_toBF2

# --- 7) Order cells by START anchor (earliest first) and plot ---
GFP_FPM = 5
shift_tbl_tp2_toBF2 = (
    df_gfp_aligned_tp2_toBF2_START
      .dropna(subset=["gfp_id_norm", "anchor_bf_start_tp"])
      .groupby("gfp_id_norm", as_index=False)["anchor_bf_start_tp"]
      .first()
)
shift_tbl_tp2_toBF2["shift_frames"] = -shift_tbl_tp2_toBF2["anchor_bf_start_tp"].astype(float) * GFP_FPM
shift_map_tp2_toBF2 = {int(r.gfp_id_norm): float(r.shift_frames) for _, r in shift_tbl_tp2_toBF2.iterrows()}

cell_ids_tp2_toBF2 = sorted(
    df_plot_tp2_toBF2["cell_id"].unique().tolist(),
    key=lambda cid: shift_map_tp2_toBF2.get(int(cid), np.inf)
)
best_shifts_aligned_tp2_toBF2 = {cid: 0 for cid in cell_ids_tp2_toBF2}

features_xcorr_tp2_toBF2 = ["pol1_int_corr", "pol2_int_corr", "pol1_minus_pol2", "cell_area"]

plot_aligned_signals(
    df_plot_tp2_toBF2,
    cell_ids_tp2_toBF2,
    best_shifts_aligned_tp2_toBF2,
    global_time_tp2_toBF2,
    time_points_tp2_toBF2,
    features_xcorr_tp2_toBF2,
    mean_trace=None,
    std_trace=None,
    title_prefix="GFP TP2 (base) aligned to BF TP2 START (frames)"
)

plot_aligned_heatmaps(
    df_plot_tp2_toBF2,
    cell_ids_tp2_toBF2,
    best_shifts_aligned_tp2_toBF2,
    global_time_tp2_toBF2,
    time_points_tp2_toBF2,
    ["pol1_int_corr", "pol2_int_corr", "pol1_minus_pol2"],
    cmap_list=["viridis", "viridis", "viridis"]
)


#%% ===== Identify GFP-TP1 ids for BF-TP2 cells and align GFP-TP1 to BF-TP2 START =====
import re, numpy as np, pandas as pd

def _norm_int(x):
    s = str(x).strip().split("_", 1)[0]
    m = re.match(r"^\d+", s)
    return int(m.group(0)) if m else None

# --- A) Build the composed map: BF2 -> GFP1 ---
# (i) invert mapping3 (GFP-TP2 -> BF-TP2) -> BF-TP2 -> GFP-TP2
bf2_to_gfp2 = {}
for g2, b2 in mapping3.items():
    gi = _norm_int(g2); bi = _norm_int(b2)
    if gi is not None and bi is not None and bi not in bf2_to_gfp2:
        bf2_to_gfp2[bi] = gi  # assume 1:1 here; keep first if duplicates

# (ii) from pairs_df2 (BF-TP1 ↔ GFP-TP2), get best BF-TP1 per GFP-TP2 (max IoU)
# pairs_df2 must have columns: ['bf_id', 'gfp_id', 'iou'] (your helper returns that)
gfp2_to_bf1 = {}
if "iou" in pairs_df2.columns:
    tmp = (pairs_df2
           .dropna(subset=["bf_id","gfp_id","iou"])
           .assign(bf_id=lambda d: d["bf_id"].map(_norm_int),
                   gfp_id=lambda d: d["gfp_id"].map(_norm_int))
           .dropna(subset=["bf_id","gfp_id"]))
    # pick the bf_id with max IoU per gfp_id
    idx = tmp.groupby("gfp_id")["iou"].idxmax()
    for _, r in tmp.loc[idx, ["gfp_id","bf_id"]].iterrows():
        gfp2_to_bf1[int(r.gfp_id)] = int(r.bf_id)
else:
    # fallback: if no IoU column, just take first occurrence per gfp_id
    tmp = (pairs_df2
           .dropna(subset=["bf_id","gfp_id"])
           .assign(bf_id=lambda d: d["bf_id"].map(_norm_int),
                   gfp_id=lambda d: d["gfp_id"].map(_norm_int))
           .dropna(subset=["bf_id","gfp_id"]))
    firsts = tmp.drop_duplicates("gfp_id")
    for _, r in firsts.iterrows():
        gfp2_to_bf1[int(r.gfp_id)] = int(r.bf_id)

# (iii) invert mapping (GFP-TP1 -> BF-TP1) -> BF-TP1 -> GFP-TP1
bf1_to_gfp1 = {}
for g1, b1 in mapping.items():
    gi = _norm_int(g1); bi = _norm_int(b1)
    if gi is not None and bi is not None and bi not in bf1_to_gfp1:
        bf1_to_gfp1[bi] = gi

# (iv) compose for your target set some_cells_2 (BF-TP2 ids)
bf2_targets = sorted({_norm_int(x) for x in some_cells_2 if _norm_int(x) is not None})
rows_bf2_gfp1 = []
lost = []
for bf2 in bf2_targets:
    g2 = bf2_to_gfp2.get(bf2)
    if g2 is None:
        lost.append((bf2, "no_gfp2"))
        continue
    bf1 = gfp2_to_bf1.get(g2)
    if bf1 is None:
        lost.append((bf2, "no_bf1_from_gfp2"))
        continue
    g1 = bf1_to_gfp1.get(bf1)
    if g1 is None:
        lost.append((bf2, "no_gfp1_from_bf1"))
        continue
    rows_bf2_gfp1.append({"bf2_id": bf2, "gfp1_id": g1, "status": "mapped"})

matched_bf2_to_gfp1 = pd.DataFrame(rows_bf2_gfp1)
print(f"[BF2→GFP1] composed pairs: {len(matched_bf2_to_gfp1)} "
      f"({matched_bf2_to_gfp1['bf2_id'].nunique()} BF2 -> {matched_bf2_to_gfp1['gfp1_id'].nunique()} GFP1)")
if lost:
    print(f"[BF2→GFP1] missing: {len(lost)} (first 10):", lost[:10])

# --- B) Gather GFP-TP1 frames (variants) for those GFP1 ids ---
gfp_dfs_tp1 = {
    "base": df_all_gfp_tp1,
    "v1":   df_all_gfp_tp1_1,
    "v2":   df_all_gfp_tp1_2,
}
tp1_frames = []
for vname, d in gfp_dfs_tp1.items():
    if d is None or len(d) == 0: 
        continue
    tmp = d.copy()
    tmp["gfp_variant"] = vname
    tmp["gfp_id_norm"] = tmp["cell_id"].map(_norm_int)
    tp1_frames.append(tmp)
df_gfp_tp1_all = pd.concat(tp1_frames, ignore_index=True) if tp1_frames else pd.DataFrame()

tp1_gfp_ids = sorted(set(matched_bf2_to_gfp1["gfp1_id"].unique().tolist()))
df_gfp_selected_tp1_forBF2 = df_gfp_tp1_all[df_gfp_tp1_all["gfp_id_norm"].isin(tp1_gfp_ids)].copy()

# --- C) Correct intensity features (same as before; optional outlier removal) ---
if not df_gfp_selected_tp1_forBF2.empty:
    df_gfp_selected_tp1_forBF2["septum_int_corr"] = df_gfp_selected_tp1_forBF2["septum_int"] - df_gfp_selected_tp1_forBF2["cyt_int"]
    df_gfp_selected_tp1_forBF2["pol1_int_corr"]   = df_gfp_selected_tp1_forBF2["pol1_int"]   - df_gfp_selected_tp1_forBF2["cyt_int"]
    df_gfp_selected_tp1_forBF2["pol2_int_corr"]   = df_gfp_selected_tp1_forBF2["pol2_int"]   - df_gfp_selected_tp1_forBF2["cyt_int"]
    # optional: drop crazy pol2 outliers (keep consistent with TP2 cleaning if you want)
    _out_tp1 = df_gfp_selected_tp1_forBF2["pol2_int_corr"] > 100
    if int(_out_tp1.sum()):
        print(f"[TP1 clean] Dropping {int(_out_tp1.sum())} row(s) with pol2_int_corr > 100")
        df_gfp_selected_tp1_forBF2 = df_gfp_selected_tp1_forBF2.loc[~_out_tp1].copy()
    df_gfp_selected_tp1_forBF2["pol1_minus_pol2"] = (
        df_gfp_selected_tp1_forBF2["pol1_int_corr"] - df_gfp_selected_tp1_forBF2["pol2_int_corr"]
    )

# --- D) Build START-anchor summary for BF-TP2 (as you did) ---
need_cols2 = ["cell_id", "start"]
need_cols2 += [c for c in ("length","slope","intercept","ssr","valid") if c in summary_2.columns]
summary_bf2_start = summary_2[need_cols2].copy().rename(columns={"start": "end"})  # 'end' holds START anchor

# --- E) Make a matched table (bf2, gfp1) for the alignment helper ---
matched_table_bf2_gfp1 = matched_bf2_to_gfp1.rename(columns={"bf2_id": "bf_id", "gfp1_id": "gfp_id"})[["bf_id","gfp_id","status"]].copy()

# --- F) Align GFP-TP1 to BF-TP2 START (timepoint-based, 1:5) ---
df_gfp_tp1_aligned_to_BF2_START = align_gfp_to_bf_end_TIMEPOINT(
    df_gfp_selected=df_gfp_selected_tp1_forBF2,
    matched_table=matched_table_bf2_gfp1,   # bf_id (BF-TP2), gfp_id (GFP-TP1)
    summary=summary_bf2_start,              # has 'end' = START timepoint
    gfp_id_col="cell_id",
    gfp_time_col="time_point",
    bf_id_col="cell_id",
    gfp_frames_per_min=5
).rename(columns={"anchor_bf_end_tp": "anchor_bf2_start_tp"})

# --- G) Prep for plotting (base-only) ---
df_base_tp1_toBF2 = df_gfp_tp1_aligned_to_BF2_START[df_gfp_tp1_aligned_to_BF2_START["gfp_variant"] == "base"].copy()
df_plot_tp1_toBF2 = df_base_tp1_toBF2.dropna(subset=["aligned_frame"]).copy()
df_plot_tp1_toBF2["aligned_frame_rounded"] = pd.to_numeric(df_plot_tp1_toBF2["aligned_frame"].round(),
                                                           errors="coerce").astype(pd.Int64Dtype())
df_plot_tp1_toBF2 = df_plot_tp1_toBF2.dropna(subset=["aligned_frame_rounded"]).copy()

df_plot_tp1_toBF2["time_point"] = df_plot_tp1_toBF2["aligned_frame_rounded"].astype(int)
df_plot_tp1_toBF2["cell_id"]    = df_plot_tp1_toBF2["gfp_id_norm"].astype(int)

global_time_tp1_toBF2 = np.sort(df_plot_tp1_toBF2["time_point"].unique())
time_points_tp1_toBF2 = global_time_tp1_toBF2

# order cells by START anchor (earliest first)
GFP_FPM = 5
shift_tbl_tp1_toBF2 = (
    df_gfp_tp1_aligned_to_BF2_START
      .dropna(subset=["gfp_id_norm", "anchor_bf2_start_tp"])
      .groupby("gfp_id_norm", as_index=False)["anchor_bf2_start_tp"]
      .first()
)
shift_tbl_tp1_toBF2["shift_frames"] = -shift_tbl_tp1_toBF2["anchor_bf2_start_tp"].astype(float) * GFP_FPM
shift_map_tp1_toBF2 = {int(r.gfp_id_norm): float(r.shift_frames) for _, r in shift_tbl_tp1_toBF2.iterrows()}

cell_ids_tp1_toBF2 = sorted(
    df_plot_tp1_toBF2["cell_id"].unique().tolist(),
    key=lambda cid: shift_map_tp1_toBF2.get(int(cid), np.inf)
)
best_shifts_aligned_tp1_toBF2 = {cid: 0 for cid in cell_ids_tp1_toBF2}

print(f"[GFP-TP1 vs BF-TP2 START] plotting {len(cell_ids_tp1_toBF2)} GFP-TP1 cells (base-only).")

# --- H) Plot (same feature set as before) ---
features_xcorr_tp1_toBF2 = ["pol1_int_corr", "pol2_int_corr", "pol1_minus_pol2", "cell_area"]

plot_aligned_signals(
    df_plot_tp1_toBF2,
    cell_ids_tp1_toBF2,
    best_shifts_aligned_tp1_toBF2,
    global_time_tp1_toBF2,
    time_points_tp1_toBF2,
    features_xcorr_tp1_toBF2,
    mean_trace=None,
    std_trace=None,
    title_prefix="GFP TP1 (base) aligned to BF TP2 START (frames)"
)

plot_aligned_heatmaps(
    df_plot_tp1_toBF2,
    cell_ids_tp1_toBF2,
    best_shifts_aligned_tp1_toBF2,
    global_time_tp1_toBF2,
    time_points_tp1_toBF2,
    ["pol1_int_corr", "pol2_int_corr", "pol1_minus_pol2"],
    cmap_list=["viridis", "viridis", "viridis"]
)
#%% Sanity print: chains GFP1 -> BF1 -> GFP2 -> BF2 for the plotted cells
import re
def _norm_int(x):
    s = str(x).strip().split("_", 1)[0]
    m = re.match(r"^\d+", s)
    return int(m.group(0)) if m else None

# mapping (GFP1 -> BF1) normalize
gfp1_to_bf1 = {}
for g1, b1 in mapping.items():
    gi = _norm_int(g1); bi = _norm_int(b1)
    if gi is not None and bi is not None and gi not in gfp1_to_bf1:
        gfp1_to_bf1[gi] = bi

# mapping3 (GFP2 -> BF2) already in that direction; normalize
gfp2_to_bf2 = {}
for g2, b2 in mapping3.items():
    gi = _norm_int(g2); bi = _norm_int(b2)
    if gi is not None and bi is not None and gi not in gfp2_to_bf2:
        gfp2_to_bf2[gi] = bi

# pairs_df2 gives BF1 <-> GFP2; build BF1 -> GFP2 using max IoU per BF1
bf1_to_gfp2 = {}
if "iou" in pairs_df2.columns:
    tmp = (pairs_df2
           .dropna(subset=["bf_id","gfp_id","iou"])
           .assign(bf_id=lambda d: d["bf_id"].map(_norm_int),
                   gfp_id=lambda d: d["gfp_id"].map(_norm_int)))
    tmp = tmp.dropna(subset=["bf_id","gfp_id"])
    idx = tmp.groupby("bf_id")["iou"].idxmax()
    for _, r in tmp.loc[idx, ["bf_id","gfp_id"]].iterrows():
        bf1_to_gfp2[int(r.bf_id)] = int(r.gfp_id)
else:
    tmp = (pairs_df2
           .dropna(subset=["bf_id","gfp_id"])
           .assign(bf_id=lambda d: d["bf_id"].map(_norm_int),
                   gfp_id=lambda d: d["gfp_id"].map(_norm_int)))
    tmp = tmp.dropna(subset=["bf_id","gfp_id"]).drop_duplicates("bf_id")
    for _, r in tmp.iterrows():
        bf1_to_gfp2[int(r.bf_id)] = int(r.gfp_id)

# Now compose and print only for the plotted GFP1 ids
chains = []
for g1 in cell_ids_tp1_toBF2:   # these are your 27 plotted GFP1 ids (ints)
    bf1 = gfp1_to_bf1.get(int(g1))
    if bf1 is None:
        continue
    g2  = bf1_to_gfp2.get(int(bf1))
    if g2 is None:
        continue
    bf2 = gfp2_to_bf2.get(int(g2))
    if bf2 is None:
        continue
    chains.append((int(g1), int(bf1), int(g2), int(bf2)))

print(f"\nChains GFP1→BF1→GFP2→BF2 ({len(chains)} cells):")
for g1, b1, g2, b2 in chains:
    print(f"GFP1 {g1} -> BF1 {b1} -> GFP2 {g2} -> BF2 {b2}")

# (optional) save to CSV for record
import pandas as pd, os
chains_df = pd.DataFrame(chains, columns=["gfp1_id","bf1_id","gfp2_id","bf2_id"])
chains_csv = os.path.join(WORKING_DIR, "chains_GFP1_to_BF2_plotted.csv")
chains_df.to_csv(chains_csv, index=False)
print(f"[saved] chains -> {chains_csv}")
#%%
import os, subprocess, shlex, shutil

def _resolve_ffmpeg():
    """
    Return a full path to an ffmpeg executable.
    Prefer system ffmpeg; fall back to imageio-ffmpeg if available (and installable).
    """
    # 1) system ffmpeg in PATH?
    path = shutil.which("ffmpeg")
    if path:
        return path

    # 2) try imageio-ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise FileNotFoundError(
            "ffmpeg not found. Install one of:\n"
            " - brew install ffmpeg  (macOS)\n"
            " - conda install -c conda-forge ffmpeg\n"
            " - pip install imageio-ffmpeg  (then re-run)\n"
            f"(original error: {e})"
        )

def merge_population_videos(WORKING_DIR,
                            files=("A14_1TP1_F1_population.mp4",
                                   "A14_1TP1_BF_F1_population.mp4",
                                   "A14_1TP2_F1_population.mp4",
                                   "A14_1TP2_BF_F1_population.mp4"),
                            out_name="merged_A14_TP1F1_BF1_TP2F1_BF2.mp4",
                            width=1280, fps=5, crf=18):
    ffmpeg = _resolve_ffmpeg()

    # Build absolute paths and sanity check
    inputs = [os.path.join(WORKING_DIR, f) for f in files]
    missing = [p for p in inputs if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing input file(s): {missing}")

    out_path = os.path.join(WORKING_DIR, out_name)

    # Build filter graph
    vf_parts, labels = [], []
    for i in range(len(inputs)):
        vf_parts.append(f"[{i}:v]scale={width}:-2,fps={fps},format=yuv420p[v{i}]")
        labels.append(f"[v{i}]")
    vf = ";".join(vf_parts) + ";" + "".join(labels) + f"concat=n={len(inputs)}:v=1:a=0[v]"

    # Assemble command
    cmd = [ffmpeg]
    for p in inputs:
        cmd += ["-i", p]
    cmd += [
        "-filter_complex", vf,
        "-map", "[v]",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        out_path
    ]

    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True)
    print(f"[ok] Wrote -> {out_path}")
merge_population_videos(WORKING_DIR)
#%%
# ===== Plot GFP1 & GFP2 for your tracked chains, both aligned to BF2 START =====

# 1) Declare the 14 chains you validated (GFP1 -> BF1 -> GFP2 -> BF2)
chains_input = [
    (266, 317, 355, 415),
    (188, 225, 257, 302),
    (248, 299, 352, 401),
    (366, 432, 507, 577),
    (220, 270, 310, 368),
    (362, 428, 500, 570),
    (198, 245, 226, 268),
    (318, 378, 402, 454),
    (235, 278, 332, 378),
    (162, 204, 249, 294),
    (9,   16,  12,  17),
    (278, 332, 375, 426),
    (96,  129, 145, 174),
    (271, 322, 361, 402),
]

import numpy as np
import pandas as pd

# 2) Build clean DataFrame and pull the GFP1/GFP2 id lists
chains_df_ok = pd.DataFrame(chains_input, columns=["gfp1_id","bf1_id","gfp2_id","bf2_id"])
gfp1_keep = chains_df_ok["gfp1_id"].astype(int).tolist()
gfp2_keep = chains_df_ok["gfp2_id"].astype(int).tolist()

# 3) Filter the already-aligned plotting tables you prepared earlier:
#    - df_plot_tp1_toBF2 : GFP1 aligned to BF2 START
#    - df_plot_tp2_toBF2 : GFP2 aligned to BF2 START
#    If any IDs are missing (e.g., variant/base filtering), we’ll drop them and report.
avail_gfp1 = sorted(set(df_plot_tp1_toBF2["cell_id"].astype(int)) & set(gfp1_keep))
miss_gfp1  = sorted(set(gfp1_keep) - set(avail_gfp1))
avail_gfp2 = sorted(set(df_plot_tp2_toBF2["cell_id"].astype(int)) & set(gfp2_keep))
miss_gfp2  = sorted(set(gfp2_keep) - set(avail_gfp2))

if miss_gfp1:
    print(f"[note] GFP1 ids not present in df_plot_tp1_toBF2 (skipped): {miss_gfp1}")
if miss_gfp2:
    print(f"[note] GFP2 ids not present in df_plot_tp2_toBF2 (skipped): {miss_gfp2}")

# Order by BF2 id (earliest start first) using your start-anchor shifts if you like.
# Here we just keep the chain order as provided.
cell_ids_gfp1 = [cid for cid in gfp1_keep if cid in avail_gfp1]
cell_ids_gfp2 = [cid for cid in gfp2_keep if cid in avail_gfp2]

# 4) Zero shifts (we already aligned to BF2 START)
best_shifts_gfp1 = {cid: 0 for cid in cell_ids_gfp1}
best_shifts_gfp2 = {cid: 0 for cid in cell_ids_gfp2}

# 5) Use the same feature sets you’ve been plotting
features_signals = ["pol1_int_corr", "pol2_int_corr", "pol1_minus_pol2", "cell_area"]
features_heatmap = ["pol1_int_corr", "pol2_int_corr", "pol1_minus_pol2"]

# 6) Plot GFP1 (aligned to BF2 START)
plot_aligned_signals(
    df_plot_tp1_toBF2,
    cell_ids_gfp1,
    best_shifts_gfp1,
    global_time_tp1_toBF2,
    time_points_tp1_toBF2,
    features_signals,
    mean_trace=None,
    std_trace=None,
    title_prefix="GFP TP1 (base) aligned to BF TP2 START — tracked chains"
)

plot_aligned_heatmaps(
    df_plot_tp1_toBF2,
    cell_ids_gfp1,
    best_shifts_gfp1,
    global_time_tp1_toBF2,
    time_points_tp1_toBF2,
    features_heatmap,
    cmap_list=["viridis","viridis","viridis"]
)

# 7) Plot GFP2 (aligned to BF2 START)
plot_aligned_signals(
    df_plot_tp2_toBF2,
    cell_ids_gfp2,
    best_shifts_gfp2,
    global_time_tp2_toBF2,
    time_points_tp2_toBF2,
    features_signals,
    mean_trace=None,
    std_trace=None,
    title_prefix="GFP TP2 (base) aligned to BF TP2 START — tracked chains"
)

plot_aligned_heatmaps(
    df_plot_tp2_toBF2,
    cell_ids_gfp2,
    best_shifts_gfp2,
    global_time_tp2_toBF2,
    time_points_tp2_toBF2,
    features_heatmap,
    cmap_list=["viridis","viridis","viridis"]
)


#%%
# ===== Paired per-cell plots (PDF, no forced 0; x-lims match data ends) =====
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Your 14 validated chains
chains_input = [
    (266, 317, 355, 415),
    (188, 225, 257, 302),
    (248, 299, 352, 401),
    (366, 432, 507, 577),
    (220, 270, 310, 368),
    (362, 428, 500, 570),
    (198, 245, 226, 268),
    (318, 378, 402, 454),
    (235, 278, 332, 378),
    (162, 204, 249, 294),
    (9,   16,  12,  17),
    (278, 332, 375, 426),
    (96,  129, 145, 174),
    (271, 322, 361, 402),
]
chains_df = pd.DataFrame(chains_input, columns=["gfp1_id","bf1_id","gfp2_id","bf2_id"])

# 2) Sanity check required columns exist
need_cols = {"time_point","cell_id","pol1_int_corr","pol2_int_corr"}
for name, dfcheck in [("df_plot_tp1_toBF2", df_plot_tp1_toBF2),
                      ("df_plot_tp2_toBF2", df_plot_tp2_toBF2)]:
    missing_cols = need_cols - set(dfcheck.columns)
    if missing_cols:
        raise KeyError(f"{name} missing columns: {sorted(missing_cols)}")

# 3) Output folder (PDF)
out_dir = os.path.join(WORKING_DIR, "paired_plots_GFP1_GFP2_vs_BF2START_PDF")
os.makedirs(out_dir, exist_ok=True)

# 4) Plot each chain
not_found = []
made = 0

for _, row in chains_df.iterrows():
    g1, b1, g2, b2 = int(row.gfp1_id), int(row.bf1_id), int(row.gfp2_id), int(row.bf2_id)

    # Slice aligned tables (already aligned to BF2 START)
    d1 = (df_plot_tp1_toBF2[df_plot_tp1_toBF2["cell_id"].astype(int) == g1]
          .sort_values("time_point"))
    d2 = (df_plot_tp2_toBF2[df_plot_tp2_toBF2["cell_id"].astype(int) == g2]
          .sort_values("time_point"))

    if d1.empty or d2.empty:
        not_found.append((g1, b1, g2, b2, "missing_gfp1" if d1.empty else "missing_gfp2"))
        continue

    # Common y-limits for comparability across the pair
    yvals = np.concatenate([
        d1["pol1_int_corr"].astype(float).values,
        d1["pol2_int_corr"].astype(float).values,
        d2["pol1_int_corr"].astype(float).values,
        d2["pol2_int_corr"].astype(float).values
    ])
    y_min, y_max = np.nanmin(yvals), np.nanmax(yvals)
    y_pad = 0.05 * (y_max - y_min if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0)
    ylims = (y_min - y_pad, y_max + y_pad)

    # X-limits = exact data ends (per subplot)
    x1_min, x1_max = np.nanmin(d1["time_point"]), np.nanmax(d1["time_point"])
    x2_min, x2_max = np.nanmin(d2["time_point"]), np.nanmax(d2["time_point"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax1, ax2 = axes

    # --- Left: GFP1 ---
    ax1.plot(d1["time_point"], d1["pol1_int_corr"], label="pol1", linewidth=1.5)
    ax1.plot(d1["time_point"], d1["pol2_int_corr"], label="pol2", linewidth=1.5, linestyle="--")
    ax1.set_xlim(x1_min, x1_max)  # exact data span
    ax1.set_title(f"GFP1 {g1}  (BF1 {b1})")
    ax1.set_xlabel("Aligned frame (BF2 START anchor)")
    ax1.set_ylabel("Intensity (corrected)")
    ax1.set_ylim(*ylims)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False, loc="best")

    # --- Right: GFP2 ---
    ax2.plot(d2["time_point"], d2["pol1_int_corr"], label="pol1", linewidth=1.5)
    ax2.plot(d2["time_point"], d2["pol2_int_corr"], label="pol2", linewidth=1.5, linestyle="--")
    ax2.set_xlim(x2_min, x2_max)  # exact data span
    ax2.set_title(f"GFP2 {g2}  (BF2 {b2})")
    ax2.set_xlabel("Aligned frame (BF2 START anchor)")
    ax2.set_ylim(*ylims)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("pol1 & pol2 vs aligned time (BF2 START anchor)", y=1.02, fontsize=12)
    fig.tight_layout()

    # Save as vector PDF
    fname = f"pair_GFP1_{g1}_BF1_{b1}__GFP2_{g2}_BF2_{b2}.pdf"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=300, bbox_inches="tight", format="pdf")  # vector output
    plt.close(fig)
    made += 1

print(f"[ok] Wrote {made} paired PDF figures to: {out_dir}")
if not_found:
    print("[warn] Skipped due to missing aligned data:", not_found)
#%% Prepare for a list of GFP1 and GFP2 from BF1 IDs that are not aligned
unaligned_cells = np.unique([15,34,42,29,41,44,50,46,63,134,
                             161,131,151,128,125,69,74,97,119,172,
                             200,198,193,164,205,217,192,191,314,260,
                             243,229,225,204,194,182,205,217,192,191,
                             243,229,225,204,194,182,205,245,231,254,
                             236,266,233,193,198,207,271,265,240,287,
                             222,294,197,195,143,183,258,238,273,180,
                             241,317,300,302,283,250,276,251,227,260,
                             278,299,329,323,348,340,364,371,331,330,
                             404,391,384,383,358,356,298,352,318,350,
                             369,353,401,395,391,349,314,
              ]).tolist()

#%% Map BF1 ids -> GFP1 and GFP2 (best and all candidates)
import numpy as np, pandas as pd, os, re

def _norm_int(x):
    s = str(x).strip().split("_", 1)[0]
    m = re.match(r"^\d+", s)
    return int(m.group(0)) if m else None

# 0) Normalize your BF1 list
bf1_list = sorted({_norm_int(x) for x in unaligned_cells if _norm_int(x) is not None})

# 1) Build BF1 -> GFP1 by inverting `mapping` (GFP1 -> BF1)
bf1_to_gfp1 = {}
for g1, b1 in mapping.items():
    gi = _norm_int(g1); bi = _norm_int(b1)
    if gi is not None and bi is not None and bi not in bf1_to_gfp1:
        bf1_to_gfp1[bi] = gi

# 2) Build BF1 -> GFP2 using pairs_df2 (pick max IoU per BF1)
bf1_to_gfp2_best = {}
bf1_to_gfp2_all = {}  # collect all candidates
if "bf_id" in pairs_df2.columns and "gfp_id" in pairs_df2.columns:
    tmp = (pairs_df2
           .dropna(subset=["bf_id","gfp_id"])
           .assign(bf_id=lambda d: d["bf_id"].map(_norm_int),
                   gfp_id=lambda d: d["gfp_id"].map(_norm_int)))
    tmp = tmp.dropna(subset=["bf_id","gfp_id"])
    # keep all candidates per BF1 (sorted by IoU desc if available)
    if "iou" in tmp.columns:
        tmp = tmp.assign(iou=pd.to_numeric(tmp["iou"], errors="coerce"))
        for bf, grp in tmp.groupby("bf_id"):
            bf1_to_gfp2_all[bf] = (grp[["gfp_id","iou"]]
                                   .sort_values("iou", ascending=False)
                                   .reset_index(drop=True))
        # best by max IoU
        idx = tmp.groupby("bf_id")["iou"].idxmax()
        for _, r in tmp.loc[idx, ["bf_id","gfp_id"]].iterrows():
            bf1_to_gfp2_best[int(r.bf_id)] = int(r.gfp_id)
    else:
        # no IoU column -> first occurrence as "best"
        for bf, grp in tmp.groupby("bf_id"):
            bf1_to_gfp2_all[bf] = grp[["gfp_id"]].assign(iou=np.nan).reset_index(drop=True)
            first = int(grp.iloc[0]["gfp_id"])
            bf1_to_gfp2_best[int(bf)] = first
else:
    raise KeyError("pairs_df2 must have columns ['bf_id','gfp_id'] (and ideally 'iou').")

# 3) Assemble main lookup table for your BF1 inputs
rows = []
for bf in bf1_list:
    g1 = bf1_to_gfp1.get(bf)
    g2 = bf1_to_gfp2_best.get(bf)
    rows.append({
        "bf1_id": int(bf),
        "gfp1_id": int(g1) if g1 is not None else np.nan,
        "gfp2_id_best": int(g2) if g2 is not None else np.nan,
        "has_gfp1": g1 is not None,
        "has_gfp2": g2 is not None
    })
bf1_map_df = pd.DataFrame(rows).sort_values("bf1_id").reset_index(drop=True)

print(f"[BF1→GFPs] {bf1_map_df['has_gfp1'].sum()}/{len(bf1_map_df)} have GFP1; "
      f"{bf1_map_df['has_gfp2'].sum()}/{len(bf1_map_df)} have GFP2 (best)")


# 5) Show a quick preview
print("\n[Preview] BF1 -> (GFP1, GFP2_best):")
print(bf1_map_df.head(15).to_string(index=False))

# 6) Save CSVs
out_main = os.path.join(WORKING_DIR, "bf1_to_gfp1_gfp2best.csv")
bf1_map_df.to_csv(out_main, index=False)
print(f"[saved] {out_main}")



#%% Build a stacked dataframe for BF1 ids (unaligned_cells) that have BOTH GFP1 and GFP2,
#    and assign new integer cell_id's for quantification
from SingleCellDataAnalysis.signal_analysis import (
    plot_simple_model_grid,  
    quantify_all_cells,
    summarize_model_distribution,
    )
import os, re
import numpy as np
import pandas as pd

def _norm_int(x):
    s = str(x).strip().split("_", 1)[0]
    m = re.match(r"^\d+", s)
    return int(m.group(0)) if m else None

# -- 0) Normalize the BF1 list
bf1_list = sorted({_norm_int(x) for x in unaligned_cells if _norm_int(x) is not None})

# -- 1) BF1 -> GFP1 by inverting `mapping` (GFP1 -> BF1)
bf1_to_gfp1 = {}
for g1, b1 in mapping.items():
    gi = _norm_int(g1); bi = _norm_int(b1)
    if gi is not None and bi is not None and bi not in bf1_to_gfp1:
        bf1_to_gfp1[bi] = gi

# -- 2) BF1 -> GFP2 (choose best IoU from pairs_df2)
bf1_to_gfp2 = {}
if {"bf_id","gfp_id"}.issubset(pairs_df2.columns):
    tmp = (pairs_df2.dropna(subset=["bf_id","gfp_id"])
                    .assign(bf_id=lambda d: d["bf_id"].map(_norm_int),
                            gfp_id=lambda d: d["gfp_id"].map(_norm_int)))
    tmp = tmp.dropna(subset=["bf_id","gfp_id"])
    if "iou" in tmp.columns:
        tmp = tmp.assign(iou=pd.to_numeric(tmp["iou"], errors="coerce"))
        idx = tmp.groupby("bf_id")["iou"].idxmax()
        best = tmp.loc[idx, ["bf_id","gfp_id"]]
    else:
        best = tmp.drop_duplicates("bf_id")[["bf_id","gfp_id"]]
    for _, r in best.iterrows():
        bf1_to_gfp2[int(r.bf_id)] = int(r.gfp_id)
else:
    raise KeyError("pairs_df2 must contain columns 'bf_id' and 'gfp_id'.")

# -- 3) Keep only BF1 ids with BOTH GFP1 and GFP2
rows_pairs = []
for bf in bf1_list:
    g1 = bf1_to_gfp1.get(bf)
    g2 = bf1_to_gfp2.get(bf)
    if g1 is not None and g2 is not None:
        rows_pairs.append({"bf1_id": int(bf), "gfp1_id": int(g1), "gfp2_id": int(g2)})
pairs_df = pd.DataFrame(rows_pairs).sort_values("bf1_id").reset_index(drop=True)

print(f"[pairs ready] {len(pairs_df)} BF1 ids with BOTH GFP1 and GFP2")

# -- 4) Prepare source frames (BASE variant) for GFP1 & GFP2
# Ensure corrected features exist (pol1_int_corr / pol2_int_corr)
def _ensure_corr_cols(df):
    if "pol1_int_corr" not in df.columns and {"pol1_int","cyt_int"}.issubset(df.columns):
        df["pol1_int_corr"] = df["pol1_int"] - df["cyt_int"]
    if "pol2_int_corr" not in df.columns and {"pol2_int","cyt_int"}.issubset(df.columns):
        df["pol2_int_corr"] = df["pol2_int"] - df["cyt_int"]
    return df

# GFP1 base only
gfp1_base = df_all_gfp_tp1.copy()
gfp1_base["gfp_id_norm"] = gfp1_base["cell_id"].map(_norm_int)
gfp1_base = gfp1_base.assign(source="GFP1", tp=1)
gfp1_base = _ensure_corr_cols(gfp1_base)

# GFP2 base only
gfp2_base = df_all_gfp_tp2.copy()
gfp2_base["gfp_id_norm"] = gfp2_base["cell_id"].map(_norm_int)
gfp2_base = gfp2_base.assign(source="GFP2", tp=2)
gfp2_base = _ensure_corr_cols(gfp2_base)

# -- 5) Build stacked dataframe with new sequential cell_id’s
stack_rows = []
map_rows   = []
new_id = 1

for _, r in pairs_df.iterrows():
    bf = int(r.bf1_id)
    g1 = int(r.gfp1_id)
    g2 = int(r.gfp2_id)

    # GFP1 slice
    d1 = gfp1_base[gfp1_base["gfp_id_norm"] == g1].copy()
    if len(d1):
        d1 = d1.rename(columns={"gfp_id_norm":"orig_gfp_id"})
        d1["cell_id"] = new_id
        d1["pair_index"] = len(map_rows) + 1
        d1["pair_bf1_id"] = bf
        d1["orig_gfp_id"] = g1
        stack_rows.append(d1[["time_point","cell_id","pol1_int_corr","pol2_int_corr","source","tp","pair_index","pair_bf1_id","orig_gfp_id"]])
        map_rows.append({"new_cell_id": new_id, "bf1_id": bf, "source": "GFP1", "orig_gfp_id": g1})
        new_id += 1

    # GFP2 slice
    d2 = gfp2_base[gfp2_base["gfp_id_norm"] == g2].copy()
    if len(d2):
        d2 = d2.rename(columns={"gfp_id_norm":"orig_gfp_id"})
        d2["cell_id"] = new_id
        d2["pair_index"] = len(map_rows) + 1  # same pair index ordering
        d2["pair_bf1_id"] = bf
        d2["orig_gfp_id"] = g2
        stack_rows.append(d2[["time_point","cell_id","pol1_int_corr","pol2_int_corr","source","tp","pair_index","pair_bf1_id","orig_gfp_id"]])
        map_rows.append({"new_cell_id": new_id, "bf1_id": bf, "source": "GFP2", "orig_gfp_id": g2})
        new_id += 1

df_unaligned_stacked = (pd.concat(stack_rows, ignore_index=True)
                          if stack_rows else
                          pd.DataFrame(columns=["time_point","cell_id","pol1_int_corr","pol2_int_corr","source","tp","pair_index","pair_bf1_id","orig_gfp_id"]))
id_map_unaligned = pd.DataFrame(map_rows)

print(f"[stacked] rows={len(df_unaligned_stacked)}, traces={id_map_unaligned['new_cell_id'].nunique()}")

# -- 6) Save artifacts
out_dir = os.path.join(WORKING_DIR, "unaligned_pairs_quant")
os.makedirs(out_dir, exist_ok=True)

map_csv  = os.path.join(out_dir, "bf1_with_gfp1_gfp2_new_ids.csv")
data_csv = os.path.join(out_dir, "stacked_gfp1_gfp2_for_unaligned_pairs.csv")
id_map_unaligned.to_csv(map_csv, index=False)
df_unaligned_stacked.to_csv(data_csv, index=False)

print(f"[saved] map -> {map_csv}")
print(f"[saved] data -> {data_csv}")

# -- 7) (Optional) run your quantifier immediately
new_ids = sorted(id_map_unaligned["new_cell_id"].unique().tolist())
print(f"[quantify] running on {len(new_ids)} new ids")
_ = quantify_all_cells(
    df_unaligned_stacked,
    new_ids,
    feature1='pol1_int_corr',
    feature2='pol2_int_corr',
    delta_threshold=4,
    filename='model_fits_unaligned_pairs.csv'
)
#%%
from SingleCellDataAnalysis.signal_cor import (
    quantify_all_cells_acor,
    )
from SingleCellDataAnalysis.clustering import (
    cluster_cells_by_amplitude_and_delay,
    )
#plot_simple_model_grid(df_unaligned_stacked, new_ids, time_points, model_type='linear', start_idx=0)
df_results = quantify_all_cells(df_unaligned_stacked, new_ids,delta_threshold=10)
periodicity_result = quantify_all_cells_acor(df_unaligned_stacked, new_ids, delta_threshold=10, visualize=False)
df_combined = pd.merge(df_results, periodicity_result, on='cell_id', how='left')
#%%
df_norm,ordered_cell_ids, row_linkage = cluster_cells_by_amplitude_and_delay(df_combined)

#%%
show_fit=False
plot_simple_model_grid(df_unaligned_stacked, ordered_cell_ids, time_points, model_type='linear', start_idx=0, show_fit=show_fit)
plot_simple_model_grid(df_unaligned_stacked, ordered_cell_ids, time_points, model_type='linear', start_idx=25, show_fit=show_fit)
plot_simple_model_grid(df_unaligned_stacked, ordered_cell_ids, time_points, model_type='linear', start_idx=50, show_fit=show_fit)
plot_simple_model_grid(df_unaligned_stacked, ordered_cell_ids, time_points, model_type='linear', start_idx=75, show_fit=show_fit)
plot_simple_model_grid(df_unaligned_stacked, ordered_cell_ids, time_points, model_type='linear', start_idx=100, show_fit=show_fit)
plot_simple_model_grid(df_unaligned_stacked, ordered_cell_ids, time_points, model_type='linear', start_idx=125, show_fit=show_fit)
plot_simple_model_grid(df_unaligned_stacked, ordered_cell_ids, time_points, model_type='linear', start_idx=150, show_fit=show_fit)
#plot_simple_model_grid(df_unaligned_stacked, ordered_cell_ids, time_points, model_type='linear', start_idx=175, show_fit=show_fit)

#%%
# === Dendrogram introspection helpers ===
from collections import defaultdict, deque
import numpy as np
import pandas as pd

def linkage_nodes_table(Z, labels):
    """
    Build a table describing every merge (node) in a SciPy linkage.
    Z: linkage matrix (shape (n-1, 4))
    labels: list/Index of row labels in the SAME order used to compute Z (df_norm.index)
    Returns: DataFrame with one row per internal node (id n..2n-2)
    """
    n = len(labels)
    Z = np.asarray(Z)
    assert Z.shape[0] == n - 1, "Linkage and label count mismatch."

    # helper to get leaf membership of a node id
    members_cache = {}
    def get_members(node_id):
        if node_id < n:   # original observation (leaf)
            return [int(node_id)]
        if node_id in members_cache:
            return members_cache[node_id]
        row = int(node_id - n)  # row in Z corresponding to this node
        left, right = int(Z[row,0]), int(Z[row,1])
        mem = get_members(left) + get_members(right)
        members_cache[node_id] = mem
        return mem

    rows = []
    for row_ix in range(n - 1):
        left = int(Z[row_ix, 0])
        right = int(Z[row_ix, 1])
        dist = float(Z[row_ix, 2])
        size = int(Z[row_ix, 3])
        node_id = n + row_ix

        left_mem = get_members(left)
        right_mem = get_members(right)
        rows.append({
            "node_id": node_id,
            "left_child": left,
            "right_child": right,
            "left_is_leaf": left < n,
            "right_is_leaf": right < n,
            "left_size": len(left_mem),
            "right_size": len(right_mem),
            "node_size": size,   # should equal left_size+right_size
            "distance": dist,
            "left_members_idx": left_mem,
            "right_members_idx": right_mem,
            "left_members_labels": [labels[i] for i in left_mem],
            "right_members_labels": [labels[i] for i in right_mem],
            "all_members_labels": [labels[i] for i in (left_mem + right_mem)],
        })
    tbl = pd.DataFrame(rows).sort_values("distance").reset_index(drop=True)
    return tbl

def describe_node(tbl, node_id):
    """
    Pretty-print how a specific node is split into left/right branches.
    """
    hit = tbl.loc[tbl["node_id"] == node_id]
    if hit.empty:
        print(f"Node {node_id} not found.")
        return
    r = hit.iloc[0]
    print(f"Node {r.node_id}  (size={r.node_size}, distance={r.distance:.3f})")
    print(f"  Left  child: {r.left_child}  (leaf={bool(r.left_is_leaf)}, size={r.left_size})")
    print(f"    members: {r.left_members_labels}")
    print(f"  Right child: {r.right_child} (leaf={bool(r.right_is_leaf)}, size={r.right_size})")
    print(f"    members: {r.right_members_labels}")

# ---- build the node table using your existing artifacts ----
labels = df_norm.index.to_list()           # these are your cell_id’s
nodes_tbl = linkage_nodes_table(row_linkage, labels)

# Peek at the largest merges (by distance)
print("\nTop 10 merges by distance:")
print(nodes_tbl.sort_values("distance", ascending=False)
               .head(10)[["node_id","distance","node_size","left_size","right_size"]])

# Example: describe a specific node (replace  e.g.  with the top merge’s node_id)
if len(nodes_tbl):
    top_node_id = nodes_tbl.sort_values("distance", ascending=False).iloc[0]["node_id"]
    print("\n=== Details for the top merge ===")
    describe_node(nodes_tbl, int(top_node_id))

# If you want a CSV of *every* node with memberships:
nodes_csv = os.path.join(WORKING_DIR, "clustering_nodes_memberships.csv")
nodes_tbl.to_csv(nodes_csv, index=False)
print(f"[saved] node breakdown -> {nodes_csv}")

#%%
from scipy.cluster.hierarchy import fcluster

# Cut by number of clusters k
k = 10
cluster_ids = fcluster(row_linkage, t=k, criterion="maxclust")
clusters = pd.Series(cluster_ids, index=labels, name="cluster")
print("\nMembers by cluster (k=4):")
for cid, grp in clusters.groupby(clusters):
    print(f"Cluster {cid} (n={len(grp)}): {list(grp.index)}")

# Or cut by distance threshold (height)
thr = nodes_tbl["distance"].median()  # example threshold
cluster_ids_thr = fcluster(row_linkage, t=thr, criterion="distance")

#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

def build_links_from_id_map(id_map_unaligned):
    """
    Return list of (new_id_gfp1, new_id_gfp2) for bf1_ids that have both.
    Requires columns: ['new_cell_id','bf1_id','source'] where source in {'GFP1','GFP2'}.
    """
    links = []
    if not {"new_cell_id","bf1_id","source"}.issubset(id_map_unaligned.columns):
        raise KeyError("id_map_unaligned must have columns: new_cell_id, bf1_id, source")
    g = id_map_unaligned.groupby("bf1_id")
    for bf1, sub in g:
        # find exactly one GFP1 and one GFP2
        s = sub.set_index("source")
        if all(k in s.index for k in ("GFP1","GFP2")):
            g1 = int(s.loc["GFP1","new_cell_id"])
            g2 = int(s.loc["GFP2","new_cell_id"])
            links.append((g1, g2))
    return links



def plot_circular_links_ordered(ordered_cell_ids, links, out_pdf,
                                r=1.0, text_r=1.08, lw=2.5, alpha=0.9,
                                label_prefix=None, tip_fs=8,
                                color_cycle=None):
    """
    Place ordered_cell_ids around a circle in the given order.
    Draw cubic Bézier curves between pairs in `links` (tuples of (idA, idB)).
    Saves as vector PDF (PDF/SVG).
    """
    # Normalize types (avoid str vs int mismatches)
    try:
        ordered_cell_ids = [int(x) for x in ordered_cell_ids]
        links = [(int(a), int(b)) for a, b in links]
    except Exception:
        pass

    # Keep only links whose endpoints are in the label set
    label_set = set(ordered_cell_ids)
    links_kept = [(a, b) for (a, b) in links if a in label_set and b in label_set and a != b]
    print(f"[links] input={len(links)} kept={len(links_kept)} (filtered by ordered_cell_ids)")
    if not links_kept:
        print("[warn] no valid links to draw (ids not found in ordered_cell_ids).")

    # Angles in the given order
    n = len(ordered_cell_ids)
    theta = {cid: 2*np.pi * (k / n) for k, cid in enumerate(ordered_cell_ids)}
    xy    = {cid: (r*np.cos(theta[cid]), r*np.sin(theta[cid])) for cid in ordered_cell_ids}

    if color_cycle is None:
        # default matplotlib cycle
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['k'])

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    # Faint guide circle
    ax.add_artist(plt.Circle((0, 0), r, fill=False, lw=0.5, alpha=0.3, ec='0.5'))

    # Tip labels
    for cid in ordered_cell_ids:
        th = theta[cid]
        label = f"{label_prefix}{cid}" if label_prefix else f"{cid}"
        ax.text(text_r*np.cos(th), text_r*np.sin(th), label,
                ha='center', va='center', fontsize=tip_fs)

    # Cubic Bézier helper (pull control points toward the center)
    def bezier_path(p, q, bend=0.30):
        px, py = p; qx, qy = q
        c1 = (px*(1-bend), py*(1-bend))
        c2 = (qx*(1-bend), qy*(1-bend))
        verts = [p, c1, c2, q]
        codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
        return Path(verts, codes)

    # Draw links
    for i, (a, b) in enumerate(links_kept):
        p, q = xy[a], xy[b]
        color = color_cycle[i % len(color_cycle)]
        try:
            patch = PathPatch(
                bezier_path(p, q, bend=0.30),
                lw=lw,
                facecolor='none',
                edgecolor=color,      # force visible edge
                alpha=alpha,
                zorder=1,
                clip_on=False
            )
            ax.add_patch(patch)
        except Exception as e:
            # Fallback to straight line if PathPatch fails
            ax.plot([p[0], q[0]], [p[1], q[1]], lw=lw, alpha=alpha, color=color, zorder=1)

    ax.set_xlim(-text_r-0.15, text_r+0.15)
    ax.set_ylim(-text_r-0.15, text_r+0.15)

    fig.savefig(out_pdf, bbox_inches='tight')  # vector
    plt.close(fig)
    print(f"[saved] circular links -> {out_pdf}")


# --- Build links from your stacked-map, then plot with your clustering order ---
links_unaligned = build_links_from_id_map(id_map_unaligned)

circ_links_pdf = os.path.join(WORKING_DIR, "circular_links_ordered_ids.pdf")
plot_circular_links_ordered(
    ordered_cell_ids=ordered_cell_ids,
    links=links_unaligned,
    out_pdf=circ_links_pdf,
    label_prefix="",     # or "ID "
)


#%% For each linked pair, collect ALL nodes whose cluster contains both leaves
def pair_nodes_long(links, nodes_tbl, valid_labels=None):
    """
    Return a long-form DataFrame with one row per (pair, node) for every internal node
    whose all_members contains BOTH leaves of the pair.

    Columns:
      gfp1_new_id, gfp2_new_id
      node_id, distance, node_size
      left_size, right_size
      status_at_node: {'split','same_left','same_right'}
         - 'split'      : the pair sits on different children at this node (this is the LCA)
         - 'same_left'  : both are already inside the left subtree
         - 'same_right' : both are already inside the right subtree
    """
    if valid_labels is None and len(nodes_tbl):
        valid_labels = set().union(*nodes_tbl["all_members_labels"])
    rows = []
    for a, b in links:
        a, b = int(a), int(b)
        if a == b or a not in valid_labels or b not in valid_labels:
            continue

        # nodes that contain both
        hits = nodes_tbl[nodes_tbl["all_members_labels"].apply(lambda L: (a in L) and (b in L))].copy()
        if hits.empty:
            continue

        # annotate split/same-left/same-right
        def _status(r):
            lm, rm = set(r["left_members_labels"]), set(r["right_members_labels"])
            in_left_a, in_left_b = a in lm, b in lm
            in_right_a, in_right_b = a in rm, b in rm
            if (in_left_a and in_left_b):
                return "same_left"
            if (in_right_a and in_right_b):
                return "same_right"
            return "split"  # one in left, one in right → this node is the LCA

        hits["status_at_node"] = hits.apply(_status, axis=1)
        hits = hits.assign(
            gfp1_new_id=a,
            gfp2_new_id=b
        )[[
            "gfp1_new_id","gfp2_new_id",
            "node_id","distance","node_size",
            "left_size","right_size","status_at_node"
        ]].sort_values("distance")  # ascending: LCA appears first
        rows.append(hits)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["gfp1_new_id","gfp2_new_id","node_id","distance","node_size",
                 "left_size","right_size","status_at_node"]
    )

pair_nodes_df = pair_nodes_long(links_unaligned, nodes_tbl, valid_labels=set(labels))
print(f"[pair×node] rows={len(pair_nodes_df)}, pairs={pair_nodes_df[['gfp1_new_id','gfp2_new_id']].drop_duplicates().shape[0]}")
print(pair_nodes_df.head().to_string(index=False))

#%% Wide view: compress to list of node_ids per pair (ordered by distance asc)
def pair_nodes_wide(pair_nodes_df):
    agg = (pair_nodes_df
           .sort_values(["gfp1_new_id","gfp2_new_id","distance"])
           .groupby(["gfp1_new_id","gfp2_new_id"])
           .agg(
               node_ids=("node_id", lambda s: list(map(int, s.tolist()))),
               node_distances=("distance", lambda s: list(map(float, s.tolist()))),
               first_together_node=("node_id", "first"),           # the LCA
               first_together_distance=("distance", "first"),
               n_nodes_together=("node_id", "size")
           )
           .reset_index())
    return agg

pair_nodes_wide_df = pair_nodes_wide(pair_nodes_df)
print("\n[pair→nodes summary]")
print(pair_nodes_wide_df.head().to_string(index=False))

#%% Cluster sharing summary: how many pairs share each node/cluster?
node_share = (pair_nodes_df
              .groupby("node_id")
              .agg(
                  n_pairs=("gfp1_new_id", "count"),
                  distance=("distance","first"),
                  node_size=("node_size","first")
               )
              .reset_index()
              .sort_values(["n_pairs","distance"], ascending=[False, True]))

print("\n[nodes ranked by how many linked pairs they contain]")
print(node_share.head(15).to_string(index=False))

#%% Save artifacts
out_dir = os.path.join(WORKING_DIR, "unaligned_pairs_quant")
os.makedirs(out_dir, exist_ok=True)

pair_nodes_long_csv = os.path.join(out_dir, "linked_pairs_all_nodes_long.csv")
pair_nodes_df.to_csv(pair_nodes_long_csv, index=False)

pair_nodes_wide_csv = os.path.join(out_dir, "linked_pairs_all_nodes_wide.csv")
pair_nodes_wide_df.to_csv(pair_nodes_wide_csv, index=False)

node_share_csv = os.path.join(out_dir, "node_pair_sharing_summary.csv")
node_share.to_csv(node_share_csv, index=False)

print(f"[saved] long  -> {pair_nodes_long_csv}")
print(f"[saved] wide  -> {pair_nodes_wide_csv}")
print(f"[saved] share -> {node_share_csv}")


#%% Classify pairs by clade membership within a chosen set of node_ids (each node = a clade)
node_ids = {293, 303, 306, 307, 308}

# Build membership maps for just the clades of interest
clades = (nodes_tbl[nodes_tbl["node_id"].isin(node_ids)]
          [["node_id","node_size","all_members_labels"]].copy())
clade_members = {int(r.node_id): set(map(int, r.all_members_labels)) for _, r in clades.iterrows()}
clade_sizes   = {int(r.node_id): int(r.node_size) for _, r in clades.iterrows()}

def assign_most_specific_clade(leaf_id):
    """
    Return the node_id (from node_ids) of the smallest clade that contains leaf_id.
    If none contain it, return None.
    """
    hits = [nid for nid, mem in clade_members.items() if leaf_id in mem]
    if not hits:
        return None
    # choose the most specific (smallest node_size)
    return min(hits, key=lambda nid: clade_sizes[nid])

rows = []
for a, b in links_unaligned:
    a = int(a); b = int(b)
    ca = assign_most_specific_clade(a)
    cb = assign_most_specific_clade(b)

    if ca is None and cb is None:
        status = "none_in_set"      # neither leaf falls inside any of the provided clades
    elif (ca is None) ^ (cb is None):
        status = "one_in_set"       # only one leaf falls inside some clade in the set
    elif ca == cb:
        status = "same_clade"       # both leaves assigned to the same clade (same node_id)
    else:
        status = "different_clades" # both assigned, but to different clades in the set

    rows.append({"gfp1_new_id": a, "gfp2_new_id": b, "clade_a": ca, "clade_b": cb, "status": status})

pairs_clade_df = pd.DataFrame(rows)

# Summary counts
summary_counts = pairs_clade_df["status"].value_counts().to_dict()
summary_counts.setdefault("same_clade", 0)
summary_counts.setdefault("different_clades", 0)
summary_counts.setdefault("one_in_set", 0)
summary_counts.setdefault("none_in_set", 0)

print("[clade assignment across selected nodes]")
for k in ["same_clade","different_clades","one_in_set","none_in_set"]:
    print(f"  {k}: {summary_counts[k]}")

# (Optional) detailed views
pairs_same      = pairs_clade_df[pairs_clade_df["status"]=="same_clade"].copy()
pairs_different = pairs_clade_df[pairs_clade_df["status"]=="different_clades"].copy()

# Save
out_dir = os.path.join(WORKING_DIR, "unaligned_pairs_quant")
os.makedirs(out_dir, exist_ok=True)
pairs_clade_df.to_csv(os.path.join(out_dir, "pairs_clade_assignment_selected_nodes.csv"), index=False)
print(f"[saved] per-pair clade assignment -> {os.path.join(out_dir, 'pairs_clade_assignment_selected_nodes.csv')}")

#%% Transition heatmap among selected clades
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Your selected clades:
node_ids = {293,303, 306, 307, 308}

# --- If pairs_clade_df isn't built yet (from previous step), build it here ---
def _ensure_pairs_clade_df(links_unaligned, nodes_tbl, node_ids):
    clades = (nodes_tbl[nodes_tbl["node_id"].isin(node_ids)]
              [["node_id","node_size","all_members_labels"]].copy())
    clade_members = {int(r.node_id): set(map(int, r.all_members_labels)) for _, r in clades.iterrows()}
    clade_sizes   = {int(r.node_id): int(r.node_size) for _, r in clades.iterrows()}

    def assign_most_specific_clade(leaf_id):
        hits = [nid for nid, mem in clade_members.items() if leaf_id in mem]
        if not hits:
            return None
        return min(hits, key=lambda nid: clade_sizes[nid])

    rows = []
    for a, b in links_unaligned:
        a = int(a); b = int(b)
        ca = assign_most_specific_clade(a)
        cb = assign_most_specific_clade(b)
        if   ca is None and cb is None: status = "none_in_set"
        elif (ca is None) ^ (cb is None): status = "one_in_set"
        elif ca == cb: status = "same_clade"
        else: status = "different_clades"
        rows.append({"gfp1_new_id": a, "gfp2_new_id": b, "clade_a": ca, "clade_b": cb, "status": status})
    return pd.DataFrame(rows)

if "pairs_clade_df" not in globals():
    pairs_clade_df = _ensure_pairs_clade_df(links_unaligned, nodes_tbl, node_ids)

# Keep pairs where both leaves are assigned to one of the chosen clades
pairs_both = pairs_clade_df.dropna(subset=["clade_a","clade_b"]).copy()
pairs_both["clade_a"] = pairs_both["clade_a"].astype(int)
pairs_both["clade_b"] = pairs_both["clade_b"].astype(int)

clade_list = sorted(node_ids)
idx = pd.Index(clade_list, name="from_clade")
cols = pd.Index(clade_list, name="to_clade")

# --- Directed transition matrix (includes diagonal = same-clade pairs) ---
M = pd.DataFrame(0, index=idx, columns=cols, dtype=int)
for _, r in pairs_both.iterrows():
    if (r.clade_a in node_ids) and (r.clade_b in node_ids):
        M.loc[r.clade_a, r.clade_b] += 1

# --- Undirected transition matrix (collapse A→B and B→A into the same bin) ---
U = pd.DataFrame(0, index=idx, columns=cols, dtype=int)
for _, r in pairs_both.iterrows():
    a, b = int(r.clade_a), int(r.clade_b)
    if (a in node_ids) and (b in node_ids):
        i, j = (a, b) if a <= b else (b, a)
        U.loc[i, j] += 1  # upper-tri (including diagonal)

# --- Save CSVs ---
out_dir = os.path.join(WORKING_DIR, "unaligned_pairs_quant")
os.makedirs(out_dir, exist_ok=True)
M.to_csv(os.path.join(out_dir, "transition_matrix_directed.csv"))
U.to_csv(os.path.join(out_dir, "transition_matrix_undirected_upper.csv"))
print(f"[saved] directed -> {os.path.join(out_dir, 'transition_matrix_directed.csv')}")
print(f"[saved] undirected (upper-tri) -> {os.path.join(out_dir, 'transition_matrix_undirected_upper.csv')}")

# --- Plot heatmaps (plain matplotlib, single-plot each, no explicit colors) ---
def plot_heatmap(mat_df, title, out_png):
    import numpy as np
    import matplotlib.pyplot as plt

    data = mat_df.values
    nrows, ncols = data.shape

    fig, ax = plt.subplots(figsize=(8, 6))

    # Use CENTER coordinates for shading="nearest" (length = ncols / nrows)
    X = np.arange(ncols)
    Y = np.arange(nrows)
    mesh = ax.pcolormesh(X, Y, data, shading="nearest", edgecolors="none")

    # Make cells square and put row 0 at the top
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, ncols - 0.5)
    ax.set_ylim(nrows - 0.5, -0.5)

    ax.set_title(title)
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels(mat_df.columns.astype(str), rotation=45, ha="right")
    ax.set_yticklabels(mat_df.index.astype(str))

    # Centered counts in WHITE at integer centers (j, i)
    for i in range(nrows):
        for j in range(ncols):
            ax.text(j, i, int(data[i, j]), ha="center", va="center", color="white")

    fig.colorbar(mesh, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] heatmap -> {out_png}")


plot_heatmap(M, "Clade Transitions (Directed)", os.path.join(out_dir, "heatmap_transitions_directed.png"))
plot_heatmap(U, "Clade Transitions (Undirected, upper-tri)", os.path.join(out_dir, "heatmap_transitions_undirected.png"))

# --- Quick textual summary ---
total_pairs = pairs_both.shape[0]
same_count = int(np.trace(M.values))
diff_count = int(M.values.sum() - same_count)
print(f"[summary] total considered={total_pairs}, same_clade={same_count}, different_clades={diff_count}")
#%% chi square
from scipy.stats import chi2_contingency
import numpy as np, pandas as pd

def chi2_on_nonzero_marginals(M):
    # Identify rows/cols with nonzero totals
    row_mask = (M.sum(axis=1) > 0)
    col_mask = (M.sum(axis=0) > 0)

    M_reduced = M.loc[row_mask, col_mask]
    if M_reduced.size == 0 or (M_reduced.shape[0] < 2 or M_reduced.shape[1] < 2):
        raise ValueError("Not enough nonzero rows/cols for chi-square.")

    obs = M_reduced.values
    chi2, p, dof, expected = chi2_contingency(obs)  # no Yates correction in RxC
    print(f"[Chi-square] chi2={chi2:.3f}, dof={dof}, p={p:.4g}")
    exp_df = pd.DataFrame(expected, index=M_reduced.index, columns=M_reduced.columns)

    # Pearson and adjusted residuals (z)
    pearson = (obs - expected) / np.sqrt(expected)
    N = obs.sum()
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    adj_denom = np.sqrt(expected * (1 - row_sums/N) * (1 - col_sums/N))
    adj = (obs - expected) / adj_denom

    pearson_df = pd.DataFrame(pearson, index=M_reduced.index, columns=M_reduced.columns)
    adj_df     = pd.DataFrame(adj,     index=M_reduced.index, columns=M_reduced.columns)

    # Reinsert dropped rows/cols as NaN (optional)
    pearson_full = pd.DataFrame(np.nan, index=M.index, columns=M.columns).combine_first(pearson_df)
    adj_full     = pd.DataFrame(np.nan, index=M.index, columns=M.columns).combine_first(adj_df)

    return chi2, p, dof, exp_df, pearson_full, adj_full

chi2, p, dof, expected_df, pearson_resid_df, adj_resid_df = chi2_on_nonzero_marginals(M)
print("[Adjusted residuals (z)]:\n", adj_resid_df.round(2))

#%% symmetry
from scipy.stats import chi2
import numpy as np
import pandas as pd

def prepare_square_nonzero(M: pd.DataFrame) -> pd.DataFrame:
    """Return a square submatrix of M with the same labels on rows/cols,
    after dropping clades with zero row AND zero column totals."""
    row_pos = (M.sum(axis=1) > 0)
    col_pos = (M.sum(axis=0) > 0)
    # keep labels that appear in either (row or col has mass)
    keep = sorted(set(M.index[row_pos]).union(set(M.columns[col_pos])))
    R = M.loc[keep, keep].copy()  # same labels, same order → square
    # if any row/col is still all zero, drop them (rare but safe)
    nz = (R.sum(0) > 0) & (R.sum(1) > 0)
    R = R.loc[nz.index[nz], nz.index[nz]]
    return R

def bowker_symmetry_test_square(Msq: pd.DataFrame):
    """Bowker’s test on a square contingency table (off-diagonal symmetry)."""
    A = Msq.values
    k = A.shape[0]
    if k < 2:
        raise ValueError("Matrix too small for Bowker (need ≥2 categories).")
    stat = 0.0
    df = 0
    for i in range(k):
        for j in range(i+1, k):
            nij, nji = A[i, j], A[j, i]
            s = nij + nji
            if s > 0:
                stat += (nij - nji)**2 / s
                df += 1
    p = 1 - chi2.cdf(stat, df)
    return stat, df, p

# --- use it ---
M_sq = prepare_square_nonzero(M)
print(f"[prep] using clades: {list(M_sq.index)}; shape={M_sq.shape}")

stat, df, p = bowker_symmetry_test_square(M_sq)
print(f"[Bowker symmetry] X2={stat:.3f}, df={df}, p={p:.4g}")




#%% flag FDR
from statsmodels.stats.multitest import multipletests

z = adj_resid_df.values.ravel()
# two-sided p-values from z
p_cells = 2 * (1 - (0.5 * (1 + np.math.erf(np.abs(z)/np.sqrt(2)))))
rej, p_fdr, _, _ = multipletests(p_cells, alpha=0.05, method="fdr_bh")
sig_mask = rej.reshape(adj_resid_df.shape)
sig_df = pd.DataFrame(sig_mask, index=M.index, columns=M.columns)
print("[Cells significant after FDR] True=significant at q<0.05")
print(sig_df)



#%% ===== Paired per-cell plots (RAW, unaligned) for BF1 pairs with BOTH GFP1 & GFP2 =====
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Expect these to already exist from your “unaligned_pairs” prep:
# - pairs_df: columns ['bf1_id','gfp1_id','gfp2_id']  (only BF1 ids with BOTH GFP1 and GFP2)
# - gfp1_base: raw GFP1 (base) table with columns: ['cell_id','time_point','pol1_int','pol2_int','cyt_int', ...]
# - gfp2_base: raw GFP2 (base) table with same columns

def _ensure_corr_cols(df):
    df = df.copy()
    if "pol1_int_corr" not in df.columns and {"pol1_int","cyt_int"}.issubset(df.columns):
        df["pol1_int_corr"] = df["pol1_int"] - df["cyt_int"]
    if "pol2_int_corr" not in df.columns and {"pol2_int","cyt_int"}.issubset(df.columns):
        df["pol2_int_corr"] = df["pol2_int"] - df["cyt_int"]
    return df

gfp1_raw = _ensure_corr_cols(gfp1_base)
gfp2_raw = _ensure_corr_cols(gfp2_base)

# sanity check
need_cols = {"time_point","cell_id","pol1_int_corr","pol2_int_corr"}
for name, dfcheck in [("gfp1_raw", gfp1_raw), ("gfp2_raw", gfp2_raw)]:
    missing = need_cols - set(dfcheck.columns)
    if missing:
        raise KeyError(f"{name} missing columns: {sorted(missing)}")

# Output folder
out_dir = os.path.join(WORKING_DIR, "paired_plots_GFP1_GFP2_RAW_unaligned_PDF")
os.makedirs(out_dir, exist_ok=True)

not_found = []
made = 0

for _, row in pairs_df.iterrows():
    bf1 = int(row["bf1_id"])
    g1  = int(row["gfp1_id"])
    g2  = int(row["gfp2_id"])

    # RAW slices (no alignment)
    d1 = gfp1_raw[gfp1_raw["cell_id"].astype(int) == g1].sort_values("time_point")
    d2 = gfp2_raw[gfp2_raw["cell_id"].astype(int) == g2].sort_values("time_point")

    if d1.empty or d2.empty:
        not_found.append((bf1, g1, g2, "missing_gfp1" if d1.empty else "missing_gfp2"))
        continue

    # Shared y-lims across the pair
    yvals = np.concatenate([
        d1["pol1_int_corr"].astype(float).values,
        d1["pol2_int_corr"].astype(float).values,
        d2["pol1_int_corr"].astype(float).values,
        d2["pol2_int_corr"].astype(float).values
    ])
    y_min, y_max = np.nanmin(yvals), np.nanmax(yvals)
    span = (y_max - y_min) if np.isfinite(y_max - y_min) and (y_max - y_min) > 0 else 1.0
    y_pad = 0.05 * span
    ylims = (y_min - y_pad, y_max + y_pad)

    # X-lims = exact data span for each subplot
    x1_min, x1_max = np.nanmin(d1["time_point"]), np.nanmax(d1["time_point"])
    x2_min, x2_max = np.nanmin(d2["time_point"]), np.nanmax(d2["time_point"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    ax1, ax2 = axes

    # --- Left: GFP1 (RAW) ---
    ax1.plot(d1["time_point"], d1["pol1_int_corr"], label="pol1", linewidth=1.5)
    ax1.plot(d1["time_point"], d1["pol2_int_corr"], label="pol2", linewidth=1.5, linestyle="--")
    ax1.set_xlim(x1_min, x1_max)
    ax1.set_title(f"GFP1 {g1}  (BF1 {bf1})")
    ax1.set_xlabel("Frame (raw)")
    ax1.set_ylabel("Intensity (corrected)")
    ax1.set_ylim(*ylims)
    ax1.grid(True, alpha=0.3)
    ax1.legend(frameon=False, loc="best")

    # --- Right: GFP2 (RAW) ---
    ax2.plot(d2["time_point"], d2["pol1_int_corr"], label="pol1", linewidth=1.5)
    ax2.plot(d2["time_point"], d2["pol2_int_corr"], label="pol2", linewidth=1.5, linestyle="--")
    ax2.set_xlim(x2_min, x2_max)
    ax2.set_title(f"GFP2 {g2}  (mapped from BF1 {bf1})")
    ax2.set_xlabel("Frame (raw)")
    ax2.set_ylim(*ylims)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("pol1 & pol2 vs raw time (no alignment)", y=1.02, fontsize=12)
    fig.tight_layout()

    # Save as vector PDF
    fname = f"RAW_pair_BF1_{bf1}__GFP1_{g1}__GFP2_{g2}.pdf"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, dpi=300, bbox_inches="tight", format="pdf")
    plt.close(fig)
    made += 1

print(f"[ok] Wrote {made} paired RAW PDF figures to: {out_dir}")
if not_found:
    print("[warn] Skipped due to missing raw data:", not_found)

