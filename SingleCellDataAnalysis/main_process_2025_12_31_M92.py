#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 12:05:22 2026

@author: user
"""

import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')



import os
import pandas as pd
import numpy as np

WORKING_DIR = "/Volumes/Movies/2025_12_31_M92/"
# Use BF images for manual alignment

print("📥 Loading data...")
#df_all = load_and_merge_csv(FILE_NAMES, WORKING_DIR)
#%%
from SingleCellDataAnalysis.alignment_board_gui import review_septum_alignment_board_gui

#BF1_DIR_F0 = os.path.join(WORKING_DIR, "A14-YES-1t-FBFBF-2_F0")
FILM_NAME = "A14-YES-1t-FBFBF-2_F0"
review_septum_alignment_board_gui(WORKING_DIR, FILM_NAME, mask_col="rle_gfp", n_rows=12, n_cols=25, tile_size=96)
#%%
FILM_NAME = "A14-YES-1t-FBFBF-2_F1"
review_septum_alignment_board_gui(WORKING_DIR, FILM_NAME, mask_col="rle_gfp", n_rows=12, n_cols=25, tile_size=96)
#%%
FILM_NAME = "A14-YES-1t-FBFBF-2_F2"
review_septum_alignment_board_gui(WORKING_DIR, FILM_NAME, mask_col="rle_gfp", n_rows=12, n_cols=25, tile_size=96)
#%%
FILM_NAME = "A14-YES-1t-FBFBF-4_F0"
review_septum_alignment_board_gui(WORKING_DIR, FILM_NAME, mask_col="rle_gfp", n_rows=24, n_cols=50, tile_size=48)
#%%
FILM_NAME = "A14-YES-1t-FBFBF-4_F1"
review_septum_alignment_board_gui(WORKING_DIR, FILM_NAME, mask_col="rle_gfp", n_rows=24, n_cols=50, tile_size=48)
#%%
FILM_NAME = "A14-YES-1t-FBFBF-4_F2"
review_septum_alignment_board_gui(WORKING_DIR, FILM_NAME, mask_col="rle_gfp", n_rows=24, n_cols=50, tile_size=48)
#%% Pairing cell IDs between BF and GFP movies
from SingleCellDataAnalysis.multi_field import run_field_sequence
from SingleCellDataAnalysis.population_movie_gui import build_global_id_maps_from_pairings

def make_field_sequence(field: str):
    # field should be "F0" / "F1" / "F2"
    return [
        ("gfp", f"A14-YES-1t-FBFBF_{field}"),
        ("bf",  f"A14-YES-1t-FBFBF-2_{field}"),
        ("gfp", f"A14-YES-1t-FBFBF-3_{field}"),
        ("bf",  f"A14-YES-1t-FBFBF-4_{field}"),
        ("gfp", f"A14-YES-1t-FBFBF-5_{field}"),
    ]

FIELDS = ["F0", "F1", "F2"]
all_res = {}
all_global_maps = {}

for field in FIELDS:
    out_dir = os.path.join(WORKING_DIR, "pipeline_outputs", f"{field}_maps_and_aligned")
    field_seq = make_field_sequence(field)

    res = run_field_sequence(
        WORKING_DIR,
        field_seq,
        out_dir,
        iou_min=0.01,
        #bf_rle_col="rle_gfp",   # your BF mask column is mislabeled as rle_gfp
    )
    all_res[field] = res
    print(f"\n==== {field} done; {len(res)} pair(s) ====")
    for r in res:
        print(r.get("pair"), r.get("aligned_csv", "no aligned_csv"))

    field_seq = make_field_sequence(field)
    pair_mappings = all_res[field]                      # <-- use the right one
    anchor_film = f"A14-YES-1t-FBFBF-2_{field}"         # <-- BF1 for that field

    global_maps = build_global_id_maps_from_pairings(
        field_seq=field_seq,
        pair_mappings=pair_mappings,
        anchor_film=anchor_film,
    )

    all_global_maps[field] = global_maps
    print(f"[global-id] {field}: built maps for {len(global_maps)} films (anchor={anchor_film})")

#%% make 15-frame-summary movies
from SingleCellDataAnalysis.population_mp4_review_gui import make_15frame_summary_mp4

for field in FIELDS:
    field_seq = make_field_sequence(field)         # your existing helper
    films = [film for _, film in field_seq]        # 5 films in order
    in_mp4 = os.path.join(WORKING_DIR,"population_movies")
    out_mp4 = os.path.join(WORKING_DIR, "pipeline_outputs", "combined_movies",
                           f"{field}__15frame_summary.mp4")
    
    make_15frame_summary_mp4(
        working_dir=in_mp4,
        films=films,
        out_mp4=out_mp4,
        seconds_per_frame=2.0,
        fps_out=10.0,          # 2 sec => 20 frames per still
        resize_to_first=True,
        force=True,            # set False once you’re happy
    )
#%% QC for F0
from SingleCellDataAnalysis.population_summary15_review_gui import review_summary15_by_global_id

field = "F0"
field_seq = make_field_sequence(field)

qc_out = review_summary15_by_global_id(
    working_dir=WORKING_DIR,
    field_seq=field_seq,
    global_maps_by_film=all_global_maps[field],
    field_label=field,
    # optional if you used different names:
    summary_mp4=f"{WORKING_DIR}/pipeline_outputs/combined_movies/{field}__15frame_summary.mp4",
    summary_index_csv=f"{WORKING_DIR}/pipeline_outputs/combined_movies/{field}__15frame_summary.index.csv",
    block=False,
)

#%% QC for F1
from SingleCellDataAnalysis.population_summary15_review_gui import review_summary15_by_global_id

field = "F1"
field_seq = make_field_sequence(field)

qc_out = review_summary15_by_global_id(
    working_dir=WORKING_DIR,
    field_seq=field_seq,
    global_maps_by_film=all_global_maps[field],
    field_label=field,
    # optional if you used different names:
    summary_mp4=f"{WORKING_DIR}/pipeline_outputs/combined_movies/{field}__15frame_summary.mp4",
    summary_index_csv=f"{WORKING_DIR}/pipeline_outputs/combined_movies/{field}__15frame_summary.index.csv",
    block=False,
)
#%% QC for F2
from SingleCellDataAnalysis.population_summary15_review_gui import review_summary15_by_global_id

field = "F2"
field_seq = make_field_sequence(field)

qc_out = review_summary15_by_global_id(
    working_dir=WORKING_DIR,
    field_seq=field_seq,
    global_maps_by_film=all_global_maps[field],
    field_label=field,
    # optional if you used different names:
    summary_mp4=f"{WORKING_DIR}/pipeline_outputs/combined_movies/{field}__15frame_summary.mp4",
    summary_index_csv=f"{WORKING_DIR}/pipeline_outputs/combined_movies/{field}__15frame_summary.index.csv",
    block=False,
)



WORKING_DIR = "/Volumes/Movies/2025_12_31_M92/"

from SingleCellDataAnalysis.multi_field_data_analysis import (
    load_good_global_ids,
    load_good_aligned_gfp,
    get_aligned_csv,
)

from SingleCellDataAnalysis.visualization import plot_aligned_signals, plot_aligned_heatmaps


# # -------------------------
# # 1) Define field sequence
# # -------------------------
# def make_field_sequence(field: str):
#     # field should be "F0" / "F1" / "F2"
#     return [
#         ("gfp", f"A14-YES-1t-FBFBF_{field}"),
#         ("bf",  f"A14-YES-1t-FBFBF-2_{field}"),  # BF1 (anchor)
#         ("gfp", f"A14-YES-1t-FBFBF-3_{field}"),  # GFP2
#         ("bf",  f"A14-YES-1t-FBFBF-4_{field}"),
#         ("gfp", f"A14-YES-1t-FBFBF-5_{field}"),
#     ]


# FIELDS = ["F0", "F1", "F2"]


# # -------------------------
# # 2) Run pairing per field
# # -------------------------
# all_res = {}         # field -> list of map_out dicts from run_field_sequence
# all_global_maps = {} # field -> film -> {local_id -> global_id}

# for field in FIELDS:
#     out_dir = os.path.join(WORKING_DIR, "pipeline_outputs", f"{field}_maps_and_aligned")
#     field_seq = make_field_sequence(field)

#     res = run_field_sequence(
#         WORKING_DIR,
#         field_seq,
#         out_dir,
#         iou_min=0.01,
#         # bf_rle_col="rle_gfp",  # enable ONLY if BF masks are mislabeled in your masks csv
#     )
#     all_res[field] = res

#     print(f"\n==== {field} done; {len(res)} pair(s) ====")
#     for r in res:
#         print(r.get("pair"), r.get("aligned_csv", "no aligned_csv"))

#     # Build global id maps (anchor = BF1 for that field)
#     anchor_film = f"A14-YES-1t-FBFBF-2_{field}"
#     global_maps = build_global_id_maps_from_pairings(
#         field_seq=field_seq,
#         pair_mappings=res,
#         anchor_film=anchor_film,
#     )
#     all_global_maps[field] = global_maps
#     print(f"[global-id] {field}: built maps for {len(global_maps)} films (anchor={anchor_film})")


# -------------------------------------------------
# 3) Load GOOD aligned GFP1 & GFP2 per field + pool
# -------------------------------------------------
def load_good_gfp1_gfp2_for_field(field: str):
    """
    Returns (df_gfp1, df_gfp2) for this field:
      - cell_id is still the field-local global_id at this point
      - time_point already aligned (whatever your helper outputs)
    """
    gfp1 = f"A14-YES-1t-FBFBF_{field}"
    bf1  = f"A14-YES-1t-FBFBF-2_{field}"
    gfp2 = f"A14-YES-1t-FBFBF-3_{field}"

    good_gids = load_good_global_ids(WORKING_DIR, field)

    global_maps = all_global_maps[field]   # film -> {local -> global}
    pair_maps   = all_res[field]           # run_field_sequence outputs (list of dicts)

    aligned_gfp1_csv = get_aligned_csv(pair_maps, gfp1, bf1)
    aligned_gfp2_csv = get_aligned_csv(pair_maps, gfp2, bf1)

    df1 = load_good_aligned_gfp(global_maps, good_gids, aligned_gfp1_csv, gfp1)
    df2 = load_good_aligned_gfp(global_maps, good_gids, aligned_gfp2_csv, gfp2)

    # tag metadata
    for df, which in ((df1, "GFP1"), (df2, "GFP2")):
        df["field"] = field
        df["which"] = which
        df["global_id_in_field"] = df["cell_id"]

        # pooled unique id (string) so F0/F1/F2 don't collide
        df["cell_id"] = df["field"].astype(str) + ":" + df["global_id_in_field"].astype(str)

    return df1, df2


dfs_gfp1 = []
dfs_gfp2 = []
for field in FIELDS:
    d1, d2 = load_good_gfp1_gfp2_for_field(field)
    dfs_gfp1.append(d1)
    dfs_gfp2.append(d2)

df_gfp1_pool = pd.concat(dfs_gfp1, ignore_index=True)
df_gfp2_pool = pd.concat(dfs_gfp2, ignore_index=True)

print("\n=== POOLED ===")
print("GFP1 pooled rows:", len(df_gfp1_pool), "cells:", df_gfp1_pool["cell_id"].nunique())
print("GFP2 pooled rows:", len(df_gfp2_pool), "cells:", df_gfp2_pool["cell_id"].nunique())


# -------------------------
# 4) Plot pooled movies
# -------------------------
features = ["pol1_int_corr", "pol2_int_corr", "pol1_minus_pol2"]

def plot_movie(df_plot: pd.DataFrame, title_prefix: str, order_by_field: bool = True):
    if order_by_field and ("field" in df_plot.columns):
        cell_ids = (df_plot[["field", "cell_id"]]
                    .drop_duplicates()
                    .sort_values(["field", "cell_id"])["cell_id"]
                    .tolist())
    else:
        cell_ids = sorted(df_plot["cell_id"].unique().tolist())

    global_time = np.sort(pd.to_numeric(df_plot["time_point"], errors="coerce").dropna().unique())
    time_points = global_time
    best_shifts = {cid: 0 for cid in cell_ids}

    plot_aligned_signals(
        df_plot, cell_ids, best_shifts,
        global_time, time_points,
        features,
        mean_trace=None, std_trace=None,
        title_prefix=title_prefix
    )

    plot_aligned_heatmaps(
        df_plot, cell_ids, best_shifts,
        global_time, time_points,
        features,
        cmap_list=["viridis"] * len(features),
        title=f"{title_prefix} heatmaps"
    )


plot_movie(df_gfp1_pool, "Pooled F0+F1+F2 GFP1 aligned to BF1", order_by_field=True)
plot_movie(df_gfp2_pool, "Pooled F0+F1+F2 GFP2 aligned to BF1", order_by_field=True)




























