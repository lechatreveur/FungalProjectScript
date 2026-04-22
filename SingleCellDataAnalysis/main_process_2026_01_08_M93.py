#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:39:16 2026

@author: user
"""
import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')



import os
import pandas as pd
import numpy as np

WORKING_DIR = "/Volumes/X10 Pro/Movies/2026_01_08_M93/"
FILM_NAMES = ["A14_FL_1",
              "A14_BF_1",
              "A14_FL_2",
              "A14_BF_2",
              "A14_FL_3"]
FIELDS = ["F0", "F1", "F2"]

# Flat list of every per-field film name — needed by export_manifest_from_json_states
ALL_FILM_NAMES = [f"{base}_{field}" for base in FILM_NAMES for field in FIELDS]
# Use BF images for manual alignment

print("📥 Loading data...")
#df_all = load_and_merge_csv(FILE_NAMES, WORKING_DIR)
#%%
from SingleCellDataAnalysis.alignment_board_gui import review_septum_alignment_board_gui

#BF1_DIR_F0 = os.path.join(WORKING_DIR, "A14-YES-1t-FBFBF-2_F0")
FILM_NAME = f"{FILM_NAMES[1]}_F0"
review_septum_alignment_board_gui(WORKING_DIR, FILM_NAME, mask_col="rle_bf", n_rows=12, n_cols=25, tile_size=96)
#%%
FILM_NAME = f"{FILM_NAMES[1]}_F1"
review_septum_alignment_board_gui(WORKING_DIR, FILM_NAME, mask_col="rle_bf", n_rows=12, n_cols=25, tile_size=96)
#%%
FILM_NAME = f"{FILM_NAMES[1]}_F2"
review_septum_alignment_board_gui(WORKING_DIR, FILM_NAME, mask_col="rle_bf", n_rows=12, n_cols=25, tile_size=96)
#%%
FILM_NAME = f"{FILM_NAMES[3]}_F0"
review_septum_alignment_board_gui(WORKING_DIR, FILM_NAME, mask_col="rle_gfp", n_rows=12, n_cols=25, tile_size=96)
#%%
FILM_NAME = f"{FILM_NAMES[2]}_F1"
review_septum_alignment_board_gui(WORKING_DIR, FILM_NAME, mask_col="rle_bf", n_rows=12, n_cols=25, tile_size=96)
#%%
FILM_NAME = f"{FILM_NAMES[3]}_F2"
review_septum_alignment_board_gui(WORKING_DIR, FILM_NAME, mask_col="rle_bf", n_rows=12, n_cols=25, tile_size=96)
#%% Export manifest from GUI JSON states (bridges new GUI → downstream pipeline)
# Run this after finishing all alignment board QC sessions above.
from SingleCellDataAnalysis.septum_gui_utils import export_manifest_from_json_states

export_manifest_from_json_states(
    WORKING_DIR, ALL_FILM_NAMES, out_relpath="training_dataset/pipeline_manifest.csv"
)

#%% Pairing cell IDs between BF and GFP movies
from SingleCellDataAnalysis.multi_field import run_field_sequence, make_field_sequence
from SingleCellDataAnalysis.population_movie_gui import build_global_id_maps_from_pairings

all_res = {}
all_global_maps = {}

for field in FIELDS:
    out_dir = os.path.join(WORKING_DIR, "pipeline_outputs", f"{field}_maps_and_aligned")
    field_seq = make_field_sequence(field, FILM_NAMES)

    res = run_field_sequence(
        WORKING_DIR,
        field_seq,
        out_dir,
        iou_min=0.01,
        manifest_relpath="training_dataset/pipeline_manifest.csv",
        #bf_rle_col="rle_gfp",   # your BF mask column is mislabeled as rle_gfp
    )
    all_res[field] = res
    print(f"\n==== {field} done; {len(res)} pair(s) ====")
    for r in res:
        print(r.get("pair"), r.get("aligned_csv", "no aligned_csv"))

    field_seq = make_field_sequence(field, FILM_NAMES)
    pair_mappings = all_res[field]                      # <-- use the right one
    anchor_film = f"A14_BF_1_{field}"         # <-- BF1 for that field

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
    field_seq = make_field_sequence(field, FILM_NAMES)       # your existing helper
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
field_seq = make_field_sequence(field, FILM_NAMES)

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
field_seq = make_field_sequence(field, FILM_NAMES)

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
field_seq = make_field_sequence(field, FILM_NAMES)

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
#%% Load GOOD aligned GFP1 & GFP2 per field + pool


from SingleCellDataAnalysis.visualization import plot_aligned_signals, plot_aligned_heatmaps, load_good_gfp1_gfp2_for_field

dfs_gfp1 = []
dfs_gfp2 = []
for field in FIELDS:
    d1, d2 = load_good_gfp1_gfp2_for_field(field, FILM_NAMES, WORKING_DIR, all_global_maps, all_res, only_has_septum_bf1=True)
    
    dfs_gfp1.append(d1)
    dfs_gfp2.append(d2)

df_gfp1_pool = pd.concat(dfs_gfp1, ignore_index=True)
df_gfp2_pool = pd.concat(dfs_gfp2, ignore_index=True)

print("\n=== POOLED ===")
print("GFP1 pooled rows:", len(df_gfp1_pool), "cells:", df_gfp1_pool["cell_id"].nunique())
print("GFP2 pooled rows:", len(df_gfp2_pool), "cells:", df_gfp2_pool["cell_id"].nunique())

# #%%
# # -------------------------
# # 4) Plot pooled movies
# # -------------------------
# features = ["pol1_int_corr", "pol2_int_corr", "pol1_minus_pol2"]

# def plot_movie(df_plot: pd.DataFrame, title_prefix: str, order_by_field: bool = True):
#     if order_by_field and ("field" in df_plot.columns):
#         cell_ids = (df_plot[["field", "cell_id"]]
#                     .drop_duplicates()
#                     .sort_values(["field", "cell_id"])["cell_id"]
#                     .tolist())
#     else:
#         cell_ids = sorted(df_plot["cell_id"].unique().tolist())

#     global_time = np.sort(pd.to_numeric(df_plot["time_point"], errors="coerce").dropna().unique())
#     time_points = global_time
#     best_shifts = {cid: 0 for cid in cell_ids}

#     plot_aligned_signals(
#         df_plot, cell_ids, best_shifts,
#         global_time, time_points,
#         features,
#         mean_trace=None, std_trace=None,
#         title_prefix=title_prefix
#     )

#     plot_aligned_heatmaps(
#         df_plot, cell_ids, best_shifts,
#         global_time, time_points,
#         features,
#         cmap_list=["viridis"] * len(features),
#         title=f"{title_prefix} heatmaps"
#     )


# plot_movie(df_gfp1_pool, "Pooled F0+F1+F2 GFP1 aligned to BF1", order_by_field=True)
# plot_movie(df_gfp2_pool, "Pooled F0+F1+F2 GFP2 aligned to BF1", order_by_field=True)



#%% plot two 
from SingleCellDataAnalysis.visualization import plot_movie_gfp1_gfp2
plot_movie_gfp1_gfp2(
    df_gfp1_pool, df_gfp2_pool,
    WORKING_DIR=WORKING_DIR,
    FILM_NAMES=FILM_NAMES,
    all_global_maps=all_global_maps,
    bf_seconds_per_frame=60.0,
    gfp_seconds_per_frame=12.0,
    bin_minutes= 0.2,#1.0,              # 1-minute bins on heatmap
    clip_x_minutes=(-200, 200),   # now interpreted as minutes
    only_has_septum_bf1=True,
)


# #%% debug
# cells1 = set(df_gfp1_pool["cell_id"].astype(str))
# cells2 = set(df_gfp2_pool["cell_id"].astype(str))
# common = sorted(cells1 & cells2)

# print("GFP1 cells:", len(cells1))
# print("GFP2 cells:", len(cells2))
# print("COMMON:", len(common))
# print("example common:", common[:10])

# # Also check per-field overlap (useful for debugging pooling)
# for f in ["F0","F1","F2"]:
#     c1 = set(df_gfp1_pool.loc[df_gfp1_pool["field"]==f, "cell_id"].astype(str))
#     c2 = set(df_gfp2_pool.loc[df_gfp2_pool["field"]==f, "cell_id"].astype(str))
#     print(f, "common:", len(c1 & c2))

# #%% debug
# bf1 = f"{FILM_NAMES[1]}_F0"
# l2g = all_global_maps["F0"][bf1]
# k0 = next(iter(l2g.keys()))
# v0 = l2g[k0]
# print("sample local_id:", k0, type(k0))
# print("sample global_id:", v0, type(v0))
# print("does g2l int lookup work?", int(str(v0)) in set(l2g.values()))
















