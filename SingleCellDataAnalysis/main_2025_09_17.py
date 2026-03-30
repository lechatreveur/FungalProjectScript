#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:06:13 2025

@author: user
"""

# main.py (sketch)
from SingleCellDataAnalysis.config import WORKING_DIR
from SingleCellDataAnalysis.visualization import plot_cells_grid, plot_aligned_signals, plot_aligned_heatmaps
from SingleCellDataAnalysis.map_gfp_bf_id import map_gfp_to_bf_ids
from SingleCellDataAnalysis.increasing_period_fit import scan_cells_summary, review_cells_gui
from common.io import load_csv, save_csv, path, ensure_dir, timestamp
from common.ids import split_variants, norm_int
from common.features import add_corrected_intensities, drop_outliers
from common.mapping import mapping_to_table, one_to_many_table, max_iou_pairs
from common.alignment import build_aligned_plot_table
from common.ordering import order_by_anchor_shift, sort_cell_ids_by_shift
from common.summary import sanitize_summary
from common.plotting import plot_cells_grid_wrapper, plot_aligned_bundle
from common.chains import compose_chains
from common.videos import concat_videos

# 1) Load BF-TP1 and subset cells
bf1_csv = path(WORKING_DIR, "A14_1TP1_BF_F1/TrackedCells_A14_1TP1_BF_F1/all_cells_time_series.csv")
df_bf1 = load_csv(bf1_csv)
some_cells = [...]  # your selection
plot_cells_grid_wrapper(plot_cells_grid, bf1_csv, cell_ids=some_cells, paginate=False)

df_bf1_sel = df_bf1[df_bf1["cell_id"].isin(some_cells)]
summary = scan_cells_summary(df_bf1_sel, some_cells, enforce_targets=True, start_target=0.0, end_target=0.9, start_tol=0.1, end_tol=0.5)
res = review_cells_gui(df_bf1_sel, some_cells, summary_df=summary, enforce_targets=True, start_target=0.0, end_target=1, start_tol=5, end_tol=5, min_len=21, block=False)
summary = res.get("summary", summary)
summary_s = sanitize_summary(summary)
save_csv(summary_s, WORKING_DIR, "manual_fits", f"A14_1TP1_BF_F1_summary_{timestamp()}.csv")

# 2) Load GFP-TP1, split variants, correct features
gfp1_csv = path(WORKING_DIR, "A14_1TP1_F1/TrackedCells_A14_1TP1_F1/all_cells_time_series.csv")
df_gfp1_raw = load_csv(gfp1_csv)
gfp1_parts = split_variants(df_gfp1_raw)
df_gfp1_all = (add_corrected_intensities(gfp1_parts["base"]))

# 3) Map GFP1→BF1 and build table
mapping1, pairs_df1, files1 = map_gfp_to_bf_ids(WORKING_DIR, gfp_timepoint="last", bf_timepoint="first", gfp_rle_col="rle_gfp", bf_rle_col="rle_bf", iou_min=0.01)
table_g1_b1 = mapping_to_table(mapping1)

# 4) Repeat analogous steps for TP2 (BF2, GFP2), build one-to-many BF1→GFP2 table using your helper

# 5) Align GFP1/2 to BF anchors (end or start), then build aligned plot tables
# (use your existing align helpers; then:)
# df_plot, global_time = build_aligned_plot_table(df_gfp_aligned, variant="base")

# 6) Sort by anchor & plot bundles
# shift_map = order_by_anchor_shift(df_gfp_aligned, anchor_col="anchor_bf_end_tp", frames_per_min=5)
# cell_ids = sort_cell_ids_by_shift(df_plot["cell_id"].unique().tolist(), shift_map)
# plot_aligned_bundle(plot_aligned_signals, plot_aligned_heatmaps, ...)

# 7) Compose chains & make paired per-cell PDFs (call a small wrapper you move into plotting.py)
