#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:31:07 2025

@author: user
"""

from typing import Sequence, Mapping
def plot_cells_grid_wrapper(plot_cells_grid_fn, df_or_path, *, cell_ids=None, channel="bf",
                            paginate=False, nrows=10, ncols=10, **kwargs):
    return plot_cells_grid_fn(df_or_path, cell_ids=cell_ids, channel=channel,
                              paginate=paginate, nrows=nrows, ncols=ncols, **kwargs)

def plot_aligned_bundle(plot_aligned_signals_fn, plot_aligned_heatmaps_fn,
                        df_plot, cell_ids: Sequence[int], shifts: Mapping[int, float],
                        global_time, time_points, features_signals, features_heatmap,
                        title_prefix, cmap="viridis"):
    plot_aligned_signals_fn(
        df_plot, cell_ids, shifts, global_time, time_points,
        features_signals, mean_trace=None, std_trace=None,
        title_prefix=title_prefix
    )
    plot_aligned_heatmaps_fn(
        df_plot, cell_ids, shifts, global_time, time_points,
        features_heatmap, cmap_list=[cmap]*len(features_heatmap)
    )
