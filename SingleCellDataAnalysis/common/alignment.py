#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:29:31 2025

@author: user
"""

import numpy as np, pandas as pd

def build_aligned_plot_table(df_aligned: pd.DataFrame,
                             variant="base",
                             aligned_col="aligned_frame",
                             gfp_id_norm_col="gfp_id_norm",
                             time_point_col="time_point"):
    d = df_aligned[df_aligned["gfp_variant"] == variant].copy()
    d = d.dropna(subset=[aligned_col])
    d["aligned_frame_rounded"] = pd.to_numeric(d[aligned_col].round(), errors="coerce").astype(pd.Int64Dtype())
    d = d.dropna(subset=["aligned_frame_rounded"]).copy()
    d[time_point_col] = d["aligned_frame_rounded"].astype(int)
    d["cell_id"] = d[gfp_id_norm_col].astype(int)
    global_time = np.sort(d[time_point_col].unique())
    return d, global_time
