#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:30:04 2025

@author: user
"""

import numpy as np, pandas as pd

def order_by_anchor_shift(df_aligned: pd.DataFrame,
                          gfp_id_norm_col="gfp_id_norm",
                          anchor_col="anchor_bf_end_tp",
                          frames_per_min=5):
    shift_tbl = (
        df_aligned
        .dropna(subset=[gfp_id_norm_col, anchor_col])
        .groupby(gfp_id_norm_col, as_index=False)[anchor_col].first()
    )
    shift_tbl["shift_frames"] = -shift_tbl[anchor_col].astype(float) * frames_per_min
    shift_map = {int(r[gfp_id_norm_col]): float(r["shift_frames"]) for _, r in shift_tbl.iterrows()}
    return shift_map

def sort_cell_ids_by_shift(cell_ids, shift_map):
    return sorted(cell_ids, key=lambda cid: shift_map.get(int(cid), float("inf")))
