#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:32:09 2025

@author: user
"""

import pandas as pd
from .ids import norm_int

def compose_chains(mapping_g1_b1: dict, pairs_b1_g2_df: pd.DataFrame, mapping_g2_b2: dict):
    g1_to_b1 = {norm_int(g): norm_int(b) for g, b in mapping_g1_b1.items() if norm_int(g) is not None and norm_int(b) is not None}
    g2_to_b2 = {norm_int(g): norm_int(b) for g, b in mapping_g2_b2.items() if norm_int(g) is not None and norm_int(b) is not None}
    # BF1 -> GFP2 (max IoU)
    bf1_to_g2 = {}
    if "bf_id" in pairs_b1_g2_df.columns and "gfp_id" in pairs_b1_g2_df.columns:
        tmp = pairs_b1_g2_df.dropna(subset=["bf_id","gfp_id"])
        if "iou" in tmp.columns:
            idx = tmp.groupby("bf_id")["iou"].idxmax()
            tmp = tmp.loc[idx]
        else:
            tmp = tmp.drop_duplicates("bf_id")
        for _, r in tmp.iterrows():
            b1, g2 = norm_int(r.bf_id), norm_int(r.gfp_id)
            if b1 is not None and g2 is not None: bf1_to_g2[b1] = g2
    # Compose chains
    rows = []
    for g1, b1 in g1_to_b1.items():
        g2 = bf1_to_g2.get(b1); b2 = g2_to_b2.get(g2) if g2 is not None else None
        if None not in (g1, b1, g2, b2):
            rows.append((g1,b1,g2,b2))
    return pd.DataFrame(rows, columns=["gfp1_id","bf1_id","gfp2_id","bf2_id"])
