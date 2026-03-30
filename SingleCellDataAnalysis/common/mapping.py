#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:29:16 2025

@author: user
"""

import pandas as pd
from .ids import norm_int

def mapping_to_table(mapping: dict, gfp_label="gfp_id", bf_label="bf_id"):
    rows = []
    for g, b in mapping.items():
        gi, bi = norm_int(g), norm_int(b)
        if gi is not None and bi is not None:
            rows.append({gfp_label: gi, bf_label: bi, "status": "mapped"})
    return pd.DataFrame(rows)

def one_to_many_table(bf_to_gfps: dict[int, list[int]]):
    rows = []
    for bf, gs in bf_to_gfps.items():
        bi = norm_int(bf)
        if bi is None: continue
        for g in gs:
            gi = norm_int(g)
            if gi is not None:
                rows.append({"bf_id": bi, "gfp_id": gi, "status": "mapped"})
    return pd.DataFrame(rows)

def max_iou_pairs(pairs_df: pd.DataFrame, key="bf_id"):
    # returns best pair per key based on max iou
    tmp = pairs_df.dropna(subset=["bf_id","gfp_id"])
    if "iou" in tmp.columns:
        idx = tmp.groupby(key)["iou"].idxmax()
        return tmp.loc[idx].copy()
    return tmp.drop_duplicates(key).copy()
