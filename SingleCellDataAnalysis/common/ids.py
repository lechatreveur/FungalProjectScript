#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:27:37 2025

@author: user
"""

import re
import pandas as pd

def norm_int(x) -> int | None:
    s = str(x).strip().split("_", 1)[0]
    m = re.match(r"^\d+", s)
    return int(m.group(0)) if m else None

def split_variants(df: pd.DataFrame, id_col="cell_id"):
    df = df.copy()
    m = df[id_col].astype(str).str.extract(r'^(?P<canonical>\d+)(?:_(?P<variant>\d+))?$')
    df["canonical_cell_id"] = m["canonical"]
    df["variant"] = m["variant"].fillna("0").astype(int)
    parts = {
        "base": df[df["variant"] == 0].assign(cell_id=lambda d: d["canonical_cell_id"]).drop(columns=["canonical_cell_id","variant"]),
        "v1":   df[df["variant"] == 1].assign(cell_id=lambda d: d["canonical_cell_id"]).drop(columns=["canonical_cell_id","variant"]),
        "v2":   df[df["variant"] == 2].assign(cell_id=lambda d: d["canonical_cell_id"]).drop(columns=["canonical_cell_id","variant"]),
    }
    return parts
