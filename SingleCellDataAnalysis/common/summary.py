#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:30:35 2025

@author: user
"""

import pandas as pd

def sanitize_summary(df):
    df = df.copy()
    for col in ["start","end","length","slope","intercept","ssr","valid"]:
        if col not in df.columns: continue
        if col in ["start","end","length"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif col == "valid":
            df[col] = df[col].astype(bool)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
