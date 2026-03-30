#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 12:28:49 2025

@author: user
"""

import pandas as pd

def add_corrected_intensities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["septum_int_corr"] = df["septum_int"] - df["cyt_int"]
    df["pol1_int_corr"]   = df["pol1_int"]   - df["cyt_int"]
    df["pol2_int_corr"]   = df["pol2_int"]   - df["cyt_int"]
    df["pol1_minus_pol2"] = df["pol1_int_corr"] - df["pol2_int_corr"]
    return df

def drop_outliers(df: pd.DataFrame, col="pol2_int_corr", thresh=100) -> pd.DataFrame:
    if col in df.columns:
        df = df.loc[~(df[col] > thresh)].copy()
    return df
