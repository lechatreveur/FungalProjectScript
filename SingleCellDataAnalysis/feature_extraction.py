#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:48:08 2025

@author: user
"""

# feature_extraction.py

import numpy as np
import pandas as pd

def extract_features(df_all):
    features = {}

    features['avg_d_cell_length'] = (
        df_all.groupby("cell_id")["d_cell_length"]
        .mean().rename("avg_d_cell_length").reset_index()
    )
    features['avg_d_cell_area'] = (
        df_all.groupby("cell_id")["d_cell_area"]
        .mean().rename("avg_d_cell_area").reset_index()
    )
    features['max_cell_area'] = (
        df_all.groupby("cell_id")["cell_area"]
        .max().rename("max_cell_area").reset_index()
    )
    features['min_cell_area'] = (
        df_all.groupby("cell_id")["cell_area"]
        .min().rename("min_cell_area").reset_index()
    )
    features['std_d_cell_area'] = (
        df_all.groupby("cell_id")["d_cell_area"]
        .std().rename("std_d_cell_area").reset_index()
    )
    features['max_d_cell_area'] = (
        df_all.groupby("cell_id")["d_cell_area"]
        .apply(lambda x: np.max(x)).rename("max_d_cell_area").reset_index()
    )
    features['max_d_nu_dis'] = (
        df_all.groupby("cell_id")["d_nu_dis"]
        .apply(lambda x: np.max(x)).rename("max_d_nu_dis").reset_index()
    )
    features['avg_pattern_score'] = (
        df_all.groupby("cell_id")["pattern_score_norm"]
        .mean().rename("avg_pattern_score").reset_index()
    )
    features['std_d_pattern_score'] = (
        df_all.groupby("cell_id")["d_pattern_score"]
        .std().rename("std_d_pattern_score").reset_index()
    )
    # Merge all into one DataFrame
    growth_matrix = features['avg_d_cell_length']
    for key in list(features.keys())[1:]:
        growth_matrix = growth_matrix.merge(features[key], on="cell_id")

    return growth_matrix
