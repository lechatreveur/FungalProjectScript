#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:49:48 2025

@author: user
"""

# filter_extremes.py

import pandas as pd
from SingleCellDataAnalysis.gumm import fit_gaussian_uniform_mixture 

def find_extreme_cells(growth_df, feature, n_sigma=1.96, mode='both', override_upper=None, override_lower=None):
    """
    Identify outliers in a feature column using GUMM fit.
    
    mode: 'both', 'upper', or 'lower'
    """
    data = growth_df[feature].values
    mu, sigma, _ = fit_gaussian_uniform_mixture(data)

    upper_thresh = override_upper if override_upper is not None else mu + n_sigma * sigma
    lower_thresh = override_lower if override_lower is not None else mu - n_sigma * sigma

    if mode == 'upper':
        mask = growth_df[feature] > upper_thresh
    elif mode == 'lower':
        mask = growth_df[feature] < lower_thresh
    else:  # both
        mask = (growth_df[feature] > upper_thresh) | (growth_df[feature] < lower_thresh)

    return growth_df.loc[mask, 'cell_id']


def get_all_extreme_cells(growth_df, n_sigma=1.96):
    """
    Apply GUMM-based thresholding across multiple features and combine all extreme cells.
    """
    extreme_ids = pd.Series(dtype=int)

    extreme_ids = pd.concat([
        extreme_ids,
        find_extreme_cells(growth_df, 'avg_d_cell_area', n_sigma=n_sigma),
        #find_extreme_cells(growth_df, 'max_cell_area', n_sigma=n_sigma, mode='upper'),
        #find_extreme_cells(growth_df, 'min_cell_area', n_sigma=2*n_sigma, mode='lower'),
        find_extreme_cells(growth_df, 'std_d_cell_area', n_sigma=n_sigma),# mode='upper'),
        find_extreme_cells(growth_df, 'max_d_cell_area', n_sigma=n_sigma),#, mode='upper'),
        #find_extreme_cells(growth_df, 'max_abs_d_nu_dis', override_upper=10, mode='upper'),
        #pd.Series([788, 899, 1061])  # Manual exclusions
        find_extreme_cells(growth_df, 'avg_pattern_score', n_sigma=n_sigma, mode='lower'),
        find_extreme_cells(growth_df, 'std_d_pattern_score', n_sigma=1.5*n_sigma, mode='lower'),
    ])

    return extreme_ids.drop_duplicates().astype(int)
