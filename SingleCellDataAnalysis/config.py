#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:44:52 2025

@author: user
"""

# config.py

# Directories
#WORKING_DIR = "/Volumes/Movies/2025_06_04_M68/"
WORKING_DIR = "/Users/user/Documents/FungalProject/TimeLapse/2025_06_25/"
FILE_NAMES = [f"A14_{i}" for i in range(1, 10)]

# Parameters
FRAME_NUMBER = 51  # Expected number of time points per cell
ROLLING_WINDOW = 20  # Smoothing window size
N_SIGMA = 1.96  # For Gaussian thresholding
SEED = 42  # For reproducibility


WORKING_DIR = "/Users/user/Documents/FungalProject/TimeLapse/2025_07_23_M77/"
FILE_NAMES = "A14_25oC_2hr_1" #[f"A14_{i}" for i in range(1, 10)]

# Parameters
FRAME_NUMBER = 601  # Expected number of time points per cell



WORKING_DIR = "/Volumes/Movies/2025_09_17/"
FILE_NAMES = ["A14_1TP1_BF_F1"] #[f"A14_{i}" for i in range(1, 10)]

# Parameters
FRAME_NUMBER = 120  # Expected number of time points per cell