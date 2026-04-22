#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_pipeline_manifest.py
============================
Standalone script: reads the per-film alignment JSON states saved by the
multi-film alignment board GUI and writes pipeline_manifest.csv for each
experiment.

  Output: <WORKING_DIR>/training_dataset/pipeline_manifest.csv

This file is read by run_field_sequence() (via load_manifest with
manifest_relpath="training_dataset/pipeline_manifest.csv") and is kept
separate from the AI training manifest (manifest.csv) to avoid clobbering it.

Usage
-----
Run the whole script once after completing all alignment board QC sessions.
Can also be run cell-by-cell in Spyder.
"""

import sys
import os

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from SingleCellDataAnalysis.septum_gui_utils import export_manifest_from_json_states

# ──────────────────────────────────────────────
# M92  (2025-12-31)
# Film order: GFP1, BF1, GFP2, BF2, GFP3
# Naming: A14-YES-1t-FBFBF[-N]_<field>
# ──────────────────────────────────────────────
#%%
M92_WORKING_DIR = "/Volumes/X10 Pro/Movies/2025_12_31_M92/"
M92_FIELDS      = ["F0", "F1", "F2"]
M92_BASE_NAMES  = [
    "A14-YES-1t-FBFBF",
    "A14-YES-1t-FBFBF-2",
    "A14-YES-1t-FBFBF-3",
    "A14-YES-1t-FBFBF-4",
    "A14-YES-1t-FBFBF-5",
]
M92_ALL_FILMS = [f"{base}_{field}"
                 for base in M92_BASE_NAMES
                 for field in M92_FIELDS]

print("=" * 60)
print("Exporting pipeline manifest for M92 ...")
df_m92 = export_manifest_from_json_states(
    working_dir=M92_WORKING_DIR,
    film_names=M92_ALL_FILMS,
    out_relpath="training_dataset/pipeline_manifest.csv",
)
print(f"M92 total rows : {len(df_m92)}")
print(f"M92 has==1     : {(df_m92['has'] == 1).sum()}")
print(f"M92 fields     : {sorted(df_m92['field'].dropna().unique())}")
print()

# ──────────────────────────────────────────────
# M93  (2026-01-08)
# Film order: FL1, BF1, FL2, BF2, FL3
# Naming: A14_FL/BF_N_<field>
# ──────────────────────────────────────────────
#%%
M93_WORKING_DIR = "/Volumes/X10 Pro/Movies/2026_01_08_M93/"
M93_FIELDS      = ["F0", "F1", "F2"]
M93_BASE_NAMES  = [
    "A14_FL_1",
    "A14_BF_1",
    "A14_FL_2",
    "A14_BF_2",
    "A14_FL_3",
]
M93_ALL_FILMS = [f"{base}_{field}"
                 for base in M93_BASE_NAMES
                 for field in M93_FIELDS]

print("=" * 60)
print("Exporting pipeline manifest for M93 ...")
df_m93 = export_manifest_from_json_states(
    working_dir=M93_WORKING_DIR,
    film_names=M93_ALL_FILMS,
    out_relpath="training_dataset/pipeline_manifest.csv",
)
print(f"M93 total rows : {len(df_m93)}")
print(f"M93 has==1     : {(df_m93['has'] == 1).sum()}")
print(f"M93 fields     : {sorted(df_m93['field'].dropna().unique())}")
print()

#%% Quick per-field summary
print("=" * 60)
print("M92 per-field breakdown:")
print(df_m92.groupby(["field", "has"])["cell_id"].count().rename("n_cells").reset_index().to_string(index=False))
print()
print("M93 per-field breakdown:")
print(df_m93.groupby(["field", "has"])["cell_id"].count().rename("n_cells").reset_index().to_string(index=False))
