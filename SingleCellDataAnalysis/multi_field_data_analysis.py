#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 11:08:00 2026

@author: user
"""
import os
import pandas as pd

def load_good_global_ids(working_dir: str, field: str) -> list[int]:
    qc_csv = os.path.join(working_dir, "pipeline_outputs", f"QC__{field}.csv")
    qc = pd.read_csv(qc_csv)
    good = qc.loc[qc["label"].astype(str).str.lower() == "good", "global_id"].dropna()
    return sorted(good.astype(int).unique().tolist())

def global_to_local_ids(global_maps_by_film: dict, film: str, good_gids: list[int]) -> dict[int, int]:
    """
    Return {global_id -> local_id} for the subset of good_gids that exist in this film.
    Keeps the first local_id if collisions exist.
    """
    l2g = global_maps_by_film[film]  # {local_id: global_id}
    g2l = {}
    for local_id, gid in l2g.items():
        if gid is None:
            continue
        gid = int(gid)
        if gid in good_gids and gid not in g2l:
            g2l[gid] = int(local_id)
    return g2l

import numpy as np
import pandas as pd

def add_polarity_corrected_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["septum_int_corr"] = df["septum_int"] - df["cyt_int"]
    df["pol1_int_corr"]   = df["pol1_int"]   - df["cyt_int"]
    df["pol2_int_corr"]   = df["pol2_int"]   - df["cyt_int"]
    df["pol1_minus_pol2"] = df["pol1_int_corr"] - df["pol2_int_corr"]
    return df

def pick_aligned_time_col(df: pd.DataFrame) -> str:
    # try common names produced by your alignment helpers
    for c in ["aligned_frame_rounded", "aligned_frame", "time_point_aligned", "aligned_time_point", "aligned_tp"]:
        if c in df.columns:
            return c
    raise KeyError("No aligned-time column found. Expected one of aligned_frame_rounded/aligned_frame/...")
    
    
def get_aligned_csv(pair_mappings: list[dict], gfp_film: str, bf_film: str) -> str:
    for item in pair_mappings:
        if item.get("gfp_film") == gfp_film and item.get("bf_film") == bf_film and "aligned_csv" in item:
            return item["aligned_csv"]
    raise FileNotFoundError(f"No aligned_csv found for {gfp_film} anchored to {bf_film}")



def keep_base_variant_only(df, cell_col="cell_id"):
    s = df[cell_col].astype(str)
    # base has no suffix OR suffix==0
    base_mask = ~s.str.contains("_")
    return df.loc[base_mask].copy()
    
def load_good_aligned_gfp(global_maps, good_gids, aligned_csv: str, film: str) -> pd.DataFrame:
    df = pd.read_csv(aligned_csv)
    df = keep_base_variant_only(df, cell_col="cell_id")
    df = add_polarity_corrected_cols(df)

    # map local -> global
    l2g = global_maps[film]  # local_id -> global_id
    # IMPORTANT: your aligned CSV might already have canonicalized cell_id;
    # if cell_id can contain suffixes, normalize first:
    cell_id_num = pd.to_numeric(df["cell_id"].astype(str).str.split("_").str[0], errors="coerce")
    df["cell_id_canon"] = cell_id_num.astype("Int64")

    df["global_id"] = df["cell_id_canon"].map(lambda x: l2g.get(int(x), np.nan) if pd.notna(x) else np.nan)
    df = df[df["global_id"].notna()].copy()
    df["global_id"] = df["global_id"].astype(int)

    # keep only QC-good
    df = df[df["global_id"].isin(good_gids)].copy()

    # use global_id as the plotting cell_id so GFP1/GFP2 match
    df["cell_id"] = df["global_id"].astype(int)

    # overwrite time_point with aligned integer time
    atc = pick_aligned_time_col(df)
    if atc != "aligned_frame_rounded":
        # if aligned_frame is float, round it for integer grid
        df["aligned_frame_rounded"] = pd.to_numeric(df[atc], errors="coerce").round().astype("Int64")
        atc = "aligned_frame_rounded"

    df = df.dropna(subset=[atc]).copy()
    df["time_point"] = df[atc].astype(int)

    return df

import pandas as pd

def report_dups(df, label, cell_col="cell_id", time_col="time_point", show=10):
    key = [cell_col, time_col]
    ndup = int(df.duplicated(key).sum())
    print(f"\n[{label}] rows={len(df)} unique_cells={df[cell_col].nunique()} dups(cell,time)={ndup}")
    if ndup:
        bad = df[df.duplicated(key, keep=False)].sort_values(key)
        print(bad[key].head(show).to_string(index=False))
        # show how many dups per cell
        per_cell = bad.groupby(cell_col)[time_col].apply(lambda s: s.duplicated().sum()).sort_values(ascending=False)
        print("top cells with duplicate times:\n", per_cell.head(10).to_string())
    return ndup
