#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 16:13:00 2026

@author: user
"""

import os
import re
import numpy as np
import pandas as pd


def make_field_sequence(field: str, FILM_NAMES):
    # field should be "F0" / "F1" / "F2"
    return [
        ("gfp", f"{FILM_NAMES[0]}_{field}"),
        ("bf",  f"{FILM_NAMES[1]}_{field}"),
        ("gfp", f"{FILM_NAMES[2]}_{field}"),
        ("bf",  f"{FILM_NAMES[3]}_{field}"),
        ("gfp", f"{FILM_NAMES[4]}_{field}"),
    ]


FIELD_RE = re.compile(r"_F(\d+)$")  # matches ..._F0 / ..._F1 / ..._F2
def bf_anchor_from_manifest(manifest: pd.DataFrame, bf_film_name: str) -> pd.DataFrame:
    """
    Returns per-BF-cell anchors from the manual alignment.
    Columns:
      bf_film, bf_id, has, bf_offset_frames
    """
    m = manifest.loc[manifest["film_name"] == bf_film_name, ["film_name","cell_id","has","offset"]].copy()
    m = m.rename(columns={
        "film_name": "bf_film",
        "cell_id": "bf_id",
        "offset": "bf_offset_frames",
    })
    # enforce NA when has!=1
    m.loc[m["has"] != 1, "bf_offset_frames"] = np.nan
    return m

def load_manifest(working_dir: str,
                  relpath: str = "training_dataset/manifest.csv") -> pd.DataFrame:
    path = os.path.join(working_dir, relpath)
    m = pd.read_csv(path)

    # types
    m["cell_id"] = pd.to_numeric(m["cell_id"], errors="coerce").astype("Int64")
    m["has"] = pd.to_numeric(m["has"], errors="coerce").fillna(0).astype(int)
    m["offset"] = pd.to_numeric(m["offset"], errors="coerce")

    # Only has==1 is meaningful
    m.loc[m["has"] != 1, "offset"] = np.nan

    # derive field from film_name
    def _field(name):
        mo = FIELD_RE.search(str(name))
        return f"F{mo.group(1)}" if mo else None
    m["field"] = m["film_name"].map(_field)

    return m

def split_cell_id(df: pd.DataFrame, cell_id_col="cell_id") -> pd.DataFrame:
    """
    Adds:
      - cell_id_canon (Int64)
      - cell_variant (int)
    """
    out = df.copy()
    s = out[cell_id_col].astype(str)

    canon = s.str.split("_", n=1).str[0]
    var = s.str.split("_", n=1).str[1]

    out["cell_id_canon"] = pd.to_numeric(canon, errors="coerce").astype("Int64")
    out["cell_variant"] = pd.to_numeric(var, errors="coerce").fillna(0).astype(int)
    return out

def find_timeseries_csv(working_dir: str, film_name: str) -> str:
    """
    Your structure:
      WORKING_DIR/<film_name>/TrackedCells_<film_name>/all_cells_time_series.csv
    """
    p = os.path.join(working_dir, film_name, f"TrackedCells_{film_name}", "all_cells_time_series.csv")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Missing time-series CSV for {film_name}: {p}")
    return p

def attach_offsets_and_align(df_ts: pd.DataFrame,
                             manifest: pd.DataFrame,
                             film_name: str,
                             time_col="time_point") -> pd.DataFrame:
    """
    Merges (offset, has) onto df_ts and computes:
      aligned_frame = time_point + offset
      aligned_frame_rounded (Int64)
    """
    df = split_cell_id(df_ts, cell_id_col="cell_id")

    m = manifest.loc[manifest["film_name"] == film_name, ["cell_id", "has", "offset"]].copy()
    m = m.rename(columns={"cell_id": "cell_id_canon"})

    df = df.merge(m, on="cell_id_canon", how="left")
    df["aligned_frame"] = pd.to_numeric(df[time_col], errors="coerce") + df["offset"]
    df["aligned_frame_rounded"] = pd.to_numeric(df["aligned_frame"].round(), errors="coerce").astype("Int64")

    return df

def process_all_films(working_dir: str,
                      out_dir: str,
                      fields=("F0","F1","F2"),
                      keep_only_has=True,
                      keep_variant="base"):
    """
    keep_variant:
      - "base": keep only cell_variant==0
      - "all": keep all variants
    """
    os.makedirs(out_dir, exist_ok=True)
    manifest = load_manifest(working_dir)

    # choose films to process
    films = (manifest.dropna(subset=["film_name"])
                    .loc[manifest["field"].isin(fields), "film_name"]
                    .drop_duplicates()
                    .tolist())

    print(f"[info] films to process: {len(films)} ({fields})")

    outputs = []
    for film in films:
        csv_path = find_timeseries_csv(working_dir, film)
        df_ts = pd.read_csv(csv_path)

        df_aln = attach_offsets_and_align(df_ts, manifest, film_name=film)

        if keep_only_has:
            df_aln = df_aln[df_aln["has"] == 1].copy()

        if keep_variant == "base":
            df_aln = df_aln[df_aln["cell_variant"] == 0].copy()

        # annotate
        df_aln["film_name"] = film
        df_aln["field"] = manifest.loc[manifest["film_name"] == film, "field"].iloc[0]

        # save per-film aligned table
        out_csv = os.path.join(out_dir, f"{film}__aligned.csv")
        df_aln.to_csv(out_csv, index=False)
        outputs.append(out_csv)
        print(f"[saved] {out_csv}  rows={len(df_aln)}")

    return outputs
import os
import numpy as np
import pandas as pd

from SingleCellDataAnalysis.map_gfp_bf_id import map_gfp_to_bf_ids

# from your module:
#   load_manifest, split_cell_id, find_timeseries_csv

def bf_anchor_from_manifest(manifest: pd.DataFrame, bf_film: str) -> pd.DataFrame:
    m = manifest.loc[manifest["film_name"] == bf_film, ["cell_id","has","offset"]].copy()
    m = m.rename(columns={"cell_id": "bf_id", "offset": "bf_offset_frames"})
    m["bf_id"] = pd.to_numeric(m["bf_id"], errors="coerce").astype("Int64")
    m.loc[m["has"] != 1, "bf_offset_frames"] = np.nan
    return m[["bf_id", "bf_offset_frames"]]

def align_gfp_ts_to_bf_anchor(df_gfp_ts: pd.DataFrame,
                             gfp_to_bf: dict,
                             bf_anchor_tbl: pd.DataFrame) -> pd.DataFrame:
    """
    Adds: gfp_id_canon, bf_id, bf_offset_frames, aligned_frame, aligned_frame_rounded
    Only meaningful for cells whose mapped BF has a non-NA offset (has==1 in manifest).
    """
    out = split_cell_id(df_gfp_ts, cell_id_col="cell_id")
    out["gfp_id_canon"] = out["cell_id_canon"].astype("Int64")

    # map canonical GFP id -> BF id
    out["bf_id"] = out["gfp_id_canon"].map(lambda x: gfp_to_bf.get(int(x)) if pd.notna(x) else np.nan)
    out["bf_id"] = pd.to_numeric(out["bf_id"], errors="coerce").astype("Int64")

    # attach BF offsets
    out = out.merge(bf_anchor_tbl, on="bf_id", how="left")

    # aligned time
    out["aligned_frame"] = pd.to_numeric(out["time_point"], errors="coerce") + out["bf_offset_frames"]
    out["aligned_frame_rounded"] = pd.to_numeric(out["aligned_frame"].round(), errors="coerce").astype("Int64")

    return out

def run_field_sequence(
    WORKING_DIR: str,
    field_seq: list,             # list of ("gfp"/"bf", film_name)
    out_dir: str,
    iou_min: float = 0.01,
    gfp_rle_col="rle_gfp",
    bf_rle_col="rle_bf",
):
    os.makedirs(out_dir, exist_ok=True)

    manifest = load_manifest(WORKING_DIR)

    bf_anchors = {}
    for kind, film in field_seq:
        if kind == "bf":
            bf_anchors[film] = bf_anchor_from_manifest(manifest, film)

    def _csv(film):
        return find_timeseries_csv(WORKING_DIR, film)

    results = []

    for (kind_a, film_a), (kind_b, film_b) in zip(field_seq[:-1], field_seq[1:]):

        if kind_a == "gfp" and kind_b == "bf":
            gfp_film, bf_film = film_a, film_b
            gfp_timepoint, bf_timepoint = "last", "first"
        elif kind_a == "bf" and kind_b == "gfp":
            bf_film, gfp_film = film_a, film_b
            gfp_timepoint, bf_timepoint = "first", "last"
        else:
            continue

        gfp_rel = f"{gfp_film}/TrackedCells_{gfp_film}/"
        bf_rel  = f"{bf_film}/TrackedCells_{bf_film}/"

        ## ---- key fix for this dataset (BF masks column mislabeled as rle_gfp) ----
        #bf_rle_col_eff = "rle_gfp" if bf_rle_col == "rle_bf" else bf_rle_col

        mapping, bf_to_gfps, pairs_df, files = map_gfp_to_bf_ids(
            WORKING_DIR,
            gfp_rel=gfp_rel,
            bf_rel=bf_rel,
            gfp_timepoint=gfp_timepoint,
            bf_timepoint=bf_timepoint,
            gfp_rle_col=gfp_rle_col,
            bf_rle_col=bf_rle_col,
            iou_min=iou_min,
        )

        pairs_csv = os.path.join(out_dir, f"pairs__{gfp_film}__to__{bf_film}.csv")
        pairs_df.to_csv(pairs_csv, index=False)

        map_out = {
            "pair": f"{gfp_film}__to__{bf_film}",
            "gfp_film": gfp_film,
            "bf_film": bf_film,
            "mapping_gfp_to_bf": mapping,
            "mapping_bf_to_gfps": bf_to_gfps,  # <-- keep this for chaining/division cases
            "pairs_csv": pairs_csv,
            "n_pairs": int(pairs_df.shape[0]),
        }

        if bf_film in bf_anchors:
            df_gfp = pd.read_csv(_csv(gfp_film))

            df_aligned = align_gfp_ts_to_bf_anchor(
                df_gfp_ts=df_gfp,
                gfp_to_bf=mapping,
                bf_anchor_tbl=bf_anchors[bf_film],
            )

            aligned_csv = os.path.join(out_dir, f"aligned__{gfp_film}__ANCHOR__{bf_film}.csv")
            df_aligned.to_csv(aligned_csv, index=False)

            map_out["aligned_csv"] = aligned_csv
            map_out["n_aligned_rows"] = int(df_aligned.shape[0])

        results.append(map_out)

    return results
from typing import Optional, Sequence, Tuple
import pandas as pd
import numpy as np

def prepare_df_for_plot_aligned_heatmaps(
    aligned_csv: str,
    value_cols: Sequence[str] = ("cyt_int",),
    time_src_cols: Sequence[str] = ("aligned_frame_rounded", "aligned_frame", "aligned_time", "time_aligned"),
    cell_src_cols: Sequence[str] = ("cell_id", "cell_id_canon"),
    out_time_col: str = "aligned_time",
    out_cell_col: str = "cell_id",
    drop_na_time: bool = True,
) -> pd.DataFrame:
    """
    Load your aligned GFP table and normalize column names so it can be fed into
    plot_aligned_heatmaps() without maintaining a separate plotting path.

    - picks first existing time column from time_src_cols
    - picks first existing cell column from cell_src_cols
    - renames them to (out_time_col, out_cell_col)
    - optionally keeps only the value_cols that exist
    """
    df = pd.read_csv(aligned_csv)

    # pick time column
    tcol = None
    for c in time_src_cols:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        raise KeyError(f"No aligned time column found. Tried: {time_src_cols}. Columns={list(df.columns)}")

    # pick cell column
    ccol = None
    for c in cell_src_cols:
        if c in df.columns:
            ccol = c
            break
    if ccol is None:
        raise KeyError(f"No cell id column found. Tried: {cell_src_cols}. Columns={list(df.columns)}")

    out = df.copy()
    out = out.rename(columns={tcol: out_time_col, ccol: out_cell_col})

    # coerce aligned time to numeric (int-ish)
    out[out_time_col] = pd.to_numeric(out[out_time_col], errors="coerce")
    if drop_na_time:
        out = out.dropna(subset=[out_time_col]).copy()

    # keep only requested value columns that exist
    keep_vals = [c for c in value_cols if c in out.columns]
    if len(keep_vals) == 0:
        raise KeyError(f"None of value_cols exist in CSV. value_cols={value_cols}. Columns={list(out.columns)}")

    # ensure needed columns exist
    cols_keep = [out_cell_col, out_time_col] + keep_vals
    # keep other useful metadata if present (optional, harmless)
    for extra in ("film_name", "field", "has", "offset", "bf_offset_frames", "cell_variant"):
        if extra in out.columns and extra not in cols_keep:
            cols_keep.append(extra)

    out = out[cols_keep].copy()

    # sort (helps heatmap ordering)
    out = out.sort_values([out_cell_col, out_time_col]).reset_index(drop=True)
    return out


def plot_shifted_gfp_heatmap_via_existing(
    aligned_csv: str,
    value_col: str = "cyt_int",
    clip_time: Optional[Tuple[float, float]] = (-40, 60),
    **kwargs,
):
    """
    Wrapper that reuses SingleCellDataAnalysis.visualization.plot_aligned_heatmaps
    by adapting columns from your aligned CSV.

    kwargs are forwarded to plot_aligned_heatmaps.
    """
    from SingleCellDataAnalysis.visualization import plot_aligned_heatmaps

    dfp = prepare_df_for_plot_aligned_heatmaps(
        aligned_csv,
        value_cols=(value_col,),
        out_time_col="aligned_time",
        out_cell_col="cell_id",
        drop_na_time=True,   # cells with has==0 in BF => time shift NA => dropped for plotting
    )

    # apply clip in aligned_time units (frames)
    if clip_time is not None:
        t0, t1 = clip_time
        dfp = dfp[(dfp["aligned_time"] >= t0) & (dfp["aligned_time"] <= t1)].copy()

    # Try the most common call patterns.
    # If your plot_aligned_heatmaps signature differs, adjust the keyword names here once.
    try:
        return plot_aligned_heatmaps(
            dfp,
            features=[value_col],
            time_col="aligned_time",
            cell_id_col="cell_id",
            **kwargs,
        )
    except TypeError:
        # fallback: some versions use slightly different keyword names
        return plot_aligned_heatmaps(
            dfp,
            feature_cols=[value_col],
            time_col="aligned_time",
            cell_col="cell_id",
            **kwargs,
        )
import numpy as np
import pandas as pd

from SingleCellDataAnalysis.visualization import plot_aligned_heatmaps


def plot_shifted_gfp_heatmap_oldstyle(
    aligned_csv,
    feature="cyt_int",
    aligned_time_col_candidates=("aligned_frame_rounded", "aligned_frame"),
    cell_col_candidates=("gfp_id_norm", "cell_id"),
    base_time_col="time_point",
    clip_time=None,                # e.g. (-40, 60) in aligned FRAME units
    include_unshifted=True,        # keep NaN-aligned cells? (will be dropped if no aligned time)
    title=None,
    cmap_list=None,
):
    """
    Old experiment behavior:
      - Use aligned_frame_rounded (int) as the plotting time axis (overwrite time_point)
      - Use integer cell ids (gfp_id_norm preferred)
      - best_shifts = 0 for all cells (we are NOT shifting via best_shifts)
      - call plot_aligned_heatmaps

    include_unshifted:
      - True  -> keep rows where aligned time is missing by falling back to base_time_col
      - False -> drop rows missing aligned time
    """

    df = pd.read_csv(aligned_csv)

    # pick cell id column
    cell_col = None
    for c in cell_col_candidates:
        if c in df.columns:
            cell_col = c
            break
    if cell_col is None:
        raise ValueError(f"No cell id col found. Tried: {cell_col_candidates}")

    # pick aligned time column
    tcol = None
    for c in aligned_time_col_candidates:
        if c in df.columns:
            tcol = c
            break
    if tcol is None:
        raise ValueError(f"No aligned time col found. Tried: {aligned_time_col_candidates}")

    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in CSV columns.")

    # Build plotting df in the same schema as old experiment: cell_id + time_point
    df_plot = df.copy()

    # cell_id must be int
    df_plot["cell_id"] = pd.to_numeric(df_plot[cell_col], errors="coerce").astype("Int64")

    # aligned time must be int (rounded)
    aligned = pd.to_numeric(df_plot[tcol], errors="coerce")

    if include_unshifted:
        # if aligned missing, fall back to original time_point
        base = pd.to_numeric(df_plot.get(base_time_col, np.nan), errors="coerce")
        use_time = aligned
        use_time = use_time.where(~use_time.isna(), base)
    else:
        use_time = aligned

    df_plot["time_point"] = pd.to_numeric(use_time, errors="coerce").round().astype("Int64")

    # Drop rows missing essential plotting keys
    df_plot = df_plot.dropna(subset=["cell_id", "time_point"]).copy()
    df_plot["cell_id"] = df_plot["cell_id"].astype(int)
    df_plot["time_point"] = df_plot["time_point"].astype(int)

    # Optional clip on aligned time axis
    if clip_time is not None:
        lo, hi = clip_time
        df_plot = df_plot[(df_plot["time_point"] >= int(lo)) & (df_plot["time_point"] <= int(hi))].copy()

    # Remove duplicates that can break alignment slicing assumptions
    df_plot = df_plot.sort_values(["cell_id", "time_point"])
    df_plot = df_plot.drop_duplicates(subset=["cell_id", "time_point"], keep="last")

    # Build global timeline
    global_time = np.sort(df_plot["time_point"].unique())
    time_points = global_time

    # Choose cells that still have data (important after clipping)
    cell_ids = df_plot["cell_id"].drop_duplicates().tolist()

    # old script: shifts are all zero (must be int, not float)
    best_shifts = {int(cid): 0 for cid in cell_ids}

    if title is None:
        title = f"Shifted heatmap: {feature}"

    return plot_aligned_heatmaps(
        df_plot,
        cell_ids,
        best_shifts,
        global_time,
        time_points,
        features=[feature],
        cmap_list=cmap_list,
        title=title,
    )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def plot_gfp1_gfp2_split_by_bf1_heatmap(
    aligned_csv_gfp1: str,
    aligned_csv_gfp2: str,
    feature: str = "cyt_int",
    # these are in "aligned frame units" (whatever your aligner wrote)
    clip_time: Tuple[int, int] = (-120, 120),
    gap: int = 10,  # blank columns between GFP1 and GFP2
    sort_by: str = "bf_offset_frames",  # if present
    title: Optional[str] = None,
):
    """
    Build a split heatmap: GFP1 (left) and GFP2 (right), same cells (rows),
    aligned to the SAME BF1 anchor.

    Requires the aligned CSVs produced by run_field_sequence():
      - aligned__<GFP1>__ANCHOR__<BF1>.csv
      - aligned__<GFP2>__ANCHOR__<BF1>.csv

    We match rows by BF cell id (a column like bf_id / bf_cell_id / anchor_bf_id).
    """

    df1 = pd.read_csv(aligned_csv_gfp1)
    df2 = pd.read_csv(aligned_csv_gfp2)

    # Identify columns robustly
    time_col_1 = _pick_col(df1, ["aligned_frame_rounded", "aligned_frame", "time_point"])
    time_col_2 = _pick_col(df2, ["aligned_frame_rounded", "aligned_frame", "time_point"])
    if time_col_1 is None or time_col_2 is None:
        raise ValueError("Could not find an aligned time column in one of the CSVs.")

    # BF-id column: used to define "same cell"
    bf_col_1 = _pick_col(df1, ["bf_id", "bf_cell_id", "anchor_bf_id", "bf_cell"])
    bf_col_2 = _pick_col(df2, ["bf_id", "bf_cell_id", "anchor_bf_id", "bf_cell"])
    if bf_col_1 is None or bf_col_2 is None:
        raise ValueError(
            "Could not find BF id column in aligned CSVs. "
            "Expected one of: bf_id / bf_cell_id / anchor_bf_id."
        )

    # Ensure numeric aligned time for slicing
    df1[time_col_1] = pd.to_numeric(df1[time_col_1], errors="coerce")
    df2[time_col_2] = pd.to_numeric(df2[time_col_2], errors="coerce")

    # Filter to the plotting window
    tmin, tmax = clip_time
    df1w = df1[(df1[time_col_1] >= tmin) & (df1[time_col_1] <= tmax)].copy()
    df2w = df2[(df2[time_col_2] >= tmin) & (df2[time_col_2] <= tmax)].copy()

    if feature not in df1w.columns or feature not in df2w.columns:
        raise ValueError(f"Feature '{feature}' not found in both CSVs.")

    # Determine common BF cells (rows)
    bf_ids_1 = set(pd.to_numeric(df1w[bf_col_1], errors="coerce").dropna().astype(int).tolist())
    bf_ids_2 = set(pd.to_numeric(df2w[bf_col_2], errors="coerce").dropna().astype(int).tolist())
    common_bf = sorted(bf_ids_1.intersection(bf_ids_2))
    if not common_bf:
        raise ValueError("No common BF ids found between GFP1 and GFP2 aligned tables.")

    # Optional ordering by BF offset if available (more meaningful than arbitrary BF id sorting)
    if sort_by in df1.columns:
        tmp = (
            df1[[bf_col_1, sort_by]]
            .copy()
        )
        tmp[bf_col_1] = pd.to_numeric(tmp[bf_col_1], errors="coerce")
        tmp[sort_by] = pd.to_numeric(tmp[sort_by], errors="coerce")
        tmp = tmp.dropna(subset=[bf_col_1]).drop_duplicates(subset=[bf_col_1])
        tmp = tmp[tmp[bf_col_1].astype(int).isin(common_bf)]
        # cells without sort_by go to end
        sort_map = {int(r[bf_col_1]): r[sort_by] for _, r in tmp.iterrows()}
        common_bf = sorted(common_bf, key=lambda x: (np.inf if pd.isna(sort_map.get(x, np.inf)) else sort_map.get(x)))
    # else: keep common_bf sorted

    # Build fixed time grids
    grid = np.arange(tmin, tmax + 1, 1, dtype=int)
    L = len(grid)

    # Pivot each table into matrices: rows=BF cell, cols=aligned time
    def build_matrix(dfw: pd.DataFrame, bf_col: str, time_col: str) -> np.ndarray:
        out = np.full((len(common_bf), L), np.nan, dtype=float)
        # speed: index by bf id
        dfw2 = dfw.copy()
        dfw2[bf_col] = pd.to_numeric(dfw2[bf_col], errors="coerce")
        dfw2 = dfw2.dropna(subset=[bf_col, time_col])
        dfw2[bf_col] = dfw2[bf_col].astype(int)
        # fill
        for i, bf in enumerate(common_bf):
            d = dfw2[dfw2[bf_col] == bf]
            if d.empty:
                continue
            # if multiple rows per time (rare), take mean
            s = d.groupby(time_col)[feature].mean()
            # map times to indices
            times = pd.to_numeric(s.index, errors="coerce").dropna().astype(int).values
            vals = s.values
            mask = (times >= tmin) & (times <= tmax)
            times = times[mask]
            vals = vals[mask]
            out[i, times - tmin] = vals
        return out

    M1 = build_matrix(df1w, bf_col_1, time_col_1)  # GFP1
    M2 = build_matrix(df2w, bf_col_2, time_col_2)  # GFP2

    # Concatenate with a blank gap in the middle
    gap_block = np.full((len(common_bf), int(gap)), np.nan, dtype=float)
    M = np.concatenate([M1, gap_block, M2], axis=1)

    # Plot
    if title is None:
        title = f"{feature} | GFP1 (left)  ||  GFP2 (right)  | aligned to BF1 anchor"

    fig, ax = plt.subplots(figsize=(14, max(3.0, len(common_bf) * 0.06)))

    im = ax.imshow(M, aspect="auto", interpolation="nearest")
    ax.set_title(title)

    # Mark the split
    split_x = L + gap / 2.0
    ax.axvline(x=L - 0.5, linewidth=2)              # boundary at end of GFP1
    ax.axvline(x=L + gap - 0.5, linewidth=2)        # boundary at end of gap

    # X ticks: show a few aligned-time ticks on each side
    # left segment
    left_ticks = np.linspace(0, L - 1, 5).astype(int)
    left_labels = (tmin + left_ticks).astype(int)

    # right segment
    right_ticks = (L + gap) + np.linspace(0, L - 1, 5).astype(int)
    right_labels = (tmin + np.linspace(0, L - 1, 5).astype(int)).astype(int)

    xticks = np.concatenate([left_ticks, right_ticks])
    xlabels = np.concatenate([left_labels, right_labels])

    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)
    ax.set_xlabel("Aligned time (BF1 anchor = 0), GFP1 left / GFP2 right")
    ax.set_ylabel("Cells (BF1 ids; common between GFP1 & GFP2)")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.show()

    return {
        "bf_ids": common_bf,
        "matrix_gfp1": M1,
        "matrix_gfp2": M2,
        "matrix_concat": M,
        "grid": grid,
    }