#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature extraction for M160 model-based dense tracking (stage 5).

Consumes the stage-4 tables written by `quantify_model_based_dense.py` and
produces the per-cell feature vector the downstream autoencoder and UMAP
explorer read.

Chain (canonical modules, P15)
------------------------------
    stage-4 quant CSV
      -> keep primary-object rows only (drop the two split rows per frame)
      -> cytoplasm-correct:  pol{1,2}_int_corr = pol{1,2}_int - cyt_int
      -> SingleCellDataAnalysis.signal_analysis.quantify_all_cells      (fits)
      -> SingleCellDataAnalysis.signal_cor.quantify_all_cells_acor      (acor)
      -> SingleCellDataAnalysis.clustering.cluster_cells_by_amplitude_and_delay
         (per-pole amplitude and midline, re-ordered so pol1 is the brighter
          pole, plus Periodicity, NC_score, a1a2, d, dd)

Identity (P12)
--------------
Cells are keyed by **global_cell_id** from `sequence_linkage.json`, for example
`5_1_N1_F0_cell_50`, with `(film, local_cell_id)` carried alongside. The M156
build used `new_cell_id` row numbers, which P12 forbids as an identity: "Never
key colour on transient UI indices ... or `new_cell_id` row numbers."

Model-only frames
-----------------
M160 is the first experiment tracked with the model-based method, so some
frames carry a mask inferred from the shape model rather than from the
segmentation. Those frames ARE included in the fits — the intensity is measured
from the best inferred segment — and the per-cell share is recorded as
`model_only_pct` so any result can be filtered or weighted by it afterwards.

Output is scratch-free and experiment-scoped: it goes to the M160 outputs folder
on the SSD (P4), never into the M156 tables.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from SingleCellDataAnalysis.signal_analysis import quantify_all_cells
from SingleCellDataAnalysis.signal_cor import quantify_all_cells_acor
from SingleCellDataAnalysis.clustering import cluster_cells_by_amplitude_and_delay

EXP_NAME = "2026_08_28_M160"
DEFAULT_EXP = Path("/Volumes/X10 Pro/Movies") / EXP_NAME
_SSD_OUT = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking") / EXP_NAME
DEFAULT_QUANT = _SSD_OUT / "quant"
DEFAULT_OUT = _SSD_OUT / "features"
SEQS = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]


FRAMES_PER_FILM = 101   # FL films run t = 0..100


def build_id_map(exp_dir):
    """(film, local_cell_id) -> global_cell_id, from sequence_linkage.json (P12).

    Also returns film -> ordinal among the FLUORESCENCE films of its sequence.
    A sequence's films are CONSECUTIVE acquisition blocks, each restarting at
    t = 0, so a global cell seen in FL1 and FL4 has two separate stretches of
    its life on the same 0..100 axis. Keying on global_cell_id without an
    offset silently stacks them into one series with duplicate time points —
    which is exactly what happened on the first run here.
    """
    linkage = json.load(open(Path(exp_dir) / "sequence_linkage.json"))
    out, order = {}, {}
    for seq in SEQS:
        if seq not in linkage:
            continue
        films = linkage[seq]["films"]
        fl_films = [f for f in films if "FL" in f]
        for k, film in enumerate(fl_films):
            order[film] = k
        for gid, locals_ in linkage[seq]["global_cells"].items():
            for film, lc in zip(films, locals_):
                if lc and lc > 0:
                    out[(film, int(lc))] = gid
    return out, order


def load_stacked(quant_dir, id_map, film_order):
    """Primary-object rows for every quantified cell, cytoplasm-corrected, on a
    sequence-continuous time axis."""
    rows, meta, unmapped = [], [], 0
    for f in sorted(Path(quant_dir).glob("*/cell_*.csv")):
        df = pd.read_csv(f)
        if df.empty:
            continue
        film = str(df.iloc[0]["film"])
        lc = int(df.iloc[0]["local_cid"])
        gid = id_map.get((film, lc))
        if gid is None:
            unmapped += 1
            continue
        k = film_order.get(film)
        if k is None:
            unmapped += 1
            continue
        # Primary (non-split) object only: cell_id is the bare local id, while
        # the two halves carry <lc>_1 and <lc>_2.
        prim = df[df["cell_id"].astype(str) == str(lc)].copy()
        if prim.empty:
            continue
        prim["pol1_int_corr"] = prim["pol1_int"] - prim["cyt_int"]
        prim["pol2_int_corr"] = prim["pol2_int"] - prim["cyt_int"]
        prim["septum_int_corr"] = prim["septum_int"] - prim["cyt_int"]
        prim["cell_id"] = gid
        prim["film"] = film
        prim["local_cid"] = lc
        prim["local_time"] = prim["time_point"]
        prim["time_point"] = k * FRAMES_PER_FILM + prim["local_time"]

        n = len(prim)
        n_model = int(prim["model_only"].fillna(False).astype(bool).sum())
        meta.append(dict(
            cell_id=gid, film=film, local_cid=lc, n_frames=n,
            n_model_only=n_model,
            model_only_pct=round(100.0 * n_model / n, 2) if n else np.nan,
            n_stage3_good=int(prim["stage3_good"].fillna(False).astype(bool).sum()),
            stage3_good_pct=round(
                100.0 * prim["stage3_good"].fillna(False).astype(bool).mean(), 2) if n else np.nan,
            t_div=(prim["stage3_t_div"].dropna().iloc[0]
                   if prim["stage3_t_div"].notna().any() else np.nan),
        ))
        rows.append(prim[["time_point", "local_time", "pol1_int_corr", "pol2_int_corr",
                          "septum_int_corr", "cell_id", "film", "local_cid"]])
    if not rows:
        raise SystemExit("no quantified cells found")
    stacked = pd.concat(rows, ignore_index=True).sort_values(["cell_id", "time_point"])

    # One feature row per global cell, so per-cell provenance is aggregated over
    # whichever films that cell was seen in.
    per_film = pd.DataFrame(meta)
    agg = (per_film.groupby("cell_id")
           .agg(n_films=("film", "nunique"),
                films=("film", lambda s: "|".join(sorted(set(s)))),
                local_cids=("local_cid", lambda s: "|".join(str(v) for v in sorted(set(s)))),
                n_frames=("n_frames", "sum"),
                n_model_only=("n_model_only", "sum"),
                n_stage3_good=("n_stage3_good", "sum"))
           .reset_index())
    agg["model_only_pct"] = (100.0 * agg.n_model_only / agg.n_frames).round(2)
    agg["stage3_good_pct"] = (100.0 * agg.n_stage3_good / agg.n_frames).round(2)
    return stacked, agg, per_film, unmapped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--quant", type=Path, default=DEFAULT_QUANT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    id_map, film_order = build_id_map(a.exp)
    print(f"id map: {len(id_map)} (film, local_cell_id) -> global_cell_id; "
          f"{len(film_order)} FL films ordered", flush=True)

    stacked, meta, per_film, unmapped = load_stacked(a.quant, id_map, film_order)
    if a.limit:
        keep = meta.cell_id.unique()[:a.limit]
        meta = meta[meta.cell_id.isin(keep)]
        per_film = per_film[per_film.cell_id.isin(keep)]
        stacked = stacked[stacked.cell_id.isin(keep)]
    dup = stacked.groupby(["cell_id", "time_point"]).size()
    print(f"duplicate (cell, time) pairs after offsetting: {int((dup > 1).sum())}", flush=True)
    print(f"cells: {meta.cell_id.nunique()}  rows: {len(stacked)}  "
          f"unmapped cells skipped: {unmapped}", flush=True)
    print(f"model-only frames: {meta.n_model_only.sum()} of {meta.n_frames.sum()} "
          f"({100*meta.n_model_only.sum()/max(meta.n_frames.sum(),1):.2f}%)", flush=True)

    stacked_path = a.out / "stacked_pol_corr.csv"
    stacked.to_csv(stacked_path, index=False)

    ids = sorted(stacked.cell_id.unique())
    fits_path = a.out / "model_fits_by_cell.csv"
    acor_path = a.out / "acor_detrended_results.csv"
    print("fitting trend / oscillation models ...", flush=True)
    quantify_all_cells(stacked, ids, feature1="pol1_int_corr", feature2="pol2_int_corr",
                       filename=str(fits_path))
    print("computing detrended autocorrelation ...", flush=True)
    quantify_all_cells_acor(stacked, ids, feature1="pol1_int_corr", feature2="pol2_int_corr",
                            filename=str(acor_path))

    fits = pd.read_csv(fits_path)
    acor = pd.read_csv(acor_path)
    # cluster_cells_by_amplitude_and_delay reads the per-pole fit rows and the
    # per-cell acor columns from one frame.
    df_result = fits.merge(acor, on="cell_id", how="left", suffixes=("", "_acor"))
    feats, _, _ = cluster_cells_by_amplitude_and_delay(df_result, verbose=False)
    feats = feats.reset_index().rename(columns={"index": "cell_id"})

    out = feats.merge(meta, on="cell_id", how="left")
    features_path = a.out / "umap_features_m160.csv"
    out.to_csv(features_path, index=False)
    meta.to_csv(a.out / "cell_provenance_m160.csv", index=False)
    per_film.to_csv(a.out / "cell_provenance_per_film_m160.csv", index=False)

    prov = dict(
        artifact=features_path.name, created=stamp,
        created_by="SingleCellQuantificationHPC/build_features_m160.py",
        experiment=EXP_NAME, identity="global_cell_id (sequence_linkage.json, P12)",
        quant_source=str(a.quant),
        n_cells=int(out.cell_id.nunique()), n_frames=int(meta.n_frames.sum()),
        model_only_frames=int(meta.n_model_only.sum()),
        model_only_included_in_fits=True,
        feature_columns=[c for c in out.columns],
    )
    (a.out / "_provenance.json").write_text(json.dumps(prov, indent=2))

    print(f"\ncells with features: {out.cell_id.nunique()}", flush=True)
    print(f"features -> {features_path}", flush=True)
    print(f"provenance -> {a.out / '_provenance.json'}", flush=True)


if __name__ == "__main__":
    main()
