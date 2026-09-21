#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feature extraction for M161 model-based dense tracking (stage 5).

Copy-to-modify variant of build_features_m160.py (P15): experiment-dated
scripts are frozen records, so the M160 original is left untouched and
only the experiment constants change here.

M161 is the replicate control for M162 - same strain, medium and
acquisition settings (101 x 12 s, 350/120 ms, laser 2 at intensity 5),
different session. The two are compared directly, so this is held
identical to build_features_m162.py apart from paths and sequence names.

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
      -> SingleCellDataAnalysis.PCA_utils.load_experiment_features
         (per-pole a / mid / v, re-ordered so pol1 is the brighter pole, plus
          NC_score, Periodicity, a1a2, d, dd — the eleven columns the
          autoencoder loader reads)

What a datapoint is
-------------------
**One cell, in ONE film, over exactly 101 frames.** The reference trajectory
loader keys on experiment + global cell + source and hard-skips any trace whose
length is not exactly 101, so a global cell followed across several consecutive
films is several datapoints, not one. Concatenating a cell's films produces a
trace of the wrong length that is silently dropped downstream.

Identity (P12)
--------------
`<global_cell_id>__<film>`, with `global_cell_id` from `sequence_linkage.json`
and the film standing in for the reference's `source`. `local_cell_id` is
carried alongside. Not `new_cell_id` row numbers, which P12 forbids as an
identity.

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
from SingleCellDataAnalysis.PCA_utils import load_experiment_features
import lineage_m160 as LIN

EXP_NAME = "2026_09_03_M161"
# Filename tag derived from EXP_NAME, never hard-coded: the M160 original
# spelled "m160" into its output names, and a copy that kept that would
# write M160-named tables into this experiment's folder.
EXP_TAG = EXP_NAME.rsplit("_", 1)[-1].lower()
DEFAULT_EXP = Path("/Volumes/X10 Pro/Movies") / EXP_NAME
_SSD_OUT = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking") / EXP_NAME
DEFAULT_QUANT = _SSD_OUT / "quant"
DEFAULT_OUT = _SSD_OUT / "features"
# Listed explicitly, never matched by substring: M161's NeonG_YES_1_
# prefix is one underscore-delimited token longer than M162's
# NeonG_YES_, so a substring test silently cross-matches the two.
SEQS = [f"NeonG_YES_1_F{i}" for i in range(4)]


FRAMES_PER_FILM = 101   # FL films run t = 0..100


def datapoint_id(gid, film):
    """One datapoint is one cell in ONE film over its 101 frames.

    This mirrors the reference loader, which keys on experiment + global cell +
    source and hard-skips any trace whose length is not exactly 101. A global
    cell followed across several films is several datapoints, not one: the films
    are consecutive acquisition blocks and concatenating them produces a trace of
    the wrong length that the loader drops outright.
    """
    return f"{gid}__{film}"


def build_id_map(exp_dir):
    """(film, local_cell_id) -> global_cell_id, from sequence_linkage.json (P12)."""
    linkage = json.load(open(Path(exp_dir) / "sequence_linkage.json"))
    out, order = {}, {}
    for seq in SEQS:
        if seq not in linkage:
            continue
        films = linkage[seq]["films"]
        for k, film in enumerate([f for f in films if "FL" in f]):
            order[film] = k
        for gid, locals_ in linkage[seq]["global_cells"].items():
            for film, lc in zip(films, locals_):
                if lc and lc > 0:
                    out[(film, int(lc))] = gid
    return out, order


PRE, POST, GAP = 12, 12, 2      # frames each side of a candidate drop
BOUNCE_MAX = 0.70               # a division does not recover; a missegmentation does
AREA_MAX = 0.75                 # the drop itself must be real


def division_event(prim):
    """Does this film contain a TRUE division, and at which frame?

    A candidate is a frame where the median area over the following 12 frames
    falls to <= AREA_MAX of the preceding 12. What separates a division from a
    missegmentation is not the depth of that drop — 0.53 against 0.59, barely
    anything — but whether the area comes back. A missegmentation recovers; a
    division does not. Measured on 208 known dividing films against 919
    negatives, `bounce <= 0.70` gives precision 0.89 at recall 0.62, against
    0.77 for the old area-only criterion.

    nu_dis and septum are deliberately NOT used. Both collapse in either class,
    because a missegmentation also loses part of the object, so they separate
    mother-state from daughter-state but not division from artefact.
    """
    d = prim.sort_values("time_point")
    t = d["time_point"].values.astype(float)
    area = d["cell_area"].values.astype(float)
    best = None
    for i in range(PRE, len(t) - POST):
        a_pre = np.nanmedian(area[i - PRE:i])
        a_post = np.nanmedian(area[i + GAP:i + GAP + POST])
        if not np.isfinite(a_pre) or a_pre <= 0:
            continue
        ratio = a_post / a_pre
        if ratio > AREA_MAX:
            continue
        window = area[i + GAP:i + GAP + POST]
        bounce = np.nanmax(window) / a_pre if np.any(np.isfinite(window)) else np.nan
        if best is None or ratio < best["area_ratio"]:
            best = dict(t=float(t[i]), area_ratio=float(ratio), bounce=float(bounce))
    if best is None:
        return dict(div_frame=np.nan, div_area_ratio=np.nan,
                    div_bounce=np.nan, is_division_film=False)
    return dict(div_frame=best["t"], div_area_ratio=round(best["area_ratio"], 4),
                div_bounce=round(best["bounce"], 4),
                is_division_film=bool(best["bounce"] <= BOUNCE_MAX))


def load_stacked(quant_dir, id_map, film_order, point_seg=None):
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
        prim["cell_id"] = datapoint_id(gid, film)
        prim["global_cell_id"] = gid
        prim["source"] = film
        prim["film"] = film
        prim["local_cid"] = lc
        # local time IS the time axis: one datapoint is one film's 101 frames
        prim = prim.sort_values("time_point")

        n = len(prim)
        n_model = int(prim["model_only"].fillna(False).astype(bool).sum())
        seg = (point_seg or {}).get((film, lc), {})
        meta.append(dict(
            cell_id=datapoint_id(gid, film), global_cell_id=gid,
            segment_id=seg.get("id"), segment_parent=seg.get("parent"),
            segment_depth=seg.get("depth"),
            **division_event(prim),
            film=film, local_cid=lc, n_frames=n,
            n_model_only=n_model,
            model_only_pct=round(100.0 * n_model / n, 2) if n else np.nan,
            n_stage3_good=int(prim["stage3_good"].fillna(False).astype(bool).sum()),
            stage3_good_pct=round(
                100.0 * prim["stage3_good"].fillna(False).astype(bool).mean(), 2) if n else np.nan,
            t_div=(prim["stage3_t_div"].dropna().iloc[0]
                   if prim["stage3_t_div"].notna().any() else np.nan),
        ))
        rows.append(prim[["time_point", "pol1_int_corr", "pol2_int_corr",
                          "septum_int_corr", "cell_id", "global_cell_id",
                          "source", "film", "local_cid"]])
    if not rows:
        raise SystemExit("no quantified cells found")
    stacked = pd.concat(rows, ignore_index=True).sort_values(["cell_id", "time_point"])
    meta = pd.DataFrame(meta)

    # A trace that is not exactly 101 frames is dropped by the downstream
    # trajectory loader, so flag it here rather than letting it vanish silently.
    short = meta[meta.n_frames != FRAMES_PER_FILM]
    if len(short):
        print(f"WARNING: {len(short)} datapoints are not {FRAMES_PER_FILM} frames "
              f"and will be dropped downstream; lengths "
              f"{sorted(short.n_frames.unique())[:6]}", flush=True)
    return stacked, meta, unmapped


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--quant", type=Path, default=DEFAULT_QUANT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--work-queue", type=Path, default=None,
                    help="optional gid filter; M160 had a scratch work "
                         "queue, this experiment has none, so the default "
                         "keeps every gid the linkage lists")
    ap.add_argument("--status", nargs="+",
                    default=["good", "corrected", "unreviewed"])
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    id_map, film_order = build_id_map(a.exp)
    print(f"id map: {len(id_map)} (film, local_cell_id) -> global_cell_id; "
          f"{len(film_order)} FL films ordered", flush=True)

    # Lineage: a mother forks into daughters, so each (film, cell) belongs to
    # exactly one segment of the tree rather than to a global_cell_id that may
    # be shared with a sister (see lineage_m160).
    wq = pd.read_csv(a.work_queue) if a.work_queue and Path(a.work_queue).exists() else None
    keep = set(wq[wq.status.isin(a.status)].gid) if wq is not None else None
    pmap, segs, edges, notes = LIN.resolve_experiment(
        a.exp, keep_gids=keep, seqs=SEQS)
    point_seg = {pt: dict(id=sid, parent=segs[sid]["parent"], depth=segs[sid]["depth"])
                 for pt, sid in pmap.items()}
    print(f"lineage: {len(segs)} segments, {len(edges)} edges, "
          f"{sum(1 for s in segs.values() if s['depth'] > 0)} daughters", flush=True)
    if notes.get("excluded_gids"):
        print(f"  excluded (re-converging tracks): {len(notes['excluded_gids'])} gids",
              flush=True)

    stacked, meta, unmapped = load_stacked(a.quant, id_map, film_order, point_seg)
    if a.limit:
        keep = meta.cell_id.unique()[:a.limit]
        meta = meta[meta.cell_id.isin(keep)]
        stacked = stacked[stacked.cell_id.isin(keep)]
    dup = stacked.groupby(["cell_id", "time_point"]).size()
    print(f"duplicate (datapoint, time) pairs: {int((dup > 1).sum())}", flush=True)
    n101 = int((meta.n_frames == FRAMES_PER_FILM).sum())
    print(f"datapoints with exactly {FRAMES_PER_FILM} frames: {n101} of {len(meta)}", flush=True)
    print(f"cells: {meta.cell_id.nunique()}  rows: {len(stacked)}  "
          f"unmapped cells skipped: {unmapped}", flush=True)
    print(f"model-only frames: {meta.n_model_only.sum()} of {meta.n_frames.sum()} "
          f"({100*meta.n_model_only.sum()/max(meta.n_frames.sum(),1):.2f}%)", flush=True)

    # load_experiment_features() looks for its three inputs under
    # <exp_dir>/unaligned_pairs_quant/, and for the stacked file it looks ONLY
    # there. Write the layout it expects rather than modifying it (P15).
    quant_out = a.out / "unaligned_pairs_quant"
    quant_out.mkdir(parents=True, exist_ok=True)
    stacked_path = quant_out / "stacked_gfp1_gfp2_for_unaligned_pairs.csv"
    stacked.to_csv(stacked_path, index=False)

    ids = sorted(stacked.cell_id.unique())
    fits_path = quant_out / "model_fits_by_cell.csv"
    acor_path = quant_out / "acor_detrended_results.csv"
    print("fitting trend / oscillation models ...", flush=True)
    quantify_all_cells(stacked, ids, feature1="pol1_int_corr", feature2="pol2_int_corr",
                       filename=str(fits_path))
    print("computing detrended autocorrelation ...", flush=True)
    quantify_all_cells_acor(stacked, ids, feature1="pol1_int_corr", feature2="pol2_int_corr",
                            filename=str(acor_path))

    # The eleven features the autoencoder reads come from load_experiment_features,
    # NOT from clustering.cluster_cells_by_amplitude_and_delay. The latter is a
    # clustering routine that builds a similar row but emits only amplitude and
    # midline per pole; it has no 'v' term and returns weight-normalised values.
    feats = load_experiment_features(str(a.out))
    feats = feats.reset_index().rename(columns={"index": "cell_id"})
    if "cell_id" not in feats.columns:
        feats = feats.rename(columns={feats.columns[0]: "cell_id"})
    want = ["pol1_a", "pol1_mid", "pol1_v", "pol2_a", "pol2_mid", "pol2_v",
            "NC_score", "Periodicity", "a1a2", "d", "dd"]
    missing = [c for c in want if c not in feats.columns]
    if missing:
        raise SystemExit(f"feature assembly is missing columns: {missing}")

    out = feats.merge(meta, on="cell_id", how="left")
    features_path = a.out / f"umap_features_{EXP_TAG}.csv"
    out.to_csv(features_path, index=False)
    meta.to_csv(a.out / f"cell_provenance_{EXP_TAG}.csv", index=False)

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
