#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cell-cycle stage per datapoint, derived from ONE division time per cell.

Replaces the per-datapoint regression in `cell_cycle_regress_m160.py`, which
predicted each film's stage independently and so ignored the one thing we know
exactly: the acquisition clock. Consecutive fluorescence films are 40.7 minutes
apart, always. The old output did not respect that — for
`5_1_N1_F1_5_1_N1_BF4_F1_cell_418` it put FL5, FL6 and FL7 all at about -19 min
despite their being 40 minutes apart, and called every film pre-division when
the cell had in fact divided at 162.0 min, in BF4.

The design here
---------------
A cell has ONE division time. Every datapoint's stage follows from it:

    stage(film) = film_midpoint - t_division

so consecutive films are exactly 40.7 minutes apart by construction, and the
incoherence above cannot recur.

For cells whose division was detected on the continuous two-channel series
(`division_detect_m160.py`), `t_division` is measured and the datapoints are
marked **curated**.

For the rest it is predicted. Each datapoint votes: a ridge regression maps that
film's summary quantities to the offset `t_division - film_midpoint`, giving one
estimate of the cell's division time per film. The cell's `t_division` is the
median of its votes, which both uses every film and yields a single coherent
answer. Those datapoints are marked **estimated**.
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
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cell_cycle_stage_m160 as CC
from cell_cycle_regress_m160 import summarise


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=CC.DEFAULT_FEATURES)
    ap.add_argument("--quant", type=Path, default=CC.DEFAULT_QUANT)
    ap.add_argument("--exp", type=Path, default=CC.DEFAULT_EXP)
    ap.add_argument("--out", type=Path, default=CC.DEFAULT_OUT)
    a = ap.parse_args()

    clock = CC.film_clock(a.exp)
    bfmap = CC.bf_local_ids(a.exp)
    feat = pd.read_csv(a.features)
    div = pd.read_csv(a.out / "division_times_m160.csv")
    tdiv = dict(zip(div[div.has_division].global_cell_id,
                    div[div.has_division].t_div))
    print(f"cells with a measured division: {len(tdiv)} of "
          f"{feat.global_cell_id.nunique()}", flush=True)

    # brightfield area per cell, as a feature
    bf_area = {}
    for gid, films in bfmap.items():
        v = [float(np.nanmedian(s.cell_area.values))
             for f, lc in films.items() if f in clock
             for s in [CC.bf_area_series(a.exp, f, lc)] if s is not None]
        if v:
            bf_area[gid] = float(np.nanmedian(v))

    rows = []
    for _, r in feat.iterrows():
        if r.film not in clock:
            continue
        sm = summarise(a.quant, r.film, int(r.local_cid))
        if sm is None:
            continue
        st, per = clock[r.film]
        mid = st + 50 * per
        sm["bf_area_med"] = bf_area.get(r.global_cell_id, np.nan)
        sm.update(dict(cell_id=r.cell_id, gid=r.global_cell_id, film=r.film,
                       mid=mid))
        rows.append(sm)
    D = pd.DataFrame(rows)
    D = D.dropna(axis=1, thresh=int(0.8 * len(D))).dropna()
    D["t_div_true"] = D.gid.map(tdiv)
    # target: how far AHEAD of this film's midpoint the division sits
    D["offset"] = D.t_div_true - D.mid
    print(f"datapoints: {len(D)}   with a measured division: "
          f"{int(D.offset.notna().sum())}", flush=True)

    Xcols = [c for c in D.columns if c not in
             ("cell_id", "gid", "film", "mid", "t_div_true", "offset")]
    tr = D[D.offset.notna()]
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))

    # honest check: hold out whole cells, then score the CELL-level division time
    groups = tr.gid.values
    cv = GroupKFold(n_splits=min(5, len(set(groups))))
    pred = np.full(len(tr), np.nan)
    for itr, ite in cv.split(tr[Xcols].values, tr.offset.values, groups):
        model.fit(tr[Xcols].values[itr], tr.offset.values[itr])
        pred[ite] = model.predict(tr[Xcols].values[ite])
    tr = tr.assign(pred_offset=pred, pred_tdiv=tr.mid.values + pred)
    per_cell = tr.groupby("gid").agg(true=("t_div_true", "first"),
                                     est=("pred_tdiv", "median"))
    err = (per_cell.est - per_cell.true).abs()
    ss = 1 - np.sum((per_cell.est - per_cell.true) ** 2) / \
        np.sum((per_cell.true - per_cell.true.mean()) ** 2)
    print(f"\nheld-out CELL division time: MAE {err.median():.1f} min (median), "
          f"{err.mean():.1f} mean;  r2 {ss:+.3f}", flush=True)
    print(f"  within one film (20.3 min): {100*(err <= 20.3).mean():.0f}% of cells",
          flush=True)
    print(f"  within two films (40.7 min): {100*(err <= 40.7).mean():.0f}%", flush=True)

    # fit on everything, then one division time per cell
    model.fit(tr[Xcols].values, tr.offset.values)
    D["pred_tdiv"] = D.mid.values + model.predict(D[Xcols].values)
    est = D.groupby("gid").pred_tdiv.median()
    t_final = {g: tdiv.get(g, est.get(g, np.nan)) for g in D.gid.unique()}

    D["t_div_cell"] = D.gid.map(t_final)
    D["stage_min"] = (D.mid - D.t_div_cell).round(2)
    D["stage_source"] = np.where(D.gid.isin(tdiv), "curated", "estimated")
    out = D[["cell_id", "gid", "film", "stage_min", "stage_source",
             "t_div_cell"]].rename(columns={"gid": "global_cell_id"})
    out.to_csv(a.out / "cell_cycle_stage_by_datapoint.csv", index=False)
    n_c = int((out.stage_source == "curated").sum())
    print(f"\nwrote {len(out)} datapoints: {n_c} curated, {len(out)-n_c} estimated",
          flush=True)

    # coherence check: consecutive films of a cell must differ by ~40.7 min
    chk = out.merge(feat[["cell_id", "film"]], on=["cell_id", "film"])
    gaps = []
    for gid, g in out.groupby("global_cell_id"):
        g = g.assign(start=g.film.map(lambda f: clock[f][0])).sort_values("start")
        if len(g) > 1:
            gaps += list(np.diff(g.stage_min.values))
    gaps = np.array(gaps)
    print(f"consecutive-film stage gaps: median {np.median(gaps):.1f} min, "
          f"sd {gaps.std():.3f}  (should be 40.7 with no spread)", flush=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(per_cell.true, per_cell.est, s=18, c="#2563eb", alpha=0.6)
    lo, hi = per_cell.true.min(), per_cell.true.max()
    ax.plot([lo, hi], [lo, hi], color="#0f172a", ls="--", lw=1.5)
    ax.fill_between([lo, hi], [lo - 20.3, hi - 20.3], [lo + 20.3, hi + 20.3],
                    color="#94a3b8", alpha=0.18, label="within one film")
    ax.set_xlabel("measured division time (min)")
    ax.set_ylabel("predicted division time (min, held-out cells)")
    ax.set_title(f"M160 cell division time: MAE {err.median():.0f} min, "
                 f"{100*(err <= 20.3).mean():.0f}% within one film")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out / "division_time_prediction_m160.png", dpi=150)
    print(f"plot -> {a.out / 'division_time_prediction_m160.png'}", flush=True)

    (a.out / "_provenance_stage.json").write_text(json.dumps(dict(
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        created_by="SingleCellQuantificationHPC/cell_cycle_stage_build.py",
        design="one division time per cell; stage = film midpoint - t_division",
        measured_cells=len(tdiv), datapoints=int(len(out)),
        curated_datapoints=n_c,
        heldout_cell_mae_min=float(err.median()),
        heldout_cell_r2=float(ss),
        within_one_film_pct=float(100 * (err <= 20.3).mean()),
        features=Xcols), indent=2))


if __name__ == "__main__":
    main()
