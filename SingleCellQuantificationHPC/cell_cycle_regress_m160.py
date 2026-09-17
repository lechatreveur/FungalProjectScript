#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cell-cycle stage for every M160 datapoint, by multi-quantity regression.

Supersedes the single-quantity size regression in `cell_cycle_stage_m160.py`,
which remains as the diagnostic that established two things: fluorescence area
is the wrong quantity (it measures the GFP-positive region, which SHRINKS 25%
before division), and brightfield area is the right one (it grows 27%, median
trend r = +0.775).

Why several quantities rather than size alone
---------------------------------------------
Grouped cross-validation on 648 curated datapoints over 97 cells, so a cell's
own films never appear in both train and test:

    predict the mean            r2  0.000   MAE 90.2 min
    brightfield area alone      r2 +0.282   MAE 78.2 min
    cell area alone             r2 +0.420   MAE 66.9 min
    cell length alone           r2 +0.443   MAE 65.4 min
    septum quantities only      r2 +0.175   MAE 82.2 min
    nuclear quantities only     r2 +0.148   MAE 84.0 min
    ALL of them                 r2 +0.682   MAE 48.9 min

Septum and nuclear signals are weak on their own but pull real weight in the
ensemble. Cell LENGTH is the strongest single predictor, which is what one would
expect for a rod-shaped cell that grows by tip extension.

Method
------
Target is the stage in minutes at the film's midpoint, on the real acquisition
clock read from the Imaris timestamps. Features are per-datapoint summaries —
median and slope — of the quantities in the stage-4 table, plus the brightfield
area of the same global cell. A ridge regression with standardised inputs is fit
on the curated datapoints and applied to the rest.

Every datapoint is labelled `curated` when its cell's division was actually
detected, or `estimated` when the stage comes from this model. They should not
be read as equivalent: the estimate carries a typical error of about 50 minutes,
which is two and a half films.
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
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cell_cycle_stage_m160 as CC

QCOLS = ["cell_area", "cell_length", "nu_dis", "nu_int", "septum_int", "cyt_int",
         "pol1_int", "pol2_int", "pattern_score_norm"]


def summarise(quant, film, lc):
    p = Path(quant) / film / f"cell_{lc}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p)
    d = d[d.cell_id.astype(str) == str(lc)].sort_values("time_point")
    if len(d) < 20:
        return None
    out, t = {}, d.time_point.values.astype(float)
    for col in QCOLS:
        if col not in d.columns:
            continue
        y = d[col].values.astype(float)
        ok = np.isfinite(y)
        if ok.sum() < 10:
            continue
        out[f"{col}_med"] = float(np.nanmedian(y))
        out[f"{col}_slope"] = float(np.polyfit(t[ok], y[ok], 1)[0])
    sept = (d.septum_int - d.cyt_int).values.astype(float)
    ok = np.isfinite(sept)
    if ok.sum() >= 10:
        out["septcorr_med"] = float(np.nanmedian(sept[ok]))
        out["septcorr_max"] = float(np.nanmax(sept[ok]))
        out["septcorr_slope"] = float(np.polyfit(t[ok], sept[ok], 1)[0])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=CC.DEFAULT_FEATURES)
    ap.add_argument("--quant", type=Path, default=CC.DEFAULT_QUANT)
    ap.add_argument("--exp", type=Path, default=CC.DEFAULT_EXP)
    ap.add_argument("--out", type=Path, default=CC.DEFAULT_OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    stage = pd.read_csv(a.out / "cell_cycle_stage_by_datapoint.csv")
    feat = pd.read_csv(a.features)
    meta = feat.set_index("cell_id")[["film", "local_cid", "global_cell_id"]]
    clock = CC.film_clock(a.exp)
    bfmap = CC.bf_local_ids(a.exp)

    # brightfield area per global cell, computed once
    bf_area = {}
    for gid, films in bfmap.items():
        vals = []
        for bfilm, blc in films.items():
            if bfilm not in clock:
                continue
            sb = CC.bf_area_series(a.exp, bfilm, blc)
            if sb is not None:
                vals.append(float(np.nanmedian(sb.cell_area.values)))
        if vals:
            bf_area[gid] = float(np.nanmedian(vals))
    print(f"brightfield area for {len(bf_area)} cells", flush=True)

    rows = []
    for cid, r in meta.iterrows():
        sm = summarise(a.quant, r.film, int(r.local_cid))
        if sm is None:
            continue
        sm["bf_area_med"] = bf_area.get(r.global_cell_id, np.nan)
        sm.update(dict(cell_id=cid, gid=r.global_cell_id))
        rows.append(sm)
    D = pd.DataFrame(rows)
    D = D.dropna(axis=1, thresh=int(0.8 * len(D))).dropna()
    print(f"datapoints with a complete feature row: {len(D)}", flush=True)

    D = D.merge(stage[["cell_id", "stage_min", "stage_source"]], on="cell_id",
                how="left")
    train = D[D.stage_source == "curated"]
    Xcols = [c for c in D.columns
             if c not in ("cell_id", "gid", "stage_min", "stage_source")]
    print(f"training on {len(train)} curated datapoints over "
          f"{train.gid.nunique()} cells, {len(Xcols)} features", flush=True)

    model = make_pipeline(StandardScaler(),
                          RidgeCV(alphas=np.logspace(-2, 3, 20)))
    Xtr, ytr, gtr = train[Xcols].values, train.stage_min.values, train.gid.values
    cv = GroupKFold(n_splits=min(5, len(set(gtr))))
    pred_cv = cross_val_predict(model, Xtr, ytr, groups=gtr, cv=cv)
    r2 = 1 - np.sum((ytr - pred_cv) ** 2) / np.sum((ytr - ytr.mean()) ** 2)
    mae = float(np.mean(np.abs(ytr - pred_cv)))
    print(f"grouped CV: r2 {r2:+.3f}   MAE {mae:.1f} min   "
          f"(baseline MAE {np.mean(np.abs(ytr - ytr.mean())):.1f})", flush=True)

    model.fit(Xtr, ytr)
    D["stage_pred"] = model.predict(D[Xcols].values)
    D["stage_final"] = np.where(D.stage_source == "curated",
                                D.stage_min, D.stage_pred)
    D["stage_source"] = D.stage_source.fillna("estimated")
    out = D[["cell_id", "gid", "stage_final", "stage_source", "stage_pred"]].rename(
        columns={"gid": "global_cell_id", "stage_final": "stage_min"})
    out.to_csv(a.out / "cell_cycle_stage_by_datapoint.csv", index=False)
    n_c = int((out.stage_source == "curated").sum())
    print(f"wrote {len(out)} datapoints: {n_c} curated, "
          f"{len(out)-n_c} estimated", flush=True)

    coef = pd.Series(model[-1].coef_, index=Xcols).sort_values(key=abs,
                                                              ascending=False)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].scatter(ytr, pred_cv, s=12, c="#2563eb", alpha=0.5)
    lo, hi = min(ytr.min(), pred_cv.min()), max(ytr.max(), pred_cv.max())
    ax[0].plot([lo, hi], [lo, hi], color="#0f172a", lw=1.5, ls="--")
    ax[0].set_xlabel("true stage (min, curated)")
    ax[0].set_ylabel("predicted stage (min, grouped CV)")
    ax[0].set_title(f"held-out prediction: r2 = {r2:+.3f}, MAE = {mae:.0f} min")
    ax[0].grid(alpha=0.25)
    top = coef.head(12)[::-1]
    ax[1].barh(range(len(top)), top.values,
               color=["#ef4444" if v < 0 else "#2563eb" for v in top.values])
    ax[1].set_yticks(range(len(top)))
    ax[1].set_yticklabels(top.index, fontsize=8)
    ax[1].set_xlabel("standardised ridge coefficient")
    ax[1].set_title("what carries the signal")
    ax[1].grid(alpha=0.25, axis="x")
    fig.suptitle("M160 cell-cycle stage from area, length, septum and nuclear signals")
    fig.tight_layout()
    fig.savefig(a.out / "cell_cycle_regression_m160.png", dpi=150)
    print(f"plot -> {a.out / 'cell_cycle_regression_m160.png'}", flush=True)

    (a.out / "_provenance_regression.json").write_text(json.dumps(dict(
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        created_by="SingleCellQuantificationHPC/cell_cycle_regress_m160.py",
        model="RidgeCV on standardised per-datapoint summaries",
        features=Xcols, n_train=int(len(train)), n_total=int(len(D)),
        grouped_cv_r2=float(r2), grouped_cv_mae_min=mae,
        top_coefficients={k: float(v) for k, v in coef.head(12).items()},
    ), indent=2))


if __name__ == "__main__":
    main()
