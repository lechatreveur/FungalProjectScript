#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find each M160 cell's division on ONE continuous time course.

Replaces the per-film detector in `build_features_m160.division_event`, which
had two faults that together made it miss most divisions and produce incoherent
stages downstream.

**It only read fluorescence films.** Those cover 20 of every 40.6 minutes, so a
division falling in a brightfield block was invisible. `5_1_N1_F1_5_1_N1_BF4_F1_
cell_418` is the case in point: its area halves from 6868 to 3350 at t = 162.0
min, the final frame of BF4, and the old detector reported no division at all
for any of that cell's seven fluorescence films.

**It worked per film.** A drop spanning a film boundary — which is exactly where
a 20-minute blind gap puts it — could not be seen from either side.

The fix is to assemble one area series per global cell across every film it
appears in, brightfield and fluorescence, on the real acquisition clock from the
Imaris timestamps, and scan that. Brightfield areas are scaled by the empirical
BF:FL ratio so the two channels are comparable within a series.

Windows are specified in MINUTES, not frames, because the two channels sample at
different rates: fluorescence every 12 s, brightfield every 30 s.

The criterion is unchanged and still the one that matters: a division is a drop
that does NOT recover. A missegmentation bounces back.
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
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cell_cycle_stage_m160 as CC

WIN_MIN = 6.0         # minutes each side of a candidate
GAP_MIN = 1.0         # minutes skipped across the transition itself
AREA_MAX = 0.75       # the drop must reach this fraction of the preceding level
BOUNCE_MAX = 0.80     # and must not recover above this fraction
BF_FL_RATIO = 1.20    # measured on cells present in both channels


def cell_series(exp, quant, clock, bfmap, feat_rows, gid):
    """One area series per cell, both channels, on the real clock (minutes)."""
    t, a, ch = [], [], []
    for _, r in feat_rows.iterrows():
        if r.film not in clock:
            continue
        s = CC.cell_area_series(quant, r.film, int(r.local_cid))
        if s is None:
            continue
        st, per = clock[r.film]
        t.append(st + s.time_point.values.astype(float) * per)
        a.append(s.cell_area.values.astype(float))
        ch.append(np.full(len(s), "FL"))
    for film, lc in bfmap.get(gid, {}).items():
        if film not in clock:
            continue
        sb = CC.bf_area_series(exp, film, lc)
        if sb is None:
            continue
        st, per = clock[film]
        t.append(st + sb.time_point.values.astype(float) * per)
        # scale brightfield onto the fluorescence area scale
        a.append(sb.cell_area.values.astype(float) / BF_FL_RATIO)
        ch.append(np.full(len(sb), "BF"))
    if not t:
        return None
    t, a, ch = np.concatenate(t), np.concatenate(a), np.concatenate(ch)
    o = np.argsort(t)
    return t[o], a[o], ch[o]


def find_division(t, a):
    """The deepest sustained, non-recovering drop. -> dict or None."""
    best = None
    for i in range(len(t)):
        ti = t[i]
        pre = (t >= ti - WIN_MIN) & (t < ti)
        post = (t > ti + GAP_MIN) & (t <= ti + GAP_MIN + WIN_MIN)
        if pre.sum() < 5 or post.sum() < 5:
            continue
        a_pre, a_post = np.nanmedian(a[pre]), np.nanmedian(a[post])
        if not np.isfinite(a_pre) or a_pre <= 0:
            continue
        ratio = a_post / a_pre
        if ratio > AREA_MAX:
            continue
        # does it come back within the post window?
        bounce = np.nanmax(a[post]) / a_pre
        if bounce > BOUNCE_MAX:
            continue
        if best is None or ratio < best["ratio"]:
            best = dict(t_div=float(ti), ratio=float(ratio), bounce=float(bounce),
                        n_pre=int(pre.sum()), n_post=int(post.sum()))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=CC.DEFAULT_FEATURES)
    ap.add_argument("--quant", type=Path, default=CC.DEFAULT_QUANT)
    ap.add_argument("--exp", type=Path, default=CC.DEFAULT_EXP)
    ap.add_argument("--out", type=Path, default=CC.DEFAULT_OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    clock = CC.film_clock(a.exp)
    bfmap = CC.bf_local_ids(a.exp)
    feat = pd.read_csv(a.features)
    print(f"cells: {feat.global_cell_id.nunique()}   films on the clock: {len(clock)}",
          flush=True)

    rows = []
    for i, (gid, g) in enumerate(feat.groupby("global_cell_id"), 1):
        ser = cell_series(a.exp, a.quant, clock, bfmap, g, gid)
        if ser is None:
            continue
        t, ar, ch = ser
        d = find_division(t, ar)
        rec = dict(global_cell_id=gid, n_points=len(t),
                   t_start=float(t.min()), t_end=float(t.max()),
                   n_bf=int((ch == "BF").sum()), n_fl=int((ch == "FL").sum()))
        if d:
            # Which film was the division in? Restrict to the cell's OWN
            # sequence: the three fields are imaged within seconds of each
            # other, so searching all 39 films by start time alone picks an
            # arbitrary field and mislabels the division.
            seq = next((q for q in CC.SEQS if str(gid).startswith(q)), None)
            cand = [f for f in clock
                    if (seq is None or f.endswith(seq.split("_")[-1]))
                    and clock[f][0] <= d["t_div"]]
            film = max(cand, key=lambda f: clock[f][0]) if cand else ""
            rec.update(t_div=d["t_div"], div_ratio=round(d["ratio"], 4),
                       div_bounce=round(d["bounce"], 4), div_film=film,
                       has_division=True)
        else:
            rec.update(t_div=np.nan, div_ratio=np.nan, div_bounce=np.nan,
                       div_film="", has_division=False)
        rows.append(rec)
        if i % 200 == 0:
            print(f"  {i} cells scanned", flush=True)

    D = pd.DataFrame(rows)
    D.to_csv(a.out / "division_times_m160.csv", index=False)
    n = int(D.has_division.sum())
    print(f"\ncells with a division found: {n} of {len(D)} "
          f"({100*n/len(D):.0f}%)", flush=True)
    if n:
        inbf = D[D.has_division & D.div_film.str.contains("BF")]
        print(f"  of those, the division falls in a BRIGHTFIELD film: "
              f"{len(inbf)} ({100*len(inbf)/n:.0f}%)", flush=True)
        print(f"  division time: median {D[D.has_division].t_div.median():.0f} min, "
              f"range {D[D.has_division].t_div.min():.0f} to "
              f"{D[D.has_division].t_div.max():.0f}", flush=True)

    (a.out / "_provenance_division.json").write_text(json.dumps(dict(
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        created_by="SingleCellQuantificationHPC/division_detect_m160.py",
        window_min=WIN_MIN, gap_min=GAP_MIN, area_max=AREA_MAX,
        bounce_max=BOUNCE_MAX, bf_fl_ratio=BF_FL_RATIO,
        cells=int(len(D)), with_division=n), indent=2))


if __name__ == "__main__":
    main()
