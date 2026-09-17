#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cell-cycle stage for M160 by cell-size regression.

Method, following `SingleCellDataAnalysis/plot_area_vs_stage_global.py`
---------------------------------------------------------------------
1. **Anchor.** A cell with a detected division gets `stage = t - t_division`,
   so stage 0 is the division and growth runs at negative stage.
2. **Regress.** Fit area against stage over the PRE-division points of anchored
   cells only, `polyfit(stage, area, 1)`, reporting slope, intercept and r.
3. **Place the rest.** A cell with no division is slid along the stage axis
   until its mean area matches where the regression says it belongs:

       tau = mean_T - (mean_A - intercept) / slope
       stage = T - tau

Differences from the M156/Sept17 original
-----------------------------------------
The anchor is the bounce-validated division film and frame (see
`build_features_m160.division_event`), not a septum endpoint: M160's septum
alignment JSONs do not exist, and the fork in the linkage dates the division
badly — only 5% of fork films contain the area drop, against 73% of t_div films.

The stage axis is in REAL MINUTES, taken from the acquisition metadata rather
than assumed. Each film's `.ims` carries a timestamp per timepoint; M160's
fluorescence films run 101 frames at 12 s, 20.0 min each, and start every
~40.6 min because a ~20.6 min brightfield block sits between consecutive ones.
Treating the fluorescence films as contiguous — the obvious assumption — would
compress the axis by a factor of two for any cell spanning several films.

Both channels are used, as in the original. The fluorescence films cover only
20 of every 40.6 minutes; the brightfield films fill the gaps almost exactly —
BF1 starts 21 s after FL1 ends — so together they give continuous coverage from
10:41 to 15:04. Brightfield areas come from the canonical `cell_<id>_masks.csv`,
because those films were quantified for the septum pattern only and carry no
area column. BF segmentations run larger than FL, so the two channels are fitted
separately and the empirical ratio is reported.
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

from Cell_tracking_functions import rle_decode

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXP_NAME = "2026_08_28_M160"
_SSD = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking") / EXP_NAME
DEFAULT_FEATURES = _SSD / "features" / "umap_features_m160.csv"
DEFAULT_QUANT = _SSD / "quant"
DEFAULT_OUT = _SSD / "cell_cycle"
DEFAULT_EXP = Path("/Volumes/X10 Pro/Movies") / EXP_NAME
SEQS = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]
FRAMES_PER_FILM = 101


def film_order(exp_dir):
    linkage = json.load(open(Path(exp_dir) / "sequence_linkage.json"))
    order = {}
    for seq in SEQS:
        if seq in linkage:
            for k, f in enumerate([x for x in linkage[seq]["films"] if "FL" in x]):
                order[f] = k
    return order


def bf_local_ids(exp_dir):
    """global_cell_id -> {brightfield film: local id}, from the linkage."""
    linkage = json.load(open(Path(exp_dir) / "sequence_linkage.json"))
    out = {}
    for seq in SEQS:
        if seq not in linkage:
            continue
        films = linkage[seq]["films"]
        for gid, track in linkage[seq]["global_cells"].items():
            d = {}
            for i, f in enumerate(films):
                if "BF" not in f or i >= len(track):
                    continue
                if track[i] and track[i] > 0:
                    d[f] = int(track[i])
            if d:
                out[gid] = d
    return out


def bf_area_series(exp_dir, film, lc):
    """Per-frame area from the canonical brightfield masks.

    The brightfield films were run through `bf_pattern_only`, which records the
    septum pattern and no geometry, so there is no area column to read — the
    mask itself is the only source.
    """
    p = Path(exp_dir) / film / f"TrackedCells_{film}" / f"cell_{lc}_masks.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
    except Exception:
        return None
    if "rle_bf" not in d.columns or d.empty:
        return None
    H, W = int(d.iloc[0]["height"]), int(d.iloc[0]["width"])
    ts, ar = [], []
    for _, r in d.sort_values("time_point").iterrows():
        v = str(r.get("rle_bf", ""))
        if not v.strip() or v.lower() == "nan":
            continue
        try:
            m = np.asarray(rle_decode(v, (H, W)), bool)
        except Exception:
            continue
        a = int(m.sum())
        if a > 0:
            ts.append(int(r["time_point"]))
            ar.append(a)
    if not ts:
        return None
    return pd.DataFrame(dict(time_point=ts, cell_area=ar))


def film_clock(exp_dir):
    """film -> (start minutes relative to the earliest film, minutes per frame).

    Read from each film's Imaris file, which stores a timestamp per timepoint.
    Nothing here is assumed: M160's fluorescence films turn out to run 101
    frames at 12 s and to start every ~40.6 min, because a brightfield block of
    about the same length sits between consecutive ones.
    """
    import h5py

    def txt(attrs, key):
        v = attrs.get(key)
        if v is None:
            return ""
        try:
            return "".join(c.decode() for c in v)
        except Exception:
            return str(v)

    out = {}
    for f in sorted(Path(exp_dir).glob("5_1_N1_*F*.ims")):
        if ("FL" not in f.stem) and ("BF" not in f.stem):
            continue
        try:
            with h5py.File(f, "r") as h:
                ti = h["DataSetInfo"]["TimeInfo"].attrs
                n = int(txt(ti, "DatasetTimePoints") or 0)
                t1 = datetime.strptime(txt(ti, "TimePoint1"), "%Y-%m-%d %H:%M:%S.%f")
                tn = datetime.strptime(txt(ti, f"TimePoint{n}"), "%Y-%m-%d %H:%M:%S.%f")
        except Exception as exc:
            print(f"  ({f.name}: {type(exc).__name__}, skipped)", flush=True)
            continue
        per = (tn - t1).total_seconds() / max(n - 1, 1) / 60.0
        out[f.stem] = (t1, per)
    if not out:
        return {}
    t0 = min(v[0] for v in out.values())
    return {k: ((v[0] - t0).total_seconds() / 60.0, v[1]) for k, v in out.items()}


def cell_area_series(quant, film, lc):
    """Per-frame area of the primary object."""
    p = Path(quant) / film / f"cell_{lc}.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, usecols=["cell_id", "time_point", "cell_area"])
    d = d[d.cell_id.astype(str) == str(lc)].sort_values("time_point")
    return d if len(d) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--quant", type=Path, default=DEFAULT_QUANT)
    ap.add_argument("--exp", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    order = film_order(a.exp)
    clock = film_clock(a.exp)
    if not clock:
        raise SystemExit("no .ims timing found; the stage axis needs real times")
    per = np.median([v[1] for v in clock.values()])
    starts = sorted({round(v[0], 1) for v in clock.values()})
    print(f"film clock: {len(clock)} films, {per*60:.1f} s/frame, "
          f"film starts at {starts[:4]} ... min", flush=True)

    def gtime(film, frames):
        """absolute minutes for these frames of this film"""
        st, pm = clock[film]
        return st + np.asarray(frames, float) * pm

    bfmap = bf_local_ids(a.exp)
    print(f"cells with brightfield local ids: {len(bfmap)}", flush=True)

    feat = pd.read_csv(a.features)
    feat["ord"] = feat.film.map(order)
    feat = feat[feat.film.isin(clock)]

    # ---- 1. anchor: global division time per cell -------------------------
    anchors = {}
    for gid, g in feat.groupby("global_cell_id"):
        d = g[g.is_division_film.fillna(False) & g.div_frame.notna()]
        src = "bounce"
        if d.empty:
            d = g[g.t_div.notna()]
            src = "t_div"
        if d.empty:
            continue
        r = d.iloc[0]
        frame = float(r.div_frame if src == "bounce" else r.t_div)
        anchors[gid] = (float(gtime(r.film, [frame])[0]), src)
    print(f"anchored cells: {len(anchors)} "
          f"(bounce {sum(1 for v in anchors.values() if v[1]=='bounce')}, "
          f"t_div {sum(1 for v in anchors.values() if v[1]=='t_div')})", flush=True)

    # ---- 2. points for anchored cells --------------------------------------
    rows = []
    for _, r in feat.iterrows():
        gid = r.global_cell_id
        if gid not in anchors or pd.isna(r["ord"]):
            continue
        s = cell_area_series(a.quant, r.film, int(r.local_cid))
        if s is None:
            continue
        t = gtime(r.film, s.time_point.values)
        rows.append(pd.DataFrame(dict(
            gid=gid, film=r.film, stage=t - anchors[gid][0],
            area=s.cell_area.values.astype(float), group="anchored",
            channel="FL")))
    # brightfield films of the same anchored cells, filling the 20-minute gaps
    n_bf = 0
    for gid in anchors:
        for film, lc in bfmap.get(gid, {}).items():
            if film not in clock:
                continue
            s_bf = bf_area_series(a.exp, film, lc)
            if s_bf is None:
                continue
            t = gtime(film, s_bf.time_point.values)
            rows.append(pd.DataFrame(dict(
                gid=gid, film=film, stage=t - anchors[gid][0],
                area=s_bf.cell_area.values.astype(float), group="anchored",
                channel="BF")))
            n_bf += 1
    print(f"brightfield series added for anchored cells: {n_bf}", flush=True)

    if not rows:
        raise SystemExit("no anchored points")
    A = pd.concat(rows, ignore_index=True)
    if "channel" not in A.columns:
        A["channel"] = "FL"
    A["channel"] = A["channel"].fillna("FL")
    A = A[np.isfinite(A.area) & (A.area > 0)]
    print(f"anchored points: {len(A)}", flush=True)

    # ---- 3. regression on PRE-division points ------------------------------
    pre = A[A.stage <= 0]
    fits = {}
    for ch, g in pre.groupby("channel"):
        if len(g) < 100:
            continue
        mm, cc = np.polyfit(g.stage.values, g.area.values, 1)
        r_ = float(np.corrcoef(g.stage.values, g.area.values)[0, 1])
        fits[ch] = (mm, cc, r_, len(g))
        print(f"pre-division fit [{ch}]: area = {mm:+.2f}*stage + {cc:.1f}   "
              f"r = {r_:+.3f}   n = {len(g)}", flush=True)
    # empirical channel scale, from the same cells measured in both
    both = pre.groupby(["gid", "channel"]).area.median().unstack()
    if {"BF", "FL"} <= set(both.columns):
        ratio = (both["BF"] / both["FL"]).median()
        print(f"empirical BF:FL area ratio (same cells): {ratio:.3f}  "
              f"n = {both.dropna().shape[0]} cells", flush=True)
    else:
        ratio = float("nan")
    # Place unanchored cells using the BRIGHTFIELD fit. FL area measures the
    # GFP-positive region, which shrinks 25% approaching division as signal
    # concentrates, so it is the wrong quantity for size: its median trend has
    # r = -0.119 against BF's +0.775.
    if "BF" not in fits:
        raise SystemExit("no brightfield fit; cannot place unanchored cells")
    m, c, rr, _ = fits["BF"]

    # ---- 4. place the unanchored cells --------------------------------------
    rows2, n_un, tau_by_gid = [], 0, {}
    for gid, g in feat.groupby("global_cell_id"):
        if gid in anchors:
            continue
        T, Ar = [], []
        for film, lc in bfmap.get(gid, {}).items():
            if film not in clock:
                continue
            sb = bf_area_series(a.exp, film, lc)
            if sb is None:
                continue
            T.append(gtime(film, sb.time_point.values))
            Ar.append(sb.cell_area.values.astype(float))
        if not T:
            continue
        T, Ar = np.concatenate(T), np.concatenate(Ar)
        ok = np.isfinite(Ar) & (Ar > 0)
        if ok.sum() < 20 or abs(m) < 1e-9:
            continue
        tau = T[ok].mean() - (Ar[ok].mean() - c) / m
        tau_by_gid[gid] = float(tau)
        rows2.append(pd.DataFrame(dict(gid=gid, film="", stage=T[ok] - tau,
                                       area=Ar[ok], group="aligned")))
        n_un += 1
    U = pd.concat(rows2, ignore_index=True) if rows2 else A.iloc[:0]
    print(f"aligned cells without a division: {n_un}", flush=True)

    # ---- 4b. one stage per DATAPOINT (cell x fluorescence film) -----------
    # The explorer colours datapoints, so each needs a single stage: the value
    # at its film's midpoint. Cells whose division was actually detected are
    # marked curated; the rest are estimated by the size regression and should
    # be read with that in mind.
    taus = {gid: t for gid, t in tau_by_gid.items()}
    dp = []
    for _, r in feat.iterrows():
        gid = r.global_cell_id
        mid = float(gtime(r.film, [50])[0])
        if gid in anchors:
            dp.append(dict(cell_id=r.cell_id, global_cell_id=gid, film=r.film,
                           stage_min=round(mid - anchors[gid][0], 2),
                           stage_source="curated", anchor=anchors[gid][1]))
        elif gid in taus:
            dp.append(dict(cell_id=r.cell_id, global_cell_id=gid, film=r.film,
                           stage_min=round(mid - taus[gid], 2),
                           stage_source="estimated", anchor=""))
    DP = pd.DataFrame(dp)
    DP.to_csv(a.out / "cell_cycle_stage_by_datapoint.csv", index=False)
    print(f"per-datapoint stages: {len(DP)}  "
          f"curated {int((DP.stage_source=='curated').sum())}  "
          f"estimated {int((DP.stage_source=='estimated').sum())}", flush=True)

    allpts = pd.concat([A, U], ignore_index=True)
    allpts.to_csv(a.out / "cell_cycle_stage_points.csv", index=False)

    # ---- 5. plot -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True)

    for ax, ch in zip(axes, ("BF", "FL")):
        g = A[A.channel == ch]
        if g.empty or ch not in fits:
            ax.set_title(f"{ch}: no data")
            continue
        mm, cc, r_, n_ = fits[ch]
        gpre, gpost = g[g.stage <= 0], g[g.stage > 0]
        ax.scatter(gpre.stage, gpre.area, s=2.5, c="#2563eb", alpha=0.35,
                   label=f"pre-division ({len(gpre)} pts)", rasterized=True)
        ax.scatter(gpost.stage, gpost.area, s=2.5, c="#ef4444", alpha=0.35,
                   label=f"post-division ({len(gpost)} pts)", rasterized=True)
        xs = np.linspace(gpre.stage.min(), 0, 100)
        ax.plot(xs, mm * xs + cc, color="#0f172a", lw=2.5,
                label=f"fit: r = {r_:+.3f}, slope = {mm:+.2f} px/min")
        # median trend in 20-minute bins, which is what the eye should follow
        b = g.assign(bin=(g.stage // 20) * 20).groupby("bin").area.median()
        ax.plot(b.index + 10, b.values, color="#f59e0b", lw=2.2, marker="o",
                ms=3.5, label="median per 20 min")
        ax.axvline(0, color="#64748b", ls="--", lw=1.2)
        ax.set_title(f"{ch} channel" + ("  (whole cell body)" if ch == "BF"
                                        else "  (GFP-positive region)"))
        ax.set_xlabel("cell-cycle stage (minutes relative to division)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    axes[0].set_ylabel("cell area (px)")
    fig.suptitle(f"M160 cell-cycle stage by size regression — {len(anchors)} anchored cells, "
                 f"both channels (BF:FL area ratio {ratio:.2f})")
    fig.tight_layout()
    png = a.out / "cell_cycle_stage_m160.png"
    fig.savefig(png, dpi=150)
    print(f"plot -> {png}", flush=True)

    (a.out / "_provenance.json").write_text(json.dumps(dict(
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        created_by="SingleCellQuantificationHPC/cell_cycle_stage_m160.py",
        experiment=EXP_NAME, method="area vs stage regression",
        anchored_cells=len(anchors), aligned_cells=n_un,
        pre_division_points=int(len(pre)), slope_px_per_frame=float(m),
        intercept_px=float(c), pearson_r=rr,
        fits_by_channel={k: dict(slope=v[0], intercept=v[1], r=v[2], n=v[3])
                         for k, v in fits.items()},
        bf_fl_area_ratio=(None if not np.isfinite(ratio) else float(ratio)),
        minutes_per_frame=float(per),
        timing_source="per-film Imaris timestamps (DataSetInfo/TimeInfo)",
    ), indent=2))


if __name__ == "__main__":
    main()
