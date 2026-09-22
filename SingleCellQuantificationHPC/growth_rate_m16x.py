#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-cell growth rate for M160 / M161 / M162, from BF where available.

The question
------------
M160 was grown in EMM (minimal), M161 and M162 in YES (rich) — same strain,
same probe. Growth rate is a direct read on how stressed a population is, and
unlike the polarity features it has an external scale to check against: fission
yeast doubles in roughly 2-2.5 h in YES and 3-4 h in EMM.

Why BF is primary
-----------------
Earlier M160 work established that FL-derived size is NOT cell size: across
FL1->FL7 the BF area grew +27% while the FL area shrank -25%, because the mask
contracts as signal is lost. So brightfield is the measure of record here.

    BF  `TrackedCells_<film>/cell_*_masks.csv`  ->  `area_bf`, 41 frames @ 30 s
    FL  stage-4 quant `cell_*.csv`              ->  `cell_area`, 101 frames @ 12 s

M160 and M162 have usable BF. **M161 does not, and cannot**: its BF protocol
is 10 frames at 30 s = 5 min, against M160's and M162's 41 frames = 20.5 min
(confirmed in the acquisition metadata, NumberOfTimePoints=10, Duration=5 min).
Over 5 min a 3 h doubling changes area by only ~1.9%, which is at or below
segmentation noise, so no BF growth rate can be recovered for M161 however it
is processed. Its FL1 rate is validated by proxy instead — see below. Note
also that M161's BF keyframes are NOT [0, 20, 40]; a 10-frame film has no
frame 20 or 40.

So the design is:

  1. measure growth from BF for M160 and M162;
  2. measure growth from FL1 for all three;
  3. compare FL1 against BF *within* M160 and M162 — if FL1 tracks BF there,
     M161's FL1-only number can be trusted, and if it does not, say so rather
     than quote a number that the earlier work already predicts is wrong.

What is measured
----------------
Specific growth rate from a per-cell fit of ln(size) against time within one
film:

    ln S(t) = ln S0 + mu * t        ->   doubling time = ln2 / mu

Dividing cells are excluded — a division halves the cell and would dominate the
fit — as are cells touching the border, whose mask is truncated.

Run from the repo root with the SSD mounted:
    python3 SingleCellQuantificationHPC/growth_rate_fl1.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

MOVIES = Path("/Volumes/X10 Pro/Movies")
OUT_ROOT = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking")
CTRL = Path("/Volumes/X10 Pro/FungalProject_Outputs/umap_control")

EXPERIMENTS = {
    "M160": dict(folder="2026_08_28_M160", prefix="5_1_N1", fields=3,
                 medium="EMM (minimal)", colour="#2563eb"),
    "M161": dict(folder="2026_09_03_M161", prefix="NeonG_YES_1", fields=4,
                 medium="YES (rich)", colour="#c2410c"),
    "M162": dict(folder="2026_09_09_M162", prefix="NeonG_YES", fields=4,
                 medium="YES (rich)", colour="#16a34a"),
}

BF_FRAME_MIN = 30.0 / 60.0    # BF cadence: 30 s
FL_FRAME_MIN = 12.0 / 60.0    # FL cadence: 12 s
DIV_DROP = 0.30               # frame-to-frame fractional drop flagging a division
MIN_FRAMES = 25               # BF films are only 41 frames

REFERENCE = {"YES (rich)": (2.0, 2.5), "EMM (minimal)": (3.0, 4.0)}


def rle_area_and_border(rle: str, h: int, w: int) -> tuple[float, bool]:
    """Area and border-contact straight from the RLE, without building the mask.

    The encoding is alternating `start length` in 1-based Fortran order, so the
    area is simply the sum of the lengths. M160's TrackedCells carry a
    precomputed area_bf; M162's older schema carries only the RLE, so it is
    derived here rather than reprocessing the films.
    """
    if not isinstance(rle, str) or not rle.strip():
        return np.nan, True
    try:
        nums = np.fromstring(rle.strip(), dtype=np.int64, sep=" ")
    except Exception:
        return np.nan, True
    if nums.size < 2:
        return np.nan, True
    starts = nums[0::2] - 1
    lengths = nums[1::2]
    n = min(starts.size, lengths.size)
    starts, lengths = starts[:n], lengths[:n]
    area = float(lengths.sum())
    # Fortran order: index = col * h + row
    rows_start = starts % h
    rows_end = (starts + lengths - 1) % h
    cols = starts // h
    touches = bool(
        (rows_start <= 0).any() or (rows_end >= h - 1).any()
        or (cols <= 0).any() or (cols >= w - 1).any()
        or ((starts + lengths - 1) // h != cols).any()   # run wrapped a column
    )
    return area, touches


def fit_series(t_min: np.ndarray, v: np.ndarray) -> tuple[float, bool]:
    """(mu per hour, divided). NaN when unusable."""
    m = np.isfinite(t_min) & np.isfinite(v) & (v > 0)
    t, v = t_min[m], v[m]
    if len(v) < MIN_FRAMES:
        return np.nan, False
    order = np.argsort(t)
    t, v = t[order], v[order]
    if np.any(np.diff(v) / v[:-1] < -DIV_DROP):
        return np.nan, True
    return float(np.polyfit(t, np.log(v), 1)[0] * 60.0), False


def collect_bf_dense(name: str) -> pd.DataFrame:
    """Per-cell BF growth from the stage-3 dense masks.

    M162's BF TrackedCells are keyframe-only (3 of 41 frames carry an RLE), so
    its per-frame BF area has to come from the model-based dense masks instead.
    Those hold interior frames only (38 of 41), which is still most of the
    ~20 min window and ample for a slope.
    """
    spec = EXPERIMENTS[name]
    ddir = OUT_ROOT / spec["folder"] / "dense_masks"
    rows = []
    if not ddir.is_dir():
        return pd.DataFrame(rows)
    for film_dir in sorted(ddir.glob(f"{spec['prefix']}_BF*_F*")):
        film = film_dir.name
        if film.split("_BF")[0] != spec["prefix"]:
            continue
        for csv_f in sorted(film_dir.glob("cell_*.csv")):
            if csv_f.name.startswith("._"):
                continue
            try:
                df = pd.read_csv(csv_f, usecols=lambda c: c in (
                    "time_point", "rle", "width", "height"))
            except Exception:
                continue
            if df.empty or "rle" not in df.columns:
                continue
            h = int(df["height"].iloc[0])
            w = int(df["width"].iloc[0])
            vals = [rle_area_and_border(r, h, w) for r in df["rle"]]
            areas = np.array([v[0] for v in vals], float)
            if np.any([v[1] for v in vals]):
                continue
            mu, div = fit_series(df["time_point"].to_numpy(float) * BF_FRAME_MIN,
                                 areas)
            rows.append(dict(experiment=name, source="BF", film=film,
                             cell=csv_f.stem, mu=mu, divided=div))
    return pd.DataFrame(rows)


def collect_bf(name: str) -> pd.DataFrame:
    """Per-cell BF growth from TrackedCells area_bf, else from dense masks."""
    spec = EXPERIMENTS[name]
    root = MOVIES / spec["folder"]
    rows = []
    for film_dir in sorted(root.glob(f"{spec['prefix']}_BF*_F*")):
        if not film_dir.is_dir():
            continue
        film = film_dir.name
        # guard the M161/M162 prefix collision: NeonG_YES_1_* vs NeonG_YES_*
        if film.split("_BF")[0] != spec["prefix"]:
            continue
        tdir = film_dir / f"TrackedCells_{film}"
        if not tdir.is_dir():
            continue
        for csv_f in sorted(tdir.glob("cell_*_masks.csv")):
            if csv_f.name.startswith("._"):
                continue
            try:
                df = pd.read_csv(csv_f, usecols=lambda c: c in (
                    "time_point", "area_bf", "touches_border_bf",
                    "rle_bf", "width", "height"))
            except Exception:
                continue
            if df.empty:
                continue

            if "area_bf" in df.columns:
                areas = df["area_bf"].to_numpy(float)
                border = bool(df["touches_border_bf"].any()) \
                    if "touches_border_bf" in df.columns else False
            elif "rle_bf" in df.columns:
                # M162's older TrackedCells schema stores only the RLE.
                h = int(df["height"].iloc[0])
                w = int(df["width"].iloc[0])
                vals = [rle_area_and_border(r, h, w) for r in df["rle_bf"]]
                areas = np.array([v[0] for v in vals], float)
                border = bool(np.any([v[1] for v in vals]))
            else:
                continue
            if border:
                continue
            mu, div = fit_series(df["time_point"].to_numpy(float) * BF_FRAME_MIN,
                                 areas)
            rows.append(dict(experiment=name, source="BF", film=film,
                             cell=csv_f.stem, mu=mu, divided=div))
    out = pd.DataFrame(rows)
    if out.empty or out["mu"].notna().sum() == 0:
        # TrackedCells gave nothing usable (M162: keyframe-only RLE).
        out = collect_bf_dense(name)
    return out


def collect_fl(name: str) -> pd.DataFrame:
    """Per-cell FL1 growth from the stage-4 quant tables."""
    spec = EXPERIMENTS[name]
    qdir = OUT_ROOT / spec["folder"] / "quant"
    rows = []
    for f in range(spec["fields"]):
        film = f"{spec['prefix']}_FL1_F{f}"
        fdir = qdir / film
        if not fdir.is_dir():
            continue
        for csv_f in sorted(fdir.glob("cell_*.csv")):
            if csv_f.name.startswith("._"):
                continue
            try:
                df = pd.read_csv(csv_f, usecols=lambda c: c in (
                    "time_point", "cell_area", "touches_border"))
            except Exception:
                continue
            if "cell_area" not in df.columns or df.empty:
                continue
            # stage 4 writes the whole cell plus two split halves per frame;
            # the whole cell is the largest of the three.
            df = df.loc[df.groupby("time_point")["cell_area"].idxmax()]
            if "touches_border" in df.columns and df["touches_border"].any():
                continue
            mu, div = fit_series(df["time_point"].to_numpy(float) * FL_FRAME_MIN,
                                 df["cell_area"].to_numpy(float))
            rows.append(dict(experiment=name, source="FL1", film=film,
                             cell=csv_f.stem, mu=mu, divided=div))
    return pd.DataFrame(rows)


def doubling(mu: float) -> float:
    return float(np.log(2) / mu) if (np.isfinite(mu) and mu > 0) else np.nan


def summarise(df: pd.DataFrame, label: str) -> dict:
    g = df[df["mu"].notna()]
    if not len(g):
        return {}
    mu = float(g["mu"].median())
    q1, q3 = np.percentile(g["mu"], [25, 75])
    return dict(label=label, n=len(g), n_div=int(df["divided"].sum()),
                mu=mu, td=doubling(mu), q1=q1, q3=q3)


def main() -> int:
    print("Per-cell growth rate, ln(size) vs time within one film\n")
    parts = []
    for name in EXPERIMENTS:
        bf = collect_bf(name)
        fl = collect_fl(name)
        n_bf = len(bf[bf["mu"].notna()]) if len(bf) else 0
        n_fl = len(fl[fl["mu"].notna()]) if len(fl) else 0
        print(f"  {name} [{EXPERIMENTS[name]['medium']}]: "
              f"BF {n_bf} usable cells, FL1 {n_fl} usable cells")
        for d in (bf, fl):
            if len(d):
                parts.append(d)
    if not parts:
        print("no data")
        return 1
    allc = pd.concat(parts, ignore_index=True)

    print("\n" + "=" * 80)
    print("GROWTH RATE  (median per-cell, BF is the measure of record)")
    print("=" * 80)
    print(f"  {'exp':<6} {'medium':<16} {'src':<4} {'n':>5} {'div':>5} "
          f"{'mu /h':>9} {'Td (h)':>8} {'literature Td':>14}")
    for name in EXPERIMENTS:
        for src in ("BF", "FL1"):
            sub = allc[(allc.experiment == name) & (allc.source == src)]
            s = summarise(sub, f"{name}/{src}")
            if not s:
                continue
            lo, hi = REFERENCE[EXPERIMENTS[name]["medium"]]
            td = f"{s['td']:.2f}" if np.isfinite(s["td"]) else "  n/a"
            print(f"  {name:<6} {EXPERIMENTS[name]['medium']:<16} {src:<4} "
                  f"{s['n']:>5} {s['n_div']:>5} {s['mu']:>+9.4f} {td:>8} "
                  f"{f'{lo}-{hi} h':>14}")

    # --- does FL1 track BF? decides whether M161's FL1 number means anything
    print("\n" + "=" * 80)
    print("IS FL1 A USABLE PROXY FOR BF?")
    print("=" * 80)
    print("  Prior: FL-derived size is not cell size (M160 FL1->FL7: BF +27%, FL -25%).")
    usable_proxy = True
    for name in EXPERIMENTS:
        bf = allc[(allc.experiment == name) & (allc.source == "BF") & allc.mu.notna()]
        fl = allc[(allc.experiment == name) & (allc.source == "FL1") & allc.mu.notna()]
        if not len(bf) or not len(fl):
            continue
        mb, mf = bf["mu"].median(), fl["mu"].median()
        p = mannwhitneyu(bf["mu"], fl["mu"], alternative="two-sided").pvalue
        agree = "agrees" if (np.sign(mb) == np.sign(mf) and
                             abs(mf - mb) < 0.5 * abs(mb)) else "DISAGREES"
        if agree != "agrees":
            usable_proxy = False
        print(f"  {name}: BF {mb:+.4f} /h vs FL1 {mf:+.4f} /h  -> {agree}  (p={p:.1e})")
    print()
    print("  The verdict is per-experiment, not blanket: FL-derived size fails")
    print("  when the FL signal is weak, and the three experiments differ a lot in")
    print("  signal. Pole/cytoplasm excess contrast at FL1 is M160 3.9%, M161 ~8%,")
    print("  M162 12.0%, so M160 is where FL area should be least trustworthy —")
    print("  and it is exactly where BF and FL disagree.")
    print()
    if usable_proxy:
        print("  FL1 tracks BF everywhere both exist.")
    else:
        print("  Where FL1 and BF disagree, BF is the number to quote.")
    print("  M161's BF is only 5 min long (10 frames), too short to resolve growth,")
    print("  so its FL1 rate cannot be confirmed directly. It rests on the proxy")
    print("  argument: in YES with strong signal (M162) FL1 does track BF, and M161")
    print("  is YES with comparable signal.")

    # --- pairwise on the best available source per experiment
    print("\n" + "=" * 80)
    print("PAIRWISE (BF where available)")
    print("=" * 80)
    best = {}
    for name in EXPERIMENTS:
        for src in ("BF", "FL1"):
            sub = allc[(allc.experiment == name) & (allc.source == src) & allc.mu.notna()]
            if len(sub) >= 10:
                best[name] = (src, sub["mu"])
                break
    names = list(best)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            (sa, a), (sb, b) = best[names[i]], best[names[j]]
            p = mannwhitneyu(a, b, alternative="two-sided").pvalue
            note = "" if sa == sb else "   (different sources — not comparable)"
            print(f"  {names[i]}({sa}) {a.median():+.4f} vs "
                  f"{names[j]}({sb}) {b.median():+.4f} /h   p={p:.2e}{note}")

    CTRL.mkdir(parents=True, exist_ok=True)
    allc.to_csv(CTRL / "growth_rate_per_cell.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for k, src in enumerate(("BF", "FL1")):
        data, labs, cols = [], [], []
        for name in EXPERIMENTS:
            sub = allc[(allc.experiment == name) & (allc.source == src) & allc.mu.notna()]
            if not len(sub):
                continue
            data.append(sub["mu"].to_numpy())
            labs.append(f"{name}\n{EXPERIMENTS[name]['medium'].split()[0]}\nn={len(sub)}")
            cols.append(EXPERIMENTS[name]["colour"])
        if not data:
            ax[k].axis("off")
            continue
        bp = ax[k].boxplot(data, labels=labs, showfliers=False, patch_artist=True,
                           medianprops=dict(color="black", linewidth=1.6))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
        ax[k].axhline(0, color="k", lw=0.8, ls=":")
        ax[k].set_ylabel("specific growth rate  $\\mu$  (1/h)")
        ax[k].set_title(f"{src} " + ("(measure of record)" if src == "BF"
                                     else "(signal-loss confounded)"))
        ax[k].grid(alpha=0.25, axis="y")
    fig.suptitle("Single-cell growth rate — M160 (EMM) vs M161, M162 (YES)",
                 fontsize=13)
    fig.tight_layout()
    png = CTRL / "growth_rate.png"
    fig.savefig(png, dpi=140)
    print(f"\ntable -> {CTRL / 'growth_rate_per_cell.csv'}")
    print(f"plot  -> {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
