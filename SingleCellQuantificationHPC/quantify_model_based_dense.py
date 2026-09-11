#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4 — polarity-site quantification on model-based dense masks (P14).

Consumes the stage-3 output of `run_model_based_dense_tracking.py` instead of
re-deriving masks by overlap tracking, so the quantified series is exactly the
mask series stage 3 emitted and every row carries its stage-3 provenance.

It calls the same routine the production GFP path uses — `quant_helpers.
quantify_one_object`, as invoked by `one_cell_quantification_1CH.py` — so the
numbers are directly comparable with the rest of the pipeline.  That routine
quantifies the whole cell, runs the touching-circles septum pattern, and splits
the cell by the minor axis through the pattern centre, emitting one row for the
cell and one for each half.

A single `ep_refs` dict is carried across the whole series, exactly as the
production script does it, which is what keeps pole 1 and pole 2 from swapping
identity down the series.

Frames whose stage-3 branch was `NO_SEG` or `BOTH_MISSED` carry no image
evidence — they are the shape model alone — and are marked `model_only` so they
can be excluded downstream (P13).

Output is scratch-only; canonical `cell_*_data.csv` files are never written
(P5).

    python SingleCellQuantificationHPC/quantify_model_based_dense.py
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from skimage.io import imread

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quant_helpers import quantify_one_object
from Cell_tracking_functions import rle_decode

_SCRATCH = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/"
                "SingleCellQuantificationHPC/scratch")
DEFAULT_EXP = Path("/Volumes/X10 Pro/Movies/2026_08_28_M160")
DEFAULT_IN = _SCRATCH / "model_based_dense_out"
DEFAULT_OUT = _SCRATCH / "model_based_quant_out"
MODEL_ONLY = {"NO_SEG", "BOTH_MISSED"}


class FrameCache:
    """Frames are shared by every cell in a film, so cache them and drop the
    cache when the film changes."""

    def __init__(self, exp, cap=140):
        self.exp, self.cap, self.film, self.d = Path(exp), cap, None, {}

    def get(self, film, t):
        if film != self.film:
            self.film, self.d = film, {}
        if t not in self.d:
            if len(self.d) >= self.cap:
                self.d.pop(next(iter(self.d)))
            p = self.exp / film / f"Frames_{film}" / f"{film}_t_{t:03d}_c_0.tif"
            self.d[t] = imread(str(p)) if p.exists() else None
        return self.d[t]

    def scale(self, film):
        """Film intensity scale, computed exactly as `FindMovieMaxMin` in
        `quantify_cell.py` does it: pool every 10th pixel of EVERY frame in the
        film, then take the 99.5th and 1st percentiles.  Pooling matters — using
        the first frame alone ignores photobleaching and shifts the scale."""
        frames = sorted((self.exp / film / f"Frames_{film}").glob(f"{film}_t_*_c_0.tif"))
        if not frames:
            return None, None
        px = []
        for p in frames:
            img = imread(str(p))
            px.append(img.ravel()[::10])
        px = np.concatenate(px)
        return float(np.percentile(px, 99.5)), float(np.percentile(px, 1))


def touches_border(mask):
    return bool(mask[0, :].any() or mask[-1, :].any()
                or mask[:, 0].any() or mask[:, -1].any())


def quantify_cell(df, cache, gmax, gmin, plot_dir):
    """Mirrors the GFP branch of `one_cell_quantification_1CH.py`: one
    `quantify_one_object` call per frame, with a single `ep_refs` dict carried
    across the whole series so the endpoint references (and therefore pole
    identity) persist.  `quantify_one_object` returns several rows per frame —
    the whole cell, then the two halves it splits by the septum pattern."""
    df = df.sort_values("time_point")
    H, W = int(df.iloc[0]["height"]), int(df.iloc[0]["width"])
    lc = int(df.iloc[0]["local_cid"])
    film = df.iloc[0]["film"]
    ep_refs = {
        "single": {"ep1": None, "ep2": None, "prev_params": None},
        "1": {"ep1": None, "ep2": None, "prev_params": None},
        "2": {"ep1": None, "ep2": None, "prev_params": None},
    }
    rows = []
    for _, r in df.iterrows():
        t = int(r["time_point"])
        rle = str(r.get("rle", ""))
        if not rle.strip() or rle.lower() == "nan":
            continue
        img = cache.get(film, t)
        if img is None:
            continue
        try:
            mask = np.asarray(rle_decode(rle, (H, W)), bool)
        except Exception:
            continue
        if mask.sum() < 30:
            continue
        tb = touches_border(mask)
        # stage-3 provenance, carried onto every row this frame emits
        extra = dict(
            film=film, local_cid=lc,
            stage3_branch=r["branch"], stage3_good=bool(r["good"]),
            model_only=bool(str(r["branch"]).split("(")[0] in MODEL_ONLY),
            stage3_theta=r.get("theta"), stage3_t_div=r.get("t_div"),
            stage3_span=r.get("out_span"), expected_span=r.get("exp_span"),
        )
        try:
            out = quantify_one_object(
                img, mask, id_suffix="", t=t,
                plot_dir=plot_dir, ep_refs=ep_refs,
                gfp_min=gmin, gfp_max=gmax, cell_id=str(lc),
                do_plot=False, touches_border_flag=tb,
                allow_split=True, extra_cols=extra,
            )
        except Exception as exc:
            rows.append(dict(extra, time_point=t, channel="gfp",
                             error=f"{type(exc).__name__}: {exc}"))
            continue
        for row in out:
            row["channel"] = "gfp"
            row.setdefault("error", "")
            for k, v in extra.items():
                row.setdefault(k, v)
        rows.extend(out)
    return rows


def _run_chunk(args):
    """One worker's slice of the cell list.  Module level so it survives spawn."""
    files, exp, out, force, offset, total = args
    cache = FrameCache(exp)
    scales = {}
    done = skipped = failed = 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        film = f.parent.name
        dst = Path(out) / film / f.name
        if dst.exists() and not force:
            skipped += 1
            continue
        try:
            df = pd.read_csv(f)
            if film not in scales:
                scales[film] = cache.scale(film)
            gmax, gmin = scales[film]
            if gmax is None:
                failed += 1
                continue
            rows = quantify_cell(df, cache, gmax, gmin,
                                 plot_dir=str(Path(out) / film / f"plots_{f.stem}"))
            if rows:
                dst.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(dst, index=False)
                done += 1
            else:
                failed += 1
        except Exception:
            traceback.print_exc()
            failed += 1
        if i % 10 == 0 or i == len(files):
            el = time.time() - t0
            rate = el / max(done, 1)
            print(f"[{offset + i}/{total}] done={done} skip={skipped} err={failed} "
                  f"{el/60:.1f}min, {rate:.1f}s/cell, "
                  f"eta {rate*(len(files)-i)/60:.0f}min", flush=True)
    return done, skipped, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--dense", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--channel", default="FL", choices=["FL", "BF", "both"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel worker processes; chunks stay contiguous so "
                         "each worker keeps its frame cache within one film")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    files = sorted(a.dense.glob("*/cell_*.csv"))
    if a.channel != "both":
        files = [f for f in files if (a.channel in f.parent.name)]
    if a.limit:
        files = files[:a.limit]
    # group by film so the frame cache is useful
    files.sort(key=lambda f: (f.parent.name, f.name))
    print(f"cells: {len(files)}  channel {a.channel}", flush=True)
    print(f"out:   {a.out}", flush=True)

    t_start = time.time()
    if a.workers <= 1:
        d, s, e = _run_chunk((files, a.exp, a.out, a.force, 0, len(files)))
    else:
        # Contiguous chunks keep each worker inside one film, so its frame cache
        # keeps paying off.
        n = a.workers
        size = (len(files) + n - 1) // n
        chunks = [(files[i:i + size], a.exp, a.out, a.force, i, len(files))
                  for i in range(0, len(files), size)]
        print(f"workers: {len(chunks)}", flush=True)
        with mp.get_context("spawn").Pool(len(chunks)) as pool:
            res = pool.map(_run_chunk, chunks)
        d = sum(r[0] for r in res)
        s = sum(r[1] for r in res)
        e = sum(r[2] for r in res)

    print(f"FINISHED done={d} skipped={s} errors={e} "
          f"{(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
