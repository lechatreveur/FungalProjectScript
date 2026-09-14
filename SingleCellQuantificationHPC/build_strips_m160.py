#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vertical cell strips for M160 model-based dense masks.

Strips are normally a by-product of stage 4: `one_cell_quantification_1CH.py`
passes a `strip_tiles` list into `quantify_one_object`, which appends one
`build_strip_tile()` output per frame, and `--make_strips` writes the PNG.
`quantify_model_based_dense.py` did not pass that list, so M160 has none.

This rebuilds them without requantifying, exactly as `build_strips_only.py`
does for tracked cells: `build_strip_tile` derives its own rotation from the
mask via `regionprops.orientation` and never touches the EM or pole fitting, so
a strip is fully determined by (frame image, mask). The only difference here is
the mask source — the stage-3 dense CSVs rather than `cell_<id>_masks.csv`.

One strip per **global cell** (P12), tiles concatenated across that cell's
films in sequence order, matching the continuous time axis used for features.

Output: <outputs>/2026_08_28_M160/strips/<global_cell_id>.png  (P4)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import label, regionprops

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Cell_tracking_functions import rle_decode
from quant_helpers import merge_disconnected_components, build_strip_tile, save_strip_from_tiles

EXP_NAME = "2026_08_28_M160"
DEFAULT_EXP = Path("/Volumes/X10 Pro/Movies") / EXP_NAME
_SSD_OUT = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking") / EXP_NAME
DEFAULT_DENSE = _SSD_OUT / "dense_masks"
DEFAULT_OUT = _SSD_OUT / "strips"
SEQS = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]


def build_id_map(exp_dir):
    """(film, local_cell_id) -> global_cell_id, and film -> FL ordinal (P12)."""
    linkage = json.load(open(Path(exp_dir) / "sequence_linkage.json"))
    gid_of, order = {}, {}
    for seq in SEQS:
        if seq not in linkage:
            continue
        films = linkage[seq]["films"]
        for k, film in enumerate([f for f in films if "FL" in f]):
            order[film] = k
        for gid, locals_ in linkage[seq]["global_cells"].items():
            for film, lc in zip(films, locals_):
                if lc and lc > 0:
                    gid_of[(film, int(lc))] = gid
    return gid_of, order


class Frames:
    """Frame cache, dropped when the film changes."""

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


def tiles_for_film(df, frames, film):
    """One tile per frame, reproducing what --make_strips feeds build_strip_tile:
    merge disconnected components, take the first labelled region's bbox, crop
    the raw frame the same way."""
    out = []
    H, W = int(df.iloc[0]["height"]), int(df.iloc[0]["width"])
    for _, r in df.sort_values("time_point").iterrows():
        rle = str(r.get("rle", ""))
        if not rle.strip() or rle.lower() == "nan":
            continue
        img = frames.get(film, int(r["time_point"]))
        if img is None:
            continue
        try:
            mask = np.asarray(rle_decode(rle, (H, W)), bool)
        except Exception:
            continue
        if mask.sum() < 30:
            continue
        mask = merge_disconnected_components(mask, keep=2, bridge_width=1)
        props = regionprops(label(mask))
        if not props:
            continue
        r0, c0, r1, c1 = props[0].bbox
        out.append(build_strip_tile(img[r0:r1, c0:c1], mask[r0:r1, c0:c1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--dense", type=Path, default=DEFAULT_DENSE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    gid_of, order = build_id_map(a.exp)

    # group the dense CSVs by global cell, films in sequence order
    by_gid = {}
    for f in sorted(a.dense.glob("*/cell_*.csv")):
        film = f.parent.name
        if "FL" not in film or film not in order:
            continue
        try:
            lc = int(f.stem.split("_")[1])
        except Exception:
            continue
        gid = gid_of.get((film, lc))
        if gid is None:
            continue
        by_gid.setdefault(gid, []).append((order[film], film, f))
    for gid in by_gid:
        by_gid[gid].sort()

    gids = sorted(by_gid)
    if a.limit:
        gids = gids[:a.limit]
    print(f"global cells: {len(gids)}", flush=True)

    frames = Frames(a.exp)
    done = skipped = failed = 0
    t0 = time.time()
    # film-major order keeps the frame cache useful
    gids.sort(key=lambda g: (by_gid[g][0][1], g))
    for i, gid in enumerate(gids, 1):
        dst = a.out / f"{gid}.png"
        if dst.exists() and not a.force:
            skipped += 1
            continue
        tiles = []
        for _, film, f in by_gid[gid]:
            try:
                tiles.extend(tiles_for_film(pd.read_csv(f), frames, film))
            except Exception as exc:
                print(f"  {gid} / {film}: {type(exc).__name__}: {exc}", flush=True)
        if not tiles:
            failed += 1
            continue
        if save_strip_from_tiles(tiles, str(dst)):
            done += 1
        else:
            failed += 1
        if i % 25 == 0 or i == len(gids):
            el = time.time() - t0
            rate = el / max(done, 1)
            print(f"[{i}/{len(gids)}] done={done} skip={skipped} err={failed} "
                  f"{el/60:.1f}min, {rate:.1f}s/cell, "
                  f"eta {rate*(len(gids)-i)/60:.0f}min", flush=True)

    print(f"FINISHED done={done} skipped={skipped} errors={failed} "
          f"{(time.time()-t0)/60:.1f} min -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
