#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run model-based dense tracking (stage 3, P14) over an experiment.

Selects cells from the QC work queue by status, expands each to its per-film
keyframe intervals, and writes one CSV of dense masks per (film, cell) plus a
per-interval summary.

Output is scratch-only; canonical masks and keyframes are never written (P5,
P14).  The run is resumable: a (film, cell) whose output CSV already exists is
skipped unless --force.

    python SingleCellQuantificationHPC/run_model_based_dense_tracking.py \
        --status good corrected unreviewed --channel FL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import model_based_dense_tracking as M

DEFAULT_EXP = Path("/Volumes/X10 Pro/Movies/2026_08_28_M160")
# Inputs: the QC work queue and the stage-1 ABBT benchmark live in the main
# checkout's scratch tree, which is not version-controlled.
_SCRATCH = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/"
                "SingleCellQuantificationHPC/scratch")
DEFAULT_WQ = _SCRATCH / "work_queue_all_statuses.csv"
DEFAULT_SEED = _SCRATCH / "bayesian_tracker_benchmark_307.csv"

# Outputs go to the external SSD, never the system disk (P4).  Mask series for a
# whole experiment run to hundreds of megabytes and the boot volume has no room.
_SSD_OUT = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking")
DEFAULT_OUT = _SSD_OUT / "2026_08_28_M160" / "dense_masks"
SEQS = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]


def keypoints(film):
    return [0, 50, 100] if "FL" in film else [0, 20, 40]


def build_tasks(exp, wq_path, seed_path, statuses, channel):
    """-> list of dicts, one per (film, local cell), each carrying its intervals."""
    linkage = {s: json.load(open(exp / "sequence_linkage.json"))[s] for s in SEQS}
    wq = pd.read_csv(wq_path)
    wq = wq[wq.status.isin(statuses)]

    # division keyframe per global cell, from the stage-1 ABBT benchmark
    div = {}
    if Path(seed_path).exists():
        sd = pd.read_csv(seed_path)
        for _, r in sd.iterrows():
            if bool(r.get("gt_is_dividing")) and pd.notna(r.get("gt_division_kf")):
                div[(r["sequence"], r["cell_key"])] = int(r["gt_division_kf"])

    seen, tasks = set(), []
    for _, row in wq.iterrows():
        seq, gid = row["seq"], row["gid"]
        gc = linkage[seq]["global_cells"]
        films = linkage[seq]["films"]
        if gid not in gc:
            continue
        pos = row["sus_pos"]
        pos = eval(pos) if isinstance(pos, str) else pos
        kf = div.get((seq, gid))
        for p in pos:
            if p >= len(films):
                continue
            lc = gc[gid][p]
            if lc <= 0:
                continue
            film = films[p]
            ch = "FL" if "FL" in film else "BF"
            if channel != "both" and ch != channel:
                continue
            if (film, lc) in seen:
                continue
            seen.add((film, lc))
            kp = keypoints(film)
            # the division sits in interval (local-1) of film index kf//3
            div_iv = None
            if kf is not None and kf // 3 == p and kf % 3 != 0:
                div_iv = kf % 3 - 1
            tasks.append(dict(seq=seq, gid=gid, film=film, lc=int(lc), channel=ch,
                              status=row["status"], keypoints=kp, div_interval=div_iv))
    return tasks


def run_one(exp, task, out_dir, force=False):
    film, lc, kp = task["film"], task["lc"], task["keypoints"]
    dst = out_dir / film / f"cell_{lc}.csv"
    if dst.exists() and not force:
        return "skip", []
    rows, summary = [], []
    for i in range(len(kp) - 1):
        Ka, Kb = kp[i], kp[i + 1]
        dividing = (task["div_interval"] == i)
        t0 = time.time()
        try:
            cell = M.Cell(exp, film, lc, Ka, Kb, task["channel"])
            res, prom, d, scan = M.solve(cell, dividing=dividing)
        except Exception as exc:
            summary.append(dict(film=film, local_cid=lc, channel=task["channel"],
                                status=task["status"], K_a=Ka, K_b=Kb,
                                dividing=dividing, error=f"{type(exc).__name__}: {exc}",
                                n_frames=0, n_good=0, n_in_band=0, t_div=None,
                                confident=None, seconds=round(time.time() - t0, 1)))
            continue
        rows.extend(M.interval_rows(cell, res, d))
        in_band = sum(1 for r in res.values()
                      if r["out"].any()
                      and M.MISS_K <= M.mask_span(r["out"]) / r["exp_span"] <= M.FUSE_K)
        summary.append(dict(
            film=film, local_cid=lc, channel=task["channel"], status=task["status"],
            K_a=Ka, K_b=Kb, dividing=dividing, error="",
            n_frames=len(res), n_good=int(sum(r["good"] for r in res.values())),
            n_in_band=int(in_band), n_anchors=len(prom),
            n_no_seg=int(sum(r["branch"] == "NO_SEG" for r in res.values())),
            n_cut=int(sum(r["branch"].startswith("LONG_INTERSECT") for r in res.values())),
            n_union=int(sum(r["branch"].startswith("SHORT_UNION") for r in res.values())),
            n_both_missed=int(sum(r["branch"] == "BOTH_MISSED" for r in res.values())),
            n_bridged=int(sum(bool(r.get("bridged")) for r in res.values())),
            n_relink_rejected=int(sum(bool(r.get("relink_rejected")) for r in res.values())),
            t_div=d, confident=(None if scan is None else bool(scan["confident"])),
            peak_frac=(None if scan is None else round(scan["peak_frac"], 3)),
            plateau=(None if scan is None else scan["plateau"]),
            seconds=round(time.time() - t0, 1)))
        cell.release()
    if rows:
        dst.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(dst, index=False)
    return "done", summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, default=DEFAULT_EXP)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--work-queue", type=Path, default=DEFAULT_WQ)
    ap.add_argument("--seed", type=Path, default=DEFAULT_SEED)
    ap.add_argument("--status", nargs="+", default=["good", "corrected", "unreviewed"])
    ap.add_argument("--channel", default="FL", choices=["FL", "BF", "both"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(a.exp, a.work_queue, a.seed, a.status, a.channel)
    # good first, then corrected, then unreviewed
    order = {"good": 0, "corrected": 1, "unreviewed": 2}
    tasks.sort(key=lambda t: (order.get(t["status"], 9), t["film"], t["lc"]))
    if a.limit:
        tasks = tasks[:a.limit]

    summary_path = a.out / "model_based_dense_summary.csv"
    print(f"tasks: {len(tasks)} (film, cell) over statuses {a.status}, channel {a.channel}",
          flush=True)
    print(f"out:   {a.out}", flush=True)

    done = skipped = failed = 0
    all_rows = []
    t_start = time.time()
    for i, task in enumerate(tasks, 1):
        try:
            state, summ = run_one(a.exp, task, a.out, force=a.force)
        except Exception:
            traceback.print_exc()
            failed += 1
            continue
        if state == "skip":
            skipped += 1
        else:
            done += 1
            all_rows.extend(summ)
            if any(s["error"] for s in summ):
                failed += 1
        if all_rows and (i % 20 == 0 or i == len(tasks)):
            pd.DataFrame(all_rows).to_csv(summary_path, index=False)
        if i % 10 == 0 or i == len(tasks):
            el = time.time() - t_start
            rate = el / max(done, 1)
            print(f"[{i}/{len(tasks)}] done={done} skip={skipped} err={failed} "
                  f"{el/60:.1f}min elapsed, {rate:.1f}s/cell, "
                  f"eta {rate*(len(tasks)-i)/60:.0f}min", flush=True)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(summary_path, index=False)
    print(f"FINISHED  done={done} skipped={skipped} errors={failed}  "
          f"{(time.time()-t_start)/60:.1f} min", flush=True)
    print(f"summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
