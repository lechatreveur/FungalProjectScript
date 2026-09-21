#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_model_based_dense_tracking_m161.py

Executes Model-Based Dense Tracking (Stage 3, P14) across the 2026_09_03_M161
cohort. Copy-to-modify variant of run_model_based_dense_tracking_m162.py (P15).

M161 is the replicate control for M162 - same strain, medium and acquisition
settings, different session - so the tracking logic is held identical and only
the paths and the experiment prefix differ.

Key Features:
1. Selects clean cohort cells (status in good, corrected, unreviewed; excludes bad/border cells)
   from sequence_linkage.json and qc_NeonG_YES_1_F{0..3}.json.
2. Incorporates division events from m161_abbt_results.csv IF PRESENT. M161 has
   not been through ABBT, so that file does not exist yet and no interval is
   marked dividing. Tracking still runs; divisions are simply not flagged,
   which is acceptable for the FL1-only feature comparison but would need
   revisiting before any division or cell-cycle analysis on M161.
3. Multi-worker parallel processing across films and cells.
4. Fully resumable: skips existing output cell CSVs unless --force.
5. Outputs to /Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking/2026_09_03_M161/dense_masks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd

HPC_DIR = Path(__file__).resolve().parent
REPO_ROOT = HPC_DIR.parent
for p in (str(HPC_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import model_based_dense_tracking as M

# Data stays on the external SSD, never the system disk (P4).
DEFAULT_EXP = Path("/Volumes/X10 Pro/Movies/2026_09_03_M161")
DEFAULT_OUT_ROOT = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking/2026_09_03_M161")
# M161 sequences are prefixed NeonG_YES_1_, one underscore-delimited token
# longer than M162's NeonG_YES_. Matching these by substring would
# cross-match the two experiments, so they are listed explicitly.
SEQS = [f"NeonG_YES_1_F{i}" for i in range(4)]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_info(repo_root: Path) -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": "unknown", "dirty": True}


def keypoints(film: str) -> list[int]:
    return [0, 50, 100] if "FL" in film else [0, 20, 40]


def load_qc_statuses(exp: Path) -> Dict[str, str]:
    """Loads QC status for all global cells across F0..F3."""
    statuses = {}
    for seq in SEQS:
        qc_f = exp / f"qc_{seq}.json"
        if qc_f.exists():
            with open(qc_f) as handle:
                data = json.load(handle)
            for gid, val in data.items():
                statuses[gid] = val.get("status", "unreviewed")
    return statuses


def load_division_calls(exp: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Loads division calls from m161_abbt_results.csv, if it exists."""
    div_file = exp / "m161_abbt_results.csv"
    calls = {}
    if div_file.exists():
        df = pd.read_csv(div_file)
        for _, row in df.iterrows():
            if bool(row.get("is_dividing")):
                seq = str(row.get("sequence"))
                ckey = str(row.get("cell_key"))
                calls[(seq, ckey)] = {
                    "film": str(row.get("division_film")),
                    "keyframe": row.get("division_keyframe"),
                    "t": row.get("division_t"),
                    "prob": row.get("map_posterior_prob")
                }
    return calls


def build_tasks(
    exp: Path,
    statuses_allowed: list[str],
    channel_choice: str,
    films_filter: Optional[list[str]] = None
) -> list[dict]:
    linkage_file = exp / "sequence_linkage.json"
    if not linkage_file.exists():
        raise FileNotFoundError(f"Missing {linkage_file}")
    with open(linkage_file) as f:
        linkage = json.load(f)

    qc_statuses = load_qc_statuses(exp)
    divisions = load_division_calls(exp)

    tasks = []
    seen = set()

    for seq in SEQS:
        if seq not in linkage:
            continue
        films = linkage[seq]["films"]
        global_cells = linkage[seq]["global_cells"]

        for gid, track in global_cells.items():
            status = qc_statuses.get(gid, "unreviewed")
            if status not in statuses_allowed:
                continue
            if len(track) != len(films) or not all(cid > 0 for cid in track):
                continue

            div_info = divisions.get((seq, gid))

            for p, film in enumerate(films):
                lc = track[p]
                if lc <= 0:
                    continue
                ch = "FL" if "FL" in film else "BF"
                if channel_choice != "both" and ch != channel_choice:
                    continue
                if films_filter and film not in films_filter:
                    continue
                if (film, lc) in seen:
                    continue
                seen.add((film, lc))

                kp = keypoints(film)
                # Determine if this film and interval contains a division
                div_iv = None
                if div_info and div_info["film"] == film:
                    div_t = div_info.get("t")
                    if pd.notna(div_t):
                        t_val = float(div_t)
                        # Determine interval
                        for iv_idx in range(len(kp) - 1):
                            if kp[iv_idx] <= t_val <= kp[iv_idx + 1]:
                                div_iv = iv_idx
                                break

                tasks.append({
                    "seq": seq,
                    "gid": gid,
                    "film": film,
                    "lc": int(lc),
                    "channel": ch,
                    "status": status,
                    "keypoints": kp,
                    "div_interval": div_iv,
                    "exp_dir": str(exp)
                })

    return tasks


def run_one_task(args: Tuple[dict, str, bool]) -> Tuple[str, list[dict]]:
    task, out_dir_str, force = args
    exp = Path(task["exp_dir"])
    out_dir = Path(out_dir_str)
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
            summary.append({
                "film": film,
                "local_cid": lc,
                "channel": task["channel"],
                "status": task["status"],
                "K_a": Ka,
                "K_b": Kb,
                "dividing": dividing,
                "error": f"{type(exc).__name__}: {exc}",
                "n_frames": 0,
                "n_good": 0,
                "n_in_band": 0,
                "t_div": None,
                "confident": None,
                "seconds": round(time.time() - t0, 2)
            })
            if "cell" in locals() and cell is not None:
                cell.release()
            continue

        rows.extend(M.interval_rows(cell, res, d))
        in_band = sum(
            1 for r in res.values()
            if r["out"].any()
            and M.MISS_K <= M.mask_span(r["out"]) / r["exp_span"] <= M.FUSE_K
        )
        summary.append({
            "film": film,
            "local_cid": lc,
            "channel": task["channel"],
            "status": task["status"],
            "K_a": Ka,
            "K_b": Kb,
            "dividing": dividing,
            "error": "",
            "n_frames": len(res),
            "n_good": int(sum(r["good"] for r in res.values())),
            "n_in_band": int(in_band),
            "n_anchors": len(prom),
            "n_no_seg": int(sum(r["branch"] == "NO_SEG" for r in res.values())),
            "n_cut": int(sum(r["branch"].startswith("LONG_INTERSECT") for r in res.values())),
            "n_union": int(sum(r["branch"].startswith("SHORT_UNION") for r in res.values())),
            "n_both_missed": int(sum(r["branch"] == "BOTH_MISSED" for r in res.values())),
            "n_bridged": int(sum(bool(r.get("bridged")) for r in res.values())),
            "n_relink_rejected": int(sum(bool(r.get("relink_rejected")) for r in res.values())),
            "t_div": d,
            "confident": (None if scan is None else bool(scan["confident"])),
            "peak_frac": (None if scan is None else round(scan["peak_frac"], 3)),
            "plateau": (None if scan is None else scan["plateau"]),
            "seconds": round(time.time() - t0, 2)
        })
        cell.release()

    if rows:
        dst.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(dst, index=False)

    return "done", summary


def main():
    parser = argparse.ArgumentParser(description="Model-based dense tracking across M162 cohort")
    parser.add_argument("--exp", type=Path, default=DEFAULT_EXP)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--status", nargs="+", default=["good", "corrected", "unreviewed"])
    parser.add_argument("--channel", default="both", choices=["FL", "BF", "both"])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--films", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    exp = args.exp.resolve()
    out_root = args.out_root.resolve()
    dense_masks_dir = out_root / "dense_masks"
    dense_masks_dir.mkdir(parents=True, exist_ok=True)

    print("=== Model-Based Dense Tracking for M161 ===")
    print(f"Experiment : {exp}")
    print(f"Output dir : {dense_masks_dir}")
    print(f"Channel    : {args.channel}")
    print(f"Workers    : {args.workers}")
    print(f"Statuses   : {args.status}")

    tasks = build_tasks(exp, args.status, args.channel, args.films)
    order = {"good": 0, "corrected": 1, "unreviewed": 2}
    tasks.sort(key=lambda t: (order.get(t["status"], 9), t["film"], t["lc"]))

    if args.limit:
        tasks = tasks[:args.limit]
    total_tasks = len(tasks)
    print(f"Total (film, cell) tasks queued: {total_tasks}")

    if total_tasks == 0:
        print("No tasks to run.")
        return 0

    film_suffix = f"_{args.films[0]}" if (args.films and len(args.films) == 1) else ""
    provenance_path = out_root / f"m161_dense_tracking{film_suffix}.provenance.json"
    git = git_info(REPO_ROOT)
    provenance = {
        "artifact": str(dense_masks_dir),
        "status": "running",
        "created": utc_now(),
        "updated": utc_now(),
        "created_by": str(Path(__file__).resolve()),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "params": {
            "channel": args.channel,
            "statuses": args.status,
            "workers": args.workers,
            "total_tasks": total_tasks,
            "films": args.films
        },
        "completed_tasks": 0,
        "skipped_tasks": 0,
        "failed_tasks": 0,
    }
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)

    work_items = [(t, str(dense_masks_dir), args.force) for t in tasks]
    all_summaries = []
    t_start = time.time()
    n_done = 0
    n_skipped = 0
    n_failed = 0

    print(f"\nLaunching {total_tasks} tasks across {args.workers} workers...")
    with Pool(processes=args.workers) as pool:
        for idx, (res_type, summary_list) in enumerate(pool.imap_unordered(run_one_task, work_items), 1):
            if res_type == "skip":
                n_skipped += 1
            elif res_type == "done":
                n_done += 1
                all_summaries.extend(summary_list)
                if any(s.get("error") for s in summary_list):
                    n_failed += 1

            if idx % 100 == 0 or idx == total_tasks:
                elapsed = time.time() - t_start
                rate = idx / elapsed if elapsed > 0 else 0
                print(f"[{idx}/{total_tasks}] Done: {n_done} | Skipped: {n_skipped} | Failed: {n_failed} | Rate: {rate:.1f} tasks/s | Elapsed: {elapsed:.1f}s", flush=True)

                provenance["completed_tasks"] = n_done
                provenance["skipped_tasks"] = n_skipped
                provenance["failed_tasks"] = n_failed
                provenance["updated"] = utc_now()
                with open(provenance_path, "w") as f:
                    json.dump(provenance, f, indent=2)

    t_total = time.time() - t_start
    print(f"\nFinished in {t_total:.1f}s. (Done: {n_done}, Skipped: {n_skipped}, Failed: {n_failed})")

    # Write summary CSV (per film if film-scoped, preventing collisions in SLURM array)
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_csv = out_root / f"dense_tracking_summary{film_suffix}.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved summary metrics to {summary_csv.name} ({len(summary_df)} intervals)")

    provenance["status"] = "complete"
    provenance["updated"] = utc_now()
    provenance["elapsed_seconds"] = round(t_total, 2)
    with open(provenance_path, "w") as f:
        json.dump(provenance, f, indent=2)

    print(f"Saved provenance to {provenance_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
