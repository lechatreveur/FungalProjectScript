#!/usr/bin/env python3
"""Locally segment every non-keyframe M161 frame with bounded MPS memory.

Copy-to-modify variant of segment_m162_all_frames_local.py (P15). M161 is the
replicate control for M162, so the segmentation must be identical between them
- same checkpoint, same diameter, same thresholds - or a difference in masks
would surface as a difference in features and be misread as biology.

Stage-1 keyframes are treated as immutable anchors. Existing non-keyframe masks
are also skipped, so interrupted runs resume without overwriting completed work.

WHY THIS STAGE IS NOT OPTIONAL: model-based dense tracking falls back to the
shape model for any frame with no segmentation. Running stage 3 on keyframes
alone produced 97% model-only frames for M161, against 0.21% for M162 and 0.0%
for M160 - intensity measured from an inferred segment rather than from image
evidence, which is not comparable with either.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
from cellpose import models
from tifffile import imread, imwrite


# Data stays on the external SSD, never the system disk (P4).
DEFAULT_EXP_DIR = Path(
    os.environ.get("FUNGAL_M161_DIR", "/Volumes/X10 Pro/Movies/2026_09_03_M161")
)
KEYFRAMES = {
    "FL": frozenset((0, 50, 100)),
    "BF": frozenset((0, 20, 40)),
}
# M161 has FL1-FL6 and BF1-BF5, against M162's FL1-FL4 and BF1-BF3.
FILM_TYPES = (("FL", range(1, 7)), ("BF", range(1, 6)))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


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


def films() -> list[str]:
    return [
        f"NeonG_YES_1_{channel}{index}_F{field}"
        for field in range(4)
        for channel, indices in FILM_TYPES
        for index in indices
    ]


def channel_for_film(film: str) -> str:
    return "FL" if "_FL" in film else "BF"


def frame_paths(exp_dir: Path, film: str) -> list[Path]:
    frame_dir = exp_dir / film / f"Frames_{film}"
    return sorted(frame_dir.glob(f"{film}_t_*_c_0.tif"))


def time_from_frame(path: Path) -> int:
    return int(path.stem.split("_t_")[-1].split("_c_")[0])


def canonical_seg_path(exp_dir: Path, film: str, t: int) -> Path:
    return (
        exp_dir
        / film
        / f"Masks_{film}"
        / f"{film}_t_{t:03d}_c_0_seg.tif"
    )


def verify_inputs(exp_dir: Path,
                  selected: list[str] | None = None) -> tuple[dict[str, int], dict[str, str]]:
    """Validate the films that will actually be processed.

    The M162 original audited every film in the experiment. M161 is being
    processed FL1-first (only those films are on the SSD), so auditing all 44
    would hard-fail on films that are deliberately absent. Scoping the audit to
    the selection keeps the check strict for what is about to be segmented
    without inventing a pass for what is not.
    """
    expected_counts: dict[str, int] = {}
    keyframe_hashes: dict[str, str] = {}
    problems: list[str] = []

    for film in (selected if selected else films()):
        paths = frame_paths(exp_dir, film)
        expected = 101 if channel_for_film(film) == "FL" else 41
        expected_counts[film] = expected
        if len(paths) != expected:
            problems.append(f"{film}: {len(paths)} frames, expected {expected}")

        for t in KEYFRAMES[channel_for_film(film)]:
            seg_path = canonical_seg_path(exp_dir, film, t)
            if not seg_path.exists():
                problems.append(f"missing immutable keyframe: {seg_path}")
            else:
                keyframe_hashes[str(seg_path)] = sha256_file(seg_path)

    if problems:
        raise RuntimeError("Input validation failed:\n  " + "\n  ".join(problems))
    return expected_counts, keyframe_hashes


def configure_device(memory_cap_gb: float, allow_cpu: bool) -> tuple[bool, dict]:
    if torch.backends.mps.is_available():
        recommended = int(torch.mps.recommended_max_memory())
        cap_bytes = int(memory_cap_gb * 1024**3)
        fraction = min(1.0, max(0.1, cap_bytes / recommended))
        torch.mps.set_per_process_memory_fraction(fraction)
        return True, {
            "device": "mps",
            "memory_cap_gib": memory_cap_gb,
            "recommended_max_bytes": recommended,
            "per_process_fraction": fraction,
        }
    if torch.cuda.is_available():
        return True, {"device": "cuda", "memory_cap_gib": None}
    if allow_cpu:
        return False, {"device": "cpu", "memory_cap_gib": None}
    raise RuntimeError(
        "Neither MPS nor CUDA is available. Re-run in a host context with MPS "
        "access, or pass --allow-cpu explicitly."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", type=Path, default=DEFAULT_EXP_DIR)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=os.environ.get("CELLPOSE_MODEL_PATH"),
        required=os.environ.get("CELLPOSE_MODEL_PATH") is None,
    )
    parser.add_argument("--memory-cap-gb", type=float, default=10.0)
    parser.add_argument("--diameter", type=float, default=80.0)
    parser.add_argument("--bsize", type=int, default=256)
    parser.add_argument("--only-film", choices=films())
    parser.add_argument("--only-time", type=int)
    parser.add_argument("--max-new-frames", type=int, default=0)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Scratch root for smoke tests; canonical mask directories are untouched.",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve() if args.output_root else None
    repo_root = Path(__file__).resolve().parents[1]

    if not exp_dir.is_dir():
        raise FileNotFoundError(exp_dir)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if args.memory_cap_gb <= 0:
        raise ValueError("--memory-cap-gb must be positive")

    selected_for_check = [args.only_film] if args.only_film else films()
    expected_counts, keyframe_hashes_before = verify_inputs(
        exp_dir, selected_for_check)
    use_gpu, device_info = configure_device(args.memory_cap_gb, args.allow_cpu)
    model_sha256 = sha256_file(model_path)
    model_provenance_path = model_path.with_name(f"{model_path.name}.provenance.json")
    model_provenance = None
    if model_provenance_path.exists():
        with model_provenance_path.open(encoding="utf-8") as handle:
            model_provenance = json.load(handle)

    selected_films = [args.only_film] if args.only_film else films()
    jobs: list[tuple[str, int, Path, Path]] = []
    for film in selected_films:
        keyframes = KEYFRAMES[channel_for_film(film)]
        for frame_path in frame_paths(exp_dir, film):
            t = time_from_frame(frame_path)
            if args.only_time is not None and t != args.only_time:
                continue
            if t in keyframes:
                continue
            if output_root is None:
                seg_path = canonical_seg_path(exp_dir, film, t)
            else:
                seg_path = output_root / film / canonical_seg_path(exp_dir, film, t).name
            if not seg_path.exists():
                jobs.append((film, t, frame_path, seg_path))

    if args.max_new_frames > 0:
        jobs = jobs[: args.max_new_frames]

    provenance_path = (
        (output_root / "_provenance.json")
        if output_root
        else (exp_dir / "m161_all_frame_segmentation.provenance.json")
    )
    git = git_info(repo_root)
    provenance = {
        "artifact": str(output_root or exp_dir),
        "status": "running",
        "created": utc_now(),
        "updated": utc_now(),
        "created_by": str(Path(__file__).resolve()),
        "command": " ".join([sys.executable, *sys.argv]),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "inputs": {
            "experiment": str(exp_dir),
            "films": selected_films,
            "expected_frame_counts": expected_counts,
        },
        "model": {
            "path": str(model_path),
            "sha256": model_sha256,
            "provenance_path": str(model_provenance_path)
            if model_provenance_path.exists()
            else None,
            "provenance": model_provenance,
        },
        "params": {
            "diameter": args.diameter,
            "flow_threshold": 0.4,
            "cellprob_threshold": 0.0,
            "bsize": args.bsize,
            "memory": device_info,
            "keyframes_immutable": True,
            "skip_existing": True,
        },
        "keyframe_sha256_before": keyframe_hashes_before,
        "planned_new_frames": len(jobs),
        "completed_new_frames": 0,
        "failures": [],
    }
    atomic_json(provenance_path, provenance)

    print(f"Experiment: {exp_dir}", flush=True)
    print(f"Model: {model_path} ({model_sha256[:12]})", flush=True)
    print(f"Device: {device_info}", flush=True)
    print(f"New frames queued: {len(jobs)}", flush=True)
    print(f"Provenance: {provenance_path}", flush=True)

    if not jobs:
        provenance["status"] = "complete"
        provenance["updated"] = utc_now()
        atomic_json(provenance_path, provenance)
        return 0

    print("Loading Cellpose-SAM checkpoint...", flush=True)
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))
    start = time.monotonic()

    try:
        for index, (film, t, frame_path, seg_path) in enumerate(jobs, start=1):
            frame_start = time.monotonic()
            img = imread(frame_path)
            out = model.eval(
                img,
                channel_axis=None,
                diameter=float(args.diameter),
                flow_threshold=0.4,
                cellprob_threshold=0.0,
                bsize=args.bsize,
            )
            masks = np.asarray(out[0])
            max_label = int(masks.max(initial=0))
            if masks.shape != img.shape:
                raise RuntimeError(
                    f"shape mismatch for {frame_path}: image={img.shape}, mask={masks.shape}"
                )
            if max_label == 0:
                raise RuntimeError(f"empty segmentation for {frame_path}")
            if max_label > np.iinfo(np.uint16).max:
                raise RuntimeError(f"too many labels ({max_label}) for uint16: {frame_path}")

            seg_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = seg_path.with_name(f".{seg_path.name}.tmp.tif")
            imwrite(tmp_path, masks.astype(np.uint16, copy=False))
            os.replace(tmp_path, seg_path)

            provenance["completed_new_frames"] = index
            provenance["updated"] = utc_now()
            provenance["last_completed"] = {
                "film": film,
                "time_point": t,
                "path": str(seg_path),
                "labels": max_label,
            }
            atomic_json(provenance_path, provenance)
            elapsed = time.monotonic() - start
            frame_elapsed = time.monotonic() - frame_start
            print(
                f"[{index}/{len(jobs)}] {film} t={t:03d} labels={max_label} "
                f"frame_s={frame_elapsed:.2f} elapsed_s={elapsed:.1f}",
                flush=True,
            )

            del img, out, masks
            if device_info["device"] == "mps":
                torch.mps.empty_cache()
            gc.collect()
    except BaseException as exc:
        provenance["status"] = "failed"
        provenance["updated"] = utc_now()
        provenance["failures"].append(repr(exc))
        atomic_json(provenance_path, provenance)
        raise

    keyframe_hashes_after = {
        path: sha256_file(Path(path)) for path in keyframe_hashes_before
    }
    changed_keyframes = sorted(
        path
        for path, before in keyframe_hashes_before.items()
        if keyframe_hashes_after[path] != before
    )
    if changed_keyframes:
        provenance["status"] = "failed"
        provenance["failures"].append(
            f"immutable keyframes changed: {changed_keyframes}"
        )
        atomic_json(provenance_path, provenance)
        raise RuntimeError("Immutable keyframe hash validation failed")

    if output_root is None and not args.only_film and args.only_time is None:
        final_counts = {}
        for film in films():
            mask_dir = exp_dir / film / f"Masks_{film}"
            final_counts[film] = len(list(mask_dir.glob(f"{film}_t_*_c_0_seg.tif")))
        mismatches = {
            film: {"actual": final_counts[film], "expected": expected}
            for film, expected in expected_counts.items()
            if final_counts[film] != expected
        }
        provenance["final_mask_counts"] = final_counts
        if mismatches:
            provenance["status"] = "failed"
            provenance["failures"].append(f"final mask-count mismatches: {mismatches}")
            atomic_json(provenance_path, provenance)
            raise RuntimeError(f"Final mask-count validation failed: {mismatches}")

    provenance["status"] = "complete"
    provenance["updated"] = utc_now()
    provenance["elapsed_seconds"] = round(time.monotonic() - start, 2)
    provenance["keyframe_sha256_after"] = keyframe_hashes_after
    atomic_json(provenance_path, provenance)
    print("Segmentation complete and keyframe hashes unchanged.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
