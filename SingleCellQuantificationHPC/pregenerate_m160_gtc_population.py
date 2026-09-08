#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pregenerate_m160_gtc_population.py

Fast offline pregeneration script for Ground Truth Corrector (GTC) and Tracking Corrector (TC).
Pre-renders and caches:
1. GTC Sequence Population Frames (39 keyframes per sequence for 5_1_N1_F0, 5_1_N1_F1, 5_1_N1_F2).
2. GTC Sequence Boundary Outlines (39 keyframes per sequence).
3. TC Film-level Population Frames (PopulationFrames_<film>).
4. TC Gallery Cell Crops (CellCrops_<film>).
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Optional

# Repo root setup
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "SingleCellQuantificationHPC"))

from ground_truth_corrector.config import Config as GTCConfig
from ground_truth_corrector.services.gt_frames_service import GTFramesService
from tracking_corrector.config import config as tc_config
from tracking_corrector.repositories.mask_repository import MaskRepository
from tracking_corrector.services.frames_service import FramesService
from pregenerate_and_jumps import generate_cell_crops_for_film


def pregenerate_gtc_sequence(
    gt_svc: GTFramesService,
    exp: str,
    sequence: str,
    force: bool = False
) -> int:
    """Pre-renders all 39 keyframe population frames and boundary PNGs for GTC."""
    print(f"\n=======================================================")
    print(f"PREGENERATING GTC KEYFRAME POPULATION FRAMES: {sequence}")
    print(f"=======================================================")
    
    k_map = gt_svc.get_sequence_keyframe_map(exp, sequence)
    total_kfs = len(k_map)
    print(f"Found {total_kfs} keyframes for sequence {sequence}.")
    
    t_start = time.time()
    count = 0
    for idx, item in enumerate(k_map):
        film = item["film"]
        local_t = item["local_t"]
        pos = item.get("keyframe_pos", "")
        
        # 1. Render & cache population frame JPEG
        try:
            jpeg_bytes = gt_svc.render_population_frame_jpeg(
                exp=exp,
                film=film,
                t_val=local_t,
                sequence=sequence,
                quality=85,
                force=force
            )
            count += 1
            if (idx + 1) % 5 == 0 or idx == 0 or (idx + 1) == total_kfs:
                print(f"  [{sequence}] Keyframe {idx+1:02d}/{total_kfs:02d} ({film} t={local_t:03d} {pos}) rendered ({len(jpeg_bytes)//1024} KB)", flush=True)
        except Exception as e:
            print(f"  [ERROR] Failed to render population frame for {sequence} {film} t={local_t}: {e}")

    t_elapsed = time.time() - t_start
    print(f"✓ Completed GTC population frame pregeneration for {sequence} ({count}/{total_kfs} frames in {t_elapsed:.2f} s)")
    return count


def main():
    parser = argparse.ArgumentParser(description="Pre-generate M160 GTC Population Frames & TC Caches")
    parser.add_argument("--experiment", type=str, default="2026_08_28_M160", help="Experiment directory name")
    parser.add_argument("--sequences", type=str, default="5_1_N1_F0,5_1_N1_F1,5_1_N1_F2", help="Comma-separated sequences")
    parser.add_argument("--force", action="store_true", help="Force overwrite existing caches")
    parser.add_argument("--include_tc", action="store_true", help="Also generate TC film-level population frames & gallery crops")
    args = parser.parse_args()

    exp = args.experiment
    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]

    print("=" * 75)
    print(f"STARTING POPULATION FRAME PREGENERATION FOR {exp}")
    print(f"Sequences: {sequences}")
    print(f"Force overwrite: {args.force}")
    print("=" * 75)

    gtc_cfg = GTCConfig()
    gt_svc = GTFramesService(gtc_cfg)

    total_gtc_frames = 0
    t_global_start = time.time()

    for seq in sequences:
        n = pregenerate_gtc_sequence(gt_svc, exp, seq, force=args.force)
        total_gtc_frames += n

    if args.include_tc:
        print("\n" + "=" * 75)
        print("PREGENERATING TC FILM-LEVEL POPULATION FRAMES & GALLERY CROPS")
        print("=" * 75)
        base_root = tc_config.local_movie_root
        mask_repo = MaskRepository(base_root)
        tc_frames_svc = FramesService(tc_config, mask_repo)

        # Collect unique films across sequences
        from ground_truth_corrector.repositories.linkage_repository import LinkageRepository
        from ground_truth_corrector.services.linkage_service import LinkageService
        link_repo = LinkageRepository(base_root)
        link_svc = LinkageService(link_repo)
        seq_data = link_svc.get_sequences(exp).get("sequences", {})

        all_films = []
        for seq in sequences:
            if seq in seq_data:
                for f in seq_data[seq].get("films", []):
                    if f not in all_films:
                        all_films.append(f)

        print(f"Processing {len(all_films)} unique films for TC caches...")
        for film in all_films:
            paths = tc_frames_svc.get_film_frame_paths(exp, film, "bf")
            cache_dir = base_root / exp / film / f"PopulationFrames_{film}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            gen_count = 0
            for t_val in sorted(paths.keys()):
                cache_file = cache_dir / f"frame_{t_val:03d}.jpg"
                if args.force or not cache_file.exists():
                    try:
                        tc_frames_svc._generate_population_frame_bytes(exp, film, t_val, cache_file)
                        gen_count += 1
                    except Exception as e:
                        pass
            print(f"[{film}] Pre-generated {gen_count} TC population frames.")
            generate_cell_crops_for_film(tc_frames_svc, exp, film, force=args.force)

    t_total = time.time() - t_global_start
    print("\n" + "=" * 75)
    print(f"ALL PREGENERATION TASKS COMPLETED IN {t_total:.2f} s")
    print(f"Total GTC Keyframe Population Frames: {total_gtc_frames}")
    print("=" * 75)


if __name__ == "__main__":
    main()
