#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parallel_cache.py
-----------------
Pre-cache TrackerDataset samples in parallel using multiprocessing to speed up epoch 1.
"""

import sys
import os
import multiprocessing
from pathlib import Path
from tqdm import tqdm

# Set start method to spawn for safety with PyTorch/macOS
multiprocessing.set_start_method('spawn', force=True)

# Add parent directories to sys.path
script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(script_dir.parent / "SingleCellDataAnalysis"))

from tracker_dataset import TrackerDataset

# Global variables in workers
worker_ds = None

def init_worker(movie_root, curated_csv, hold_out_film):
    global worker_ds
    # Redirect print outputs to sys.stderr to avoid messing up pool output
    print(f"[Worker {os.getpid()}] Initializing dataset...", file=sys.stderr)
    try:
        worker_ds = TrackerDataset(
            movie_root=movie_root,
            hold_out_film=hold_out_film,
            curated_csv=curated_csv,
            augment=False
        )
        print(f"[Worker {os.getpid()}] Dataset initialized successfully.", file=sys.stderr)
    except Exception as e:
        print(f"[Worker {os.getpid()}] Failed to initialize dataset: {e}", file=sys.stderr)

def cache_sample(idx):
    global worker_ds
    if worker_ds is None:
        return False
    try:
        # Check if already exists to avoid loading images
        cache_dir = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/tracker_dataset_cache")
        cache_path = cache_dir / f"sample_{idx}.npz"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return True
        
        # Load sample to trigger caching
        _ = worker_ds[idx]
        return True
    except Exception as e:
        print(f"[Worker {os.getpid()}] Error caching index {idx}: {e}", file=sys.stderr)
        return False

def main():
    movie_root = "/Volumes/X10 Pro/Movies"
    curated_csv = "/Volumes/X10 Pro/Movies/2026_01_08_M93/curated_training_samples.csv"
    hold_out_film = "A14_BF_1_F2"
    
    print("Pre-loading dataset in main process to get length...")
    main_ds = TrackerDataset(
        movie_root=movie_root,
        hold_out_film=hold_out_film,
        curated_csv=curated_csv,
        augment=False
    )
    n = len(main_ds)
    print(f"Total samples to process: {n}")
    
    num_workers = min(multiprocessing.cpu_count(), 8)
    print(f"Starting pool with {num_workers} workers...")
    
    indices = list(range(n))
    
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(movie_root, curated_csv, hold_out_film)
    ) as pool:
        # Use tqdm to show progress
        results = list(tqdm(pool.imap_unordered(cache_sample, indices), total=n))
        
    success = sum(1 for r in results if r)
    print(f"Finished caching. Success: {success}/{n}")

if __name__ == "__main__":
    main()
