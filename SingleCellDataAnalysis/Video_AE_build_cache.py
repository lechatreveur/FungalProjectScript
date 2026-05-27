#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')

from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from SingleCellDataAnalysis.Video_AE_data_loader import load_video_dataset, EXPERIMENT_BASES

EXPERIMENTS = {
    'Sept17':     '/Volumes/X10 Pro/Movies/2025_09_17/',
    'M92':        '/Volumes/X10 Pro/Movies/2025_12_31_M92/',
    'M93':        '/Volumes/X10 Pro/Movies/2026_01_08_M93/',
    'June25_20m': '/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/',
}

OUTPUT_DIR   = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CACHE_VIDEOS = os.path.join(OUTPUT_DIR, "video_cache_32x112_padded.npy")
CACHE_GIDS   = os.path.join(OUTPUT_DIR, "video_gids.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    print("📥 Loading curated 431-cell list...")
    _, _, gids, labels, _, _ = load_feature_constrained_data(EXPERIMENTS)
    print(f"   Total curated cells: {len(gids)}")

    print("\n🎥 Building video tensors (PADDED 32x112)...")
    videos, valid_gids = load_video_dataset(gids, EXPERIMENT_BASES, frame_h=32, frame_w=112)

    print(f"\n💾 Saving cache: {CACHE_VIDEOS}")
    np.save(CACHE_VIDEOS, videos)

    with open(CACHE_GIDS, 'w') as f:
        for gid in valid_gids:
            f.write(gid + '\n')

    print(f"✅ Cache saved. Shape: {videos.shape}")
