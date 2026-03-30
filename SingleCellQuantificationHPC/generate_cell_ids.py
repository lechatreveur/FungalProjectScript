#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General script to generate cell_ids.txt from brightfield segmentations.
"""

import os
import argparse
from skimage.io import imread
from skimage.measure import label, regionprops

def generate_cell_ids(movie_root, file_name, output_base_dir, z_index=2, min_area=2500):
    masks_folder = os.path.join(movie_root, f"{file_name}/Masks_{file_name}")
    brightfield_seg_folder = os.path.join(masks_folder, "brightfield_seg")
    seg_filename = f"{file_name}_t_00_z_{z_index}_c_1_seg.tif"
    seg_path = os.path.join(brightfield_seg_folder, seg_filename)

    if not os.path.exists(seg_path):
        print(f"⚠️ Segmentation file missing: {seg_path} — skipping.")
        return False

    segmentation = imread(seg_path)
    labeled = label(segmentation)
    regions = regionprops(labeled)
    filtered = [r.label for r in regions if r.area >= min_area]

    cell_id_file = os.path.join(output_base_dir, file_name, "cell_ids.txt")
    os.makedirs(os.path.dirname(cell_id_file), exist_ok=True)
    with open(cell_id_file, 'w') as f:
        for cid in filtered:
            f.write(f"{cid}\n")

    print(f"✅ {len(filtered)} cell IDs written to: {cell_id_file}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cell_ids.txt from a given movie segmentation.")
    parser.add_argument('--movie_root', type=str, required=True, help="Base folder with movie subfolders.")
    parser.add_argument('--file_name', type=str, required=True, help="Movie base name (e.g., A14_5_F1).")
    parser.add_argument('--output_base_dir', type=str, required=True, help="Where to write cell_ids.txt (e.g., path to 2025_05_15_M63).")
    parser.add_argument('--z_index', type=int, default=2, help="Z-slice index to load for segmentation.")
    parser.add_argument('--min_area', type=int, default=2500, help="Minimum area threshold for cell filtering.")
    args = parser.parse_args()

    generate_cell_ids(args.movie_root, args.file_name, args.output_base_dir, args.z_index, args.min_area)

