#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 7 2025

Performs Cellpose segmentation from pre-exported TIFFs using the original output folder structure.
"""

import os
import gc
import numpy as np
from tifffile import imread, imwrite
from tqdm import tqdm
from cellpose import models

# Settings
working_dir = "/Volumes/Movies/2025_09_17/"
custom_model_path = "/Volumes/Movies/AI_training_set/models/CP_20250517_152934"
channels_for_segmentation = [0, 0]
diameter = 80

def segment_frames_for_file(file_name):
    print(f"Segmenting: {file_name}")
    
    # Define folder paths using original structure
    output_frames_folder = os.path.join(working_dir, f"{file_name}/Frames_{file_name}")
    output_masks_folder = os.path.join(working_dir, f"{file_name}/Masks_{file_name}")
    GFP_seg_folder = os.path.join(output_masks_folder, "GFP_seg")
    brightfield_seg_folder = os.path.join(output_masks_folder, "brightfield_seg")

    # Create necessary output folders
    for folder in [output_masks_folder, GFP_seg_folder, brightfield_seg_folder]:
        os.makedirs(folder, exist_ok=True)

    # Load Cellpose model
    model = models.CellposeModel(gpu=True, pretrained_model=custom_model_path)

    # List all TIFF frames in the Frames_... folder
    all_files = sorted([f for f in os.listdir(output_frames_folder) if f.endswith('.tif')])

    with tqdm(total=len(all_files), desc=f"Segmenting {file_name}") as pbar:
        for fname in all_files:
            img_path = os.path.join(output_frames_folder, fname)
            image = imread(img_path)

            # Determine if this is channel 0 (GFP) or channel 1 (brightfield)
            if "_c_0" in fname:
                output_path = os.path.join(GFP_seg_folder, fname.replace(".tif", "_seg.tif"))
            elif "_c_1" in fname:
                output_path = os.path.join(brightfield_seg_folder, fname.replace(".tif", "_seg.tif"))
            else:
                print(f"Skipping unrecognized file: {fname}")
                pbar.update(1)
                continue

            # Run segmentation
            masks, *_ = model.eval(image, channels=channels_for_segmentation, diameter=diameter)
            imwrite(output_path, masks.astype(np.uint16))
            del image, masks
            gc.collect()
            pbar.update(1)

# Batch process all directories in working_dir
if __name__ == "__main__":
    all_subdirs = [d for d in os.listdir(working_dir) if os.path.isdir(os.path.join(working_dir, d))]
    for subdir in all_subdirs:
        frame_dir = os.path.join(working_dir, subdir, f"Frames_{subdir}")
        if os.path.exists(frame_dir):
            segment_frames_for_file(subdir)
