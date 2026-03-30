#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import argparse
import numpy as np
from tifffile import imread, imwrite
from tqdm import tqdm
from cellpose import models
from cellpose import core

import torch
torch.backends.mkldnn.enabled = False
import cv2
import fnmatch
from imaris_ims_file_reader import ims

# ----------------------------
# Settings (non-working_dir)
# ----------------------------
#custom_model_path = "/RAID1/working/R402/hsushen/FungalProject/Movies/models/CP_20250517_152934"
channels_for_segmentation = [0, 0]
diameter = 80
fps = 10


# ----------------------------
# Helpers
# ----------------------------
def is_ims_completed(file_name, working_dir, time_points, require_movie=True):
    """
    Returns True if this ims appears fully processed:
    - all segmentation masks exist for t=0..time_points-1
    - optionally the mp4 movie exists
    """
    output_masks_folder = os.path.join(working_dir, f"{file_name}/Masks_{file_name}")
    
    if not os.path.isdir(output_masks_folder):
        return False
    
    for t in range(time_points):
        seg_path = os.path.join(output_masks_folder, f"{file_name}_t_{t:03d}_c_0_seg.tif")


    if require_movie:
        movie_path = os.path.join(
            working_dir, f"{file_name}/TrackedCells_{file_name}", f"{file_name}_c_0.mp4"
        )
        if not os.path.exists(movie_path):
            return False

    return True

def FindMovieMaxMin(file_name, channel, frame_dir):
    # Keep c_0 naming convention
    if channel != 0:
        raise ValueError("This simplified script only supports channel=0")

    frame_pattern = f"{file_name}_t_??_c_{channel}.tif"
    movie_name = f"{file_name}_c_{channel:01d}.mp4"

    frame_files = sorted(
        [f for f in os.listdir(frame_dir) if f.endswith(".tif") and fnmatch.fnmatch(f, frame_pattern)]
    )
    if not frame_files:
        print(f"No frames found for channel {channel}")
        return None, None, None, None

    sample_pixels = []
    for fname in frame_files:
        frame = imread(os.path.join(frame_dir, fname))
        sample_pixels.extend(frame.ravel()[::10])
    sample_pixels = np.array(sample_pixels)

    global_min = np.percentile(sample_pixels, 1)
    global_max = np.percentile(sample_pixels, 99.5)

    return global_max, global_min, frame_files, movie_name


def create_movie(file_name, channel, frame_dir, output_dir):
    global_max, global_min, frame_files, movie_name = FindMovieMaxMin(file_name, channel, frame_dir)
    if frame_files is None:
        return

    print(f"Global intensity range: min={global_min}, max={global_max}")
    first_frame = imread(os.path.join(frame_dir, frame_files[0]))
    height, width = first_frame.shape

    out_path = os.path.join(output_dir, movie_name)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height), isColor=False)

    for fname in frame_files:
        frame = imread(os.path.join(frame_dir, fname)).astype(np.float32)
        frame_clipped = np.clip(frame, global_min, global_max)
        frame_normalized = (frame_clipped - global_min) / (global_max - global_min + 1e-12)
        frame_uint8 = (frame_normalized * 255).astype(np.uint8)
        out.write(frame_uint8)

    out.release()
    print(f"Saved movie: {out_path}")


def process_ims_file(ims_path, working_dir):
    file_name = os.path.splitext(os.path.basename(ims_path))[0]
    print(f"Processing {file_name}")
    done_flag = os.path.join(working_dir, f"{file_name}/DONE_segmentation.txt")
    if os.path.exists(done_flag):
        print(f"SKIP (DONE flag): {file_name}")
        return

    output_frames_folder = os.path.join(working_dir, f"{file_name}/Frames_{file_name}")
    output_masks_folder = os.path.join(working_dir, f"{file_name}/Masks_{file_name}")
    output_tracked_cells_folder = os.path.join(working_dir, f"{file_name}/TrackedCells_{file_name}")
    

    for folder in [output_frames_folder, output_masks_folder, output_tracked_cells_folder]:
        os.makedirs(folder, exist_ok=True)

    #model = models.CellposeModel(gpu=True, pretrained_model=custom_model_path)
    # Use GPU if available (on a SLURM GPU node this should be True)
    use_gpu = core.use_gpu()
    print(f"[cellpose] use_gpu={use_gpu}")
    
    # Cellpose v4 default is pretrained_model='cpsam' (Cellpose-SAM)
    model = models.CellposeModel(gpu=use_gpu)  # same as pretrained_model="cpsam"

    data = ims(ims_path)
    time_points, n_channels, z_stacks, height, width = data.shape
    
    # Skip fully completed ims
    if is_ims_completed(file_name, working_dir, time_points, require_movie=True):
        print(f"SKIP (already completed): {file_name}")
        del data
        gc.collect()
        return

    if n_channels != 1 or z_stacks != 1:
        print(f"WARNING: Expected (C=1, Z=1) but got (C={n_channels}, Z={z_stacks}) for {file_name}. Proceeding with [0,0].")

    with tqdm(total=time_points, desc=f"Segmenting {file_name}") as pbar:
        for t in range(time_points):
            frame_path = os.path.join(output_frames_folder, f"{file_name}_t_{t:03d}_c_0.tif")
            seg_path = os.path.join(output_masks_folder, f"{file_name}_t_{t:03d}_c_0_seg.tif")

            if not os.path.exists(frame_path) or not os.path.exists(seg_path):
                img = data[t, 0, 0, :, :]  # single channel, single z
                os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                imwrite(frame_path, img)

                out = model.eval(
                    img,
                    channels=channels_for_segmentation,  # [0,0] for single-channel images
                    channel_axis=None,                   # img is (Y,X)
                    diameter=diameter,                   # or set None if you want less assumption
                )
                masks = out[0]

                os.makedirs(os.path.dirname(seg_path), exist_ok=True)
                imwrite(seg_path, masks)

                del img, masks
                gc.collect()

            pbar.update(1)

    del data
    gc.collect()

    # Movie for c_0
    create_movie(file_name, 0, output_frames_folder, output_tracked_cells_folder)
    # Create done marker
    done_flag = os.path.join(working_dir, f"{file_name}/DONE_segmentation.txt")
    with open(done_flag, "w") as f:
        f.write("completed\n")


def parse_args():
    p = argparse.ArgumentParser(description="Batch segment all .ims files inside a working directory (C=1, Z=1).")
    p.add_argument("working_dir", help="Directory containing .ims files to process")
    return p.parse_args()


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    args = parse_args()
    working_dir = os.path.abspath(args.working_dir)

    if not os.path.isdir(working_dir):
        raise SystemExit(f"ERROR: working_dir does not exist or is not a directory: {working_dir}")

    ims_files = [f for f in os.listdir(working_dir) if f.endswith(".ims")]
    if not ims_files:
        raise SystemExit(f"ERROR: No .ims files found in: {working_dir}")

    for f in ims_files:
        process_ims_file(os.path.join(working_dir, f), working_dir)
