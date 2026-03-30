#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 15:00:46 2025

@author: user
"""

import os
import gc
import numpy as np
from tifffile import imread, imwrite
from tqdm import tqdm
from cellpose import models
import torch
torch.backends.mkldnn.enabled = False
from skimage.measure import regionprops, label
import cv2
import fnmatch
from imaris_ims_file_reader import ims

# Settings
working_dir = "/RAID1/working/R402/hsushen/FungalProject/Movies/2025_06_25"
custom_model_path = "/RAID1/working/R402/hsushen/FungalProject/Movies/models/CP_20250517_152934"
z_index = 1
channels_for_segmentation = [0, 0]
diameter = 80
fps = 10

# Helpers
def load_segmentation(path):
    return np.load(path) if path.endswith('.npy') else imread(path)

def FindMovieMaxMin(file_name, channel, z_index, frame_dir):
    if channel == 0:
        frame_pattern = f"{file_name}_t_??_c_{channel}.tif"
        movie_name = f"{file_name}_c_{channel:01d}.mp4"
    elif channel == 1:
        frame_pattern = f"{file_name}_t_??_z_{z_index}_c_{channel}.tif"
        movie_name = f"{file_name}_z_{z_index:01d}_c_{channel:01d}.mp4"
    else:
        raise ValueError("Unsupported channel or missing z_index")

    frame_files = sorted([f for f in os.listdir(frame_dir)
                          if f.endswith('.tif') and fnmatch.fnmatch(f, frame_pattern)])
    if not frame_files:
        print(f"No frames found for channel {channel} (z={z_index})")
        return

    sample_pixels = []
    for fname in frame_files:
        frame = imread(os.path.join(frame_dir, fname))
        sample_pixels.extend(frame.ravel()[::10])
    sample_pixels = np.array(sample_pixels)
    global_min = np.percentile(sample_pixels, 1)
    global_max = np.percentile(sample_pixels, 99.5)
    return global_max, global_min, frame_files, movie_name

def create_movie(file_name, channel, z_index, frame_dir, output_dir):
    global_max, global_min, frame_files, movie_name = FindMovieMaxMin(file_name, channel, z_index, frame_dir)
    print(f"Global intensity range: min={global_min}, max={global_max}")
    first_frame = imread(os.path.join(frame_dir, frame_files[0]))
    height, width = first_frame.shape
    out_path = os.path.join(output_dir, movie_name)
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height), isColor=False)

    for fname in frame_files:
        frame = imread(os.path.join(frame_dir, fname)).astype(np.float32)
        frame_clipped = np.clip(frame, global_min, global_max)
        frame_normalized = (frame_clipped - global_min) / (global_max - global_min)
        frame_uint8 = (frame_normalized * 255).astype(np.uint8)
        out.write(frame_uint8)
    out.release()
    print(f"Saved movie: {out_path}")

def process_ims_file(ims_path):
    file_name = os.path.splitext(os.path.basename(ims_path))[0]
    print(f"Processing {file_name}")

    output_frames_folder = os.path.join(working_dir, f"{file_name}/Frames_{file_name}")
    output_masks_folder = os.path.join(working_dir, f"{file_name}/Masks_{file_name}")
    output_tracked_cells_folder = os.path.join(working_dir, f"{file_name}/TrackedCells_{file_name}")
    GFP_seg_folder = os.path.join(output_masks_folder, "GFP_seg")
    brightfield_seg_folder = os.path.join(output_masks_folder, "brightfield_seg")

    for folder in [output_frames_folder, output_masks_folder, output_tracked_cells_folder, GFP_seg_folder, brightfield_seg_folder]:
        os.makedirs(folder, exist_ok=True)

    model = models.CellposeModel(gpu=True, pretrained_model=custom_model_path)
    data = ims(ims_path)
    time_points, n_channels, z_stacks, height, width = data.shape

    with tqdm(total=time_points * 2, desc=f"Segmenting {file_name}") as pbar:
        for t in range(time_points):
            # ---------- Channel 0 ----------
            frame_path_ch0 = os.path.join(output_frames_folder, f"{file_name}_t_{t:02d}_c_0.tif")
            seg_path_ch0 = os.path.join(GFP_seg_folder, f"{file_name}_t_{t:02d}_c_0_seg.tif")
            if not os.path.exists(frame_path_ch0) or not os.path.exists(seg_path_ch0):
                projected_img = np.mean(data[t, 0, :, :, :], axis=0)
                imwrite(frame_path_ch0, projected_img)
                masks, *_ = model.eval(projected_img, channels=channels_for_segmentation, diameter=diameter)
                imwrite(seg_path_ch0, masks)
                del projected_img, masks
                gc.collect()
            pbar.update(1)

            # ---------- Channel 1 ----------
            z = z_index if z_index < z_stacks else z_stacks - 1
            frame_path_ch1 = os.path.join(output_frames_folder, f"{file_name}_t_{t:02d}_z_{z:01d}_c_1.tif")
            seg_path_ch1 = os.path.join(brightfield_seg_folder, f"{file_name}_t_{t:02d}_z_{z:01d}_c_1_seg.tif")
            if not os.path.exists(frame_path_ch1) or not os.path.exists(seg_path_ch1):
                selected_img = data[t, 1, z, :, :]
                imwrite(frame_path_ch1, selected_img)
                masks, *_ = model.eval(selected_img, channels=channels_for_segmentation, diameter=diameter)
                imwrite(seg_path_ch1, masks)
                del selected_img, masks
                gc.collect()
            pbar.update(1)

    del data
    gc.collect()
    create_movie(file_name, 0, z_index, output_frames_folder, output_tracked_cells_folder)
    create_movie(file_name, 1, z_index, output_frames_folder, output_tracked_cells_folder)


# Batch process all .ims files
if __name__ == "__main__":
    ims_files = [f for f in os.listdir(working_dir) if f.endswith(".ims")]
    for f in ims_files:
        full_path = os.path.join(working_dir, f)
        process_ims_file(full_path)
