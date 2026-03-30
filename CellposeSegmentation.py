#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:03:20 2025

@author: user
"""

import os
from tifffile import TiffFile, imsave
from cellpose import models
import numpy as np
from skimage.color import rgb2gray  # Import for RGB to grayscale conversion

# Paths
input_tif = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/20250314-1DIC.tiff"
output_frames_folder = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/Frames"
output_masks_folder = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/Masks"
custom_model_path = "/Users/user/.cellpose/models/yeast_BF_cp3"

# Create output directories if they don't exist
os.makedirs(output_frames_folder, exist_ok=True)
os.makedirs(output_masks_folder, exist_ok=True)

# Initialize Cellpose model with a custom model
model = models.CellposeModel(gpu=True, pretrained_model=custom_model_path)

# Parameters for segmentation
channels = [0, 0]  # Assuming grayscale images; adjust if needed
diameter = 100  # Set a specific diameter for cells

# Load the multi-frame TIFF
with TiffFile(input_tif) as tif:
    frames = tif.asarray()
# Process each frame
for i, frame in enumerate(frames):
    # Convert to grayscale if RGB
    if frame.ndim == 3:
        # Ensure correct axis order and convert to grayscale
        frame_grayscale = rgb2gray(frame.transpose(1, 2, 0))
        frame_grayscale = (frame_grayscale * 65535).astype(np.uint16)  # Scale to 16-bit
    else:
        frame_grayscale = frame  # Already grayscale

    # Save each frame as an individual TIFF
    frame_filename = os.path.join(output_frames_folder, f"brightfield_{i:03d}.tif")
    imsave(frame_filename, frame_grayscale)
    print(f"Saved grayscale frame: {frame_filename}")

    # Perform segmentation on the frame
    masks, flows, styles = model.eval(
        frame_grayscale, diameter=diameter, channels=channels
    )

    # Save the segmented mask
    mask_filename = os.path.join(output_masks_folder, f"segmented_timepoint_{i:03d}.tif")
    imsave(mask_filename, masks.astype(np.uint16))
    print(f"Saved segmentation mask: {mask_filename}")

print("Timelapse segmentation completed.")
#%%
# Paths
input_tif = "/Users/user/Documents/FungalProject/TimeLapse/MAX_P1_Lng_LVCC-C1C2.tif"
output_frames_folder = "/Users/user/Documents/FungalProject/TimeLapse/Frames"

# Create output directory if it doesn't exist
os.makedirs(output_frames_folder, exist_ok=True)

# Load the multi-frame TIFF
with TiffFile(input_tif) as tif:
    frames = tif.asarray()

# Ensure correct shape
num_frames, num_channels, height, width = frames.shape
fluorescent_C1_frames = frames[:, 0, :, :]
fluorescent_C2_frames = frames[:, 1, :, :]

# Process each frame for both fluorescent channels
for i in range(num_frames):
    fluorescent_C1_frame = fluorescent_C1_frames[i]
    fluorescent_C2_frame = fluorescent_C2_frames[i]

    # Save each channel's frame as an individual TIFF without modifying dtype
    fluorescent_C1_filename = os.path.join(output_frames_folder, f"fluorescent_C1_{i:03d}.tif")
    fluorescent_C2_filename = os.path.join(output_frames_folder, f"fluorescent_C2_{i:03d}.tif")
    
    imsave(fluorescent_C1_filename, fluorescent_C1_frame)
    imsave(fluorescent_C2_filename, fluorescent_C2_frame)
    
    print(f"Saved fluorescent channel frames: {fluorescent_C1_filename}, {fluorescent_C2_filename}")

print("Fluorescent channel frame extraction completed.")





