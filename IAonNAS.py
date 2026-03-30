#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 14:06:58 2025

@author: user
"""


import os
from tifffile import TiffFile, imsave
from cellpose import models
import numpy as np
import gc

# Paths
input_tif = "/Volumes/Movies/20250320--M49/1/MAX_20250320_wt tea1_F1-007_DIC.tif"
output_frames_folder = "/Volumes/Movies/20250320--M49/1/Frames-007"
output_masks_folder = "/Volumes/Movies/20250320--M49/1/Masks-007"

custom_model_path = "/Volumes/Movies/20250314/20250314_wt tea1/Frames/models/CP_20250319_101813"

# Create output directories if they don't exist
os.makedirs(output_frames_folder, exist_ok=True)
os.makedirs(output_masks_folder, exist_ok=True)

# Initialize the Cellpose model with your custom model
model = models.CellposeModel(gpu=True, pretrained_model=custom_model_path)

# Parameters for segmentation
channels = [0, 0]  # For grayscale images
diameter = 100     # Specific diameter for cells

# Load the multi-dimensional TIFF file
# Expected shape: (timepoints, z-stacks, width, height)
with TiffFile(input_tif) as tif:
    data = tif.asarray()

# Define the z-index you want to process for each time point
z_index = 2
print(f"Processing segmentation on z-stack index {z_index} for each time point.")

# Loop over each time point
for t in range(data.shape[0]):
    # Check if the requested z-index exists for this time point
    if z_index >= data.shape[1]:
        print(f"Time point {t} does not have z-index {z_index}. Skipping.")
        continue

    # Extract the slice at the chosen z-index for the current time point
    #slice_img = data[t, z_index, :, :]
    slice_img = data[t, :, :]

    # Save the brightfield image for reference
    #frame_filename = os.path.join(output_frames_folder, f"brightfield_time{t:03d}_z{z_index:03d}.tif")
    frame_filename = os.path.join(output_frames_folder, f"20250320_DIC_time{t:03d}.tif")
    imsave(frame_filename, slice_img)
    print(f"Saved brightfield image: {frame_filename}")

    # Perform segmentation on the selected slice
    masks, flows, styles = model.eval(slice_img, diameter=diameter, channels=channels)

    # Save the segmentation mask
    mask_filename = os.path.join(output_masks_folder, f"segmented_time{t:03d}_z{z_index:03d}.tif")
    #imsave(mask_filename, masks.astype(np.uint16))
    print(f"Saved segmentation mask: {mask_filename}")

print("Segmentation completed for all time points at z-index 2.")

# Cleanup
del data
gc.collect()