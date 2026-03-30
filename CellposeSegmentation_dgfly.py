#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:50:56 2025

@author: user
"""

import os
from tifffile import TiffFile, imsave
from cellpose import models
import numpy as np
import gc

# Paths
input_tif = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/20250314-1DIC.tiff"
output_frames_folder = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/Frames"
output_masks_folder = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/Masks"
#custom_model_path = "/Users/user/.cellpose/models/yeast_BF_cp3"
custom_model_path = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/Frames/models/CP_20250319_101813"

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
    slice_img = data[t, z_index, :, :]

    # Save the brightfield image for reference
    frame_filename = os.path.join(output_frames_folder, f"brightfield_time{t:03d}_z{z_index:03d}.tif")
    imsave(frame_filename, slice_img)
    print(f"Saved brightfield image: {frame_filename}")

    # Perform segmentation on the selected slice
    masks, flows, styles = model.eval(slice_img, diameter=diameter, channels=channels)

    # Save the segmentation mask
    mask_filename = os.path.join(output_masks_folder, f"segmented_time{t:03d}_z{z_index:03d}.tif")
    imsave(mask_filename, masks.astype(np.uint16))
    print(f"Saved segmentation mask: {mask_filename}")

print("Segmentation completed for all time points at z-index 2.")

# Cleanup
del data
gc.collect()

#%% export fluorescent channels
import os
from tifffile import TiffFile, imwrite

# Paths for the two fluorescent channels
input_GFP_tif = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/20250314-1GFP.tif"
input_RFP_tif = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/20250314-1RFP.tif"
output_frames_folder = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/Frames"

# Create the output directory if it doesn't exist
os.makedirs(output_frames_folder, exist_ok=True)

def process_fluorescent_tif(input_tif, channel_name):
    """
    Process a fluorescent TIFF file and save each frame with a channel-specific filename.
    The function handles TIFF files that are either 3D (num_frames, height, width)
    or 4D (num_frames, 1, height, width).
    """
    with TiffFile(input_tif) as tif:
        frames = tif.asarray()

    # Check and adjust for array shape.
    if frames.ndim == 4:
        num_frames, num_channels, height, width = frames.shape
        # If there is a singleton channel dimension, extract the first channel.
        frames = frames[:, 0, :, :]
    elif frames.ndim == 3:
        num_frames, height, width = frames.shape
    else:
        raise ValueError(f"Unexpected TIFF shape: {frames.shape}")

    print(f"Processing {num_frames} frames for {channel_name} with shape: {frames.shape}")

    # Iterate through frames and save each one individually
    for i in range(num_frames):
        frame = frames[i]
        output_filename = os.path.join(output_frames_folder, f"{channel_name}_{i:03d}.tif")
        imwrite(output_filename, frame)
        print(f"Saved {channel_name} frame: {output_filename}")

# Process the GFP channel
process_fluorescent_tif(input_GFP_tif, "fluorescent_GFP")

# Process the RFP channel
process_fluorescent_tif(input_RFP_tif, "fluorescent_RFP")

print("Fluorescent channel frame extraction completed.")

del tif
gc.collect()

#%%
import os
import numpy as np
from tifffile import imread
from skimage.measure import regionprops, label
import matplotlib.pyplot as plt

# Paths
input_masks_folder = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/Masks"
output_tracked_cells_folder = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/TrackedCells"

# Create necessary output directories
os.makedirs(output_tracked_cells_folder, exist_ok=True)

# Get list of mask files (sorted by timepoint)
mask_files = sorted([f for f in os.listdir(input_masks_folder) if f.endswith('.tif')])
mask_paths = [os.path.join(input_masks_folder, f) for f in mask_files]

# Load the first mask and detect cells
first_mask = imread(mask_paths[0])
labeled_mask = label(first_mask)
regions = regionprops(labeled_mask)

# Extract area measurements from each region
areas = [region.area for region in regions]

# Filter out small areas; adjust min_area threshold as needed
min_area = 2500  
filtered_areas = [area for area in areas if area >= min_area]

# Plot the histogram of filtered cell areas
plt.figure(figsize=(8, 6))
plt.hist(filtered_areas, bins=20, color='blue', edgecolor='black')
plt.xlabel("Cell Area (pixels)")
plt.ylabel("Frequency")
plt.title("Histogram of Filtered Cell Areas")
plt.show()

# Plot the segmentation with cell IDs
plt.figure(figsize=(10, 8))
# Use a colormap that differentiates labels (each cell gets a different color)
plt.imshow(labeled_mask, cmap='nipy_spectral')
for region in regions:
    # Only label cells with area above the threshold
    if region.area >= min_area:
        # region.centroid returns (row, col) coordinates
        y, x = region.centroid
        plt.text(x, y, str(region.label), color='white', fontsize=10,
                 ha='center', va='center', weight='bold')
plt.title("Segmented Cells with Cell IDs")
plt.axis('off')
plt.show()

#%%
from skimage.measure import regionprops, label, find_contours
import cv2
import numpy as np
import os
from tifffile import imread

def compute_overlap(mask1, mask2):
    """Computes the percentage of overlap between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    total_area = mask1.sum()
    return intersection / total_area if total_area > 0 else 0

def draw_dashed_contour(image, mask, color, thickness=1, gap=5):
    """Draws a dashed outline around the mask on the given image."""
    contours = find_contours(mask, 0.5)
    for contour in contours:
        contour = np.round(contour).astype(int)
        for i in range(0, len(contour) - 1, gap * 2):  # Draw segments with gaps
            start = tuple(contour[i][::-1])  # Reverse (y, x) to (x, y) for OpenCV
            end = tuple(contour[min(i + gap, len(contour) - 1)][::-1])
            cv2.line(image, start, end, int(color), thickness, cv2.LINE_AA)

# Assuming 'regions', 'labeled_mask', 'first_mask', 'mask_paths', and 'output_frames_folder' are defined earlier.
# For example, you might have something like:
# first_mask = imread(path_to_first_mask)
# labeled_mask = label(first_mask)
# regions = regionprops(labeled_mask)
# mask_paths = sorted([...])
# output_frames_folder = "path/to/frames"

# Define the minimum area threshold to filter out small cells
min_area = 2500  

# Initialize prev_bbox before entering the loop (used if tracking is lost)
prev_bbox = None  

# Process each detected cell (only process cells with area >= min_area)
for cell in regions:
    if cell.area < min_area:
        continue  # Skip cells that do not meet the area threshold

    cell_id = cell.label
    centroid = np.round(cell.centroid).astype(int)
    y, x = centroid

    brightfield_timelapse_frames = []
    fluorescent_timelapse_frames_C1 = []
    fluorescent_timelapse_frames_C2 = []

    prev_cell_mask = (labeled_mask == cell_id)

    # Get initial bounding box
    minr, minc, maxr, maxc = cell.bbox
    bbox_size = max(maxr - minr, maxc - minc)
    bbox_size = max(bbox_size, 10)  # Ensure a minimum bbox size

    y_min = max(0, y - bbox_size // 2)
    y_max = min(first_mask.shape[0], y + bbox_size // 2)
    x_min = max(0, x - bbox_size // 2)
    x_max = min(first_mask.shape[1], x + bbox_size // 2)

    prev_bbox = (y_min, y_max, x_min, x_max)  # Initialize bounding box

    for t, mask_path in enumerate(mask_paths):
        brightfield_frame_path = os.path.join(output_frames_folder, f"brightfield_time{t:03d}_z002.tif")
        fluorescent_GFP_frame_path = os.path.join(output_frames_folder, f"fluorescent_GFP_{t:03d}.tif")
        fluorescent_RFP_frame_path = os.path.join(output_frames_folder, f"fluorescent_RFP_{t:03d}.tif")


        if not os.path.exists(brightfield_frame_path) or not os.path.exists(fluorescent_GFP_frame_path) or not os.path.exists(fluorescent_RFP_frame_path):
            print(f"Skipping missing frame at time {t}")
            continue

        current_masks = imread(mask_path)
        labeled_current = label(current_masks)
        current_regions = regionprops(labeled_current)

        best_match = None
        max_overlap = 0

        for candidate in current_regions:
            candidate_mask = (labeled_current == candidate.label)
            overlap = compute_overlap(prev_cell_mask, candidate_mask)
            if overlap >= 0.7 and overlap > max_overlap:
                best_match = candidate
                max_overlap = overlap

        if best_match is None:
            print(f"Cell {cell_id} lost at frame {t}. Using previous segmentation.")
            # Keep using the previous mask and bounding box
            current_cell_mask = prev_cell_mask
            y_min, y_max, x_min, x_max = prev_bbox  # Use last known bbox
        else:
            # Update mask tracking when a match is found
            current_cell_mask = (labeled_current == best_match.label)
            centroid = np.round(best_match.centroid).astype(int)
            y, x = centroid
            minr, minc, maxr, maxc = best_match.bbox
            bbox_size = max(maxr - minr, maxc - minc)
            bbox_size = max(bbox_size, 10)
            # Save bbox for reuse if tracking is lost later
            y_min = max(0, y - bbox_size // 2)
            y_max = min(current_cell_mask.shape[0], y + bbox_size // 2)
            x_min = max(0, x - bbox_size // 2)
            x_max = min(current_cell_mask.shape[1], x + bbox_size // 2)
            prev_bbox = (y_min, y_max, x_min, x_max)  # Update bbox

        brightfield_frame = imread(brightfield_frame_path)
        fluorescent_frame_C1 = imread(fluorescent_GFP_frame_path)
        fluorescent_frame_C2 = imread(fluorescent_RFP_frame_path)
        
        BFmin = brightfield_frame.min()
        BFmax = brightfield_frame.max()
        C1min = 98  # Adjust as needed
        C1max = 211  # Adjust as needed
        C2min = 98  # Adjust as needed
        C2max = 211  # Adjust as needed

        brightfield_frame = np.clip(((brightfield_frame - BFmin) / (BFmax - BFmin) * 255), 0, 255).astype(np.uint8)
        fluorescent_frame_C1 = np.clip(((fluorescent_frame_C1 - C1min) / (C1max - C1min) * 255), 0, 255).astype(np.uint8)
        fluorescent_frame_C2 = np.clip(((fluorescent_frame_C2 - C2min) / (C2max - C2min) * 255), 0, 255).astype(np.uint8)
        
        draw_dashed_contour(brightfield_frame, current_cell_mask, 0)
        draw_dashed_contour(fluorescent_frame_C1, current_cell_mask, 255)
        draw_dashed_contour(fluorescent_frame_C2, current_cell_mask, 255)
        
        brightfield_frame = brightfield_frame[y_min:y_max, x_min:x_max]
        fluorescent_frame_C1 = fluorescent_frame_C1[y_min:y_max, x_min:x_max]
        fluorescent_frame_C2 = fluorescent_frame_C2[y_min:y_max, x_min:x_max]

        brightfield_frame = cv2.putText(brightfield_frame.copy(), f'{t}', (0, bbox_size),
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        fluorescent_frame_C1 = cv2.putText(fluorescent_frame_C1.copy(), f'{t}', (0, bbox_size),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        fluorescent_frame_C2 = cv2.putText(fluorescent_frame_C2.copy(), f'{t}', (0, bbox_size),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        brightfield_timelapse_frames.append(brightfield_frame)
        fluorescent_timelapse_frames_C1.append(fluorescent_frame_C1)
        fluorescent_timelapse_frames_C2.append(fluorescent_frame_C2)
        
        prev_cell_mask = current_cell_mask

    def pad_frame(frame, target_size):
        """Pads a frame to match the target size (height, width) while centering the image."""
        h, w = frame.shape
        target_h, target_w = target_size
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        return np.pad(frame, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    
    for channel_name, frames in zip(["brightfield", "fluorescent_C1", "fluorescent_C2"],
                                    [brightfield_timelapse_frames, fluorescent_timelapse_frames_C1, fluorescent_timelapse_frames_C2]):
        if not frames:
            print(f"No frames found for cell {cell_id} in {channel_name}. Skipping movie creation.")
            continue
    
        # Find the maximum height and width across all frames
        max_height = max(frame.shape[0] for frame in frames)
        max_width = max(frame.shape[1] for frame in frames)
        target_size = (max_height, max_width)
    
        # Define video output path
        video_path = os.path.join(output_tracked_cells_folder, f"cell_{cell_id}_{channel_name}_timelapse.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video = cv2.VideoWriter(video_path, fourcc, 10, (max_width, max_height), False)
    
        for frame in frames:
            padded_frame = pad_frame(frame, target_size)  # Ensure all frames have the same size
            video.write(padded_frame)
    
        video.release()
        print(f"Saved complete {channel_name} timelapse for Cell {cell_id} at {video_path}")
