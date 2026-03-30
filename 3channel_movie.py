#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 14:32:07 2025

@author: user
"""

import os
from cellpose import models, io
import numpy as np

# Define base and model directories
base_dir = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/Frames"
model_dir = os.path.join(base_dir, "models/CP_20250319_101813")

# Define output directories for each channel
brightfield_out = os.path.join(base_dir, "brightfield_seg")
GFP_out = os.path.join(base_dir, "GFP_seg")
RFP_out = os.path.join(base_dir, "RFP_seg")

# Create output directories if they don't exist
os.makedirs(brightfield_out, exist_ok=True)
os.makedirs(GFP_out, exist_ok=True)
os.makedirs(RFP_out, exist_ok=True)

# Load your custom cellpose model.
# If your custom model was saved via cellpose, you can load it by giving the model file or directory.
# Here we assume the model is in model_dir (adjust the parameters as needed).
model = models.CellposeModel(pretrained_model=model_dir)

# Loop through all .tif files in the base directory
for file in os.listdir(base_dir):
    if file.endswith(".tif"):
        file_path = os.path.join(base_dir, file)
        
        # Determine which channel folder to use based on the filename
        if "brightfield" in file:
            out_dir = brightfield_out
            channels = [0, 0]  # cellpose default for grayscale images
        elif "GFP" in file:
            out_dir = GFP_out
            channels = [0, 0]  # adjust if needed for your GFP setup
        elif "RFP" in file:
            out_dir = RFP_out
            channels = [0, 0]  # adjust if needed for your RFP setup
        else:
            continue  # skip files that do not match any channel
        
        # Read image
        img = io.imread(file_path)
        # Run segmentation
        masks, flows, styles = model.eval(img, 100, channels=channels)

        
        # Save the segmentation mask; here saving as a TIFF file (you can also save as .npy if preferred)
        out_file = os.path.join(out_dir, file.replace(".tif", "_seg.tif"))
        io.imsave(out_file, masks.astype(np.uint16))
        
        print(f"Segmented {file} and saved result to {out_file}")
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
#%%
import os
import numpy as np
from tifffile import imread
from skimage.measure import regionprops, label, find_contours
import matplotlib.pyplot as plt
import cv2

# Define directories for raw images and segmentation masks
frames_folder = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/Frames"
brightfield_seg_folder = os.path.join(frames_folder, "brightfield_seg")
GFP_seg_folder = os.path.join(frames_folder, "GFP_seg")
RFP_seg_folder = os.path.join(frames_folder, "RFP_seg")
output_tracked_cells_folder = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/TrackedCells"
output_frames_folder = frames_folder  # raw images are in the frames folder

os.makedirs(output_tracked_cells_folder, exist_ok=True)

# Helper function to load a segmentation file (either .npy or .tif)
def load_segmentation(path):
    if path.endswith('.npy'):
        return np.load(path)
    else:
        return imread(path)

# Get list of brightfield segmentation files (used for tracking)
mask_files = sorted([f for f in os.listdir(brightfield_seg_folder) 
                     if f.endswith('_seg.npy') or f.endswith('_seg.tif')])
mask_paths = [os.path.join(brightfield_seg_folder, f) for f in mask_files]

# Load the first brightfield segmentation mask and detect cells
first_mask = load_segmentation(mask_paths[0])
labeled_mask = label(first_mask)
regions = regionprops(labeled_mask)

# Filter out small cells based on area threshold
min_area = 2500  
filtered_regions = [r for r in regions if r.area >= min_area]

# (Optional) Plot histogram of filtered cell areas
areas = [r.area for r in filtered_regions]
plt.figure(figsize=(8, 6))
plt.hist(areas, bins=20, color='blue', edgecolor='black')
plt.xlabel("Cell Area (pixels)")
plt.ylabel("Frequency")
plt.title("Histogram of Filtered Cell Areas")
plt.show()

# (Optional) Plot the brightfield segmentation with cell IDs
plt.figure(figsize=(10, 8))
plt.imshow(labeled_mask, cmap='nipy_spectral')
for r in filtered_regions:
    y, x = r.centroid
    plt.text(x, y, str(r.label), color='white', fontsize=10, ha='center', va='center', weight='bold')
plt.title("Segmented Cells with Cell IDs (Brightfield)")
plt.axis('off')
plt.show()

# Function to compute the percentage overlap between two binary masks
def compute_overlap(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    total_area = mask1.sum()
    return intersection / total_area if total_area > 0 else 0

# Function to draw a dashed contour on an image from a binary mask
def draw_dashed_contour(image, mask, color, thickness=1, gap=5):
    contours = find_contours(mask, 0.5)
    for contour in contours:
        contour = np.round(contour).astype(int)
        for i in range(0, len(contour) - 1, gap * 2):
            start = tuple(contour[i][::-1])
            end = tuple(contour[min(i + gap, len(contour) - 1)][::-1])
            cv2.line(image, start, end, int(color), thickness, cv2.LINE_AA)

# Given a segmentation mask and a reference mask, find the region that best overlaps
def get_best_match(segmentation, reference_mask, threshold=0.7):
    labeled = label(segmentation)
    regions = regionprops(labeled)
    best_region = None
    max_overlap = 0
    for region in regions:
        candidate_mask = (labeled == region.label)
        overlap = compute_overlap(reference_mask, candidate_mask)
        if overlap >= threshold and overlap > max_overlap:
            best_region = candidate_mask
            max_overlap = overlap
    return best_region

def get_cell_bbox(segmentation, prev_cell_mask, prev_bbox, threshold=0.7):
    """
    Given a segmentation image, the previous cell mask, and the previous bounding box,
    this function finds the best matching cell region based on overlap.
    
    Parameters:
      segmentation (ndarray): The segmentation image (binary or labeled) for the current frame.
      prev_cell_mask (ndarray): The binary mask of the cell from the previous frame.
      prev_bbox (tuple): Previous bounding box as (y_min, y_max, x_min, x_max).
      threshold (float): Minimum required overlap fraction to accept a candidate.
      
    Returns:
      current_cell_mask (ndarray): The binary mask of the current cell region.
      bbox (tuple): The bounding box for the current cell region, in the form (y_min, y_max, x_min, x_max).
    """
    # Label the segmentation and extract regions.
    labeled_current = label(segmentation)
    current_regions = regionprops(labeled_current)
    best_candidate = None
    max_overlap = 0
    for candidate in current_regions:
        candidate_mask = (labeled_current == candidate.label)
        overlap = compute_overlap(prev_cell_mask, candidate_mask)
        if overlap >= threshold and overlap > max_overlap:
            best_candidate = candidate
            max_overlap = overlap

    if best_candidate is None:
        print("Cell lost in segmentation. Using previous segmentation.")
        return prev_cell_mask, prev_bbox
    else:
        current_cell_mask = (labeled_current == best_candidate.label)
        # Use the candidate's centroid and bounding box dimensions to define a new box.
        centroid = np.round(best_candidate.centroid).astype(int)
        y, x = centroid
        minr, minc, maxr, maxc = best_candidate.bbox
        bbox_size = max(maxr - minr, maxc - minc)
        bbox_size = max(bbox_size, 10)  # Ensure a minimum size.
        y_min = max(0, y - bbox_size // 2)
        y_max = min(current_cell_mask.shape[0], y + bbox_size // 2)
        x_min = max(0, x - bbox_size // 2)
        x_max = min(current_cell_mask.shape[1], x + bbox_size // 2)
        return current_cell_mask, (y_min, y_max, x_min, x_max)

# Process each detected cell (tracking based on brightfield segmentation)
for cell in filtered_regions:
    cell_id = cell.label
    centroid = np.round(cell.centroid).astype(int)
    y, x = centroid

    # Initialize timelapse frame lists for each channel
    brightfield_timelapse_frames = []
    fluorescent_timelapse_frames_C1 = []
    fluorescent_timelapse_frames_C2 = []

    # Use the brightfield segmentation mask for tracking
    prev_cell_mask = (labeled_mask == cell_id)
    
    # Define initial bounding box based on the cell's properties
    minr, minc, maxr, maxc = cell.bbox
    bbox_size = max(maxr - minr, maxc - minc)
    bbox_size = max(bbox_size, 10)
    y_min = max(0, y - bbox_size // 2)
    y_max = min(first_mask.shape[0], y + bbox_size // 2)
    x_min = max(0, x - bbox_size // 2)
    x_max = min(first_mask.shape[1], x + bbox_size // 2)
    prev_bbox = (y_min, y_max, x_min, x_max)
    prev_bbox_gfp = prev_bbox
    prev_bbox_rfp = prev_bbox

    # Optionally, initialize previous masks for GFP and RFP channels
    prev_cell_mask_gfp = None
    prev_cell_mask_rfp = None

    # Loop over each timepoint using the brightfield segmentation masks
    for t, bf_mask_path in enumerate(mask_paths):
        # Define paths for the raw images for each channel
        brightfield_frame_path = os.path.join(output_frames_folder, f"brightfield_time{t:03d}_z002.tif")
        fluorescent_GFP_frame_path = os.path.join(output_frames_folder, f"fluorescent_GFP_{t:03d}.tif")
        fluorescent_RFP_frame_path = os.path.join(output_frames_folder, f"fluorescent_RFP_{t:03d}.tif")

        # Skip frame if any image is missing
        if not (os.path.exists(brightfield_frame_path) and 
                os.path.exists(fluorescent_GFP_frame_path) and 
                os.path.exists(fluorescent_RFP_frame_path)):
            print(f"Skipping missing frame at time {t}")
            continue

        # ----- Brightfield Tracking -----
       
        bf_seg = load_segmentation(bf_mask_path)
        current_cell_mask, (y_min, y_max, x_min, x_max) = get_cell_bbox(bf_seg, prev_cell_mask, prev_bbox, threshold=0.7)
        prev_bbox = (y_min, y_max, x_min, x_max)

        
        # ----- Load Raw Images -----
        brightfield_frame = imread(brightfield_frame_path)
        fluorescent_frame_C1 = imread(fluorescent_GFP_frame_path)
        fluorescent_frame_C2 = imread(fluorescent_RFP_frame_path)
        
        # ----- Load and Match GFP Segmentation -----
        gfp_seg_path = os.path.join(GFP_seg_folder, f"fluorescent_GFP_{t:03d}_seg.tif")
        if os.path.exists(gfp_seg_path):
            gfp_seg = load_segmentation(gfp_seg_path)
            current_cell_mask_gfp, bbox_gfp = get_cell_bbox(gfp_seg, current_cell_mask, prev_bbox_gfp, threshold=0.7)
            prev_bbox_gfp = bbox_gfp

        else:
            current_cell_mask_gfp = prev_cell_mask_gfp

        # ----- Load and Match RFP Segmentation -----
        rfp_seg_path = os.path.join(RFP_seg_folder, f"fluorescent_RFP_{t:03d}_seg.tif")
        if os.path.exists(rfp_seg_path):
            rfp_seg = load_segmentation(rfp_seg_path)
            current_cell_mask_rfp, bbox_rfp = get_cell_bbox(rfp_seg, current_cell_mask, prev_bbox_rfp, threshold=0.7)
            prev_bbox_rfp = bbox_rfp
        else:
            current_cell_mask_rfp = prev_cell_mask_rfp
            


        # ----- Normalize Images for Display -----
        BFmin, BFmax = brightfield_frame.min(), brightfield_frame.max()
        C1min, C1max = 98, 211  # Adjust based on your data
        C2min, C2max = 98, 211  # Adjust based on your data
        brightfield_frame = np.clip(((brightfield_frame - BFmin) / (BFmax - BFmin) * 255), 0, 255).astype(np.uint8)
        fluorescent_frame_C1 = np.clip(((fluorescent_frame_C1 - C1min) / (C1max - C1min) * 255), 0, 255).astype(np.uint8)
        fluorescent_frame_C2 = np.clip(((fluorescent_frame_C2 - C2min) / (C2max - C2min) * 255), 0, 255).astype(np.uint8)
        
        # ----- Draw Dashed Contours for Each Channel -----
        draw_dashed_contour(brightfield_frame, current_cell_mask, 0)
        if current_cell_mask_gfp is not None:
            #print("yes")
            draw_dashed_contour(fluorescent_frame_C1, current_cell_mask_gfp, 255)
        if current_cell_mask_rfp is not None:
            draw_dashed_contour(fluorescent_frame_C2, current_cell_mask_rfp, 255)
        
        # Crop each frame to the bounding box
        brightfield_frame_crop = brightfield_frame[y_min:y_max, x_min:x_max]
        fluorescent_frame_C1_crop = fluorescent_frame_C1[bbox_gfp[0]:bbox_gfp[1], bbox_gfp[2]:bbox_gfp[3]]
        fluorescent_frame_C2_crop = fluorescent_frame_C2[bbox_rfp[0]:bbox_rfp[1], bbox_rfp[2]:bbox_rfp[3]]

        # Add a timestamp (frame number) on the images
        brightfield_frame_crop = cv2.putText(brightfield_frame_crop.copy(), f'{t}', (0, bbox_size),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        fluorescent_frame_C1_crop = cv2.putText(fluorescent_frame_C1_crop.copy(), f'{t}', (0, bbox_size),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        fluorescent_frame_C2_crop = cv2.putText(fluorescent_frame_C2_crop.copy(), f'{t}', (0, bbox_size),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        brightfield_timelapse_frames.append(brightfield_frame_crop)
        fluorescent_timelapse_frames_C1.append(fluorescent_frame_C1_crop)
        fluorescent_timelapse_frames_C2.append(fluorescent_frame_C2_crop)
        
        prev_cell_mask = current_cell_mask
        prev_cell_mask_gfp = current_cell_mask_gfp
        prev_cell_mask_rfp = current_cell_mask_rfp

    # Function to pad frames to a uniform target size
    def pad_frame(frame, target_size):
        h, w = frame.shape
        target_h, target_w = target_size
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        return np.pad(frame, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
        
       # Save timelapse videos for each channel
    for channel_name, frames in zip(["brightfield", "fluorescent_C1", "fluorescent_C2"],
                                    [brightfield_timelapse_frames, fluorescent_timelapse_frames_C1, fluorescent_timelapse_frames_C2]):
        if not frames:
            print(f"No frames found for cell {cell_id} in {channel_name}. Skipping movie creation.")
            continue
    
        # Calculate maximum dimensions for this channel's frames
        channel_max_height = max(frame.shape[0] for frame in frames)
        channel_max_width = max(frame.shape[1] for frame in frames)
        channel_target_size = (channel_max_height, channel_max_width)
    
        video_path = os.path.join(output_tracked_cells_folder, f"cell_{cell_id}_{channel_name}_timelapse.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        video = cv2.VideoWriter(video_path, fourcc, 10, (channel_max_width, channel_max_height), False)
    
        for frame in frames:
            # Pad each frame to the target dimensions for this channel
            padded_frame = pad_frame(frame, channel_target_size)
            video.write(padded_frame)
    
        video.release()
        print(f"Saved complete {channel_name} timelapse for Cell {cell_id} at {video_path}")

