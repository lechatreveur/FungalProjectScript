#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 14:30:20 2025

@author: user
"""

import os
import gc
import numpy as np
from tifffile import imwrite  # imwrite is recommended over imsave
from tqdm import tqdm
from cellpose import models

# Import the IMS file reader package
from imaris_ims_file_reader import ims

# Paths
input_ims = "/Volumes/Movies/20250314/20250314_wt tea1_1/20250314_wt tea1_1_F2-005.ims"
output_frames_folder = "/Volumes/Movies/20250314/20250314_wt tea1_1/Frames_F2-005"
output_masks_folder = "/Volumes/Movies/20250314/20250314_wt tea1_1/Masks_F2-005"
custom_model_path = "/Volumes/Movies/20250314/20250314_wt tea1/Frames/models/CP_20250319_101813"

# Create output directories if they don't exist
os.makedirs(output_frames_folder, exist_ok=True)
os.makedirs(output_masks_folder, exist_ok=True)

# Create segmentation subfolders based on channel
GFP_seg_folder = os.path.join(output_masks_folder, "GFP_seg")
RFP_seg_folder = os.path.join(output_masks_folder, "RFP_seg")
brightfield_seg_folder = os.path.join(output_masks_folder, "brightfield_seg")
os.makedirs(GFP_seg_folder, exist_ok=True)
os.makedirs(RFP_seg_folder, exist_ok=True)
os.makedirs(brightfield_seg_folder, exist_ok=True)

#%% export images and perform segmentation
# Initialize the Cellpose model with your custom model
model = models.CellposeModel(gpu=True, pretrained_model=custom_model_path)
channels_for_segmentation = [0, 0]  # For grayscale images
diameter = 100

# Load the IMS file using imaris-ims-file-reader.
# Data shape is: (time, channel, z, height, width)
data = ims(input_ims)
time_points, n_channels, z_stacks, height, width = data.shape

# Set the chosen z-stack index for channel 2 segmentation (if available)
z_index = 4

# Total segmentation tasks: channels 0 & 1 (max projection) and channel 2 (single z-stack) per time point
total_tasks = time_points * 3

with tqdm(total=total_tasks, desc="Processing time points") as pbar:
    for t in range(time_points):
        # Process channels 0 and 1: maximum projection across z-stacks
        for c in [0, 1]:
            # data[t, c, :, :, :] has shape (z, height, width)
            projected_img = np.max(data[t, c, :, :, :], axis=0)
            # Save the projected image in the frames folder
            projected_filename = f"20250314_wt tea1_1_F2-005_t_{t:02d}_c_{c:01d}.tif"
            projected_path = os.path.join(output_frames_folder, projected_filename)
            imwrite(projected_path, projected_img)
            
            # Perform segmentation using Cellpose on the projected image
            masks, flows, styles, *extra = model.eval(projected_img,
                                                      channels=channels_for_segmentation,
                                                      diameter=diameter)
            # Choose the appropriate segmentation folder based on channel
            seg_folder = GFP_seg_folder if c == 0 else RFP_seg_folder
                
            seg_filename = f"20250314_wt tea1_1_F2-005_t_{t:02d}_c_{c:01d}_seg.tif"
            seg_path = os.path.join(seg_folder, seg_filename)
            imwrite(seg_path, masks)
            
            # Clean up memory for this task
            del projected_img, masks, flows, styles, extra
            gc.collect()
            pbar.update(1)

        # Process channel 2: select a specific z-stack for segmentation
        c = 2
        # Ensure the chosen z_index is within bounds
        z = z_index if z_index < z_stacks else z_stacks - 1
        selected_img = data[t, c, z, :, :]  # shape: (height, width)
        img_name = f"20250314_wt tea1_1_F2-005_t_{t:02d}_z_{z:01d}_c_{c:01d}.tif"
        img_path = os.path.join(output_frames_folder, img_name)
        imwrite(img_path, selected_img)
        
        masks, flows, styles, *extra = model.eval(selected_img,
                                                  channels=channels_for_segmentation,
                                                  diameter=diameter)
        seg_filename = f"20250314_wt tea1_1_F2-005_t_{t:02d}_z_{z:01d}_c_{c:01d}_seg.tif"
        seg_path = os.path.join(brightfield_seg_folder, seg_filename)
        imwrite(seg_path, masks)
        
        del selected_img, masks, flows, styles, extra
        gc.collect()
        pbar.update(1)

del data
gc.collect()

#%% Make movies with exported frames

import cv2
import fnmatch


# Define output movie directory
out_dir = "/Volumes/Movies/20250314/20250314_wt tea1_1/TrackedCells_F2-005"
os.makedirs(out_dir, exist_ok=True)

# Function to generate movie from frames
def create_movie(channel, z_index=None, frame_dir=output_frames_folder, output_dir=out_dir, fps=10):
    if channel in [0, 1]:
        frame_pattern = f"20250314_wt tea1_1_F2-005_t_??_c_{channel}.tif"
        movie_name = f"20250314_wt tea1_c_{channel:01d}.mp4"
    elif channel == 2 and z_index is not None:
        frame_pattern = f"20250314_wt tea1_1_F2-005_t_??_z_{z_index}_c_{channel}.tif"
        movie_name = f"20250314_wt tea1_1_F2-005_z_{z_index:01d}_c_{channel:01d}.mp4"
    else:
        raise ValueError("Unsupported channel or missing z_index for channel 2")

    frame_files = sorted([
        f for f in os.listdir(frame_dir)
        if f.endswith('.tif') and fnmatch.fnmatch(f, frame_pattern)
    ])

    if not frame_files:
        print(f"No frames found for channel {channel} (z={z_index})")
        return

    # Read first frame for size
    first_frame = imread(os.path.join(frame_dir, frame_files[0]))
    height, width = first_frame.shape
    out_path = os.path.join(output_dir, movie_name)

    # Use avc1 codec (compatible with QuickTime) and set grayscale (isColor=False)
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=False)

    for fname in frame_files:
        frame = imread(os.path.join(frame_dir, fname))
        # Normalize and convert to 8-bit grayscale
        frame_uint8 = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        out.write(frame_uint8)
    
    out.release()
    print(f"Saved QuickTime-compatible movie: {out_path}")


# Create movies for each channel
create_movie(channel=0)
create_movie(channel=1)
create_movie(channel=2, z_index=z_index)
#%%
import os
import numpy as np
from tifffile import imread, imwrite
from skimage.measure import regionprops, label, find_contours
import matplotlib.pyplot as plt
import cv2

# Define directories for raw images and segmentation masks
#frames_folder = "/Users/user/Documents/FungalProject/TimeLapse/20250314_wt tea1/Frames"



output_tracked_cells_folder = "/Volumes/Movies/20250314/20250314_wt tea1_1/TrackedCells_F2-005"


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
histogram_path = os.path.join(output_tracked_cells_folder, "cell_area_histogram.png")
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')  # Save the figure
plt.close()  # Close the figure to avoid displaying it in GUI

# (Optional) Plot the brightfield segmentation with cell IDs
plt.figure(figsize=(10, 8))
plt.imshow(labeled_mask, cmap='nipy_spectral')
for r in filtered_regions:
    y, x = r.centroid
    plt.text(x, y, str(r.label), color='white', fontsize=10, ha='center', va='center', weight='bold')
plt.title("Segmented Cells with Cell IDs (Brightfield)")
plt.axis('off')
segmentation_plot_path = os.path.join(output_tracked_cells_folder, "labeled_segmentation.png")
plt.savefig(segmentation_plot_path, dpi=300, bbox_inches='tight')  # Save the figure
plt.close()


#%% Make single-cell movies
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
        c = 0 
        fluorescent_GFP_frame_path = os.path.join(output_frames_folder, f"20250314_wt tea1_1_F2-005_t_{t:02d}_c_{c:01d}.tif")
        c = 1
        fluorescent_RFP_frame_path = os.path.join(output_frames_folder, f"20250314_wt tea1_1_F2-005_t_{t:02d}_c_{c:01d}.tif")
        c = 2
        z = 4
        brightfield_frame_path = os.path.join(output_frames_folder, f"20250314_wt tea1_1_F2-005_t_{t:02d}_z_{z:01d}_c_{c:01d}.tif")
        
        

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
        
        # # ----- Load and Match GFP Segmentation -----
        # c = 0
        # gfp_seg_path = os.path.join(GFP_seg_folder, f"20250314_wt tea1_1_F2-005_t_{t:02d}_c_{c:01d}_seg.tif")
        # if os.path.exists(gfp_seg_path):
        #     gfp_seg = load_segmentation(gfp_seg_path)
        #     current_cell_mask_gfp, bbox_gfp = get_cell_bbox(gfp_seg, current_cell_mask, prev_bbox_gfp, threshold=0.7)
        #     prev_bbox_gfp = bbox_gfp

        # else:
        #     current_cell_mask_gfp = prev_cell_mask_gfp

        # # ----- Load and Match RFP Segmentation -----
        # c = 1
        # rfp_seg_path = os.path.join(RFP_seg_folder, f"20250314_wt tea1_1_F2-005_t_{t:02d}_c_{c:01d}_seg.tif")
        # if os.path.exists(rfp_seg_path):
        #     rfp_seg = load_segmentation(rfp_seg_path)
        #     current_cell_mask_rfp, bbox_rfp = get_cell_bbox(rfp_seg, current_cell_mask, prev_bbox_rfp, threshold=0.7)
        #     prev_bbox_rfp = bbox_rfp
        # else:
        #     current_cell_mask_rfp = prev_cell_mask_rfp
            


        # ----- Normalize Images for Display -----
        BFmin, BFmax = brightfield_frame.min(), brightfield_frame.max()
        C1min, C1max = 98, 211  # Adjust based on your data
        C2min, C2max = 98, 211  # Adjust based on your data
        brightfield_frame = np.clip(((brightfield_frame - BFmin) / (BFmax - BFmin) * 255), 0, 255).astype(np.uint8)
        fluorescent_frame_C1 = np.clip(((fluorescent_frame_C1 - C1min) / (C1max - C1min) * 255), 0, 255).astype(np.uint8)
        fluorescent_frame_C2 = np.clip(((fluorescent_frame_C2 - C2min) / (C2max - C2min) * 255), 0, 255).astype(np.uint8)
        
        # ----- Draw Dashed Contours for Each Channel -----
        draw_dashed_contour(brightfield_frame, current_cell_mask, 0)
        draw_dashed_contour(fluorescent_frame_C1, current_cell_mask, 255)
        draw_dashed_contour(fluorescent_frame_C2, current_cell_mask, 255)
        # if current_cell_mask_gfp is not None:
        #     draw_dashed_contour(fluorescent_frame_C1, current_cell_mask_gfp, 255)
        # if current_cell_mask_rfp is not None:
        #     draw_dashed_contour(fluorescent_frame_C2, current_cell_mask_rfp, 255)
        
        # Crop each frame to the bounding box
        brightfield_frame_crop = brightfield_frame[y_min:y_max, x_min:x_max]
        fluorescent_frame_C1_crop = fluorescent_frame_C1[y_min:y_max, x_min:x_max]
        fluorescent_frame_C2_crop = fluorescent_frame_C2[y_min:y_max, x_min:x_max]
        # fluorescent_frame_C1_crop = fluorescent_frame_C1[bbox_gfp[0]:bbox_gfp[1], bbox_gfp[2]:bbox_gfp[3]]
        # fluorescent_frame_C2_crop = fluorescent_frame_C2[bbox_rfp[0]:bbox_rfp[1], bbox_rfp[2]:bbox_rfp[3]]
        

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
        prev_cell_mask_gfp = current_cell_mask
        prev_cell_mask_rfp = current_cell_mask
        #prev_cell_mask_gfp = current_cell_mask_gfp
        #prev_cell_mask_rfp = current_cell_mask_rfp

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


