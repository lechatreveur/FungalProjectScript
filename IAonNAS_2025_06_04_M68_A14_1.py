#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 11:30:26 2025

@author: user
"""

import os
import gc
import numpy as np
from tifffile import imread,imwrite  # imwrite is recommended over imsave
from tqdm import tqdm
from cellpose import models
from skimage.measure import regionprops, label, find_contours
import matplotlib.pyplot as plt
import cv2
import fnmatch

# Import the IMS file reader package
from imaris_ims_file_reader import ims

# Paths
working_dir = "/Volumes/Movies/2025_06_04_M68/"
file_name = "A14_1"
input_ims = f"{working_dir}{file_name}.ims"
output_frames_folder = f"{working_dir}{file_name}/Frames_{file_name}"
output_masks_folder = f"{working_dir}{file_name}/Masks_{file_name}"
output_tracked_cells_folder = f"{working_dir}{file_name}/TrackedCells_{file_name}"
custom_model_path = "/Volumes/Movies/AI_training_set/models/CP_20250517_152934"





# Create output directories if they don't exist
os.makedirs(output_frames_folder, exist_ok=True)
os.makedirs(output_masks_folder, exist_ok=True)
os.makedirs(output_tracked_cells_folder, exist_ok=True)

# Create segmentation subfolders based on channel
GFP_seg_folder = os.path.join(output_masks_folder, "GFP_seg")
#RFP_seg_folder = os.path.join(output_masks_folder, "RFP_seg")
brightfield_seg_folder = os.path.join(output_masks_folder, "brightfield_seg")
os.makedirs(GFP_seg_folder, exist_ok=True)
#os.makedirs(RFP_seg_folder, exist_ok=True)
os.makedirs(brightfield_seg_folder, exist_ok=True)

# Set the chosen z-stack index for channel 2 segmentation (if available)
z_index = 2

# Helper function to load a segmentation file (either .npy or .tif)
def load_segmentation(path):
    if path.endswith('.npy'):
        return np.load(path)
    else:
        return imread(path)
# Helper function to get initial filter regions
def GetFilteredRegions(min_area = 2500):
    # Get list of brightfield segmentation files (used for tracking)
    mask_files = sorted([f for f in os.listdir(brightfield_seg_folder) 
                         if f.endswith('_seg.npy') or f.endswith('_seg.tif')])
    mask_paths = [os.path.join(brightfield_seg_folder, f) for f in mask_files]

    # Load the first brightfield segmentation mask and detect cells
    first_mask = load_segmentation(mask_paths[0])
    labeled_mask = label(first_mask)
    regions = regionprops(labeled_mask)

    # Filter out small cells based on area threshold
    
    filtered_regions = [r for r in regions if r.area >= min_area]
    return first_mask, labeled_mask, filtered_regions
def FindMovieMaxMin(channel, z_index=None, frame_dir=output_frames_folder,):
    if channel in [0]:
        frame_pattern = f"{file_name}_t_??_c_{channel}.tif"
        movie_name = f"{file_name}_c_{channel:01d}.mp4"
    elif channel == 1 and z_index is not None:
        frame_pattern = f"{file_name}_t_??_z_{z_index}_c_{channel}.tif"
        movie_name = f"{file_name}_z_{z_index:01d}_c_{channel:01d}.mp4"
    else:
        raise ValueError("Unsupported channel or missing z_index for channel 2")

    frame_files = sorted([
        f for f in os.listdir(frame_dir)
        if f.endswith('.tif') and fnmatch.fnmatch(f, frame_pattern)
    ])

    if not frame_files:
        print(f"No frames found for channel {channel} (z={z_index})")
        return

    # Estimate global min/max using percentiles
    sample_pixels = []
    for fname in frame_files:
        frame = imread(os.path.join(frame_dir, fname))
        sample_pixels.extend(frame.ravel()[::10])  # sample every 10th pixel for speed

    sample_pixels = np.array(sample_pixels)
    global_min = np.percentile(sample_pixels, 1)    # 1st percentile
    global_max = np.percentile(sample_pixels, 99.5) # 99.5th percentile
    return global_max, global_min, frame_files, movie_name
# Function to generate movie from frames
def create_movie(channel, z_index=None, frame_dir=output_frames_folder, output_dir=output_tracked_cells_folder, fps=10):
    
    
    global_max, global_min, frame_files, movie_name = FindMovieMaxMin(channel)
    print(f"Global intensity range after clipping: min={global_min}, max={global_max}")

    # Read first frame for size
    first_frame = imread(os.path.join(frame_dir, frame_files[0]))
    height, width = first_frame.shape
    out_path = os.path.join(output_dir, movie_name)

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height), isColor=False)

    for fname in frame_files:
        frame = imread(os.path.join(frame_dir, fname)).astype(np.float32)

        # Clip to the global min/max range
        frame_clipped = np.clip(frame, global_min, global_max)
        frame_normalized = (frame_clipped - global_min) / (global_max - global_min)
        frame_uint8 = (frame_normalized * 255).astype(np.uint8)

        out.write(frame_uint8)

    out.release()
    print(f"Saved QuickTime-compatible movie: {out_path}")
    
    

# #%% export images and perform segmentation
# # Initialize the Cellpose model with your custom model
# model = models.CellposeModel(gpu=True, pretrained_model=custom_model_path)
# channels_for_segmentation = [0, 0]  # For grayscale images
# diameter = 80

# # Load the IMS file using imaris-ims-file-reader.
# # Data shape is: (time, channel, z, height, width)
# data = ims(input_ims)
# time_points, n_channels, z_stacks, height, width = data.shape



# # Total segmentation tasks: channels 0 (max projection) and channel 1 (single z-stack) per time point
# total_tasks = time_points * 2

# with tqdm(total=total_tasks, desc="Processing time points") as pbar:
#     for t in range(time_points):
#         # Process channels 0 and 1: maximum projection across z-stacks
#         for c in [0]:
#             # Average projection across z-stacks
#             projected_img = np.mean(data[t, c, :, :, :], axis=0)
#             projected_filename = f"{file_name}_t_{t:02d}_c_{c:01d}.tif"
#             projected_path = os.path.join(output_frames_folder, projected_filename)
#             imwrite(projected_path, projected_img)
        
#             # Perform segmentation
#             masks, flows, styles, *extra = model.eval(projected_img,
#                                                       channels=channels_for_segmentation,
#                                                       diameter=diameter)
#             seg_folder = GFP_seg_folder
#             seg_filename = f"{file_name}_t_{t:02d}_c_{c:01d}_seg.tif"
#             seg_path = os.path.join(seg_folder, seg_filename)
#             imwrite(seg_path, masks)
        
#             del projected_img, masks, flows, styles, extra
#             gc.collect()
#             pbar.update(1)


#         # Process channel 1: select a specific z-stack for segmentation
#         c = 1
#         # Ensure the chosen z_index is within bounds
#         z = z_index if z_index < z_stacks else z_stacks - 1
#         selected_img = data[t, c, z, :, :]  # shape: (height, width)
#         img_name = f"{file_name}_t_{t:02d}_z_{z:01d}_c_{c:01d}.tif"
#         img_path = os.path.join(output_frames_folder, img_name)
#         imwrite(img_path, selected_img)
        
#         masks, flows, styles, *extra = model.eval(selected_img,
#                                                   channels=channels_for_segmentation,
#                                                   diameter=diameter)
#         seg_filename = f"{file_name}_t_{t:02d}_z_{z:01d}_c_{c:01d}_seg.tif"
#         seg_path = os.path.join(brightfield_seg_folder, seg_filename)
#         imwrite(seg_path, masks)
        
#         del selected_img, masks, flows, styles, extra
#         gc.collect()
#         pbar.update(1)

# del data
# gc.collect()



#%% Image quantification v2
import os
import pandas as pd
from skimage.io import imread
import imageio.v2 as imageio
import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript')

from Image_quantification_functions import ImageQuantification, plot_cell_and_gamma_overlay

# Function to compute the percentage overlap between two binary masks
def compute_overlap(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    total_area = mask1.sum()
    return intersection / total_area if total_area > 0 else 0
def SingleCellMovieWithGamma(cell_plot_folder):
    # Collect and sort images
    image_files = sorted([f for f in os.listdir(cell_plot_folder) if f.endswith(".png")])
    image_paths = [os.path.join(cell_plot_folder, f) for f in image_files]
    
    # Read the first image to get dimensions
    first_frame = cv2.imread(image_paths[0])
    height, width, layers = first_frame.shape
    
    # Define output path and writer
    video_path = os.path.join(output_tracked_cells_folder, f"cell_{cell_id}_movie.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    fps = 4
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    # Write frames
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Could not read frame {img_path}")
            continue
        video_writer.write(frame)
    
    video_writer.release()
    print(f"Saved movie for cell {cell_id} at: {video_path}")


def get_cell_mask(segmentation, prev_cell_mask, threshold=0.7):
    labeled_current = label(segmentation)
    current_regions = regionprops(labeled_current)

    overlaps = []
    for candidate in current_regions:
        candidate_mask = (labeled_current == candidate.label)
        overlap = compute_overlap(prev_cell_mask, candidate_mask)
        overlaps.append((overlap, candidate, candidate_mask))
    
    # Try to find the best single candidate
    best_candidate = None
    max_overlap = 0
    for overlap, candidate, candidate_mask in overlaps:
        if overlap >= threshold and overlap > max_overlap:
            best_candidate = candidate
            best_mask = candidate_mask
            max_overlap = overlap
    
    # If no single candidate passes, try combining top 5 masks
    if best_candidate is None:
        print("Trying combination of top 2 candidates.")
        overlaps.sort(key=lambda x: x[0], reverse=True)
        combined_mask = np.zeros_like(segmentation, dtype=bool)
        combined_regions = []
        for _, candidate, candidate_mask in overlaps[:2]:
            combined_mask = np.logical_or(combined_mask, candidate_mask)
            combined_regions.append(candidate)
        
        combined_overlap = compute_overlap(prev_cell_mask, combined_mask)
        if combined_overlap >= threshold:
            print("Using combined mask of top candidates.")
           
            return combined_mask

        print("Cell still not found. Using previous segmentation.")
        return prev_cell_mask

    else:
        return best_mask
    
def TrackCell(t, initial_cell_mask, prev_cell_mask, prev_cell_mask_gfp):#, prev_bbox, prev_bbox_gfp):
    # ----- BF SEG -----
    c = 1
    z = z_index
    bf_seg_path = os.path.join(brightfield_seg_folder, f"{file_name}_t_{t:02d}_z_{z:01d}_c_{c:01d}_seg.tif")
    
    if os.path.exists(bf_seg_path):
        bf_seg = load_segmentation(bf_seg_path)
        if t == 0:
            current_cell_mask = get_cell_mask(bf_seg, initial_cell_mask, threshold=0.7)
        else:
            current_cell_mask = get_cell_mask(bf_seg, prev_cell_mask, threshold=0.7)
    else:
        current_cell_mask = prev_cell_mask

    # Check if current cell mask touches the border
    touches_border = False
    if current_cell_mask is not None:
        touches_border = (
            np.any(current_cell_mask[0, :]) or
            np.any(current_cell_mask[-1, :]) or
            np.any(current_cell_mask[:, 0]) or
            np.any(current_cell_mask[:, -1])
        )

    # ----- GFP SEG -----
    c = 0
    fluorescent_GFP_frame_path = os.path.join(output_frames_folder, f"{file_name}_t_{t:02d}_c_{c:01d}.tif")
    fluorescent_frame_C1 = imread(fluorescent_GFP_frame_path)
    gfp_seg_path = os.path.join(GFP_seg_folder, f"{file_name}_t_{t:02d}_c_{c:01d}_seg.tif")
    
    if os.path.exists(gfp_seg_path):
        gfp_seg = load_segmentation(gfp_seg_path)
        if t == 0:
            current_cell_mask_gfp = get_cell_mask(gfp_seg, initial_cell_mask, threshold=0.5)
        else:
            current_cell_mask_gfp = get_cell_mask(gfp_seg, prev_cell_mask_gfp, threshold=0.7)
    else:
        current_cell_mask_gfp = prev_cell_mask_gfp

    return fluorescent_frame_C1, current_cell_mask, current_cell_mask_gfp, touches_border





# Read existing csv file or creat a new one
cell_data_path = os.path.join(output_tracked_cells_folder, "all_cells_time_series.csv")

if os.path.exists(cell_data_path):
    df_all = pd.read_csv(cell_data_path)
else:
    df_all = pd.DataFrame(columns=[
            'cell_id', 'time_point', 'cell_length', 'nu_dis',
            'nu_int', 'cyt_int', 'septum_int', 'pol1_int', 'pol2_int'
            ])  


# # Final dataset for all cells
# all_cells_time_series = []

cell_ids = [95] #46 #53 #82
#cell_ids = [region.label for region in filtered_regions]

plot_output_root = os.path.join(output_tracked_cells_folder, "cell_plots")
first_mask, labeled_mask, filtered_regions = GetFilteredRegions(500)
gfp_max, gfp_min,_,_ =FindMovieMaxMin(0)


for cell_id in cell_ids:
    print(f"Processing cell {cell_id}...")
    
    cell = next((cell for cell in filtered_regions if cell.label == cell_id), None)
    if cell is None:
        print(f"Warning: Cell ID {cell_id} not found in filtered_regions.")
        continue
    
    cell_plot_folder = os.path.join(plot_output_root, f"cell_{cell_id}")
    os.makedirs(cell_plot_folder, exist_ok=True)
    
    initial_cell_mask = (labeled_mask == cell_id)
    prev_cell_mask = None
    prev_cell_mask_gfp = None

    time_series_data = []
    list_of_timepoints = list(range(0, 51))
    skip_cell = False  # flag to skip the rest of the loop if border is touched

    for t in list_of_timepoints:
        img, current_cell_mask, current_cell_mask_gfp, touches_border = TrackCell(
            t, initial_cell_mask, prev_cell_mask, prev_cell_mask_gfp)
            

        if touches_border:
            print(f"Cell {cell_id} touches border at t={t}. Skipping this cell.")
            # Optionally save a minimal record for tracking
            time_series_data.append({
                'cell_id': cell_id,
                'time_point': t,
                'touches_border': True
            })
            skip_cell = True
            break  # skip this cell completely

        try:
            if t == 0:
                par, par_fixed, plot_data, ref_ep1, ref_ep2 = ImageQuantification(img, current_cell_mask_gfp, cell_id, gfp_max, gfp_min, t)
            else:
                par, par_fixed, plot_data, _, _ = ImageQuantification(img, current_cell_mask_gfp, cell_id, gfp_max, gfp_min, t, ref_ep1=ref_ep1, ref_ep2=ref_ep2)

            prop = {
                'cell_id': cell_id,
                'time_point': t,
                'cell_length': par_fixed['major_axis_length'],
                'nu_dis': par['mu_mn_Y2'][1],
                'nu_int': par['mu_I_Y2'],
                'cyt_int': par['mu_bg_Y2'],
                'septum_int': (par['mu_S1_Y2'] + par['mu_S2_Y2']) / 2,
                'pol1_int': par['mu_P1_Y2'],
                'pol2_int': par['mu_P2_Y2'],
                'touches_border': touches_border
            }
            time_series_data.append(prop)

            plot_filename = os.path.join(cell_plot_folder, f"frame_t_{t:03d}.png")
            plot_cell_and_gamma_overlay(plot_data, plot_filename=plot_filename)

            prev_cell_mask = current_cell_mask
            prev_cell_mask_gfp = current_cell_mask_gfp
            
        except Exception as e:
            print(f"Error quantifying cell {cell_id} at t={t}: {e}")

    if skip_cell:
        continue  # move to next cell

    df_cell = pd.DataFrame(time_series_data)
    df_all = df_all[df_all['cell_id'] != cell_id]
    df_all = pd.concat([df_all, df_cell], ignore_index=True)

    SingleCellMovieWithGamma(cell_plot_folder)

cell_data_path = os.path.join(output_tracked_cells_folder, "all_cells_time_series.csv")
df_all.to_csv(cell_data_path, index=False)









# #%% Imports
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import statsmodels.api as sm

# #%% Load data

# csv_path = os.path.join(output_tracked_cells_folder, "all_cells_time_series.csv")
# #csv_path = os.path.join(output_tracked_cells_folder, "all_cells_with_aligned_time.csv")
# df_all = pd.read_csv(csv_path)

# #%% Preprocessing: sort and compute derivative
# df_all = df_all.sort_values(by=['cell_id', 'time_point'])
# df_all['d_cell_length'] = df_all.groupby('cell_id')['cell_length'].diff()

# df_all['d_cell_length_avg5'] = (
#     df_all.groupby('cell_id')['d_cell_length']
#     .rolling(window=20, center=True, min_periods=1)
#     .mean()
#     .reset_index(level=0, drop=True)
# )
# df_all['d_cell_area'] = df_all.groupby('cell_id')['cell_area'].diff()

# df_all['d_cell_area_avg5'] = (
#     df_all.groupby('cell_id')['d_cell_area']
#     .rolling(window=20, center=True, min_periods=1)
#     .mean()
#     .reset_index(level=0, drop=True)
# )

# #%% Filtering: keep only cells with 41 time points
# valid_cells = df_all.groupby("cell_id")["time_point"].count()
# valid_cells = valid_cells[valid_cells == 41].index
# df_all = df_all[df_all['cell_id'].isin(valid_cells)]
# # Count how many cells have exactly 41 time points
# num_cells = df_all['cell_id'].nunique()
# print(f"✅ Number of cells with exactly 41 time points: {num_cells}")

# #%% Compute average d_cell_length and d_cell_area per cell
# avg_d_length = (
#     df_all
#     .groupby("cell_id")["d_cell_length"]
#     .mean()
#     .rename("avg_d_cell_length")
#     .reset_index()
# )

# avg_d_area = (
#     df_all
#     .groupby("cell_id")["d_cell_area"]
#     .mean()
#     .rename("avg_d_cell_area")
#     .reset_index()
# )

# # Merge both metrics into one DataFrame
# avg_growth_per_cell = pd.merge(avg_d_length, avg_d_area, on="cell_id")

# #%% Plot histograms with two subplots
# fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# # First subplot: avg_d_cell_length
# axes[0].hist(avg_growth_per_cell["avg_d_cell_length"], bins=120, edgecolor='black')
# axes[0].axvline(0, color='red', linestyle='--', label='Zero Growth')
# axes[0].set_title("Distribution of Smoothed Average Δ Cell Length per Cell")
# axes[0].set_xlabel("Average Smoothed d(cell_length) (px)")
# axes[0].set_ylabel("Number of Cells")
# axes[0].grid(True)
# axes[0].legend()

# # Second subplot: avg_d_cell_area
# axes[1].hist(avg_growth_per_cell["avg_d_cell_area"], bins=120, edgecolor='black')
# axes[1].axvline(0, color='red', linestyle='--', label='Zero Growth')
# axes[1].set_title("Distribution of Smoothed Average Δ Cell Area per Cell")
# axes[1].set_xlabel("Average Smoothed d(cell_area) (px²)")
# axes[1].set_ylabel("Number of Cells")
# axes[1].grid(True)
# axes[1].legend()

# plt.tight_layout()
# plt.show()


# plt.tight_layout()
# plt.show()

# e1 = 1
# e2 = -0.2
# # Identify cells with strong positive or negative average growth
# extreme_growth_cells = avg_growth_per_cell[
#     (avg_growth_per_cell['avg_d_cell_length'] > e1) |
#     (avg_growth_per_cell['avg_d_cell_length'] < e2)
# ]

# # Display the results
# print(f"📌 Cells with avg_d_cell_length > {e1} or < {e2}:")
# print(extreme_growth_cells)

# extreme_cell_ids = extreme_growth_cells['cell_id']
# df_all = df_all[~df_all['cell_id'].isin(extreme_cell_ids)]


# # #%% Clean NaNs
# # df_clean = df_all[['time_point','cell_id', 'cell_length','cell_area','nu_dis','d_cell_length_avg5',
# #                    'nu_int', 'septum_int', 'pol1_int', 'pol2_int', 'cyt_int','avg_d_cell_length']].dropna()



# #%% Feature time course plots
# features = ['cell_area', 'nu_dis', 'nu_int', 'cyt_int', 'septum_int', 'pol1_int', 'pol2_int']
# cell_ids = df_all['cell_id'].unique()

# for feature in features:
#     plt.figure(figsize=(8, 5))
#     for cell_id in cell_ids:
#         df_cell = df_all[df_all['cell_id'] == cell_id]
#         plt.plot(df_cell['time_point'], df_cell[feature], marker='o', label=f"Cell {cell_id}")
#     plt.title(f"{feature} Over Time")
#     plt.xlabel("Time Point")
#     plt.ylabel(feature)
#     #plt.legend()
#     plt.tight_layout()
#     plt.show()

# #%% Global time warping alignment with MCMC for MSE

# import seaborn as sns
# import matplotlib.pyplot as plt

# # Ensure seaborn styles
# sns.set(style="whitegrid")


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import random

# df_all['septum_int_corr'] = df_all['septum_int'] - df_all['cyt_int']
# df_all['pol1_int_corr'] = df_all['pol1_int'] - df_all['cyt_int']
# df_all['pol2_int_corr'] = df_all['pol2_int'] - df_all['cyt_int']
# df_all['pol1_minus_pol2'] = df_all['pol1_int_corr']  - df_all['pol2_int_corr']
# df_all['weighted_area'] = df_all['cell_area']/500
# # ---- Setup ----
# features_xcorr = ['nu_dis', 'weighted_area', 'septum_int_corr']
# cell_ids = df_all['cell_id'].unique()
# time_points = sorted(df_all['time_point'].unique())
# T = len(time_points)
# n_features = len(features_xcorr)
# padding = 4 * T
# global_time = np.arange(0, T + padding)
# shift_range = len(global_time) - T

# # Create signal dict (features × timepoints)
# cell_signals = {}
# for cell_id in cell_ids:
#     df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
#     matrix = np.array([
#         df_cell.set_index("time_point").reindex(time_points)[f].values
#         for f in features_xcorr
#     ])
#     cell_signals[cell_id] = matrix

# # ---- MSE computation function ----
# def compute_total_mse(shifts):
#     acc = np.zeros((n_features, len(global_time)))
#     weights = np.zeros_like(acc)

#     for cell_id in cell_ids:
#         signal = cell_signals[cell_id]
#         shift = shifts[cell_id]
#         valid = ~np.isnan(signal)
#         acc[:, shift:shift+T] += np.where(valid, signal, 0)
#         weights[:, shift:shift+T] += valid

#     avg = np.divide(acc, weights, where=weights != 0)

#     total_mse = 0
#     for cell_id in cell_ids:
#         signal = cell_signals[cell_id]
#         shift = shifts[cell_id]
#         valid = ~np.isnan(signal) & (weights[:, shift:shift+T] > 0)
#         mse = np.mean((signal[valid] - avg[:, shift:shift+T][valid])**2)
#         total_mse += mse

#     return total_mse, avg

# # ---- MCMC Optimization ----
# # Step 1: Compute mean cell length per cell
# mean_cell_lengths = {
#     cell_id: np.nanmean(cell_signals[cell_id][features_xcorr.index('nu_dis')])
#     for cell_id in cell_ids
# }

# # Step 2: Sort cells by increasing mean cell length
# sorted_cells = sorted(mean_cell_lengths, key=mean_cell_lengths.get)

# # Step 3: Evenly space shifts across allowed range
# num_cells = len(sorted_cells)
# available_shift_range = shift_range  # already = len(global_time) - T

# # Evenly space shifts, e.g. 0, gap, 2*gap, ..., up to shift_range
# shift_values = np.linspace(0, available_shift_range, num_cells).astype(int)
# initial_shifts = {
#     cell_id: shift for cell_id, shift in zip(sorted_cells, shift_values)
# }

# current_shifts = initial_shifts.copy()
# best_shifts = current_shifts.copy()
# best_score, best_mean = compute_total_mse(best_shifts)


# # ---- Plot Initial Alignment ----
# fig, axs = plt.subplots(nrows=n_features, figsize=(10, 2.5 * n_features), sharex=True)

# for i, feature in enumerate(features_xcorr):
#     ax = axs[i]

#     for cell_id in sorted_cells:
#         df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
#         values = df_cell.set_index("time_point").reindex(time_points)[feature].values

#         shift = initial_shifts[cell_id]
#         aligned = np.full_like(global_time, np.nan, dtype=np.float64)
#         aligned[shift:shift+T] = values

#         ax.plot(global_time, aligned, alpha=0.3)

#     ax.set_title(f"Initial Aligned: {feature}")
#     ax.set_ylabel("Value")
#     ax.grid(True)

# axs[-1].set_xlabel("Global Master Timeline (Initial Alignment)")
# plt.suptitle("Initial Condition: Aligned Cell Tracks Before MCMC", fontsize=16)
# plt.tight_layout()
# plt.show()
# #
# n_iter = 25000
# #temperature = 0.1
# mse_trace = [best_score]
# initial_temp = 1.0
# for i in range(n_iter):
#     temperature = initial_temp * (0.99 ** (i/1))

#     proposal = best_shifts.copy()
#     cell = random.choice(list(cell_ids))
#     direction = random.choice([ -10, -1, 1, 10])

#     new_shift = np.clip(proposal[cell] + direction, 0, shift_range)
#     proposal[cell] = new_shift

#     proposed_score, proposed_mean = compute_total_mse(proposal)
#     delta = proposed_score - best_score

#     if delta < 0 or np.exp(-delta / temperature) > np.random.rand():
#         best_shifts = proposal
#         best_score = proposed_score
#         best_mean = proposed_mean  


#     mse_trace.append(best_score)

#     if i % 100 == 0:
#         print(f"Step {i}: total MSE = {best_score:.4f}")

# # ---- Plot MSE trace ----
# plt.figure(figsize=(8, 4))
# plt.plot(mse_trace)
# plt.xlabel("Iteration")
# plt.ylabel("Total MSE")
# plt.title("MCMC Optimization Trace")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # ---- Plot aligned signals ----
# fig, axs = plt.subplots(nrows=n_features, figsize=(10, 2.5 * n_features), sharex=True)

# for i, feature in enumerate(features_xcorr):
#     ax = axs[i]

#     for cell_id in cell_ids:
#         df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
#         values = df_cell.set_index("time_point").reindex(time_points)[feature].values

#         shift = best_shifts[cell_id]
#         aligned = np.full_like(global_time, np.nan, dtype=np.float64)
#         aligned[shift:shift+T] = values

#         ax.plot(global_time, aligned, alpha=0.3)

#     ax.plot(global_time, best_mean[i], color='black', linewidth=2, label='Mean')
#     ax.set_title(f"{feature}")
#     ax.set_ylabel("Value")
#     ax.grid(True)

# axs[-1].set_xlabel("Global Master Timeline (Aligned Time Points)")
# plt.suptitle("Aligned Cell Tracks on Master Timeline (MCMC)", fontsize=16)
# plt.tight_layout()
# plt.show()

# #%

# # Define all other features to visualize
# additional_features = [
#     'pol1_int_corr', 'pol2_int_corr', 'pol1_minus_pol2'
# ]

# n_additional = len(additional_features)

# fig, axs = plt.subplots(nrows=n_additional, figsize=(10, 2.5 * n_additional), sharex=True)

# for i, feature in enumerate(additional_features):
#     ax = axs[i]

#     # Initialize accumulator and weights
#     acc = np.zeros(len(global_time))
#     weight = np.zeros(len(global_time))

#     for cell_id in cell_ids:
#         df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
#         values = df_cell.set_index("time_point").reindex(time_points)[feature].values

#         shift = best_shifts[cell_id]
#         aligned = np.full_like(global_time, np.nan, dtype=np.float64)
#         aligned[shift:shift+T] = values

#         ax.plot(global_time, aligned, alpha=0.3)

#         # Update mean accumulator
#         valid = ~np.isnan(aligned)
#         acc[valid] += aligned[valid]
#         weight[valid] += 1

#     # Compute and plot mean
#     mean_trace = np.divide(acc, weight, out=np.zeros_like(acc), where=weight != 0)
#     ax.plot(global_time, mean_trace, color='black', linewidth=2, label='Mean')

#     ax.set_title(f"{feature}")
#     ax.set_ylabel("Value")
#     ax.grid(True)

# axs[-1].set_xlabel("Global Master Timeline (Aligned Time Points)")
# plt.suptitle("Additional Aligned Cell Tracks with Mean (MCMC)", fontsize=16)
# plt.tight_layout()
# plt.show()


# #%% Heatmap for aligned time points
# import seaborn as sns

# # Sort cells by first appearance on the master timeline
# ordered_cells_by_shift = sorted(cell_ids, key=lambda cid: best_shifts[cid])

# # Create heatmap layout
# fig, axs = plt.subplots(nrows=n_additional, figsize=(10, 2.5 * n_additional), sharex=True)

# for i, feature in enumerate(additional_features):
#     heatmap_data = []

#     for cell_id in ordered_cells_by_shift:
#         df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
#         values = df_cell.set_index("time_point").reindex(time_points)[feature].values

#         shift = best_shifts[cell_id]
#         aligned = np.full_like(global_time, np.nan, dtype=np.float64)
#         aligned[shift:shift+T] = values

#         # For "pol1_minus_pol2", flip if mean is negative
#         if i == 2:  # third plot
#             mean_val = np.nanmean(values)
#             if mean_val < 0:
#                 aligned = -aligned

#         heatmap_data.append(aligned)

#     heatmap_array = np.array(heatmap_data)

#     # Apply fixed color scale only to the third feature
#     if i == 2:
#         sns.heatmap(
#             heatmap_array,
#             ax=axs[i],
#             cmap="RdBu",
#             cbar=True,
#             vmin=-4,
#             vmax=4,
#             xticklabels=False,
#             yticklabels=False
#         )
#     else:
#         sns.heatmap(
#             heatmap_array,
#             ax=axs[i],
#             cmap="viridis",
#             cbar=True,
#             xticklabels=False,
#             yticklabels=False
#         )

#     axs[i].set_title(f"{feature}")
#     axs[i].set_ylabel("Cells (Ordered by Appearance)")

# axs[-1].set_xlabel("Global Master Timeline (Aligned Time Points)")
# plt.suptitle("Aligned Feature Heatmaps (MCMC)", fontsize=16, y=1.02)
# plt.tight_layout()
# plt.show()

# #%%
# import numpy as np
# import matplotlib.pyplot as plt

# # Define features of interest
# features_fft = ['pol1_int_corr', 'pol2_int_corr']#, 'pol1_minus_pol2']
# colors = ['tab:blue', 'tab:orange']#, 'tab:green']

# # Sampling frequency assumptions (1 unit per time point)
# fs = 1.0
# freqs = np.fft.rfftfreq(len(time_points), d=1/fs)

# # Create a plot
# plt.figure(figsize=(10, 5))

# for idx, feature in enumerate(features_fft):
#     aligned_positions = []
#     dominant_freqs = []

#     for cell_id in cell_ids:
#         df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
#         values = df_cell.set_index("time_point").reindex(time_points)[feature].values

#         if np.sum(~np.isnan(values)) >= 3:
#             values_detrended = values - np.nanmean(values)
#             values_filled = np.nan_to_num(values_detrended)

#             fft_values = np.fft.rfft(values_filled)
#             power = np.abs(fft_values)

#             dominant_freq = freqs[np.argmax(power[1:]) + 1]  # skip zero-frequency
#             shift = best_shifts[cell_id]
#             aligned_positions.append(global_time[shift])
#             dominant_freqs.append(dominant_freq)

#     # Plot
#     plt.scatter(aligned_positions, dominant_freqs, label=feature, alpha=0.7, s=30, color=colors[idx])

# # Final plot adjustments
# plt.xlabel("Global Master Timeline (Aligned Start Time)")
# plt.ylabel("Dominant Frequency")
# plt.title("Most Significant Frequency per Cell (FFT)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# #%%
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Define features
# features_fft = ['pol1_int_corr', 'pol2_int_corr']
# colors = ['tab:blue', 'tab:orange']

# # Sampling frequency
# fs = 1.0
# freqs = np.fft.rfftfreq(len(time_points), d=1/fs)

# # Define 3 time periods
# period_edges = np.array_split(global_time, 3)
# period_bounds = [(p[0], p[-1]) for p in period_edges]

# # ---- Precompute dominant frequencies per cell and feature ----
# dominant_freqs_all = {f: [[] for _ in range(3)] for f in features_fft}
# all_dominant_freqs = []

# for cell_id in cell_ids:
#     df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
#     shift = best_shifts[cell_id]
#     start_time = global_time[shift]

#     for f_idx, feature in enumerate(features_fft):
#         values = df_cell.set_index("time_point").reindex(time_points)[feature].values
#         if np.sum(~np.isnan(values)) >= 3:
#             detrended = values - np.nanmean(values)
#             filled = np.nan_to_num(detrended)
#             fft_values = np.fft.rfft(filled)
#             power = np.abs(fft_values)
#             dom_freq = freqs[np.argmax(power[1:]) + 1]  # skip zero

#             for p_idx, (start, end) in enumerate(period_bounds):
#                 if start <= start_time <= end:
#                     dominant_freqs_all[feature][p_idx].append(dom_freq)
#                     all_dominant_freqs.append(dom_freq)
#                     break

# # ---- Determine shared bins ----
# shared_bins = np.histogram_bin_edges(all_dominant_freqs, bins=15)

# # ---- Plot ----
# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

# for p_idx in range(3):
#     for f_idx, feature in enumerate(features_fft):
#         sns.histplot(
#             dominant_freqs_all[feature][p_idx],
#             ax=axs[p_idx],
#             bins=shared_bins,
#             kde=True,
#             stat='probability',
#             color=colors[f_idx],
#             label=feature,
#             alpha=0.5
#         )

#     axs[p_idx].set_title(f"Period {p_idx+1} ({period_bounds[p_idx][0]}–{period_bounds[p_idx][1]})")
#     axs[p_idx].set_xlabel("Dominant Frequency")
#     axs[p_idx].set_xlim(shared_bins[0], shared_bins[-1])
#     axs[p_idx].grid(True)
#     if p_idx == 0:
#         axs[p_idx].set_ylabel("Probability")
#     axs[p_idx].legend()

# plt.suptitle("Probability Distribution of Dominant Frequencies Across Periods", fontsize=16)
# plt.tight_layout()
# plt.show()
# #%%
# import GPy
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# features_gp = ['pol1_int_corr', 'pol2_int_corr']
# colors = ['tab:blue', 'tab:orange']

# # Define 3 global time periods
# period_edges = np.array_split(global_time, 3)
# period_bounds = [(p[0], p[-1]) for p in period_edges]

# # Store learned periods
# learned_periods = {f: [[] for _ in range(3)] for f in features_gp}
# all_periods = []

# for cell_id in cell_ids:
#     df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
#     shift = best_shifts[cell_id]
#     start_time = global_time[shift]
#     t_obs = np.arange(T).reshape(-1, 1)

#     for f_idx, feature in enumerate(features_gp):
#         y = df_cell.set_index("time_point").reindex(time_points)[feature].values.reshape(-1, 1)

#         # Skip if too many NaNs
#         if np.sum(~np.isnan(y)) < 5:
#             continue

#         y = np.nan_to_num(y - np.nanmean(y))  # Detrend and fill NaNs

#         # Define periodic kernel
#         kernel = GPy.kern.StdPeriodic(input_dim=1)
        
#         # Set Gamma prior on lengthscale to encourage smoother functions
#         # Gamma(a=5.0, b=1.0) → mean = a * b = 5.0
#         kernel.lengthscale.set_prior(GPy.priors.Gamma(5.0, 1.0))
        
#         # Optional: set reasonable bounds for period
#         kernel.period.constrain_bounded(2.0, 40.0)  # You can adjust bounds

#         model = GPy.models.GPRegression(t_obs, y, kernel)
#         model.optimize(messages=False, max_iters=500)

#         period = float(model.kern.period.values[0])

#         # Assign to global time period
#         for p_idx, (start, end) in enumerate(period_bounds):
#             if start <= start_time <= end:
#                 learned_periods[feature][p_idx].append(period)
#                 all_periods.append(period)
#                 break


# #%% ---- Plot ----
# import numpy as np
# import matplotlib.pyplot as plt

# # Define shared bin edges
# shared_bins = np.histogram_bin_edges(all_periods, bins=30)
# bin_centers = 0.5 * (shared_bins[:-1] + shared_bins[1:])
# n_bins = len(bin_centers)

# fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5), sharey=True)

# for p_idx in range(3):
#     ax = axs[p_idx]
    
#     bottoms = np.zeros(n_bins)  # For stacking
#     for f_idx, feature in enumerate(features_gp):
#         # Get histogram for current feature and period
#         data = np.array(learned_periods[feature][p_idx])
#         hist, _ = np.histogram(data, bins=shared_bins, density=False)
#         prob = hist / hist.sum() if hist.sum() > 0 else np.zeros_like(hist)  # Normalize to probability

#         # Plot stacked bars
#         ax.bar(
#             bin_centers,
#             prob,
#             width=np.diff(shared_bins),
#             bottom=bottoms,
#             color=colors[f_idx],
#             alpha=0.7,
#             label=feature,
#             edgecolor='black'
#         )
#         bottoms += prob  # Update stack base

#     ax.set_title(f"Period {p_idx + 1} ({period_bounds[p_idx][0]}–{period_bounds[p_idx][1]})")
#     ax.set_xlabel("Inferred GP Period")
#     ax.set_xlim(shared_bins[0], shared_bins[-1])
#     ax.grid(True)
#     if p_idx == 0:
#         ax.set_ylabel("Probability")
#     ax.legend()

# plt.suptitle("Stacked Inferred GP Period Distributions", fontsize=16)
# plt.tight_layout()
# plt.show()
# #%% ---- Plot Period vs Aligned Start Time ----
# plt.figure(figsize=(10, 5))

# for f_idx, feature in enumerate(features_gp):
#     x_vals = []  # global time at first alignment
#     y_vals = []  # learned period

#     for cell_id in cell_ids:
#         df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")
#         shift = best_shifts[cell_id]
#         start_time = global_time[shift]

#         # Check if we have a learned period
#         t_obs = np.arange(T).reshape(-1, 1)
#         y = df_cell.set_index("time_point").reindex(time_points)[feature].values.reshape(-1, 1)
#         if np.sum(~np.isnan(y)) < 5:
#             continue

#         y = np.nan_to_num(y - np.nanmean(y))

#         kernel = GPy.kern.StdPeriodic(input_dim=1)
#         kernel.lengthscale.set_prior(GPy.priors.Gamma(5.0, 1.0))
#         kernel.period.constrain_bounded(2.0, 40.0)

#         model = GPy.models.GPRegression(t_obs, y, kernel)
#         model.optimize(messages=False, max_iters=500)

#         period = float(model.kern.period.values[0])
#         x_vals.append(start_time)
#         y_vals.append(period)

#     # Scatter plot for current feature
#     plt.scatter(x_vals, y_vals, color=colors[f_idx], label=feature, alpha=0.7, s=30)

# plt.xlabel("Aligned Start Time on Global Timeline")
# plt.ylabel("Inferred GP Period")
# plt.title("GP Periodicity at First Aligned Time Point")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()

# #%% 1-D GP
# import GPy
# import numpy as np
# import matplotlib.pyplot as plt
# from GPy.util.multioutput import LCM

# # Select one cell for 2D modeling
# cell_id = cell_ids[1]
# df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")

# t = df_cell["time_point"].values.reshape(-1, 1)
# y1 = df_cell["pol1_int_corr"].values.reshape(-1, 1)
# y2 = df_cell["pol2_int_corr"].values.reshape(-1, 1)

# # Only keep rows where both are valid
# valid = ~np.isnan(y1[:, 0]) & ~np.isnan(y2[:, 0])
# t = t[valid]
# y1 = y1[valid]
# y2 = y2[valid]


# y = np.nan_to_num(y1 - np.nanmean(y1))  # Detrend and fill NaNs
 
# # Define periodic kernel
# kernel = GPy.kern.StdPeriodic(input_dim=1)

# # Set Gamma prior on lengthscale to encourage smoother functions
# # Gamma(a=5.0, b=1.0) → mean = a * b = 5.0
# kernel.lengthscale.set_prior(GPy.priors.Gamma(5.0, 1.0))

# # Optional: set reasonable bounds for period
# kernel.period.constrain_bounded(2.0, 40.0)  # You can adjust bounds
   
# model = GPy.models.GPRegression(t_obs, y, kernel)
# model.optimize(messages=False, max_iters=500)
# # Create prediction input (dense time grid)
# t_pred = np.linspace(t.min(), t.max(), 200)[:, None]

# # Predict mean and variance from the model
# mu, var = model.predict(t_pred)

# # Plot
# plt.figure(figsize=(10, 5))
# plt.title(f"Periodic GP Fit for Cell {cell_id} (pol1_int_corr)")
# plt.plot(t, y, 'kx', label='Observed')
# plt.plot(t_pred, mu, 'b-', label='GP Mean')
# plt.fill_between(t_pred[:, 0],
#                  mu[:, 0] - 2 * np.sqrt(var[:, 0]),
#                  mu[:, 0] + 2 * np.sqrt(var[:, 0]),
#                  color='blue', alpha=0.2, label='95% CI')
# plt.xlabel("Time")
# plt.ylabel("Detrended Signal")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.show()
# #%% 2-D GP
# import GPy
# import numpy as np
# import matplotlib.pyplot as plt
# from GPy.util.multioutput import LCM

# # Select one cell for 2D modeling
# cell_id = cell_ids[6]
# df_cell = df_all[df_all["cell_id"] == cell_id].sort_values("time_point")

# t = df_cell["time_point"].values.reshape(-1, 1)
# y1 = df_cell["pol1_int_corr"].values.reshape(-1, 1)
# y2 = df_cell["pol2_int_corr"].values.reshape(-1, 1)

# # Only keep rows where both are valid
# valid = ~np.isnan(y1[:, 0]) & ~np.isnan(y2[:, 0])
# t = t[valid]
# y1 = y1[valid] - np.nanmean(y1[valid])
# y2 = y2[valid] - np.nanmean(y2[valid])

# # Simulate toy sine wave signals for testing GP
# #y1 = np.sin(2 * np.pi * t / 10.0) + 0.1 * np.random.randn(*t.shape)  # Period = 10
# #y2 = np.sin(2 * np.pi * t / 12.0 + np.pi / 4) + 0.1 * np.random.randn(*t.shape)  # Period = 12, phase offset

# X_list = [t, t]
# Y_list = [y1, y2]



# # # Define shared periodic kernel
# # latent_kern = GPy.kern.StdPeriodic(1, variance=1., lengthscale=2., period=3.)
# # latent_kern.period.set_prior(GPy.priors.Uniform(0.5, 50.0))
# # latent_kern.lengthscale.set_prior(GPy.priors.Gamma(1.0, 1.0))
# # latent_kern.variance.set_prior(GPy.priors.Gamma(5.0, 1.0))

# # # Build LCM kernel
# # lcm = LCM(input_dim=1, num_outputs=2, kernels_list=[latent_kern])
# latent_k1 = GPy.kern.StdPeriodic(1, variance=1., lengthscale=2., period=5.)
# #latent_k1.period.set_prior(GPy.priors.Uniform(5, 50.0))
# latent_k1.period.constrain_bounded(5.0, 50.0)
# latent_k1.lengthscale.set_prior(GPy.priors.Gamma(5.0, 1.0))
# latent_k1.variance.set_prior(GPy.priors.Gamma(5.0, 1.0))

# latent_k2 = GPy.kern.StdPeriodic(1, variance=1., lengthscale=2., period=2.)
# #latent_k2.period.set_prior(GPy.priors.Uniform(5, 50.0))
# latent_k2.period.constrain_bounded(5.0, 50.0)
# latent_k2.lengthscale.set_prior(GPy.priors.Gamma(5.0, 1.0))
# latent_k2.variance.set_prior(GPy.priors.Gamma(5.0, 1.0))

# lcm = LCM(input_dim=1, num_outputs=2, kernels_list=[latent_k1, latent_k2])

# # Access the shared latent kernel and apply constraints
# latent_k = lcm.parts[0]
# #latent_k.lengthscale.set_prior(GPy.priors.Gamma(5.0, 1.0))
# #latent_k.period.constrain_bounded(2.0, 40.0)



# # Build and train model
# model = GPy.models.GPCoregionalizedRegression(X_list, Y_list, kernel=lcm)
# model.optimize(messages=True, max_iters=2000)

# # --- Prediction ---
# t_pred = np.linspace(t.min(), t.max(), 200)[:, None]

# # Prepare input for each output
# # Make prediction inputs for each output (note: shape = [N, 2])
# X_pred_0 = np.hstack([t_pred, np.zeros_like(t_pred)])
# X_pred_1 = np.hstack([t_pred, np.ones_like(t_pred)])

# # Provide Y_metadata as required by MixedNoise likelihood
# Y_metadata_0 = {'output_index': np.zeros((t_pred.shape[0], 1), dtype=int)}
# Y_metadata_1 = {'output_index': np.ones((t_pred.shape[0], 1), dtype=int)}

# # Predict
# mu1, var1 = model.predict(X_pred_0, Y_metadata=Y_metadata_0)
# mu2, var2 = model.predict(X_pred_1, Y_metadata=Y_metadata_1)




# # --- Plotting ---
# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.title("pol1_int_corr")
# plt.plot(t, y1, 'kx', label='Observed')
# plt.plot(t_pred, mu1, 'b-', label='GP Mean')
# plt.fill_between(t_pred[:, 0], mu1[:, 0] - 2*np.sqrt(var1[:, 0]), mu1[:, 0] + 2*np.sqrt(var1[:, 0]), color='b', alpha=0.2)
# plt.grid(True)
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.title("pol2_int_corr")
# plt.plot(t, y2, 'kx', label='Observed')
# plt.plot(t_pred, mu2, 'orange', label='GP Mean')
# plt.fill_between(t_pred[:, 0], mu2[:, 0] - 2*np.sqrt(var2[:, 0]), mu2[:, 0] + 2*np.sqrt(var2[:, 0]), color='orange', alpha=0.2)
# plt.grid(True)
# plt.legend()

# plt.suptitle(f"2-D Periodic GP Fit for Cell {cell_id}", fontsize=16)
# plt.tight_layout()
# plt.show()




# #%% Create a list to collect aligned time info
# aligned_time_records = []

# for cell_id in cell_ids:
#     shift = best_shifts[cell_id]

#     # Extract the original time points for this cell
#     df_cell = df_all[df_all["cell_id"] == cell_id][["cell_id", "time_point"]].copy()

#     # Compute global aligned time for each time_point
#     df_cell["aligned_time"] = df_cell["time_point"] + shift

#     # Append to list
#     aligned_time_records.append(df_cell)

# # Concatenate all results
# df_aligned_time = pd.concat(aligned_time_records, ignore_index=True)

# # Merge aligned_time back into df_all
# df_all = df_all.merge(df_aligned_time, on=["cell_id", "time_point"])

# # Define the output file path
# output_path = os.path.join(output_tracked_cells_folder, "all_cells_with_aligned_time.csv")

# # Save to CSV
# df_all.to_csv(output_path, index=False)

# print(f"✅ Saved aligned data to: {output_path}")

# #%% Summary plots per cell
# cell_ids = [14,177,77,76,35,163,157,210,42]
# intensity_cols = [col for col in df_all.columns if col.endswith('_int')]
# intensity_colors = {
#     'nu_int': 'lightgreen',
#     'cyt_int': 'pink',
#     'pol1_int': 'blue',
#     'pol2_int': 'brown',
#     'septum_int': 'magenta'
# }

# output_plot_folder = "cell_plots"
# os.makedirs(output_plot_folder, exist_ok=True)

# for cell_id in cell_ids:
#     df_cell = df_clean[df_clean['cell_id'] == cell_id]
#     if df_cell.empty:
#         continue

#     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

#     ax1.plot(df_cell['time_point'], df_cell['cell_length'], marker='o', color='red', label='cell_length')
#     ax1.set_title(f'Cell {cell_id} – Length')
#     ax1.set_ylabel('Length (px)')
#     ax1.set_ylim(50, 250)
#     ax1.legend()
#     ax1.grid(True)

#     ax2.plot(df_cell['time_point'], df_cell['nu_dis'], marker='s', color='green', label='nu_dis')
#     ax2.set_title(f'Cell {cell_id} – Nucleus Distance')
#     ax2.set_ylabel('Distance (px)')
#     ax2.set_ylim(0, 70)
#     ax2.legend()
#     ax2.grid(True)

#     for col in intensity_cols:
#         color = intensity_colors.get(col, 'gray')
#         ax3.plot(df_cell['time_point'], df_cell[col], marker='o', color=color, label=col)
#     ax3.set_title(f'Cell {cell_id} – Fluorescence Intensities')
#     ax3.set_xlabel('Time Point')
#     ax3.set_ylabel('Intensity (a.u.)')
#     ax3.legend()
#     ax3.grid(True)

#     plt.tight_layout()
#     plt.savefig(os.path.join(output_plot_folder, f"cell_{cell_id}_summary.png"))
#     plt.show()

# #%% Regression: polarity vs. growth rate
# selected_cells = [69]
# X_all = []
# y_all = []

# plt.figure(figsize=(7, 6))
# for cell_id in selected_cells:
#     df_cell = df_clean[df_clean['cell_id'] == cell_id]
#     if df_cell.empty:
#         continue

#     int_cell = df_cell['pol1_int'] + df_cell['pol2_int'] - 2 * df_cell['cyt_int']
#     dL = df_cell['d_cell_length_avg5']

#     plt.scatter(int_cell, dL, label=f'Cell {cell_id}', alpha=1, s=5)
#     X_all.extend(int_cell.tolist())
#     y_all.extend(dL.tolist())

# X_all = pd.Series(X_all, name="pol_int")
# y_all = pd.Series(y_all, name="d_length")
# X_with_const = sm.add_constant(X_all)

# model = sm.OLS(y_all, X_with_const).fit()

# # Regression line
# x_fit = np.linspace(X_all.min(), X_all.max(), 100)
# x_fit_const = sm.add_constant(x_fit)
# y_fit = model.predict(x_fit_const)

# plt.plot(x_fit, y_fit, color='black', linewidth=2,
#          label=f'Fit: $R^2$={model.rsquared:.2f}, p={model.pvalues["pol_int"]:.2e}')
# plt.xlabel('Pol1 + Pol2 - 2×Cyt Intensity (a.u.)')
# plt.ylabel('Smoothed Δ Cell Length (px)')
# plt.title('Polarity vs. Growth Rate (Statsmodels OLS)')
# plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Summary
# print("📈 Regression Summary:")
# print(model.summary())
