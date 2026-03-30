#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 13:46:09 2025

@author: user
"""
import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.measure import regionprops, label
import cv2
import argparse
import sys
sys.path.append('/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC')
from Image_quantification_functions import ImageQuantification, plot_cell_and_gamma_overlay

# Parse arguments
parser = argparse.ArgumentParser(description="Quantify a single cell.")
parser.add_argument('--cell_id', type=int, required=True, help='Cell ID to process')
parser.add_argument('--experiment_path', type=str, required=True, help='Path to experiment')
parser.add_argument('--file_name', type=str, required=True, help='File name of .ims file')
parser.add_argument('--z_index', type=int, default=2, help="Z-slice index to load for segmentation.")
parser.add_argument('--min_area', type=int, default=2500, help="Minimum area threshold for cell filtering.")
parser.add_argument('--update_existing', action='store_true', help="Only update existing properties without running EM")


args = parser.parse_args()
cell_id = args.cell_id
working_dir = args.experiment_path
file_name = args.file_name
update_existing = args.update_existing
min_area = args.min_area
z_index = args.z_index

# Paths
#working_dir = "/Volumes/Movies/2025_05_15_M63/"
#working_dir = "/RAID1/working/R402/hsushen/FungalProject/Movies/2025_05_15_M63/"
#file_name = "A14_5_F1"
output_frames_folder = f"{working_dir}{file_name}/Frames_{file_name}"
output_masks_folder = f"{working_dir}{file_name}/Masks_{file_name}"
output_tracked_cells_folder = f"{working_dir}{file_name}/TrackedCells_{file_name}"
#custom_model_path = "/Volumes/Movies/AI_training_set/models/CP_20250517_152934"

GFP_seg_folder = os.path.join(output_masks_folder, "GFP_seg")
brightfield_seg_folder = os.path.join(output_masks_folder, "brightfield_seg")
os.makedirs(GFP_seg_folder, exist_ok=True)
os.makedirs(brightfield_seg_folder, exist_ok=True)


frame_number = len([
    f for f in os.listdir(GFP_seg_folder)
    if os.path.isfile(os.path.join(GFP_seg_folder, f)) and f.endswith('_seg.tif')
])

# check existing data
data_path = os.path.join(output_tracked_cells_folder, f"cell_{cell_id}_data.csv")
if update_existing and os.path.exists(data_path):
    df_existing = pd.read_csv(data_path)
    existing_times = df_existing['time_point'].tolist()
else:
    df_existing = pd.DataFrame()
    existing_times = []


# Helper functions
def load_segmentation(path):
    return np.load(path) if path.endswith('.npy') else imread(path)

def GetFilteredRegions(min_area=2500):
    mask_files = sorted([f for f in os.listdir(brightfield_seg_folder) if f.endswith('_seg.npy') or f.endswith('_seg.tif')])
    first_mask = load_segmentation(os.path.join(brightfield_seg_folder, mask_files[0]))
    labeled_mask = label(first_mask)
    regions = regionprops(labeled_mask)
    filtered_regions = [r for r in regions if r.area >= min_area]
    return first_mask, labeled_mask, filtered_regions

def FindMovieMaxMin(channel):
    from fnmatch import fnmatch
    if channel == 0:
        pattern = f"{file_name}_t_??_c_{channel}.tif"
    else:
        raise ValueError("Only channel 0 supported in this script.")
    frame_files = sorted([f for f in os.listdir(output_frames_folder) if fnmatch(f, pattern)])
    pixels = []
    for fname in frame_files:
        frame = imread(os.path.join(output_frames_folder, fname))
        pixels.extend(frame.ravel()[::10])
    pixels = np.array(pixels)
    return np.percentile(pixels, 99.5), np.percentile(pixels, 1), frame_files

def compute_overlap(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    return intersection / mask1.sum() if mask1.sum() > 0 else 0

def get_cell_mask(segmentation, prev_mask, threshold=0.7):
    labeled = label(segmentation)
    regions = regionprops(labeled)
    best_overlap = 0
    best_mask = None
    for r in regions:
        candidate_mask = (labeled == r.label)
        overlap = compute_overlap(prev_mask, candidate_mask)
        if overlap >= threshold and overlap > best_overlap:
            best_mask = candidate_mask
            best_overlap = overlap
    if best_mask is not None:
        return best_mask
    print("Trying combined top-2 overlap mask")
    regions.sort(key=lambda r: compute_overlap(prev_mask, (labeled == r.label)), reverse=True)
    combined_mask = np.zeros_like(segmentation, dtype=bool)
    for r in regions[:2]:
        combined_mask |= (labeled == r.label)
    return combined_mask if compute_overlap(prev_mask, combined_mask) >= threshold else prev_mask

def TrackCell(t, initial_mask, prev_mask, prev_mask_gfp):
    bf_seg_path = os.path.join(brightfield_seg_folder, f"{file_name}_t_{t:02d}_z_{z_index}_c_1_seg.tif")
    current_mask = get_cell_mask(load_segmentation(bf_seg_path), initial_mask if t == 0 else prev_mask) if os.path.exists(bf_seg_path) else prev_mask
    touches_border = np.any(current_mask[0, :]) or np.any(current_mask[-1, :]) or np.any(current_mask[:, 0]) or np.any(current_mask[:, -1])
    gfp_frame_path = os.path.join(output_frames_folder, f"{file_name}_t_{t:02d}_c_0.tif")
    img = imread(gfp_frame_path)
    gfp_seg_path = os.path.join(GFP_seg_folder, f"{file_name}_t_{t:02d}_c_0_seg.tif")
    current_mask_gfp = get_cell_mask(load_segmentation(gfp_seg_path), initial_mask if t == 0 else prev_mask_gfp, threshold=0.5 if t == 0 else 0.7) if os.path.exists(gfp_seg_path) else prev_mask_gfp
    return img, current_mask, current_mask_gfp, touches_border

# Setup
plot_output_root = os.path.join(output_tracked_cells_folder, "cell_plots")
os.makedirs(plot_output_root, exist_ok=True)
first_mask, labeled_mask, filtered_regions = GetFilteredRegions(min_area)
gfp_max, gfp_min, _ = FindMovieMaxMin(0)

cell = next((c for c in filtered_regions if c.label == cell_id), None)
if cell is None:
    print(f"Cell {cell_id} not found.")
    sys.exit(1)

cell_plot_folder = os.path.join(plot_output_root, f"cell_{cell_id}")
os.makedirs(cell_plot_folder, exist_ok=True)

initial_mask = (labeled_mask == cell_id)
prev_mask = None
prev_mask_gfp = None
time_series_data = []

for t in range(frame_number):
    img, mask, mask_gfp, touches = TrackCell(t, initial_mask, prev_mask, prev_mask_gfp)

    if touches:
        print(f"Cell {cell_id} touches border at t={t}. Skipping.")
        time_series_data.append({
            'cell_id': cell_id,
            'time_point': t,
            'touches_border': True
        })
        break

    try:
        skip_em_and_plot = update_existing and t in existing_times

        if skip_em_and_plot:
            # Only run fixed computation to get updated par_fixed
            if t == 0:
                _, par_fixed, _, ep1, ep2 = ImageQuantification(
                    img, mask_gfp, cell_id, gfp_max, gfp_min, t, skip_em=True
                )
            else:
                _, par_fixed, _, _, _ = ImageQuantification(
                    img, mask_gfp, cell_id, gfp_max, gfp_min, t,
                    ref_ep1=ep1, ref_ep2=ep2, skip_em=True
                )

            # Reuse previous row and update cell_area and cell_length
            old_row = df_existing[df_existing['time_point'] == t].iloc[0].to_dict()
            old_row['cell_area'] = par_fixed.get('area', None)
            old_row['cell_length'] = par_fixed.get('major_axis_length', None)
            time_series_data.append(old_row)

        else:
            if t == 0:
                par, par_fixed, plot_data, ep1, ep2 = ImageQuantification(
                    img, mask_gfp, cell_id, gfp_max, gfp_min, t, skip_em=False
                )
            else:
                par, par_fixed, plot_data, _, _ = ImageQuantification(
                    img, mask_gfp, cell_id, gfp_max, gfp_min, t,
                    ref_ep1=ep1, ref_ep2=ep2, skip_em=False
                )

            prop = {
                'cell_id': cell_id,
                'time_point': t,
                'cell_length': par_fixed.get('major_axis_length', None),
                'cell_area': par_fixed.get('area', None),
                'nu_dis': par.get('mu_mn_Y2', [None, None])[1],
                'nu_int': par.get('mu_I_Y2', None),
                'cyt_int': par.get('mu_bg_Y2', None),
                'septum_int': None if not par.get('mu_S1_Y2') or not par.get('mu_S2_Y2')
                                else (par['mu_S1_Y2'] + par['mu_S2_Y2']) / 2,
                'pol1_int': par.get('mu_P1_Y2', None),
                'pol2_int': par.get('mu_P2_Y2', None),
                'touches_border': touches
            }
            time_series_data.append(prop)

            # Only create plot in full EM mode
            plot_file = os.path.join(cell_plot_folder, f"frame_t_{t:03d}.png")
            plot_cell_and_gamma_overlay(plot_data, plot_filename=plot_file)

        prev_mask, prev_mask_gfp = mask, mask_gfp

    except Exception as e:
        print(f"Error quantifying cell {cell_id} at t={t}: {e}")


# Save data
df_cell = pd.DataFrame(time_series_data)
df_cell.to_csv(os.path.join(output_tracked_cells_folder, f"cell_{cell_id}_data.csv"), index=False)
print(f"Saved quantification data for cell {cell_id}")



