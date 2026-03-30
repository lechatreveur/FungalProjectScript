#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 10:47:34 2025

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from skimage.measure import find_contours, regionprops, label
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
import os
from tifffile import TiffFile, imsave
from cellpose import models
from skimage.color import rgb2gray  # Import for RGB to grayscale conversion

# Paths
input_tif = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/P3_BF.tif"
output_frames_folder = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Frames"
output_masks_folder = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Masks"
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
for i, frame in enumerate(frames[:1]):
    # Convert to grayscale if RGB
    if frame.ndim == 3:
        if frame.shape[-1] == 3:
            # This branch is if you actually had a 3-channel RGB image
            from skimage.color import rgb2gray
            frame_grayscale = rgb2gray(frame.transpose(1, 2, 0))
            frame_grayscale = (frame_grayscale * 65535).astype(np.uint16)
        else:
            # For images with a different number of channels, select one channel
            frame_grayscale = frame[3,...]
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

#%% load single segmentation
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
cell_ID = 85
# Load the segmentation mask saved from your script
mask = imread("/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Masks/segmented_timepoint_000.tif")

# Find unique labels (each cell should have a unique label, excluding background 0)
cell_labels = np.unique(mask)
cell_labels = cell_labels[cell_labels != 0]  # Remove background

print("Found cell labels:", cell_labels)

# Check if any cells were segmented
if cell_labels.size > 0:
    # Select a single cell, for example, the first one (pixel value - 1)
    selected_label = cell_labels[cell_ID]
    single_cell_mask = (mask == selected_label)
    
    # Visualize the single cell segmentation
    plt.imshow(single_cell_mask, cmap="gray")
    plt.title(f"Segmentation for cell label: {selected_label}")
    plt.axis("off")
    plt.show()
else:
    print("No cells were segmented.")




#%% save singel-cell fluorescent image
import os
from tifffile import TiffFile, imsave

# Paths
input_tif = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/P3_GFP_MIP.tif"
output_frames_folder = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Frames"

# Create output directory if it doesn't exist
os.makedirs(output_frames_folder, exist_ok=True)

# Load the multi-frame TIFF
with TiffFile(input_tif) as tif:
    frames = tif.asarray()

# Ensure correct shape
# If your TIFF has shape (num_frames, num_channels, height, width), you can slice as needed.
# For now, we're using the full frames array for fluorescent channel 1.
fluorescent_C1_frames = frames  # or use frames[:, 0, :, :] if necessary
# fluorescent_C2_frames = frames[:, 1, :, :]

# Process only the first frame (adjust slicing as needed)
for i, frame in enumerate(fluorescent_C1_frames[:1]):
    fluorescent_C1_frame = frame  # Use the current frame directly
    # fluorescent_C2_frame = fluorescent_C2_frames[i]

    # Save each channel's frame as an individual TIFF without modifying dtype
    fluorescent_C1_filename = os.path.join(output_frames_folder, f"fluorescent_C1_{i:03d}.tif")
    # fluorescent_C2_filename = os.path.join(output_frames_folder, f"fluorescent_C2_{i:03d}.tif")
    
    imsave(fluorescent_C1_filename, fluorescent_C1_frame)
    # imsave(fluorescent_C2_filename, fluorescent_C2_frame)
    
    print(f"Saved fluorescent channel frame: {fluorescent_C1_filename}")
    # print(f"Saved fluorescent channel frame: {fluorescent_C2_filename}")

print("Fluorescent channel frame extraction completed.")

#%% overlaid cell mask to fluorescent image and crop to single cell
import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imsave
from skimage.measure import regionprops, label

# File paths
fluorescent_image_file = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Frames/fluorescent_C1_000.tif"
segmentation_mask_file = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Masks/segmented_timepoint_000.tif"
output_extracted_folder = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/ExtractedCells"

os.makedirs(output_extracted_folder, exist_ok=True)

# Load the fluorescent image and segmentation mask
fluorescent_img = imread(fluorescent_image_file)
segmentation_mask = imread(segmentation_mask_file)

# Identify unique cell labels (exclude background label 0)
cell_labels = np.unique(segmentation_mask)
cell_labels = cell_labels[cell_labels != 0]

if cell_labels.size == 0:
    print("No cells were segmented in the mask.")
else:
    # Select a single cell (e.g., the first one)
    selected_label = cell_labels[85]
    print(f"Selected cell label: {selected_label}")
    
    # Create a binary mask for the selected cell
    cell_mask = (segmentation_mask == selected_label)
    
    # Use regionprops to find the bounding box of the cell
    labeled_mask = label(cell_mask)  # Ensure the mask is labeled for regionprops
    props = regionprops(labeled_mask)
    
    if props:
        region = props[0]  # There should only be one region for this single cell mask
        min_row, min_col, max_row, max_col = region.bbox
        print(f"Extracting bounding box: rows {min_row} to {max_row}, cols {min_col} to {max_col}")
        
        # Extract the corresponding region from the fluorescent image
        extracted_cell = fluorescent_img[min_row:max_row, min_col:max_col]
        
        # Optionally, mask the fluorescent image to only show the cell (setting other pixels to 0)
        # This step applies the binary cell mask to the fluorescent region.
        cell_region_mask = cell_mask[min_row:max_row, min_col:max_col]
        extracted_cell_masked = extracted_cell.copy()
        extracted_cell_masked[~cell_region_mask] = 0
        
        # Save the extracted cell image (both cropped and masked version)
        output_cropped_file = os.path.join(output_extracted_folder, f"extracted_cell_{selected_label:03d}_cropped.tif")
        output_masked_file = os.path.join(output_extracted_folder, f"extracted_cell_{selected_label:03d}_masked.tif")
        imsave(output_cropped_file, extracted_cell)
        imsave(output_masked_file, extracted_cell_masked)
        
        print(f"Extracted cropped cell image saved to: {output_cropped_file}")
        print(f"Extracted masked cell image saved to: {output_masked_file}")
        
        # Visualize the extracted cell region (masked)
        plt.figure(figsize=(6,6))
        plt.imshow(extracted_cell_masked, cmap='gray')
        plt.title(f"Extracted Cell Label: {selected_label}")
        plt.axis("off")
        plt.show()
    else:
        print("No region found for the selected cell mask.")
#%% use slide box to quantify membrane intensity
import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imsave
from skimage.measure import find_contours, regionprops, label

# Helper function to compute average intensity in a square window (5x5) centered at (y, x)
# Only pixels inside the segmentation mask (seg_mask==True) are included in the average.
def average_intensity_at_point(img, seg_mask, y, x, window_size=5):
    half = window_size // 2
    y_min = max(y - half, 0)
    y_max = min(y + half + 1, img.shape[0])
    x_min = max(x - half, 0)
    x_max = min(x + half + 1, img.shape[1])
    region = img[y_min:y_max, x_min:x_max]
    region_mask = seg_mask[y_min:y_max, x_min:x_max]
    if np.any(region_mask):
        return np.mean(region[region_mask])
    else:
        return np.nan  # or return 0

def enforce_ccw(contour):
    """
    Ensure the contour is in counter-clockwise (CCW) order.
    The contour is an array of shape (n_points, 2) where each row is [row, col].
    This function uses the shoelace formula to compute the signed area.
    If the area is negative, the contour is clockwise and is reversed.
    """
    # Extract x and y coordinates; note: contour[:, 1] is x (columns) and contour[:, 0] is y (rows)
    x = contour[:, 1]
    y = contour[:, 0]
    # Compute signed area using the shoelace formula (with wrap-around)
    area = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    if area > 0:
        contour = contour[::-1]
    return contour

# File paths
fluorescent_image_file = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Frames/fluorescent_C1_000.tif"
segmentation_mask_file = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Masks/segmented_timepoint_000.tif"
output_extracted_folder = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/ExtractedCells"

os.makedirs(output_extracted_folder, exist_ok=True)

# Load the fluorescent image and segmentation mask
fluorescent_img = imread(fluorescent_image_file)
segmentation_mask = imread(segmentation_mask_file)

# Identify unique cell labels (exclude background label 0)
cell_labels = np.unique(segmentation_mask)
cell_labels = cell_labels[cell_labels != 0]

if cell_labels.size == 0:
    print("No cells were segmented in the mask.")
else:
    # Select a single cell (for example, the first one)
    selected_label = cell_labels[85]
    print(f"Selected cell label: {selected_label}")
    
    # Create a binary mask for the selected cell
    cell_mask = (segmentation_mask == selected_label)
    
    # Extract the ordered inner boundary using find_contours
    contours = find_contours(cell_mask.astype(float), 0.5)
    if len(contours) == 0:
        print("No contour found for the selected cell.")
    else:
        # Select the longest contour (if multiple exist)
        contour = max(contours, key=len)
        # Enforce counter-clockwise order using the shoelace formula
        contour = enforce_ccw(contour)
        
        # Round coordinates for indexing and clip them to image boundaries
        contour_int = np.rint(contour).astype(int)
        contour_int[:, 0] = np.clip(contour_int[:, 0], 0, fluorescent_img.shape[0]-1)
        contour_int[:, 1] = np.clip(contour_int[:, 1], 0, fluorescent_img.shape[1]-1)
        
        # Compute the average intensity for each contour point using a 5x5 window
        intensities = []
        for point in contour_int:
            y, x = point
            avg_intensity = average_intensity_at_point(fluorescent_img, cell_mask, y, x, window_size=20)
            intensities.append(avg_intensity)
        intensities = np.array(intensities)
        
        # Plot the intensity profile along the ordered boundary
        plt.figure(figsize=(10, 4))
        plt.plot(intensities, marker='o', linestyle='-')
        plt.xlabel("Boundary Position (ordered)")
        plt.ylabel("Average Fluorescent Intensity (5-pixel window)")
        plt.title(f"Fluorescent Intensity Profile Along Cell {selected_label} Inner Boundary (CCW)")
        plt.tight_layout()
        plt.show()
        
        # Optionally, save the intensity profile to a text file
        output_intensity_file = os.path.join(output_extracted_folder, f"cell_{selected_label:03d}_intensity.txt")
        np.savetxt(output_intensity_file, intensities, fmt="%.2f")
        print(f"Intensity profile saved to: {output_intensity_file}")
        
        # Determine the bounding box of the cell using regionprops
        labeled_cell = label(cell_mask)
        props = regionprops(labeled_cell)
        if props:
            region = props[0]
            min_row, min_col, max_row, max_col = region.bbox
        else:
            # Fallback: use the min/max of the contour coordinates
            min_row = np.min(contour_int[:, 0])
            min_col = np.min(contour_int[:, 1])
            max_row = np.max(contour_int[:, 0]) + 1
            max_col = np.max(contour_int[:, 1]) + 1
        
        # Crop the fluorescent image around the cell's bounding box
        cropped_img = fluorescent_img[min_row:max_row, min_col:max_col]
        
        # Adjust contour coordinates to the cropped image coordinate system
        contour_cropped = contour_int.copy()
        contour_cropped[:, 0] -= min_row
        contour_cropped[:, 1] -= min_col
        
        # Plot the cropped fluorescent image with the thick (5-pixel) contour overlaid
        plt.figure(figsize=(8, 8))
        plt.imshow(cropped_img, cmap='gray')
        plt.plot(contour_cropped[:, 1], contour_cropped[:, 0], 'r-', linewidth=5, label='Cell Contour')
        # Mark the starting point of the contour
        start_y, start_x = contour_cropped[0]
        plt.plot(start_x, start_y, 'bo', markersize=8, label='Start')
        plt.text(start_x, start_y, " Start", color='blue', fontsize=12, fontweight='bold')
        plt.title(f"Cropped Cell {selected_label} with Thick Contour (CCW)")
        plt.axis("off")
        plt.legend()
        plt.tight_layout()
        plt.show()
#%% find end points 
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.segmentation import find_boundaries

# Create a cropped cell mask from the full cell mask:
cropped_cell_mask = cell_mask[min_row:max_row, min_col:max_col]
# Ensure the first and last rows and columns of cropped_cell_mask are False.
cropped_cell_mask[0, :] = False
cropped_cell_mask[-1, :] = False
cropped_cell_mask[:, 0] = False
cropped_cell_mask[:, -1] = False
# Get region properties from the cropped mask.
labeled_cell = label(cropped_cell_mask)
props = regionprops(labeled_cell)

if not props:
    raise ValueError("No region found in the mask.")

# Use the first region.
region = props[0]
# Get centroid and orientation; note: region.centroid returns (row, col)
P0 = np.array(region.centroid)
major_axis_length = region.major_axis_length
orientation = region.orientation  # in radians

# Define the unit direction vector along the major axis.
# Note: In skimage, orientation is defined as the angle between the 0th axis (rows) and the major axis.
d = np.array([np.cos(orientation), np.sin(orientation)])  # direction in (row, col)

# Instead of using find_contours, get all boundary pixels using find_boundaries.
boundary_mask = find_boundaries(cropped_cell_mask, mode='inner')
boundary_coords = np.column_stack(np.nonzero(boundary_mask))  # shape (M, 2) in (row, col) coordinates

# Set a tolerance (in pixels) to decide if a boundary pixel is "on" the major axis.
tol = 2.0

on_line_t = []  # will store the projection parameter t for each boundary pixel
on_line_points = []  # store the boundary pixel coordinates that lie near the line

for Q in boundary_coords:
    # Q is a boundary pixel in (row, col) coordinates.
    # Compute projection parameter: t = d dot (Q - P0)
    t = np.dot(Q - P0, d)
    # Compute the corresponding point on the line: P0 + t*d
    Q_line = P0 + t * d
    # Compute perpendicular distance from Q to the line.
    dist = np.linalg.norm(Q - Q_line)
    if dist < tol:
        on_line_t.append(t)
        on_line_points.append(Q)

if len(on_line_points) == 0:
    raise ValueError("No boundary points found on the major axis within the specified tolerance.")

on_line_t = np.array(on_line_t)
on_line_points = np.array(on_line_points)

# Select endpoints as the points with the minimum and maximum t values.
t_min_idx = np.argmin(on_line_t)
t_max_idx = np.argmax(on_line_t)
endpoint1 = on_line_points[t_min_idx]  # in (row, col)
endpoint2 = on_line_points[t_max_idx]

# For plotting, convert (row, col) to (x, y) where x = col and y = row.
ep1_plot = (endpoint1[1], endpoint1[0])
ep2_plot = (endpoint2[1], endpoint2[0])

print("Computed Endpoint 1 (row, col):", endpoint1)
print("Computed Endpoint 2 (row, col):", endpoint2)

# --- Visualization ---
plt.figure(figsize=(8,8))
plt.imshow(cropped_cell_mask, cmap='gray')
# Optionally, overlay all boundary points for reference.
plt.scatter(boundary_coords[:,1], boundary_coords[:,0], s=1, c='lime', alpha=0.5, label='All Boundary Points')
# Plot the major axis line for reference.
L = 0.5 * major_axis_length
line_pts = np.array([P0 - L*d, P0 + L*d])
plt.plot(line_pts[:,1], line_pts[:,0], 'b--', linewidth=2, label='Major Axis')
# Plot the computed endpoints.
plt.scatter(ep1_plot[0], ep1_plot[1], color='red', s=100, marker='x', label='Endpoint 1')
plt.scatter(ep2_plot[0], ep2_plot[1], color='blue', s=100, marker='x', label='Endpoint 2')
plt.text(ep1_plot[0], ep1_plot[1], ' E1', color='red', fontsize=12, weight='bold')
plt.text(ep2_plot[0], ep2_plot[1], ' E2', color='blue', fontsize=12, weight='bold')
plt.title("Cropped Cell Mask with Major Axis Intersection Endpoints")
plt.legend()
plt.show()
#%% try distance to membrane 5-D 4 compartments 2 laten model
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal, norm

#% Prepare Data

# --- Assume these variables are defined from your segmentation workflow:
# cell_mask: full binary segmentation mask (2D array)
# cropped_img: cropped fluorescent image (2D array) corresponding to the cell region.
# min_row, min_col, max_row, max_col: bounding box coordinates for the cell.
# Also assume you have computed the fixed polarity endpoints:
#   endpoint1: (row, col) for polarity site 1
#   endpoint2: (row, col) for polarity site 2

# Create a cropped cell mask from the full cell mask:
cropped_cell_mask = cell_mask[min_row:max_row, min_col:max_col]

# Ensure the mask's border is false so that the boundary does not touch image edges.
cropped_cell_mask[0, :] = False
cropped_cell_mask[-1, :] = False
cropped_cell_mask[:, 0] = False
cropped_cell_mask[:, -1] = False

# Extract pixel coordinates (within the cropped image) that belong to the cell.
y_idx, x_idx = np.where(cropped_cell_mask)
intensities = cropped_img[y_idx, x_idx]

# Build the 3-D dataset: each row is [x, y, intensity] 
# (Note: x corresponds to column, y corresponds to row.)
X_data = np.column_stack((x_idx, y_idx, intensities))
print("Extracted", X_data.shape[0], "pixels from the cell.")

# Define region bounds based on extracted data.
region_min_data = np.array([np.min(x_idx), np.min(y_idx), np.min(intensities)])
region_max_data = np.array([np.max(x_idx), np.max(y_idx), np.max(intensities)])
print("Uniform region bounds:", region_min_data, region_max_data)

# Define area A as the sum of True pixels in the cropped mask.
A = cropped_cell_mask.sum()
print("Area A (number of pixels in cell):", A)



import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from scipy.stats import multivariate_normal, norm

#% (u,v) Transformation for Polarity Modeling
# (Assuming cropped_cell_mask is already defined and its border set to False)

# Extract the cell boundary.
contours = find_contours(cropped_cell_mask.astype(float), level=0.5)
if len(contours) == 0:
    raise ValueError("No contour found in the cell mask.")
# Use the longest contour.
boundary = max(contours, key=len)  # in (row, col) coordinates
# Ensure the boundary is closed.
if np.linalg.norm(boundary[0] - boundary[-1]) > 1e-3:
    boundary = np.vstack([boundary, boundary[0]])

# Compute arc-length along the boundary.
diffs = np.diff(boundary, axis=0)
segment_lengths = np.sqrt(np.sum(diffs**2, axis=1))
arc_length = np.concatenate(([0], np.cumsum(segment_lengths)))
L = arc_length[-1]  # Total perimeter
u_boundary = arc_length  # u coordinate for each boundary point, in [0,L]

# For each polarity endpoint, find its corresponding u coordinate.
tree = cKDTree(boundary)
dist1, idx1 = tree.query(endpoint1)  # endpoint1 in (row, col)
u_endpoint1 = u_boundary[idx1]
dist2, idx2 = tree.query(endpoint2)
u_endpoint2 = u_boundary[idx2]
print("Polarity endpoint 1 (row, col):", endpoint1, "u_E1:", u_endpoint1)
print("Polarity endpoint 2 (row, col):", endpoint2, "u_E2:", u_endpoint2)
print("Total perimeter L:", L)

# Compute v (distance from boundary) for each pixel using a distance transform.
v_map = distance_transform_edt(cropped_cell_mask)

# For u: for each pixel inside the cell, assign the u value of its nearest boundary point.
rows_in, cols_in = np.nonzero(cropped_cell_mask)
cell_coords = np.column_stack((rows_in, cols_in))
boundary_tree = cKDTree(boundary)
dists, indices = boundary_tree.query(cell_coords)
u_cell = u_boundary[indices]

# Build a u_map (same shape as cropped_cell_mask).
u_map = np.zeros_like(cropped_cell_mask, dtype=float)
u_map[rows_in, cols_in] = u_cell

# (Optional visualization of u_map and v_map)
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.imshow(u_map, cmap='viridis', origin='upper')
plt.title("u_map (Arc-length coordinate)")
plt.colorbar(label='u')
plt.subplot(1,2,2)
plt.imshow(v_map, cmap='magma', origin='upper')
plt.title("v_map (Distance from boundary)")
plt.colorbar(label='v')
plt.show()

#% Define PDF Functions for Background and Nucleus (unchanged)

def pdf_background(X, area, mu_bg, sigma_bg):
    I = X[:, 2]
    f_int = 1/np.sqrt(2*np.pi*(sigma_bg**2)) * np.exp(-0.5*((I - mu_bg)/(sigma_bg))**2)
    return (1.0/area)*f_int

def pdf_circular_nucleus(X, mu_xy_nuc, sigma_xy, mu_I, sigma_I):
    X_xy = X[:, :2]
    diff = X_xy - mu_xy_nuc
    f_spatial = 1.0/(2*np.pi*(sigma_xy**2)) * np.exp(-0.5*np.sum(diff**2,axis=1)/(sigma_xy**2))
    I = X[:, 2]
    f_int = 1/np.sqrt(2*np.pi*(sigma_I**2)) * np.exp(-0.5*((I - mu_I)/(sigma_I))**2)
    return f_spatial * f_int

#% Define the Polarity PDF in (u,v) Space with Intensity Constrained to the Nucleus

def circular_diff(u, u0, L):
    """Compute the circular difference between u and u0 given total length L."""
    diff = np.abs(u - u0)
    return np.minimum(diff, L - diff)

def pdf_polarity_uv(X, u_map, v_map, u_endpoint, sigma_u, sigma_v, mu_I, sigma_I, L):
    """
    Evaluate the full PDF for a polarity site using the (u,v) model.
    Spatially, we assume:
      - A Gaussian in u (using a circular difference) centered at u_endpoint.
      - A half-normal (truncated Gaussian on [0, ∞)) in v.
    For intensity, we use the same Gaussian as the nucleus (with parameters mu_I and sigma_I).
    
    Parameters:
      X : array, shape (N,3)
          Each row is [x, y, intensity] (with x = column, y = row).
      u_map, v_map : 2D arrays (same shape as cropped_cell_mask)
      u_endpoint : scalar, target u coordinate.
      sigma_u : scalar, standard deviation for u.
      sigma_v : scalar, standard deviation for v.
      mu_I, sigma_I : scalars, intensity parameters (shared with nucleus).
      L : total perimeter (for circular difference).
      
    Returns:
      pdf_vals : array, shape (N,), the full PDF values.
    """
    # Extract pixel coordinates.
    rows = X[:, 1].astype(int)
    cols = X[:, 0].astype(int)
    u_vals = u_map[rows, cols]
    v_vals = v_map[rows, cols]
    
    # Spatial PDF:
    du = circular_diff(u_vals, u_endpoint, L)
    f_u = np.exp(- (du**2) / (2 * sigma_u**2))
    f_v = (np.sqrt(2) / (sigma_v * np.sqrt(np.pi))) * np.exp(- (v_vals**2) / (2 * sigma_v**2))
    spatial_pdf = f_u * f_v
    
    # Intensity PDF: use the same Gaussian as for the nucleus.
    I = X[:, 2]
    f_int = 1 / np.sqrt(2*np.pi*sigma_I**2) * np.exp(- ((I - mu_I)**2) / (2 * sigma_I**2))
    
    return spatial_pdf * f_int

def pdf_cyto(X, area, mu_cyto, sigma_cyto):
    """
    Cytoplasm (non-expressing cell): spatially uniform over the cell and with a Gaussian intensity.
    """
    I = X[:, 2]
    f_int = norm.pdf(I, loc=mu_cyto, scale=sigma_cyto)
    return (1.0 / area) * f_int

#% Hierarchical Probability Model

def hierarchical_pdf(X, area, p_express, pi, params):
    """
    Compute the hierarchical probability of data X given the model.
    
    Parameters:
      X: (N,3) array with columns [x, y, intensity]
      area: total number of cell pixels (A)
      p_express: P(Y=1), prior probability that a cell expresses the fluorescent protein.
      pi: mixing weights for the compartments when Y=1, as an array of length 4
          (for background, nucleus, polarity site 1, polarity site 2).
      params: dictionary containing parameters for each component.
      
    Returns:
      P(D): the overall probability of the data.
    """
    # For Y = 1 (cell expresses fluorescent protein)
    f_bg = pdf_background(X, area, params['mu_bg'], params['sigma_bg'])
    f_nuc = pdf_circular_nucleus(X, params['mu_xy_nuc'], params['sigma_xy'], params['mu_I'], params['sigma_I'])
    f_pol1 = pdf_polarity_uv(X, params['u_map'], params['v_map'], params['u_endpoint1'],
                              params['sigma_u1'], params['sigma_v1'], params['mu_I'], params['sigma_I'], params['L'])
    f_pol2 = pdf_polarity_uv(X, params['u_map'], params['v_map'], params['u_endpoint2'],
                              params['sigma_u2'], params['sigma_v2'], params['mu_I'], params['sigma_I'], params['L'])
    f_Y1 = pi[0]*f_bg + pi[1]*f_nuc + pi[2]*f_pol1 + pi[3]*f_pol2
    
    # For Y = 0 (cell does not express, only cytoplasm)
    f_Y0 = pdf_cyto(X, area, params['mu_cyto'], params['sigma_cyto'])
    
    return p_express * f_Y1 + (1 - p_express) * f_Y0

#% Set Up Dummy Parameters (replace these with your estimates)

# Prior probability that cell expresses fluorescent protein.
p_express = 0.6

# Mixture weights for Y=1 (should sum to 1).
pi = np.array([0.3, 0.4, 0.15, 0.15])  # background, nucleus, polarity1, polarity2

# For f_bg, f_nuc, and the shared nucleus intensity:
mu_bg = 100.0
sigma_bg = 10.0
# Nucleus spatial parameters:
mu_xy_nuc = np.mean(X_data[:, :2], axis=0)
sigma_xy = max(np.std(X_data[:,0]), eps)
# Shared nucleus intensity parameters:
mu_I = np.mean(X_data[:,2])
sigma_I = max(np.std(X_data[:,2]), eps)

# For cytoplasmic cells (Y=0):
mu_cyto = 80.0
sigma_cyto = 15.0



# Fixed endpoints (in (row, col) coordinates) for polarity sites.
endpoint1 = np.array([50, 100])  # dummy value
endpoint2 = np.array([110, 80])  # dummy value

# Using a KD-tree on the boundary (we assume the (u,v) transformation code has been run),
# here we provide dummy u coordinates for the endpoints:
u_endpoint1 = 100.0
u_endpoint2 = 400.0

# For the polarity sites, spatial parameters:
sigma_u1 = 15.0  # fixed
sigma_v1 = 4.0   # initial value (to be updated in EM, if desired)
sigma_u2 = 15.0  # fixed
sigma_v2 = 4.0   # initial value

# Build a dictionary with all parameters.
params = {
    'mu_bg': mu_bg,
    'sigma_bg': sigma_bg,
    'mu_xy_nuc': mu_xy_nuc,
    'sigma_xy': sigma_xy,
    'mu_I': mu_I,
    'sigma_I': sigma_I,
    'u_map': u_map,
    'v_map': v_map,
    'u_endpoint1': u_endpoint1,
    'u_endpoint2': u_endpoint2,
    'sigma_u1': sigma_u1,
    'sigma_v1': sigma_v1,
    'sigma_u2': sigma_u2,
    'sigma_v2': sigma_v2,
    'L': L,
    'mu_cyto': mu_cyto,
    'sigma_cyto': sigma_cyto
}

#% Evaluate the Hierarchical Probability for the Cell
P_D = hierarchical_pdf(X_data, A, p_express, pi, params)
print("Hierarchical probability P(D):", P_D)

# Optionally, if you wish to compute the log likelihood:
log_likelihood = np.sum(np.log(P_D + 1e-10))
print("Log likelihood of cell data:", log_likelihood)


#%%


