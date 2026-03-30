#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:38:28 2025

@author: user
"""
import os
import gc
from tifffile import TiffFile, imwrite  # Using imwrite instead of deprecated imsave
from cellpose import models
import numpy as np
from skimage.color import rgb2gray

# Monkey-patch cellpose.transforms with a normalize99 function that accepts extra parameters
import cellpose.transforms as transforms
def normalize99(img, lower=1, upper=99, copy=False):
    if copy:
        img = np.copy(img)
    p_lower = np.percentile(img, lower)
    p_upper = np.percentile(img, upper)
    norm = (img - p_lower) / (p_upper - p_lower + 1e-6)
    return np.clip(norm, 0, 1)
transforms.normalize99 = normalize99

# Paths
input_tif = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/P3_BF.tif"
# Set output filenames that reflect the selected time and z slice:
output_frame_filename = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Frames/brightfield_t001_z003.tif"
output_mask_filename  = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Masks/segmented_t001_z003.tif"
custom_model_path     = "/Users/user/.cellpose/models/yeast_BF_cp3"

# Create output directories if they don't exist
os.makedirs(os.path.dirname(output_frame_filename), exist_ok=True)
os.makedirs(os.path.dirname(output_mask_filename), exist_ok=True)

# Initialize Cellpose model with your custom model
model = models.CellposeModel(gpu=True, pretrained_model=custom_model_path)

# Parameters for segmentation
channels = [0, 0]  # Assuming grayscale images; adjust if needed
diameter = 100     # Set a specific cell diameter for segmentation

# Load the multi-frame TIFF and extract the specific frame
with TiffFile(input_tif) as tif:
    frames = tif.asarray()
    print("Frames shape:", frames.shape)
    # Here we assume frames is organized as (time, z, height, width)
    # For t: 1/60, use index 0 (first time point)
    # For z: 3/13, use index 2 (third z-slice)
    t_index = 0  # First time point
    z_index = 3  # Third z-slice (zero-indexed)
    frame = frames[t_index, z_index, :, :]

# If the frame is RGB, convert it to grayscale; otherwise, assume it's already grayscale.
if frame.ndim == 3:
    if frame.shape[-1] == 3:
        frame_grayscale = rgb2gray(frame)
        frame_grayscale = (frame_grayscale * 65535).astype(np.uint16)
    else:
        frame_grayscale = frame[0, ...]  # Fallback: select the first channel
else:
    frame_grayscale = frame

# Save the selected frame as a TIFF file
imwrite(output_frame_filename, frame_grayscale)
print(f"Saved grayscale frame: {output_frame_filename}")

# Perform segmentation on the selected frame
masks, flows, styles = model.eval(frame_grayscale, diameter=diameter, channels=channels)

# Save the segmented mask
imwrite(output_mask_filename, masks.astype(np.uint16))
print(f"Saved segmentation mask: {output_mask_filename}")

# Cleanup
del frames, frame, frame_grayscale, masks, flows, styles
gc.collect()

print("Selected frame segmentation completed.")


#%% load single segmentation
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.measure import regionprops, label
from skimage.measure import find_contours
cell_ID = 81 #81(Y=2,2Po), 116(Y=0), 105(Y=1),114(Y=2,1Po)
# Load the segmentation mask saved from your script
mask = imread("/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Masks/segmented_t001_z003.tif")

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


# File paths
fluorescent_image_file = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Frames/fluorescent_C1_000.tif"
segmentation_mask_file = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Masks/segmented_t001_z003.tif"
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
    selected_label = cell_labels[cell_ID]
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
        imwrite(output_cropped_file, extracted_cell)
        imwrite(output_masked_file, extracted_cell_masked)
        
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
# use slide box to quantify membrane intensity


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
    selected_label = cell_labels[cell_ID]
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
# plot histogram of intensity


data = intensities 

# Plot the histogram
plt.hist(data, bins=25, edgecolor='black')  # 'bins' determines the number of bins (bars)
plt.title('Histogram of Data')  # Title of the plot
plt.xlabel('Value')  # Label for the x-axis
plt.ylabel('Frequency')  # Label for the y-axis

# Show the plot
plt.show()

#%% v3 optimize for 3 states
# find end points 
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

from scipy.stats import norm
from scipy.stats import uniform
from scipy.special import logsumexp
import numpy as np
import matplotlib.pyplot as plt

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
def pdf_uniform_1d(X, mu_bg):
    """
    Computes the PDF of a uniform distribution in 1D along the third dimension (Z-axis).

    Parameters:
        X (numpy.ndarray): Nx3 array where the third column represents the 1D variable.

    Returns:
        numpy.ndarray: PDF values for each point.
    """
    I = X[:, 2]  # Extract the Z-dimension (1D)

    # Find the min and max values of the distribution region
    I_min, I_max = np.min(I), np.max(I)
    
    # Compute interval length
    interval_length = I_max - mu_bg #I_min
    
    # Prevent division by zero if all values are the same
    if interval_length == 0:
        return np.zeros_like(I)

    # Check which points are inside the interval
    #in_region = (I >= I_min) & (I <= I_max)
    in_region = (I >= mu_bg) & (I <= I_max)

    # Compute uniform probability density
    pdf_vals = np.zeros(len(I))
    pdf_vals[in_region] = 1.0 / interval_length

    return pdf_vals

def pdf_cytoplasm(X, area, mu_bg, sigma_bg):
    I = X[:, 2]
    f_int = 1/np.sqrt(2*np.pi*(sigma_bg**2)) * np.exp(-0.5*((I - mu_bg)/(sigma_bg))**2)
    # fit for background noise
    #min_3D = np.array([min(X[:,0]),min(X[:,1]),min(X[:,2])])
    #max_3D = np.array([max(X[:,0]),max(X[:,1]),max(X[:,2])])
    #f_int_uni = pdf_uniform_3d(X, min_3D, max_3D)
    return (1.0/area)*f_int# *0.5 + f_int_uni*0.5



def pdf_circular_nucleus(X, mu_xy_nuc, sigma_xy, mu_I, sigma_I):
    X_xy = X[:, :2]
    diff = X_xy - mu_xy_nuc
    f_spatial = 1.0/(2*np.pi*(sigma_xy**2)) * np.exp(-0.5*np.sum(diff**2,axis=1)/(sigma_xy**2))
    I = X[:, 2]
    f_int = 1/np.sqrt(2*np.pi*(sigma_I**2)) * np.exp(-0.5*((I - mu_I)/(sigma_I))**2)
    return f_spatial * f_int

# Define the Polarity PDF in (u,v) Space with Intensity Constrained to the Nucleus

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
    

# EM Algorithm Initialization

N = X_data.shape[0]
eps = 1e-6

# For polarity sites, we use the endpoints via the (u,v) model.
# Their u coordinates are given by u_endpoint1 and u_endpoint2 (computed previously).
# Initialize sigma_u for both sites (fixed) and sigma_v for each polarity site:
sigma_u1 = 10.0  # fixed
sigma_v1 = 5   # to be updated
sigma_u2 = 10.0  # fixed
sigma_v2 = 5   # to be updated

# Mixing weights for 3 states: background only (Y=0), background and nucleus 
# (Y=1), background, nucleus, and polarity site (Y=2)
Yi = np.array([0.1, 0.45, 0.45])

# Mixing weights for 2 components when Y = 1: background (Z=0), nucleus (Z=1)
#piY1 = np.array([0.8, 0.2])

# Mixing weights for 4 components: background (Z=0), nucleus (Z=1),
# polarity site 1 (Z=2), polarity site 2 (Z=3)
pi = np.array([0.2, 0.2, 0.3, 0.3])

# Background intensity parameters:
mu_bg = np.median(X_data[:,2])
sigma_bg = max(np.std(X_data[:,2]), eps)

mu_bg_int = np.median(X_data[:,2])
sigma_bg_int = max(np.std(X_data[:,2]), eps)

# Nucleus spatial parameters:
mu_xy_nuc = np.mean(X_data[:, :2], axis=0)
sigma_xy = max(np.std(X_data[:,0]), eps)

# Shared nucleus intensity parameters:
mu_I = np.mean(X_data[:,2])
sigma_I = max(np.std(X_data[:,2]), eps)

mu_I_int = np.mean(X_data[:,2])
sigma_I_int = max(np.std(X_data[:,2]), eps)
# For polarity sites, the intensity model is constrained to the nucleus,
# so we do not have separate intensity parameters for polarity.

# Assume A is the sum of cropped_cell_mask pixels.
A = cropped_cell_mask.sum()

max_iter = 10
tol = 1e-10
log_likelihoods_Y0 = []
log_likelihoods_Y1 = []
log_likelihoods_Y2 = []

#----- EM Algorithm for the 4-Compartment Model -----
for iteration in range(max_iter):
    #------
    #E-step:
    #------    
    
    # Intensity only
    X_sorted = X_data[np.argsort(X_data[:, 2])]
    X_small = X_data#X_sorted[::1,:]
    Xmin = np.mean(X_small[:,2]) #min(X_small[:,2])
    Xscale = max(X_small[:,2])-Xmin
    #PX_Y0 = pdf_cytoplasm(X_small, 1, mu_bg_int, sigma_bg_int)
    PX_Y0 = norm.pdf(X_small[:,2], loc=mu_bg_int, scale=sigma_bg_int)
    #PX_Y1 = (pdf_uniform_1d(X_small, mu_bg_int)*pi[1] + pdf_cytoplasm(X_small, A, mu_bg_int, sigma_bg_int)*pi[0])
    #PX_Y1 = uniform.pdf(X_small[:,2], loc=Xmin, scale=Xscale)*pi[1] + norm.pdf(X_small[:,2], loc=mu_bg_int, scale=sigma_bg_int)*pi[0]
    PX_Y1 = norm.pdf(X_small[:,2], loc=mu_I_int, scale=sigma_I_int)*pi[1] + norm.pdf(X_small[:,2], loc=mu_bg_int, scale=sigma_bg_int)*(1-pi[1])
    
    epsilon = 1e-60  # Small constant to avoid division by zero
    PX_Y0_safe = np.maximum(PX_Y0, epsilon)  
    PX_Y1_safe = np.maximum(PX_Y1, epsilon)
    
    #PZ0_Y1X = pdf_cytoplasm(X_small, A, mu_bg_int, sigma_bg_int) * pi[0] / PX_Y1_safe
    PZ0_Y1X = norm.pdf(X_small[:,2], loc=mu_bg_int, scale=sigma_bg_int)*pi[0]/ PX_Y1_safe
    #PZ1_Y1X = uniform.pdf(X_small[:,2], loc=Xmin, scale=Xscale)*pi[1] / PX_Y1_safe
    PZ1_Y1X = norm.pdf(X_small[:,2], loc=mu_I_int, scale=sigma_I_int)*pi[1] / PX_Y1_safe

    #PY0_X = np.prod(PX_Y0) / (np.prod(PX_Y0) + np.prod(PX_Y1))
    #PY1_X = np.prod(PX_Y1) / (np.prod(PX_Y0) + np.prod(PX_Y1))
    # Compute log probabilities
    log_PX_Y0 = np.sum(np.log(PX_Y0_safe))
    log_PX_Y1 = np.sum(np.log(PX_Y1_safe))

    # Compute PY0_X in log-space
    PY0_X = np.exp(log_PX_Y0 - logsumexp([log_PX_Y0, log_PX_Y1]))
    PY1_X = np.exp(log_PX_Y1 - logsumexp([log_PX_Y0, log_PX_Y1]))
    # Ensure sum(PZ1_Y1X) is not zero
    sum_PZ1_Y1X = np.maximum(np.sum(PZ1_Y1X), epsilon)
    # Ensure sum(PZ0_Y1X) is not zero
    sum_PZ0_Y1X = np.maximum(np.sum(PZ0_Y1X), epsilon)
    E_nuc_int = PY1_X * np.sum(PZ1_Y1X * X_small[:,2]) / sum_PZ1_Y1X + PY0_X * np.sum(X_small[:,2])/X_small[:,2].shape[0]
    E_cyt_int = PY1_X * np.sum(PZ0_Y1X * X_small[:,2]) / sum_PZ0_Y1X + PY0_X * np.sum(X_small[:,2])/X_small[:,2].shape[0]    
    log_likelihoods_Y1.append([PY0_X, PY1_X])
    
    
    
    # With location   
    fZ0 = pdf_cytoplasm(X_data, A, mu_bg, sigma_bg)
    
    fY0Z1 = pdf_circular_nucleus(X_data, mu_xy_nuc, sigma_xy, mu_bg, sigma_bg)
    fY0Z2 = pdf_polarity_uv(X_data, u_map, v_map, u_endpoint=u_endpoint1, 
                         sigma_u=sigma_u1, sigma_v=sigma_v1, mu_I=mu_bg, sigma_I=sigma_bg, L=L)
    fY0Z3 = pdf_polarity_uv(X_data, u_map, v_map, u_endpoint=u_endpoint2, 
                         sigma_u=sigma_u2, sigma_v=sigma_v2, mu_I=mu_bg, sigma_I=sigma_bg, L=L)
    
    
    fZ1 = pdf_circular_nucleus(X_data, mu_xy_nuc, sigma_xy, mu_I, sigma_I)
    fY1Z2 = pdf_polarity_uv(X_data, u_map, v_map, u_endpoint=u_endpoint1, 
                         sigma_u=sigma_u1, sigma_v=sigma_v1, mu_I=mu_bg, sigma_I=sigma_bg, L=L)
    fY1Z3 = pdf_polarity_uv(X_data, u_map, v_map, u_endpoint=u_endpoint2, 
                         sigma_u=sigma_u2, sigma_v=sigma_v2, mu_I=mu_bg, sigma_I=sigma_bg, L=L)
    
    fY2Z2 = pdf_polarity_uv(X_data, u_map, v_map, u_endpoint=u_endpoint1, 
                         sigma_u=sigma_u1, sigma_v=sigma_v1, mu_I=mu_I, sigma_I=sigma_I, L=L)
    fY2Z3 = pdf_polarity_uv(X_data, u_map, v_map, u_endpoint=u_endpoint2, 
                         sigma_u=sigma_u2, sigma_v=sigma_v2, mu_I=mu_I, sigma_I=sigma_I, L=L)
    #f_total_Y0 = Yi[0] * fZ0 #+ 1e-10
    f_total_Y0 = Yi[0] * (pi[0]*fZ0 + pi[1]*fY0Z1 + pi[2]*fY0Z2 + pi[3]*fY0Z3)
    #f_total_Y1 = Yi[1] * (piY1[0]*fZ0 + piY1[1]*fZ1) #  + 1e-10
    f_total_Y1 = Yi[1] * (pi[0]*fZ0 + pi[1]*fZ1 + pi[2]*fY1Z2 + pi[3]*fY1Z3) # + 1e-10
    f_total_Y2 = Yi[2] * (pi[0]*fZ0 + pi[1]*fZ1 + pi[2]*fY2Z2 + pi[3]*fY2Z3) # + 1e-10
    
    gammaY0Z0 = Yi[0]*pi[0]*fZ0 / f_total_Y0
    gammaY0Z1 = Yi[0]*pi[1]*fY0Z1 / f_total_Y0
    gammaY0Z2 = Yi[0]*pi[2]*fY0Z2 / f_total_Y0
    gammaY0Z3 = Yi[0]*pi[3]*fY0Z3 / f_total_Y0
    
    #gammaY1Z0 = Yi[1]*piY1[0]*fZ0 / f_total_Y1
    #gammaY1Z1 = Yi[1]*piY1[1]*fZ1 / f_total_Y1
    gammaY1Z0 = Yi[1]*pi[0]*fZ0 / f_total_Y1
    gammaY1Z1 = Yi[1]*pi[1]*fZ1 / f_total_Y1
    gammaY1Z2 = Yi[1]*pi[2]*fY1Z2 / f_total_Y1
    gammaY1Z3 = Yi[1]*pi[3]*fY1Z3 / f_total_Y1
    
    gammaY2Z0 = Yi[2]*pi[0]*fZ0 / f_total_Y2
    gammaY2Z1 = Yi[2]*pi[1]*fZ1 / f_total_Y2
    #gammaY2Z2 = Yi[2]*piY2[2]*fZ2 / f_total_Y2
    #gammaY2Z3 = Yi[2]*piY2[3]*fZ3 / f_total_Y2
    gammaY2Z2 = Yi[2]*pi[2]*fY2Z2 / f_total_Y2
    gammaY2Z3 = Yi[2]*pi[3]*fY2Z3 / f_total_Y2
    
    # Compute log-likelihood to monitor convergence
    #log_likelihood_Y0 = np.sum(np.log(fZ0))
    #log_likelihoods_Y0.append(log_likelihood_Y0)
    
    #log_likelihood_Y1 = np.sum(np.log(piY1[0]*fZ0  + piY1[1]*fZ1))
    #log_likelihoods_Y1.append(log_likelihood_Y1)
    
    #log_likelihood_Y2 = np.sum(np.log(piY2[0]*fZ0  + piY2[1]*fZ1 + piY2[2]*fZ2 + piY2[3]*fZ3))
    #log_likelihoods_Y2.append(log_likelihood_Y2)
    
    #-----
    # M-step:
    #-----
    
    # Intensity only
    #print(PY0_X)
    sum1minus_PY1_X_PZ1_Y1X = np.maximum(np.sum((1- PY1_X * PZ1_Y1X)), epsilon)
    mu_bg_int_new = np.sum(X_small[:,2] * (1-(PY1_X * PZ1_Y1X))) / sum1minus_PY1_X_PZ1_Y1X#np.sum((PY0_X) * PZ1_Y1X)
    sigma_bg_int_new = np.sqrt(np.sum((X_small[:,2]-mu_bg_int)**2 * (1 - PY1_X * PZ1_Y1X)) / sum1minus_PY1_X_PZ1_Y1X )
    mu_bg_int = mu_bg_int_new
    sigma_bg_int = max(sigma_bg_int_new,eps)
    
    sum_PY1_X_PZ1_Y1X = np.maximum(np.sum((PY1_X * PZ1_Y1X)), epsilon)
    mu_I_int_new = np.sum(X_small[:,2] * PY1_X * PZ1_Y1X) / sum_PY1_X_PZ1_Y1X#np.sum((PY0_X) * PZ1_Y1X)
    sigma_I_int_new = np.sqrt(np.sum((X_small[:,2]-mu_I_int)**2 *  PY1_X * PZ1_Y1X) / sum_PY1_X_PZ1_Y1X )
    mu_I_int = mu_I_int_new
    sigma_I_int = max(sigma_I_int_new,eps)    
    
    #pi_new = np.array([np.mean(gamma0), np.mean(gamma1), np.mean(gamma2), np.mean(gamma3)])
    #gamma_total = np.mean(gammaY0Z0+gammaY1Z0+gammaY2Z0) + np.mean(gammaY0Z1+gammaY1Z1+gammaY2Z1) + np.mean(gammaY0Z2+gammaY1Z2+gammaY2Z2) + np.mean(gammaY0Z3+gammaY1Z3+gammaY2Z3)
    #pi_new = np.array([np.mean(gammaY0Z0+gammaY1Z0+gammaY2Z0)/gamma_total, np.mean(gammaY0Z1+gammaY1Z1+gammaY2Z1)/gamma_total, np.mean(gammaY0Z2+gammaY1Z2+gammaY2Z2)/gamma_total, np.mean(gammaY0Z3+gammaY1Z3+gammaY2Z3)/gamma_total])
    #pi_new = np.array([np.mean(gamma0), pi[1], np.mean(gamma2), np.mean(gamma3)])
    #pi_new = np.array(pi)
    #piY2_new = np.array([np.mean(gammaY2Z0), piY2[1], np.mean(gammaY2Z2), np.mean(gammaY2Z3)])
    
    # Update pi for the polarity sites by using only their intensity
    # N(I|Z=2,mu_I,sigma_I) vs N(I|Z=3,mu_I,sigma_I), which is fY2Z2 vs fY2Z3
    p_Po_total = 1 - pi[0] - pi[1]
    pi2_new = p_Po_total * np.sum(fY2Z2) / (np.sum(fY2Z2 + fY2Z3))
    pi3_new = p_Po_total * np.sum(fY2Z3) / (np.sum(fY2Z2 + fY2Z3))
    pi_new = np.array([pi[0], pi[1], pi2_new, pi3_new])    
    
    # Update background intensity parameters for Y=0, Y=1 and Y = 2jointly.
    # For Y=0, the four component share the same background parameters.
    # For Y=1, the effective weight for the background part is gammaY1Z0 and gammaY1Z2+3
    # For Y=2, the effective weight for the backgorund part is gammaY2Z0

    # Combine Y = 0, Y = 1, and Y = 2 contributions:
    #effective_gammaZ0_total = Yi[0]*gammaY0 + Yi[1]*gammaY1Z0 + Yi[2]*gammaY2Z0
    effective_gammaZ0_total = gammaY0Z0  + gammaY1Z0  + gammaY2Z0 #+ gammaY0Z1 + gammaY0Z2 + gammaY0Z3 + gammaY1Z2 + gammaY1Z3
    sum_gammaZ0 = np.sum(effective_gammaZ0_total)
    mu_bg_new = np.sum(effective_gammaZ0_total * X_data[:,2]) / sum_gammaZ0
    sigma_bg_new = np.sqrt(np.sum(effective_gammaZ0_total * (X_data[:,2]-mu_bg_new)**2) / sum_gammaZ0)
    sigma_bg_new = max(sigma_bg_new, eps)
    
    # Update nucleus location parameters for Y = 0, Y = 1 and Y = 2 jointly.
    # For Y = 0, the effective weight for the nucleus location is gammaY0Z1
    # For Y = 1, the effective weight for the nucleus location is gammaY1Z1
    # For Y = 2, the effective weight for the nucleus location is gammaY2Z1
    
    # Combine Y = 0, Y = 1 and Y = 2 contributions:
    # effective_gammaZ1_total = Yi[1]*gammaY1Z1 + Yi[2]*gammaY2Z1
    effective_gammaZ1_total =  gammaY1Z1 + gammaY2Z1 #gammaY0Z1 +
    sum_gammaZ1 = np.sum(effective_gammaZ1_total)
    mu_xy_nuc_new = np.sum(effective_gammaZ1_total[:, np.newaxis] * X_data[:, :2], axis=0) / sum_gammaZ1
    var_x = np.sum(effective_gammaZ1_total * (X_data[:,0]-mu_xy_nuc_new[0])**2) / sum_gammaZ1
    var_y = np.sum(effective_gammaZ1_total * (X_data[:,1]-mu_xy_nuc_new[1])**2) / sum_gammaZ1
    sigma_xy_new = np.sqrt((var_x+var_y)/2)
    sigma_xy_new = max(sigma_xy_new, eps)
    
    # Update nucleus intensity parameters for Y = 1 and Y = 2 jointly.
    # For Y = 1, the effective weight for the nucleus intensity is gammaY1Z1
    # For Y = 2, the effective weight for the nucleus intensity is gammaY2Z1, gammaY2Z2, gammaY2Z3
    gamma_I = gammaY1Z1 + gammaY2Z1 + gammaY2Z2 + gammaY2Z3
    sum_gamma_I = np.sum(gamma_I)
    mu_I_new = np.sum(gamma_I * X_data[:,2]) / sum_gamma_I #+ 1 * sigma_bg_new
    #mu_I_new = max(mu_I_new,(mu_bg_new + 1 * sigma_bg_new))
    sigma_I_new = np.sqrt(np.sum(gamma_I * (X_data[:,2]-mu_I_new)**2) / sum_gamma_I)
    sigma_I_new = max(sigma_I_new, eps)
    
    # Constrain polarity site intensities to the nucleus intensity:
    mu_I_pol1_new = mu_I_new
    mu_I_pol2_new = mu_I_new
    
    # For polarity site 1: update sigma_v1 only (keep sigma_u1 fixed).
    # Combine Y = 0, Y = 1, and Y = 2 contributions
    # For Y = 0, the effective weight for the polarity site is gammaY0Z2
    # For Y = 1, the effective weight for the polarity site is gammaY1Z2
    # For Y = 2, the effective weight for the polarity site is gammaY2Z2

    rows_data = X_data[:,1].astype(int)
    cols_data = X_data[:,0].astype(int)
    u_vals = u_map[rows_data, cols_data]
    v_vals = v_map[rows_data, cols_data]
    effective_gammaPol1 = gammaY0Z2 + gammaY1Z2 + gammaY2Z2
    sum_gammaZ2 = np.sum(effective_gammaPol1)
    sigma_v1_new = np.sqrt(np.sum(effective_gammaPol1 * (v_vals)**2) / sum_gammaZ2)
    sigma_v1_new = max(sigma_v1_new, eps)
    sigma_u1_new = sigma_u1  # fixed
    
    # For polarity site 2: update sigma_v2 only (keep sigma_u2 fixed).
    # Combine Y = 0, Y = 1, and Y = 2 contributions
    # For Y = 0, the effective weight for the polarity site is gammaY0Z3
    # For Y = 1, the effective weight for the polarity site is gammaY1Z3
    # For Y = 2, the effective weight for the polarity site is gammaY2Z3
    effective_gammaPol2 = gammaY0Z3 + gammaY1Z3 + gammaY2Z3
    sum_gammaZ3 = np.sum(effective_gammaPol2)
    sigma_v2_new = np.sqrt(np.sum(effective_gammaPol2 * (v_vals)**2) / sum_gammaZ3)
    sigma_v2_new = max(sigma_v2_new, eps)
    sigma_u2_new = sigma_u2  # fixed
    
    # #Update Y: the new mixing fraction is the average effective fraction
    #f_Ytotal = np.sum(f_total_Y0) + np.sum(f_total_Y1) + np.sum(f_total_Y2)
    #Yi_new = np.array([np.sum(f_total_Y0)/f_Ytotal, np.sum(f_total_Y1)/f_Ytotal, np.sum(f_total_Y2)/f_Ytotal])
    
    # Update Y: use only the intensity data
    # Update Y = 0 and Y > 0 by using the ratio between likelihoods of monodistribution and bimodal distribution
    # Monodistribution is N(I|mu_bg,sigma_bg) = fZ0
    # Bimodal distribution is 0.8fZ0 + 0.2N(I|mu_I,sigma_I)
    fZn0 = pdf_cytoplasm(X_data, A, mu_I, sigma_I)
    fYn0 = 0.8 * fZ0 + 0.2 * fZn0
    fY_total = np.sum(fZ0) + np.sum(fYn0)
    Yi0_new = np.sum(fZ0) / fY_total
    # Update Y = 1 and 2 by using the ratio between likelihoods of polarity site
    # N(I|Z>1,mu_bg, sigma_bg) vs N(I|Z>1,mu_I,sigma_I), which is (fY1Z2+fY1Z3) vs (fY2Z2+fY2Z3)
    fY_total = np.sum(fY1Z2 + fY1Z3) + np.sum(fY2Z2 + fY2Z3)
    Yi1_new = (1-Yi0_new) * np.sum(fY1Z2 + fY1Z3) / fY_total
    Yi2_new = (1-Yi0_new) * np.sum(fY2Z2 + fY2Z3) / fY_total
    Yi_new = np.array([Yi0_new, Yi1_new, Yi2_new])
    
    # Convergence check.
    if (np.linalg.norm(mu_xy_nuc_new - mu_xy_nuc) < tol and
        abs(sigma_xy_new - sigma_xy) < tol and
        abs(mu_I_new - mu_I) < tol and
        abs(sigma_I_new - sigma_I) < tol and
        abs(mu_bg_new - mu_bg) < tol and
        abs(sigma_bg_new - sigma_bg) < tol and
        abs(sigma_v1_new - sigma_v1) < tol and
        abs(sigma_v2_new - sigma_v2) < tol):
        print(f"Convergence reached at iteration {iteration}")
        break
    
    # Update parameters.
    #pi = pi_new
    mu_bg, sigma_bg = mu_bg_new, sigma_bg_new
    mu_xy_nuc, sigma_xy = mu_xy_nuc_new, sigma_xy_new
    mu_I, sigma_I = mu_I_new, sigma_I_new
    # Polarity site intensity parameters are constrained:
    mu_I_pol1, sigma_I_pol1 = mu_I_new, sigma_I_new
    mu_I_pol2, sigma_I_pol2 = mu_I_new, sigma_I_new
    #sigma_v1 = sigma_v1_new
    #sigma_v2 = sigma_v2_new
    # sigma_u1 and sigma_u2 remain fixed.
    #Yi = Yi_new
    
# ----- Final Estimated Parameters -----
# print("Estimated mixing weights for Y:", Yi)
#print("Estimated mixing weights for pi:", pi)
print("Estimated background intensity: mu =", mu_bg, ", sigma =", sigma_bg)
# print("Estimated nucleus center (row, col):", mu_xy_nuc)
# print("Estimated nucleus sigma_xy:", sigma_xy)
print("Estimated nucleus intensity: mu =", mu_I, ", sigma =", sigma_I)
# print("Estimated polarity site 1 intensity: mu =", mu_I_pol1, ", sigma =", sigma_I_pol1)
# print("Estimated polarity site 2 intensity: mu =", mu_I_pol2, ", sigma =", sigma_I_pol2)
# print("Estimated polarity site 1: sigma_u =", sigma_u1, " (fixed), sigma_v =", sigma_v1)
# print("Estimated polarity site 2: sigma_u =", sigma_u2, " (fixed), sigma_v =", sigma_v2)
print("Estimated background intensity: mu =", mu_bg_int, ", sigma =", sigma_bg_int)
print("Estimated nucleus intensity: mu =", mu_I_int, ", sigma =", sigma_I_int)

#%% Create and Visualize the Nucleus Mask



# quantify prabability of polarity pitches
# Create empty images for the polarity responsibilities.
gamma_pol1_img = np.zeros_like(cropped_cell_mask, dtype=float)
gamma_pol2_img = np.zeros_like(cropped_cell_mask, dtype=float)

# x_idx and y_idx were extracted earlier (for pixels inside the cell).
# They index the cropped mask.
gamma_pol1_img[y_idx, x_idx] = gammaY2Z2  # gamma2: responsibilities for polarity site 1
gamma_pol2_img[y_idx, x_idx] = gammaY2Z3  # gamma3: responsibilities for polarity site 2

# Visualization
plt.figure(figsize=(10,5))

plt.subplot(3,4,1)
plt.imshow(gamma_pol1_img, cmap='jet', origin='upper')
plt.title("GammaY2Z2 Polarity 1")
plt.colorbar(label="Probability")

plt.subplot(3,4,2)
plt.imshow(gamma_pol2_img, cmap='jet', origin='upper')
plt.title("GammaY2Z3 Polarity 2")
plt.colorbar(label="Probability")

# plt.tight_layout()
# plt.show()
#
# Create empty images for the polarity responsibilities.
gamma_cy_img = np.zeros_like(cropped_cell_mask, dtype=float)
gamma_nu_img = np.zeros_like(cropped_cell_mask, dtype=float)

# x_idx and y_idx were extracted earlier (for pixels inside the cell).
# They index the cropped mask.
gamma_cy_img[y_idx, x_idx] = gammaY2Z0  # gamma2: responsibilities for polarity site 1
gamma_nu_img[y_idx, x_idx] = gammaY2Z1  # gamma3: responsibilities for polarity site 2


plt.subplot(3,4,3)
plt.imshow(gamma_cy_img, cmap='jet', origin='upper')
plt.title("GammaY2Z0 Cytoplasm")
plt.colorbar(label="Probability")

plt.subplot(3,4,4)
plt.imshow(gamma_nu_img, cmap='jet', origin='upper')
plt.title("GammaY2Z1 Nucleus")
plt.colorbar(label="Probability")

plt.tight_layout()

# quantify prabability of polarity pitches
# Create empty images for the polarity responsibilities.
gamma_pol1_img = np.zeros_like(cropped_cell_mask, dtype=float)
gamma_pol2_img = np.zeros_like(cropped_cell_mask, dtype=float)

# x_idx and y_idx were extracted earlier (for pixels inside the cell).
# They index the cropped mask.
gamma_pol1_img[y_idx, x_idx] = gammaY1Z2  # gamma2: responsibilities for polarity site 1
gamma_pol2_img[y_idx, x_idx] = gammaY1Z3  # gamma3: responsibilities for polarity site 2



plt.subplot(3,4,5)
plt.imshow(gamma_pol1_img, cmap='jet', origin='upper')
plt.title("GammaY1Z2 Polarity 1")
plt.colorbar(label="Probability")

plt.subplot(3,4,6)
plt.imshow(gamma_pol2_img, cmap='jet', origin='upper')
plt.title("GammaY1Z3 Polarity 2")
plt.colorbar(label="Probability")

# plt.tight_layout()
# plt.show()
#
# Create empty images for the polarity responsibilities.
gamma_cy_img = np.zeros_like(cropped_cell_mask, dtype=float)
gamma_nu_img = np.zeros_like(cropped_cell_mask, dtype=float)

# x_idx and y_idx were extracted earlier (for pixels inside the cell).
# They index the cropped mask.
gamma_cy_img[y_idx, x_idx] = gammaY1Z0  # gamma2: responsibilities for polarity site 1
gamma_nu_img[y_idx, x_idx] = gammaY1Z1  # gamma3: responsibilities for polarity site 2


plt.subplot(3,4,7)
plt.imshow(gamma_cy_img, cmap='jet', origin='upper')
plt.title("GammaY1Z0 Cytoplasm")
plt.colorbar(label="Probability")

plt.subplot(3,4,8)
plt.imshow(gamma_nu_img, cmap='jet', origin='upper')
plt.title("GammaY1Z1 Nucleus")
plt.colorbar(label="Probability")

plt.tight_layout()

# quantify prabability of polarity pitches
# Create empty images for the polarity responsibilities.
gamma_pol1_img = np.zeros_like(cropped_cell_mask, dtype=float)
gamma_pol2_img = np.zeros_like(cropped_cell_mask, dtype=float)

# x_idx and y_idx were extracted earlier (for pixels inside the cell).
# They index the cropped mask.
gamma_pol1_img[y_idx, x_idx] = gammaY0Z2  # gamma2: responsibilities for polarity site 1
gamma_pol2_img[y_idx, x_idx] = gammaY0Z3  # gamma3: responsibilities for polarity site 2



plt.subplot(3,4,9)
plt.imshow(gamma_pol1_img, cmap='jet', origin='upper')
plt.title("GammaY0Z2 Polarity 1")
plt.colorbar(label="Probability")

plt.subplot(3,4,10)
plt.imshow(gamma_pol2_img, cmap='jet', origin='upper')
plt.title("GammaY0Z3 Polarity 2")
plt.colorbar(label="Probability")

# plt.tight_layout()
# plt.show()
#
# Create empty images for the polarity responsibilities.
gamma_cy_img = np.zeros_like(cropped_cell_mask, dtype=float)
gamma_nu_img = np.zeros_like(cropped_cell_mask, dtype=float)

# x_idx and y_idx were extracted earlier (for pixels inside the cell).
# They index the cropped mask.
gamma_cy_img[y_idx, x_idx] = gammaY0Z0  # gamma2: responsibilities for polarity site 1
gamma_nu_img[y_idx, x_idx] = gammaY0Z1  # gamma3: responsibilities for polarity site 2


plt.subplot(3,4,11)
plt.imshow(gamma_cy_img, cmap='jet', origin='upper')
plt.title("GammaY0Z0 Cytoplasm")
plt.colorbar(label="Probability")

plt.subplot(3,4,12)
plt.imshow(gamma_nu_img, cmap='jet', origin='upper')
plt.title("GammaY0Z1 Nucleus")
plt.colorbar(label="Probability")

plt.tight_layout()
plt.show()
# 
#%%
import os
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread, imsave
from skimage.measure import find_contours, regionprops, label

# Helper function to compute the average value in a square window (e.g., 20x20) centered at (y, x)
# Only pixels where the mask is True are included.
def average_value_at_point(data, mask, y, x, window_size=10):
    half = window_size // 2
    y_min = max(y - half, 0)
    y_max = min(y + half + 1, data.shape[0])
    x_min = max(x - half, 0)
    x_max = min(x + half + 1, data.shape[1])
    region = data[y_min:y_max, x_min:x_max]
    region_mask = mask[y_min:y_max, x_min:x_max]
    if np.any(region_mask):
        return np.mean(region[region_mask])
    else:
        return np.nan

def enforce_ccw(contour):
    """
    Ensure the contour is in counter-clockwise (CCW) order.
    """
    x = contour[:, 1]
    y = contour[:, 0]
    area = 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    if area > 0:
        contour = contour[::-1]
    return contour

# ----------------- Data Preparation -----------------
# File paths
fluorescent_image_file = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Frames/fluorescent_C1_000.tif"
segmentation_mask_file = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/Masks/segmented_timepoint_000.tif"
output_extracted_folder = "/Users/user/Documents/FungalProject/TimeLapse/Tilescan2/ExtractedCells"
os.makedirs(output_extracted_folder, exist_ok=True)

# Load images
fluorescent_img = imread(fluorescent_image_file)
segmentation_mask = imread(segmentation_mask_file)

# Identify cell labels and select one cell (here label 85 is chosen as an example)
cell_labels = np.unique(segmentation_mask)
cell_labels = cell_labels[cell_labels != 0]
selected_label = cell_labels[cell_ID]
print(f"Selected cell label: {selected_label}")

# Create binary mask for the selected cell.
cell_mask = (segmentation_mask == selected_label)

# Extract the ordered inner boundary using find_contours.
contours = find_contours(cell_mask.astype(float), 0.5)
if len(contours) == 0:
    print("No contour found for the selected cell.")
else:
    contour = max(contours, key=len)
    contour = enforce_ccw(contour)
    
    # Round coordinates for indexing.
    contour_int = np.rint(contour).astype(int)
    contour_int[:, 0] = np.clip(contour_int[:, 0], 0, fluorescent_img.shape[0]-1)
    contour_int[:, 1] = np.clip(contour_int[:, 1], 0, fluorescent_img.shape[1]-1)
    
    # (Optional) Plot the boundary.
    plt.figure(figsize=(8,8))
    plt.imshow(fluorescent_img, cmap='gray')
    plt.plot(contour_int[:,1], contour_int[:,0], 'r-', linewidth=5, label='Boundary')
    plt.title(f"Cell {selected_label} Boundary")
    plt.axis("off")
    plt.legend()
    plt.show()
    
    # Extract average intensity profile along the boundary (from a 20x20 window)
    intensities = []
    for point in contour_int:
        y, x = point
        avg_intensity = average_value_at_point(fluorescent_img, cell_mask, y, x, window_size=20)
        intensities.append(avg_intensity)
    intensities = np.array(intensities)
    
    plt.figure(figsize=(10,4))
    plt.plot(intensities, marker='o', linestyle='-')
    plt.xlabel("Boundary Position (ordered)")
    plt.ylabel("Average Fluorescent Intensity")
    plt.title(f"Fluorescent Intensity Profile Along Cell {selected_label} Boundary")
    plt.tight_layout()
    plt.show()
    
    # Determine the bounding box of the cell using regionprops.
    labeled_cell = label(cell_mask)
    props = regionprops(labeled_cell)
    if props:
        region_prop = props[0]
        min_row, min_col, max_row, max_col = region_prop.bbox
    else:
        min_row = np.min(contour_int[:, 0])
        min_col = np.min(contour_int[:, 1])
        max_row = np.max(contour_int[:, 0]) + 1
        max_col = np.max(contour_int[:, 1]) + 1
    
    # Crop the fluorescent image and cell mask.
    cropped_img = fluorescent_img[min_row:max_row, min_col:max_col]
    cropped_cell_mask = cell_mask[min_row:max_row, min_col:max_col]
    
    # Adjust contour coordinates to the cropped image.
    contour_cropped = contour_int.copy()
    contour_cropped[:, 0] -= min_row
    contour_cropped[:, 1] -= min_col
    
    # Plot the cropped image with contour.
    plt.figure(figsize=(8,8))
    plt.imshow(cropped_img, cmap='gray')
    plt.plot(contour_cropped[:,1], contour_cropped[:,0], 'r-', linewidth=5, label='Boundary')
    plt.axis("off")
    plt.title(f"Cropped Cell {selected_label} with Boundary")
    plt.legend()
    plt.show()

# ----------------- Assume that the EM algorithm has been run and produced gamma values for polarity sites -----------------
# Here, we assume that the arrays gamma2 and gamma3 (each of length equal to the number of cell pixels in X_data)
# represent the posterior probabilities (responsibilities) for polarity site 1 and 2, respectively.
# Also, x_idx and y_idx were extracted from the cropped_cell_mask (above).
# Build gamma images for polarity sites:
gamma_pol1_img = np.zeros_like(cropped_cell_mask, dtype=float)
gamma_pol2_img = np.zeros_like(cropped_cell_mask, dtype=float)
gamma_cy_img = np.zeros_like(cropped_cell_mask, dtype=float)
#gamma_pol1_img[y_idx, x_idx] = gammaY2Z2  # from EM algorithm
#gamma_pol2_img[y_idx, x_idx] = gammaY2Z3  # from EM algorithm
gamma_pol1_img[y_idx, x_idx] = gammaY2Z2 * PZ1_Y1X * (cropped_img[y_idx, x_idx] - mu_bg) / sigma_bg #+ gammaY1Z2 * (mu_I - cropped_img[y_idx, x_idx]) / sigma_I)/2# * (cropped_img[y_idx, x_idx] - mu_I) / sigma_I_pol1 # / (gammaY2Z0)  # from EM algorithm
gamma_pol2_img[y_idx, x_idx] = gammaY2Z3 * PZ1_Y1X * (cropped_img[y_idx, x_idx] - mu_bg) / sigma_bg #+ gammaY1Z3 * (mu_I - cropped_img[y_idx, x_idx]) / sigma_I)/2# * (cropped_img[y_idx, x_idx] - mu_I) / sigma_I_pol2# / (gammaY2Z0) # from EM algorithm
gamma_cy_img[y_idx, x_idx] = gammaY0Z0 * PZ0_Y1X * (cropped_img[y_idx, x_idx] - mu_bg) / sigma_bg 


# ----------------- Sliding Box Quantification of Polarity Site Gamma -----------------
# We use the ordered boundary (contour_cropped) to slide a window along the cell boundary.
window_size = 20  # adjust window size as needed
gamma_pol1_profile = []
gamma_pol2_profile = []
gamma_cy_profile = []

for point in contour_cropped:
    y, x = point
    avg_gamma1 = average_value_at_point(gamma_pol1_img, cropped_cell_mask, y, x, window_size)
    avg_gamma2 = average_value_at_point(gamma_pol2_img, cropped_cell_mask, y, x, window_size)
    avg_gamma0 = average_value_at_point(gamma_cy_img, cropped_cell_mask, y, x, window_size)
    gamma_pol1_profile.append(avg_gamma1)
    gamma_pol2_profile.append(avg_gamma2)
    gamma_cy_profile.append(avg_gamma0)

gamma_pol1_profile = np.array(gamma_pol1_profile)
gamma_pol2_profile = np.array(gamma_pol2_profile)
gamma_cy_profile = np.array(gamma_cy_profile)

# Plot both gamma profiles along the ordered boundary on the same plot.
plt.figure(figsize=(10,6))

plt.plot(gamma_cy_profile, 'g-o', label='Cytoplasm')
plt.plot(gamma_pol1_profile, 'r-o', label='Polarity Site 1')
plt.plot(gamma_pol2_profile, 'b-o', label='Polarity Site 2')
plt.xlabel("Boundary Position (ordered)")
plt.ylabel("Average Gamma (Responsibility)")
plt.title("Polarity Site Gamma Profiles Along the Cell Boundary")
plt.legend()
plt.tight_layout()
plt.show()
