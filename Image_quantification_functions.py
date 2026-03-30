#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 12:02:36 2025

@author: user
"""
import numpy as np
from tifffile import imread
import matplotlib.pyplot as plt
import os
import numpy as np
from skimage.measure import regionprops, label
from skimage.measure import find_contours
from skimage.segmentation import find_boundaries
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree






# use slide box to quantify membrane intensity
# Helper function to compute average intensity in a square window (5x5) centered at (y, x)
# Only pixels inside the segmentation mask (seg_mask==True) are included in the average.



def plot_cell_and_gamma_overlay(plot_data, 
                                plot_filename = None):
    
    # cropped_img=cropped_img,
    # contour_cropped=boundary,
    # gammas=gammas_unlinked,
    # cropped_cell_mask=cropped_cell_mask,
    # y_idx=y_idx,
    # x_idx=x_idx,
    # mu=Mu,
    # start_point=boundary[0],
    # ep1=(endpoint1[1], endpoint1[0]),
    # ep2=(endpoint2[1], endpoint2[0]),
    # mp1=(midpoint1[1], midpoint1[0]),
    # mp2=(midpoint2[1], midpoint2[0]),
    # selected_label=selected_label
    cropped_img=plot_data[0]
    contour_cropped=plot_data[1]
    gammas=plot_data[2]
    cropped_cell_mask=plot_data[3]
    y_idx=plot_data[4]
    x_idx=plot_data[5]
    mu=plot_data[6]
    start_point=plot_data[7]
    ep1=plot_data[8]
    ep2=plot_data[9]
    mp1=plot_data[10]
    mp2=plot_data[11]
    selected_label=plot_data[12]
    alpha_scale=1.0

    """
    Plots: [1] fluorescent cell image with landmarks, and [2] gamma overlay with transparency.
    Layout depends on aspect ratio (side-by-side if tall, top-bottom if wide).
    """

    H, W = cropped_img.shape
    if H > W:
        nrows, ncols = 1, 2
        figsize = (12, 6)
    else:
        nrows, ncols = 2, 1
        figsize = (6, 12)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    # -------------------- Plot 1: Cell with Landmarks --------------------
    ax0 = axes[0]
    ax0.imshow(cropped_img, cmap='gray')

    # Cell contour
    ax0.plot(contour_cropped[:, 1], contour_cropped[:, 0], 'r-', linewidth=5, label='Cell Contour')

    # Start point
    if start_point is not None:
        sy, sx = start_point
        ax0.plot(sx, sy, 'go', markersize=8)
        ax0.text(sx, sy, f"Nu/Cy={mu[1]:.1f}/{mu[0]:.1f}", color='green', fontsize=10, fontweight='bold')

    # Endpoints
    if ep1 is not None:
        ax0.plot(ep1[0], ep1[1], 'co', markersize=10)
        ax0.text(ep1[0], ep1[1], f"({mu[2]:.1f})", color='cyan', fontsize=10, fontweight='bold')
    if ep2 is not None:
        ax0.plot(ep2[0], ep2[1], 'yo', markersize=10)
        ax0.text(ep2[0], ep2[1], f"({mu[3]:.1f})", color='yellow', fontsize=10, fontweight='bold')

    # Midpoints
    if mp1 is not None:
        ax0.plot(mp1[0], mp1[1], 'ms', markersize=10)
        ax0.text(mp1[0], mp1[1], f"({mu[4]:.1f})", color='magenta', fontsize=10, fontweight='bold')
    if mp2 is not None:
        ax0.plot(mp2[0], mp2[1], 'ms', markersize=10)
        ax0.text(mp2[0], mp2[1], f"({mu[5]:.1f})", color='magenta', fontsize=10, fontweight='bold')

    ax0.set_title(f"Cell {selected_label}" if selected_label is not None else "Cell with Landmarks", fontsize=10,color='white')
    ax0.axis("off")
    ax0.set_facecolor('black')  # for the cell/landmark subplot

    # -------------------- Plot 2: Gamma Overlay --------------------
    # Create RGBA overlay image
    overlay = np.zeros((H, W, 4), dtype=float)

    # Unique RGB colors per gamma key
    color_map = {
        'Y2Z0': (1.0, 0.0, 0.0),  # red
        'Y2Z1': (0.0, 1.0, 0.0),  # green
        #'Y2Z2': (0.0, 0.0, 1.0),  # blue
        'Y2Z2': (0.0, 1.0, 1.0),  # cyan
        'Y2Z3': (1.0, 1.0, 0.0),  # yellow
        'Y2Z4': (1.0, 0.0, 1.0),  # magenta
        #'Y2Z5': (0.0, 1.0, 1.0),  # cyan
        'Y2Z5': (1.0, 0.0, 1.0),  # magenta
    }

    for key, color in color_map.items():
        if key not in gammas:
            continue
        gamma = np.zeros_like(cropped_cell_mask, dtype=float)
        gamma[y_idx, x_idx] = gammas[key]
        alpha = np.clip(gamma * alpha_scale, 0, 1)

        for c in range(3):
            overlay[:, :, c] += alpha * color[c] * (1 - overlay[:, :, 3])
        overlay[:, :, 3] += alpha * (1 - overlay[:, :, 3])

    overlay = np.clip(overlay, 0, 1)

    ax1 = axes[1]
    ax1.imshow(overlay)
    ax1.set_title("Gamma Overlay (Color by Component, Alpha by Value)", fontsize=10)
    ax1.axis("off")
    
    ax1.set_facecolor('black')  # for the gamma overlay subplot
    fig.patch.set_facecolor('black')  # for the entire figure background

    plt.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename, dpi=150)
    #plt.show()
    plt.close()



    
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



def plot_mask(gammas, cropped_cell_mask, y_idx, x_idx, mu):
    fs = 6     # title font size
    fs2 = 4    # tick label and colorbar font size

    # Extract gamma arrays
    gamma_keys = ['Y2Z0', 'Y2Z1','Y2Z2', 'Y2Z3',  'Y2Z4', 'Y2Z5']
    titles = [f"Gamma{k}, mu={mu[i]:.2f}" for i, k in enumerate(gamma_keys)]
    gamma_arrays = [gammas[k] for k in gamma_keys]

    # Fill gamma images
    gamma_imgs = []
    for gamma in gamma_arrays:
        img = np.zeros_like(cropped_cell_mask, dtype=float)
        img[y_idx, x_idx] = gamma
        gamma_imgs.append(img)

    # Decide layout based on aspect ratio
    H, W = cropped_cell_mask.shape
    if H > W:
        nrows, ncols = 1, 6  # wider → plot left to right
        figsize = (12, 6)
    else:
        nrows, ncols = 6, 1  # taller → plot top to bottom
        figsize = (6, 12)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, ax in enumerate(axes):
        im = ax.imshow(gamma_imgs[i], cmap='jet', origin='upper', vmin=0, vmax=1)
        ax.set_title(titles[i], fontsize=fs)
        ax.tick_params(labelsize=fs2)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Probability", fontsize=fs2)
        cbar.ax.tick_params(labelsize=fs2)

    plt.tight_layout()
    plt.show()


    
def transform_to_uv_space(boundary,endpoint1,endpoint2, midpoint1, midpoint2,cropped_cell_mask):
    
   
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
    dist3, idx3 = tree.query(midpoint1)
    u_midpoint1 = u_boundary[idx3]
    dist4, idx4 = tree.query(midpoint2)
    u_midpoint2 = u_boundary[idx4]
    
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
    return u_map, v_map, u_endpoint1, u_endpoint2, u_midpoint1, u_midpoint2, L



def transform_to_mn_space(midpoint1, midpoint2, cropped_cell_mask, reflect=True):
    """
    Transform coordinates from image space to (m, n) space for nucleus modeling.
    
    Parameters:
    - midpoint1, midpoint2: midpoints in (row, col) format (y, x)
    - cropped_cell_mask: 2D binary mask of the cropped cell
    - reflect: if True, reflects all points to one side of the m-axis (absolute n)
    
    Returns:
    - m_map: 2D array of m-coordinates (same shape as input mask, 0 outside cell)
    - n_map: 2D array of n-coordinates (same shape as input mask, 0 outside cell)
    """
    # Convert midpoints from (row, col) to (x, y)
    p1 = np.array([midpoint1[1], midpoint1[0]])  # (x1, y1)
    p2 = np.array([midpoint2[1], midpoint2[0]])  # (x2, y2)

    # Define m-axis direction vector
    d = p2 - p1
    d = d / np.linalg.norm(d)

    # Perpendicular direction (n-axis)
    n = np.array([-d[1], d[0]])

    # Get coordinates of all pixels inside the mask
    rows_in, cols_in = np.nonzero(cropped_cell_mask)
    coords = np.column_stack((cols_in, rows_in))  # (x, y)

    # Translate all coordinates relative to p1
    rel_coords = coords - p1

    # Project onto m and n axes
    m_vals = np.dot(rel_coords, d)
    n_vals = np.dot(rel_coords, n)
    if reflect:
        n_vals = np.abs(n_vals)

    # Build m_map and n_map
    m_map = np.zeros_like(cropped_cell_mask, dtype=float)
    n_map = np.zeros_like(cropped_cell_mask, dtype=float)
    m_map[rows_in, cols_in] = m_vals
    n_map[rows_in, cols_in] = n_vals

    return m_map, n_map


def check_convergence(params, params_new, tol):
    """
    Check convergence between two dictionaries of parameters.

    Parameters:
        params (dict): Dictionary containing the original parameters.
        params_new (dict): Dictionary containing the updated parameters.
        tol (float): Tolerance threshold for convergence.

    Returns:
        bool: True if all parameters have changed less than tol, False otherwise.
    """
    for key in params:
        if key not in params_new:
            raise KeyError(f"Key '{key}' not found in updated parameters.")

        old_val = params[key]
        new_val = params_new[key]

        # Use norm if the parameter is an array; otherwise, use absolute difference.
        if isinstance(old_val, np.ndarray):
            diff = np.linalg.norm(new_val - old_val)
        else:
            diff = abs(new_val - old_val)
        
        if diff >= tol:
            return False
    return True

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

def pdf_circular_nucleus2(X, mu_xy_nuc, sigma_xy, mu_bg, sigma_bg, eps):
    X_xy = X[:, :2]
    diff = X_xy - mu_xy_nuc
    f_spatial = 1.0/(2*np.pi*(sigma_xy**2)) * np.exp(-0.5*np.sum(diff**2,axis=1)/(sigma_xy**2))
    I = X[:, 2]
    I_min, I_max = mu_bg - 1*sigma_bg, np.max(I)
    interval_length = I_max - I_min
    if interval_length == 0:
        return np.zeros_like(I)
    in_region = (I >= I_min) & (I <= I_max)
    f_int = np.zeros(len(I))
    f_int[in_region] = 1.0 / interval_length + eps
    
    return f_spatial * f_int




# Define the Polarity PDF in (u,v) Space with Intensity Constrained to the Nucleus

def circular_diff(u, u0, L):
    """Compute the circular difference between u and u0 given total length L."""
    diff = np.abs(u - u0)
    return np.minimum(diff, L - diff)

def pdf_polarity_uv(X, u_map, v_map, u_endpoint, sigma_u, sigma_v, mu_I, sigma_I, L):
   
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
def pdf_polarity_uv2(X, u_map, v_map, u_endpoint, sigma_u, sigma_v, mu_bg, sigma_bg, L, eps):
   
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
    
    I = X[:, 2]
    I_min, I_max = mu_bg - 1*sigma_bg, np.max(I)
    interval_length = I_max - I_min
    if interval_length == 0:
        return np.zeros_like(I)
    in_region = (I >= I_min) & (I <= I_max)
    f_int = np.zeros(len(I))
    f_int[in_region] = 1.0 / interval_length + eps
    
    return spatial_pdf * f_int
def pdf_movie_background_uv(X, v_map, sigma_v, mu_bg, sigma_bg, eps):
    # Extract pixel coordinates.
    rows = X[:, 1].astype(int)
    cols = X[:, 0].astype(int)
    
    v_vals = v_map[rows, cols]
    
    # Spatial PDF:
    
    f_v = (np.sqrt(2) / (sigma_v * np.sqrt(np.pi))) * np.exp(- (v_vals**2) / (2 * sigma_v**2))

    I = X[:, 2]
    I_max = mu_bg - sigma_bg
    I_min = np.min(I)
    interval_length = max(I_max - I_min, eps)
    f_int = np.zeros(len(I))
    in_range = (I >= I_min) & (I <= I_max)
    f_int[in_range] = 1.0 / interval_length + eps
    return f_int * f_v

def mu_xy_to_mu_mn(midpoint1,midpoint2,mu_xy, reflect=True):
    p1 = np.array([midpoint1[1], midpoint1[0]])  # (x1, y1)
    p2 = np.array([midpoint2[1], midpoint2[0]])  # (x2, y2)
    d = p2 - p1
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])

    mu_xy = np.array([mu_xy[0], mu_xy[1]])  # (col, row)
    rel_mu = mu_xy - p1
    mu_m = np.dot(rel_mu, d)
    mu_n = np.dot(rel_mu, n)
    if reflect:
        mu_n = abs(mu_n)
    mu_mn = [mu_m, mu_n]
    return mu_mn

def pdf_circular_nucleus_mn(X, m_map, n_map, mu_mn, sigma_mn, mu_bg, sigma_bg, eps):
    
    # Extract pixel coordinates
    rows = X[:, 1].astype(int)
    cols = X[:, 0].astype(int)
    m_vals = m_map[rows, cols]
    n_vals = n_map[rows, cols]

    
    # ---- Spatial PDF (circular Gaussian)
    r2 = (m_vals - mu_mn[0]) ** 2 + (n_vals - mu_mn[1]) ** 2
    sigma2 = np.maximum(np.asarray(sigma_mn, float)**2, float(eps)**2)
    f_spatial = (1.0 / (2.0 * np.pi * sigma2)) * np.exp(-0.5 * r2 / sigma2)
    #f_spatial = (1.0 / (2 * np.pi * sigma_mn**2)) * np.exp(-0.5 * r2 / (sigma_mn**2))

    # ---- Intensity PDF (uniform + eps over truncated background)
    I = X[:, 2]
    I_min = mu_bg - sigma_bg
    I_max = np.max(I)
    interval_length = max(I_max - I_min, eps)
    f_int = np.zeros(len(I))
    in_range = (I >= I_min) & (I <= I_max)
    f_int[in_range] = 1.0 / interval_length + eps

    return f_spatial * f_int
def pdf_movie_background(X, mu_bg, sigma_bg, eps):
    I = X[:, 2]
    I_max = mu_bg - sigma_bg
    I_min = np.min(I)
    interval_length = max(I_max - I_min, eps)
    f_int = np.zeros(len(I))
    in_range = (I >= I_min) & (I <= I_max)
    f_int[in_range] = 1.0 / interval_length + eps
    return f_int

def _safe(x, eps=1e-6):
    return max(float(x), eps)

def _kl_shrink_mu(mu_ml, N_eff, var_ml, mu_prev, var_prev, lam):
    """
    KL-consistent-ish shrinkage for Gaussian mean:
      mu = argmin [ (N/2var_ml)*(mu-mu_ml)^2 + (lam/2var_prev)*(mu-mu_prev)^2 ]
    => weighted average.
    """
    if mu_prev is None or var_prev is None:
        return mu_ml
    var_prev = _safe(var_prev)
    var_ml = _safe(var_ml)
    w_ml = N_eff / var_ml
    w_prev = lam / var_prev
    return (w_ml * mu_ml + w_prev * mu_prev) / (w_ml + w_prev + 1e-12)

def _log_smooth_sigma(sigma_ml, sigma_prev, alpha, eps=1e-6):
    if sigma_prev is None:
        return sigma_ml
    a = _safe(sigma_ml, eps)
    b = _safe(sigma_prev, eps)
    return float(np.exp((1 - alpha) * np.log(a) + alpha * np.log(b)))

def state_model(X_data, A, s_par, pi, eps):
    fZ0 = pdf_cytoplasm(X_data, A, s_par['mu_bg_Z0'], s_par['sigma_bg_Z0'])
    #fZ1 = pdf_circular_nucleus(X_data, s_par['mu_xy_Z1'], s_par['sigma_xy_Z1'], s_par['mu_I_Z1'], s_par['sigma_I_Z1'])
    #fZ1 = pdf_circular_nucleus2(X_data, s_par['mu_xy_Z1'], s_par['sigma_xy_Z1'], s_par['mu_bg_Z0'], s_par['sigma_bg_Z0'], eps)
    fZ1 = pdf_circular_nucleus_mn(X_data, s_par['m_map'], s_par['n_map'],  s_par['mu_mn_Z1'], s_par['sigma_mn_Z1'], s_par['mu_bg_Z0'], s_par['sigma_bg_Z0'],eps)
    #fZ2 = pdf_polarity_uv(X_data, s_par['u_map'], s_par['v_map'], s_par['u_endpoint1'],s_par['sigma_u1_Z2'], s_par['sigma_v1_Z2'], s_par['mu_P_Z2_Z3'], s_par['sigma_P_Z2_Z3'], s_par['L'])
    #fZ3 = pdf_polarity_uv(X_data, s_par['u_map'], s_par['v_map'], s_par['u_endpoint2'],s_par['sigma_u2_Z3'], s_par['sigma_v2_Z3'], s_par['mu_P_Z2_Z3'], s_par['sigma_P_Z2_Z3'], s_par['L'])
    fZ2 = pdf_polarity_uv2(X_data, s_par['u_map'], s_par['v_map'], s_par['u_endpoint1'],s_par['sigma_u1'], s_par['sigma_v1'], s_par['mu_bg_Z0'], s_par['sigma_bg_Z0'], s_par['L'], eps)
    fZ3 = pdf_polarity_uv2(X_data, s_par['u_map'], s_par['v_map'], s_par['u_endpoint2'],s_par['sigma_u1'], s_par['sigma_v1'], s_par['mu_bg_Z0'], s_par['sigma_bg_Z0'], s_par['L'], eps)
    fZ4 = pdf_polarity_uv2(X_data, s_par['u_map'], s_par['v_map'], s_par['u_midpoint1'],s_par['sigma_u2'], s_par['sigma_v2'], s_par['mu_bg_Z0'], s_par['sigma_bg_Z0'], s_par['L'], eps)
    fZ5 = pdf_polarity_uv2(X_data, s_par['u_map'], s_par['v_map'], s_par['u_midpoint2'],s_par['sigma_u2'], s_par['sigma_v2'], s_par['mu_bg_Z0'], s_par['sigma_bg_Z0'], s_par['L'], eps)
    #fZ6 = pdf_movie_background(X_data, s_par['mu_bg_Z0'], s_par['sigma_bg_Z0'], eps)
    fZ6 = pdf_movie_background_uv(X_data, s_par['v_map'], s_par['sigma_v1'], s_par['mu_bg_Z0'], s_par['sigma_bg_Z0'], eps)
    
    f_total = pi[0]*fZ0 + pi[1]*fZ1 + pi[2]*fZ2 + pi[3]*fZ3 + pi[4]*fZ4 + pi[5]*fZ5 + pi[6]*fZ6
    gammaZ0 = pi[0]*fZ0 / f_total
    gammaZ1 = pi[1]*fZ1 / f_total
    gammaZ2 = pi[2]*fZ2 / f_total
    gammaZ3 = pi[3]*fZ3 / f_total
    gammaZ4 = pi[4]*fZ4 / f_total
    gammaZ5 = pi[5]*fZ5 / f_total
    gammaZ6 = pi[6]*fZ6 / f_total
      
    
    return gammaZ0, gammaZ1, gammaZ2, gammaZ3, gammaZ4, gammaZ5, gammaZ6

def E_step(X_data, A, params, Yi, pi, eps, params_fixed):
     
    
    s_par_Y2 = {
        'mu_bg_Z0': params['mu_bg_Y2'],
        'sigma_bg_Z0': params['sigma_bg_Y2'],
        #'mu_xy_Z1': params['mu_xy_Y2'],
        #'sigma_xy_Z1': params['sigma_xy_Y2'],
        'mu_mn_Z1': params['mu_mn_Y2'],
        'sigma_mn_Z1': params['sigma_mn_Y2'],
        'mu_I_Z1': params['mu_I_Y2'],
        'sigma_I_Z1': params['sigma_I_Y2'],
        'mu_P1_Z2': params['mu_P1_Y2'],
        'sigma_P1_Z2': params['sigma_P1_Y2'],
        'mu_P2_Z3': params['mu_P2_Y2'],
        'sigma_P2_Z3': params['sigma_P2_Y2'],
        'sigma_u1': params_fixed['sigma_u1_Y2'],
        'sigma_v1': params['sigma_v1_Y2'],
        'sigma_u2': params_fixed['sigma_u2_Y2'],
        'sigma_v2': params['sigma_v2_Y2'],
        'mu_mv_Z6':params['mu_mv_Y2' ],
        'sigma_mv_Z6':params['sigma_mv_Y2'],
        
        'u_map': params_fixed['u_map'],
        'v_map': params_fixed['v_map'],
        'u_endpoint1': params_fixed['u_endpoint1'],
        'u_endpoint2': params_fixed['u_endpoint2'],
        'u_midpoint1': params_fixed['u_midpoint1'],
        'u_midpoint2': params_fixed['u_midpoint2'],
        
        'm_map': params_fixed['m_map'],
        'n_map': params_fixed['n_map'],
        
        'L': params_fixed['L']
        }
    gammaY2Z0, gammaY2Z1, gammaY2Z2, gammaY2Z3, gammaY2Z4, gammaY2Z5, gammaY2Z6= state_model(X_data, A, s_par_Y2, pi, eps)     
            
   # Build a dictionary for all gamma
    gammas = {
        'Y2Z0': gammaY2Z0,
        'Y2Z1': gammaY2Z1,
        'Y2Z2': gammaY2Z2,
        'Y2Z3': gammaY2Z3,
        'Y2Z4': gammaY2Z4,
        'Y2Z5': gammaY2Z5,
        'Y2Z6': gammaY2Z6
        
        }
    return gammas


def M_step_unlinked(X_data, gammas, Yi, pi, eps, params_fixed,
                    prev_params=None,
                    lam_mn=0.2, lam_I=0.2,
                    alpha_sigma=0.2):

    
    gammaY2Z0 = gammas['Y2Z0']
    gammaY2Z1 = gammas['Y2Z1']
    gammaY2Z2 = gammas['Y2Z2']
    gammaY2Z3 = gammas['Y2Z3']
    gammaY2Z4 = gammas['Y2Z4']
    gammaY2Z5 = gammas['Y2Z5']
    gammaY2Z6 = gammas['Y2Z6']
    m_map = params_fixed['m_map']
    n_map = params_fixed['n_map']
    major_axis_length = params_fixed['major_axis_length']
    sigma_v1 = 3
    
   
    
    effective_gamma_total = gammaY2Z0    
    sum_gamma = np.sum(effective_gamma_total)
    mu_bg_Y2_new = np.sum(effective_gamma_total * X_data[:,2]) / sum_gamma
    sigma_bg_Y2_new = np.sqrt(np.sum(effective_gamma_total * (X_data[:,2]-mu_bg_Y2_new)**2) / sum_gamma)
    sigma_bg_Y2_new = max(sigma_bg_Y2_new, eps)
    
    effective_gamma_total = gammaY2Z6    
    sum_gamma = np.sum(effective_gamma_total)
    mu_mv_Y2_new = np.sum(effective_gamma_total * X_data[:,2]) / sum_gamma
    sigma_mv_Y2_new = np.sqrt(np.sum(effective_gamma_total * (X_data[:,2]-mu_mv_Y2_new)**2) / sum_gamma)
    sigma_mv_Y2_new = max(sigma_mv_Y2_new, eps)
    
    
    # mu_xy_Y2_new = np.sum(gammaY2Z1[:, np.newaxis] * X_data[:, :2], axis=0) / np.sum(gammaY2Z1)
    # var_x = np.sum(gammaY2Z1 * (X_data[:,0]-mu_xy_Y2_new[0])**2) / np.sum(gammaY2Z1)
    # var_y = np.sum(gammaY2Z1 * (X_data[:,1]-mu_xy_Y2_new[1])**2) / np.sum(gammaY2Z1)
    # sigma_xy_Y2_new = np.sqrt((var_x+var_y)/2)
    # sigma_xy_Y2_new = max(sigma_xy_Y2_new, eps)
    
    #Update mu_mn in mn space
    # Extract pixel coordinates
    rows = X_data[:, 1].astype(int)
    cols = X_data[:, 0].astype(int)
    m_vals = m_map[rows, cols]
    n_vals = n_map[rows, cols]
    # Stack into (N, 2) array
    X_mn = np.column_stack((m_vals, n_vals))
    # Compute new mu_mn using responsibilities as weights
    mu_mn_Y2_new = np.sum(gammaY2Z1[:, np.newaxis] * X_mn, axis=0) / np.sum(gammaY2Z1)
    #mu_mn_Y2_new[0]=np.mean(X_mn[0,:])
    # Responsibility-weighted variance in m and n
    var_m = np.sum(gammaY2Z1 * (m_vals - mu_mn_Y2_new[0])**2) / np.sum(gammaY2Z1)
    var_n = np.sum(gammaY2Z1 * (n_vals - mu_mn_Y2_new[1])**2) / np.sum(gammaY2Z1)
    # Isotropic standard deviation (average of variances)
    sigma_mn_Y2_new = np.sqrt((var_m + var_n) / 2)
    # Ensure minimum variance to avoid collapse
    sigma_mn_Y2_new = max(sigma_mn_Y2_new,3)

    
    
    
    
        
    gamma_I =  gammaY2Z1 + gammaY2Z2 + gammaY2Z3
    sum_gamma_I = np.sum(gamma_I)
    mu_I_Y2_new = np.sum(gamma_I * X_data[:,2]) / sum_gamma_I #+ 1 * sigma_bg_new
    #mu_I_new = max(mu_I_new,(mu_bg_new + 1 * sigma_bg_new))
    sigma_I_Y2_new = np.sqrt(np.sum(gamma_I * (X_data[:,2]-mu_I_Y2_new)**2) / sum_gamma_I)
    sigma_I_Y2_new = max(sigma_I_Y2_new, eps)
    
    gamma_I = gammaY2Z1 
    
    sum_gamma_I = np.sum(gamma_I)
    mu_nu_Y2_new = np.sum(gamma_I * X_data[:,2]) / sum_gamma_I 
    sigma_nu_Y2_new = np.sqrt(np.sum(gamma_I * (X_data[:,2]-mu_nu_Y2_new)**2) / sum_gamma_I)
    #sigma_nu_Y2_new = max(sigma_nu_Y2_new, eps)
    
    gamma_I = gammaY2Z2 
    sum_gamma_I = np.sum(gamma_I)
    mu_P1_Y2_new = np.sum(gamma_I * X_data[:,2]) / sum_gamma_I 
    sigma_P1_Y2_new = np.sqrt(np.sum(gamma_I * (X_data[:,2]-mu_P1_Y2_new)**2) / sum_gamma_I)
    #sigma_P1_Y2_new = max(sigma_P1_Y2_new, eps)
    
    gamma_I = gammaY2Z3 
    sum_gamma_I = np.sum(gamma_I)
    mu_P2_Y2_new = np.sum(gamma_I * X_data[:,2]) / sum_gamma_I 
    sigma_P2_Y2_new = np.sqrt(np.sum(gamma_I * (X_data[:,2]-mu_P2_Y2_new)**2) / sum_gamma_I)
    
    gamma_I = gammaY2Z4 
    sum_gamma_I = np.sum(gamma_I)
    mu_S1_Y2_new = np.sum(gamma_I * X_data[:,2]) / sum_gamma_I 
    sigma_S1_Y2_new = np.sqrt(np.sum(gamma_I * (X_data[:,2]-mu_S1_Y2_new)**2) / sum_gamma_I)
    
    gamma_I = gammaY2Z5 
    sum_gamma_I = np.sum(gamma_I)
    mu_S2_Y2_new = np.sum(gamma_I * X_data[:,2]) / sum_gamma_I 
    sigma_S2_Y2_new = np.sqrt(np.sum(gamma_I * (X_data[:,2]-mu_S2_Y2_new)**2) / sum_gamma_I)
    
    
    #rows_data = X_data[:,1].astype(int)
    #cols_data = X_data[:,0].astype(int)
    # #u_vals = u_map[rows_data, cols_data]
    # v_vals = v_map[rows_data, cols_data]
    
    
    # #sigma_v1_Y2_new = np.sqrt(np.sum(gammaY2Z2 * (v_vals)**2) / np.sum(gammaY2Z2))
    # #sigma_v1_Y2_new = max(sigma_v1_Y2_new, eps)
    v1_lim = max((1-min(mu_mn_Y2_new[1],0.75*major_axis_length)/(0.75*major_axis_length)),eps)
    #print(v1_lim)
    sigma_v1_Y2_new = sigma_v1 * v1_lim #(1-mu_mn_Y2_new[1]/(0.5*major_axis_length))  # fixed
    # sigma_u1_Y2_new = sigma_u1  # fixed
    
    
    # #sigma_v2_Y2_new = np.sqrt(np.sum(gammaY2Z3 * (v_vals)**2) / np.sum(gammaY2Z3))
    # #sigma_v2_Y2_new = max(sigma_v2_Y2_new, eps)
    sigma_v2_Y2_new = mu_mn_Y2_new[1]/5  # fixed
    #sigma_u2_Y2_new = mu_mn_Y2_new  # fixed
    
    # Smooth
    # Effective counts (how many pixels support the component)
    N_nu = float(np.sum(gammaY2Z1))   # nucleus responsibilities count
    
    if prev_params is not None:
        # ---- smooth nucleus position (mn) ----
        prev_mu_mn = prev_params.get('mu_mn_Y2', None)
        prev_sigma_mn = prev_params.get('sigma_mn_Y2', None)
    
        if prev_mu_mn is not None and prev_sigma_mn is not None and N_nu > 1:
            prev_mu_mn = np.array(prev_mu_mn, dtype=float)
    
            # treat mn as isotropic with var = sigma^2 (your model)
            var_prev = float(_safe(prev_sigma_mn)**2)
            var_ml   = float(_safe(sigma_mn_Y2_new)**2)
    
            # shrink each dimension’s mean toward previous
            mu_mn_Y2_new = np.array([
                _kl_shrink_mu(mu_mn_Y2_new[0], N_nu, var_ml, prev_mu_mn[0], var_prev, lam_mn),
                _kl_shrink_mu(mu_mn_Y2_new[1], N_nu, var_ml, prev_mu_mn[1], var_prev, lam_mn),
            ], dtype=float)
    
            # smooth sigma (super robust)
            sigma_mn_Y2_new = _log_smooth_sigma(sigma_mn_Y2_new, prev_sigma_mn, alpha_sigma, eps=eps)
    
        # ---- smooth nucleus intensity ----
        prev_mu_I = prev_params.get('mu_I_Y2', None)
        prev_sigma_I = prev_params.get('sigma_I_Y2', None)
    
        if prev_mu_I is not None and prev_sigma_I is not None and N_nu > 1:
            var_prev_I = float(_safe(prev_sigma_I)**2)
            var_ml_I   = float(_safe(sigma_nu_Y2_new)**2)
    
            mu_nu_Y2_new = _kl_shrink_mu(mu_nu_Y2_new, N_nu, var_ml_I, float(prev_mu_I), var_prev_I, lam_I)
            sigma_nu_Y2_new = _log_smooth_sigma(sigma_nu_Y2_new, prev_sigma_I, alpha_sigma, eps=eps)

    
    params_new = {
       
        
        'mu_bg_Y2': mu_bg_Y2_new,
        'sigma_bg_Y2': sigma_bg_Y2_new,
        #'mu_xy_Y2': mu_xy_Y2_new,
        #'sigma_xy_Y2': sigma_xy_Y2_new,
        'mu_mn_Y2': mu_mn_Y2_new,
        'sigma_mn_Y2': sigma_mn_Y2_new,
        'mu_I_Y2': mu_nu_Y2_new,
        'sigma_I_Y2': sigma_nu_Y2_new,
        'mu_P1_Y2': mu_P1_Y2_new,
        'sigma_P1_Y2': sigma_P1_Y2_new,
        'mu_P2_Y2': mu_P2_Y2_new,
        'sigma_P2_Y2': sigma_P2_Y2_new,
        'mu_S1_Y2': mu_S1_Y2_new,
        'sigma_S1_Y2': sigma_S1_Y2_new,
        'mu_S2_Y2': mu_S2_Y2_new,
        'sigma_S2_Y2': sigma_S2_Y2_new,
        # 'sigma_u1_Y2': sigma_u1_Y2_new,
        'sigma_v1_Y2': sigma_v1_Y2_new,
        # 'sigma_u2_Y2': sigma_u2_Y2_new,
        'sigma_v2_Y2': sigma_v2_Y2_new,
        'mu_mv_Y2': mu_mv_Y2_new,
        'sigma_mv_Y2': sigma_mv_Y2_new
        
        # 'u_map': u_map,
        # 'v_map': v_map,
        # 'u_endpoint1': u_endpoint1,
        # 'u_endpoint2': u_endpoint2,
        # 'u_midpoint1': u_midpoint1,
        # 'u_midpoint2': u_midpoint2,
        
        # 'm_map': m_map,
        # 'n_map': n_map,

        # 'L': L
        
    }
    
    return params_new

def order_endpoints_by_proximity(ep1, ep2, ref_ep1, ref_ep2):
    dist1 = np.linalg.norm(np.array(ep1) - ref_ep1)
    dist2 = np.linalg.norm(np.array(ep2) - ref_ep1)
    if dist2 < dist1:
        # Swap to maintain consistency with reference
        ep1, ep2 = ep2, ep1
    return ep1, ep2



def axis_extrema_points(
    boundary_coords: np.ndarray,
    P0: np.ndarray,
    direction: np.ndarray,
    tol: float = 2.0,
):
    """
    Return the two boundary points with min/max projection along a given axis,
    restricted to boundary pixels lying within `tol` pixels of that axis.

    Parameters
    ----------
    boundary_coords : (M, 2) array
        Boundary pixel coordinates in (row, col).
    P0 : (2,) array
        Point on the axis (e.g., centroid) in (row, col).
    direction : (2,) array
        Axis direction vector in (row, col). Can be any nonzero vector;
        it is normalized internally.
    tol : float
        Distance tolerance (in pixels) to consider a boundary pixel “on” the axis.

    Returns
    -------
    p_min : (2,) array
        Boundary point with minimum projection along the axis.
    p_max : (2,) array
        Boundary point with maximum projection along the axis.
    t_min : float
        Minimum projection scalar.
    t_max : float
        Maximum projection scalar.

    Raises
    ------
    ValueError
        If no boundary points lie within `tol` of the axis.
    """
    direction = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("`direction` must be nonzero.")
    d = direction / norm  # ensure unit vector

    rel = boundary_coords - P0  # (M, 2)
    t = rel @ d                # (M,)
    q_line = P0 + np.outer(t, d)
    dist = np.linalg.norm(boundary_coords - q_line, axis=1)

    on_axis = dist < tol
    if not np.any(on_axis):
        raise ValueError("No boundary points found within tolerance of the axis.")

    t_sel = t[on_axis]
    pts_sel = boundary_coords[on_axis]

    i_min = np.argmin(t_sel)
    i_max = np.argmax(t_sel)
    return pts_sel[i_min], pts_sel[i_max], float(t_sel[i_min]), float(t_sel[i_max])


def find_midpoints_on_minor_axis(
    boundary_coords: np.ndarray,
    P0: np.ndarray,
    major_direction: np.ndarray,
    tol: float = 2.0,
):
    """
    Convenience wrapper for the minor axis.
    `major_direction` is the unit vector along the major axis in (row, col).
    """
    # rotate +90° CCW to get the minor axis direction
    d_perp = np.array([-major_direction[1], major_direction[0]], dtype=float)
    p_min, p_max, _, _ = axis_extrema_points(boundary_coords, P0, d_perp, tol)
    return p_min, p_max


def ImageQuantification(
    fluorescent_img, cell_mask, selected_label, C1max, C1min, tp,
    ref_ep1=None, ref_ep2=None, skip_em=False,
    init_params_unlinked=None,   # NEW
    init_blend=0.7               # NEW
):

    #%% load single segmentation
    # Use regionprops to find the bounding box of the cell
    labeled_mask = label(cell_mask)  # Ensure the mask is labeled for regionprops
    props = regionprops(labeled_mask)
    region = props[0]  # There should only be one region for this single cell mask
    min_row, min_col, max_row, max_col = region.bbox
    
    # Crop the fluorescent image around the cell's bounding box
    cropped_img = fluorescent_img[min_row:max_row, min_col:max_col]
    # Create a cropped cell mask from the full cell mask:
    cropped_cell_mask = cell_mask[min_row:max_row, min_col:max_col]
    # Ensure the first and last rows and columns of cropped_cell_mask are False.
    cropped_cell_mask[0, :] = False
    cropped_cell_mask[-1, :] = False
    cropped_cell_mask[:, 0] = False
    cropped_cell_mask[:, -1] = False
    

    labeled_cell = label(cropped_cell_mask)
    #print(labeled_cell)
    props = regionprops(labeled_cell)[0]

    if not props:
        raise ValueError("No region found in the mask.")

    #---- Extract basic properties from the cropped cell mask and cropped image
    # Extract the cell boundary wiht find_contours.
    contours = find_contours(cropped_cell_mask.astype(float), level=0.5)
    if len(contours) == 0:
        raise ValueError("No contour found in the cell mask.")
    # Use the longest contour.
    boundary = max(contours, key=len)  # in (row, col) coordinates
    
    # Extract pixel coordinates (within the cropped image) that belong to the cell.
    y_idx, x_idx = np.where(cropped_cell_mask)
    intensities = cropped_img[y_idx, x_idx]
    # Build the 3-D dataset: each row is [x, y, intensity] 
    # (Note: x corresponds to column, y corresponds to row.)
    X_data = np.column_stack((x_idx, y_idx, intensities))
    #print("Extracted", X_data.shape[0], "pixels from the cell.")
    # Define area A as the sum of True pixels in the cropped mask.
    A = cropped_cell_mask.sum()
    # Get centroid and orientation; note: region.centroid returns (row, col)
    P0 = np.array(props.centroid)
    major_axis_length = props.major_axis_length
    minor_axis_length = props.minor_axis_length
    orientation = props.orientation  # in radians

    # Define the unit direction vector along the major axis.
    # Note: In skimage, orientation is defined as the angle between the 0th axis (rows) and the major axis.
    d = np.array([np.cos(orientation), np.sin(orientation)])  # direction in (row, col)

    # Instead of using find_contours, get all boundary pixels using find_boundaries.
    boundary_mask = find_boundaries(cropped_cell_mask, mode='inner')
    boundary_coords = np.column_stack(np.nonzero(boundary_mask))  # shape (M, 2) in (row, col) coordinates
    
    # Set a tolerance (in pixels) to decide if a boundary pixel is "on" the major axis.
    tol = 2.0

  
    # Endpoints along the major axis
    endpoint1, endpoint2, _, _ = axis_extrema_points(boundary_coords, P0, d, tol)

    
    
    #---- find middle points for mirror space (mn space) transformation
    # Perpendicular direction to the major axis
    
    # boundary_coords already computed from find_boundaries(...)
    midpoint1, midpoint2 = find_midpoints_on_minor_axis(
        boundary_coords=boundary_coords,
        P0=P0,
        major_direction=d,
        tol=tol,  # same tolerance you used for endpoints
    )

 
    

  
  


   
    


    #---- (u,v) Transformation for Polarity Modeling and Septum Modelin
    
    u_map, v_map, u_endpoint1, u_endpoint2, u_midpoint1, u_midpoint2, L = transform_to_uv_space(boundary, endpoint1, endpoint2, midpoint1, midpoint2, cropped_cell_mask)

   

    #---- (m,n) Transformation for Nuclei Modeling
    
    m_map, n_map = transform_to_mn_space(midpoint1, midpoint2, cropped_cell_mask)
  
        
    
    #----- EM Algorithm Initialization ------

    N = X_data.shape[0]
    eps = 1e-6

    # uv 1 for polarity sites, uv2 for septum. 
    sigma_u1 = 10.0  # fixed
    sigma_v1 = 3   # fixed
    sigma_u2 = 3  # fixed
    sigma_v2 = 3   # increase with mu_mn[1] 

    # Mixing weights for 3 states: background only (Y=0), background and nucleus 
    # (Y=1), background, nucleus, and polarity site (Y=2)
    Yi = np.array([0.1, 0.45, 0.45])

    # Mixing weights for 2 components when Y = 1: background (Z=0), nucleus (Z=1)
    #piY1 = np.array([0.8, 0.2])

    # Mixing weights for 4 components: background (Z=0), nucleus (Z=1),
    # polarity site 1 (Z=2), polarity site 2 (Z=3)
    # septum end 1 (Z=4), septum end 2 (Z=5)
    # Movie background (Z=6)
    pi = np.array([0.7, 0.2, 0.035, 0.035, 0.01, 0.01, 0.01])

    # Background (cytosol) intensity parameters:
    mu_bg = np.median(X_data[:,2])
    sigma_bg = max(np.std(X_data[:,2]), eps)
    # Movie background intensity parameters:
    mu_mv = np.min(X_data[:,2])
    sigma_mv = max(np.std(X_data[:,2]), eps)


    # Nucleus spatial parameters:
    mu_xy = np.mean(X_data[:, :2], axis=0)
    sigma_xy = max(np.std(X_data[:,0]), eps)
    
    # Convert mu_xy to mu_mn
    mu_mn = mu_xy_to_mu_mn(midpoint1,midpoint2,mu_xy)
    sigma_mn = minor_axis_length/4
    # Shared nucleus intensity parameters:
    mu_I = np.mean(X_data[:,2])
    sigma_I = max(np.std(X_data[:,2]), eps)

    # For polarity sites, the intensity model is constrained to the nucleus,
    # so we do not have separate intensity parameters for polarity.

    # Assume A is the sum of cropped_cell_mask pixels.
    A = cropped_cell_mask.sum()

    max_iter = 100
    tol = 1e-4
    # Build a dictionary with fixed parameters.
    params_fixed = {
            
        'sigma_S2_Y2': sigma_I,
        'sigma_u1_Y2': sigma_u1,
        #'sigma_v1_Y2': sigma_v1,
        'sigma_u2_Y2': sigma_u2,
        #'sigma_v2_Y2': sigma_v2,
        
        'u_map': u_map,
        'v_map': v_map,
        'u_endpoint1': u_endpoint1,
        'u_endpoint2': u_endpoint2,
        'u_midpoint1': u_midpoint1,
        'u_midpoint2': u_midpoint2,
        
        'm_map': m_map,
        'n_map': n_map,
        
        'major_axis_length':major_axis_length,
        'area':A,
        'L': L
        
    }
    # Build a dictionary with all parameters.
    params_unlinked = {
        
        'mu_bg_Y2': mu_bg,
        'sigma_bg_Y2': sigma_bg,
        #'mu_xy_Y2':  mu_xy,
        #'sigma_xy_Y2': sigma_xy,
        'mu_mn_Y2': mu_mn,
        'sigma_mn_Y2': sigma_mn,
        'mu_I_Y2': mu_I,
        'sigma_I_Y2': sigma_I,
        'mu_P1_Y2': mu_I,
        'sigma_P1_Y2': sigma_I,
        'mu_P2_Y2': mu_I,
        'sigma_P2_Y2': sigma_I,
        'mu_S1_Y2': mu_I,
        'sigma_S1_Y2': sigma_I,
        'mu_S2_Y2': mu_I,
        'sigma_S2_Y2': sigma_I,
        'sigma_v1_Y2': sigma_v1,
        'sigma_v2_Y2': sigma_v2,
        'mu_mv_Y2': mu_mv,
        'sigma_mv_Y2': sigma_mv
        
        
    }
    def _log_blend(a, b, w, eps=1e-6):
        # blend positive scalars in log space (good for sigmas)
        a = max(float(a), eps)
        b = max(float(b), eps)
        return float(np.exp((1.0 - w) * np.log(a) + w * np.log(b)))
    
    def _lin_blend(a, b, w):
        return (1.0 - w) * a + w * b
    
    if init_params_unlinked is not None:
        w = float(init_blend)
    
        # ---- means (linear blend) ----
        for kname in [
            'mu_bg_Y2', 'mu_I_Y2', 'mu_P1_Y2', 'mu_P2_Y2',
            'mu_S1_Y2', 'mu_S2_Y2', 'mu_mv_Y2'
        ]:
            if kname in init_params_unlinked and init_params_unlinked[kname] is not None:
                params_unlinked[kname] = float(_lin_blend(params_unlinked[kname], init_params_unlinked[kname], w))
    
        # nucleus location in mn-space is a 2-vector
        if 'mu_mn_Y2' in init_params_unlinked and init_params_unlinked['mu_mn_Y2'] is not None:
            prev_mu_mn = np.array(init_params_unlinked['mu_mn_Y2'], dtype=float)
            cur_mu_mn  = np.array(params_unlinked['mu_mn_Y2'], dtype=float)
            if prev_mu_mn.shape == (2,) and cur_mu_mn.shape == (2,):
                params_unlinked['mu_mn_Y2'] = _lin_blend(cur_mu_mn, prev_mu_mn, w)
    
        # ---- stds (log blend) ----
        for kname in [
            'sigma_bg_Y2', 'sigma_I_Y2', 'sigma_mn_Y2',
            'sigma_P1_Y2', 'sigma_P2_Y2', 'sigma_S1_Y2', 'sigma_S2_Y2',
            'sigma_v1_Y2', 'sigma_v2_Y2', 'sigma_mv_Y2'
        ]:
            if kname in init_params_unlinked and init_params_unlinked[kname] is not None:
                params_unlinked[kname] = _log_blend(params_unlinked[kname], init_params_unlinked[kname], w, eps=eps)

    plot_data = []
    if skip_em:
        print("Skipping EM computation; extracting only fixed features.")
        return params_unlinked, params_fixed, plot_data, ref_ep1, ref_ep2
    else: 
        #----- EM Algorithm for the 4-Compartment Model -----
        for iteration in range(max_iter):
            
    
            #------
            #E-step:
            #------    
            
          
            gammas_unlinked = E_step(X_data, A, params_unlinked, Yi, pi, eps, params_fixed)
            
            #-----
            # M-step:
            #-----
          
            params_unlinked_new = M_step_unlinked(
                X_data, gammas_unlinked, Yi, pi, eps, params_fixed,
                prev_params=init_params_unlinked,   # <-- pass the t-1 params here
                lam_mn=0.3, lam_I=0.2,
                alpha_sigma=0.2
            )

            
            #print(params_unlinked_new['sigma_mn_Y2']) 
            
            #Convergence check.
            if check_convergence(params_unlinked, params_unlinked_new, tol):
                print(f"Convergence reached at iteration {iteration}")
                break
            
          
            params_unlinked = params_unlinked_new
            
     
    
        # ----- Fix polarity sites ----
        # Get endpoints
            ep1 = endpoint1
            ep2 = endpoint2
            pol1_int = params_unlinked['mu_P1_Y2']
            pol2_int = params_unlinked['mu_P2_Y2']
            
            if tp == 0:
                # Save reference endpoint for comparison
                ref_ep1 = endpoint1
                ref_ep2 = endpoint2
            # Reorder based on reference (skip t=0 since it's the reference)
            else:
                ep1, ep2 = order_endpoints_by_proximity(ep1, ep2, ref_ep1, ref_ep2)
                if np.allclose(ep1, endpoint1):
                    params_unlinked['mu_P1_Y2'] = pol1_int 
                    params_unlinked['mu_P2_Y2'] = pol2_int 
    
                else:
                    # Endpoints were swapped
                    pol1_int = params_unlinked['mu_P2_Y2']
                    pol2_int = params_unlinked['mu_P1_Y2']
                    params_unlinked['mu_P1_Y2'] = pol1_int 
                    params_unlinked['mu_P2_Y2'] = pol2_int 
    
                    pol1_gamma = gammas_unlinked['Y2Z3']
                    pol2_gamma = gammas_unlinked['Y2Z2']
                    gammas_unlinked['Y2Z2']=pol1_gamma
                    gammas_unlinked['Y2Z3']=pol2_gamma
    
            
        Mu = [params_unlinked['mu_bg_Y2'],params_unlinked['mu_I_Y2'] ,params_unlinked['mu_P1_Y2'],params_unlinked['mu_P2_Y2'],params_unlinked['mu_S1_Y2'],params_unlinked['mu_S2_Y2']]
        #plot_mask(gammas_unlinked,cropped_cell_mask,y_idx, x_idx, Mu)
        #plot_gamma_overlay(gammas_unlinked,cropped_cell_mask,y_idx, x_idx)#, Mu)
        
        # change contract for visualization
        cropped_img = np.clip(((cropped_img - C1min) / (C1max - C1min) * 255), 0, 255).astype(np.uint8)
        plot_data = [cropped_img,
                     boundary,
                     gammas_unlinked,
                     cropped_cell_mask,
                     y_idx,
                     x_idx,
                     Mu,
                     boundary[0],
                     (ep1[1], ep1[0]),
                     (ep2[1], ep2[0]),
                     (midpoint1[1], midpoint1[0]),
                     (midpoint2[1], midpoint2[0]),
                     selected_label
            ]
      
        return params_unlinked, params_fixed, plot_data, ref_ep1, ref_ep2
 
