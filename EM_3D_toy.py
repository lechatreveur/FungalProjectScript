#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:38:28 2025

@author: user
"""

import numpy as np

# --- 1. Define the 3-D Gaussian and Uniform PDFs ---
def pdf_gaussian_3d(z, mu, Sigma):
    diff = z - mu
    inv_Sigma = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    norm_const = 1.0 / ((2*np.pi)**1.5 * np.sqrt(det_Sigma))
    exponent = -0.5 * np.sum(diff @ inv_Sigma * diff, axis=1)
    return norm_const * np.exp(exponent)

def pdf_uniform_3d(z, region_min, region_max):
    # region_min and region_max are 3-D vectors (for x, y, intensity)
    in_region = np.all((z >= region_min) & (z <= region_max), axis=1)
    volume = np.prod(region_max - region_min)
    pdf_vals = np.zeros(len(z))
    pdf_vals[in_region] = 1.0 / volume
    return pdf_vals

# --- 2. Generate Example 3-D Data (Simulated Image Pixels) ---
np.random.seed(0)

# Simulate the Gaussian object (e.g., a bright circular blob)
mu_true = np.array([50, 50, 200])  # center at (50,50) with high intensity 200
Sigma_true = np.array([[100, 0, 0],
                       [0, 100, 0],
                       [0,   0, 25]])  # spatial spread and intensity variance

N_object = 300
object_pixels = np.random.multivariate_normal(mu_true, Sigma_true, N_object)

# Simulate background pixels uniformly spread over the image domain
N_background = 700
# Assume image dimensions: x and y in [0, 100], intensity in [0, 255]
region_min = np.array([0, 0, 0])
region_max = np.array([100, 100, 255])
background_pixels = np.random.rand(N_background, 3) * (region_max - region_min) + region_min

# Combine into one dataset
X = np.vstack((object_pixels, background_pixels))
N = X.shape[0]

# True labels (1 for object, 0 for background) -- for evaluation only
true_labels = np.array([1]*N_object + [0]*N_background)

# --- 3. EM Algorithm Initialization ---
p = 0.5  # initial guess for mixture weight
# Initialize mu and Sigma using all data (or a random subset)
mu = np.mean(X, axis=0)
Sigma = np.cov(X.T)

max_iter = 100
tol = 1e-4
log_likelihoods = []

for iteration in range(max_iter):
    # E-step: Compute responsibilities gamma_i for each pixel z_i
    f_obj = pdf_gaussian_3d(X, mu, Sigma)
    f_bg = pdf_uniform_3d(X, region_min, region_max)
    
    gamma = (p * f_obj) / (p * f_obj + (1-p) * f_bg)
    
    # Compute the log-likelihood (for monitoring convergence)
    log_likelihood = np.sum(np.log(p * f_obj + (1-p) * f_bg))
    log_likelihoods.append(log_likelihood)
    
    # M-step: Update parameters
    p_new = np.mean(gamma)
    mu_new = np.sum(gamma[:, np.newaxis] * X, axis=0) / np.sum(gamma)
    diff = X - mu_new
    Sigma_new = (gamma[:, np.newaxis] * diff).T @ diff / np.sum(gamma)
    
    # Check convergence
    if np.linalg.norm(mu_new - mu) < tol and np.linalg.norm(Sigma_new - Sigma) < tol:
        print(f'Convergence reached at iteration {iteration}')
        break
    
    p, mu, Sigma = p_new, mu_new, Sigma_new

print("Estimated mixture weight p:", p)
print("Estimated mean mu:", mu)
print("Estimated covariance Sigma:\n", Sigma)
#%%
import matplotlib.pyplot as plt

# Create a binary mask from the responsibilities: 
# True (1) if the pixel is more likely from the object (gamma > 0.5), False (0) otherwise.
mask = gamma > 0.5

# Plot the simulated image using a scatter plot.
plt.figure(figsize=(12, 6))

# Left subplot: Simulated image
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(X[:, 0], X[:, 1], c=X[:, 2], cmap='gray', edgecolor='k', s=40)
plt.title('Simulated Image')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(scatter1, label='Intensity')

# Right subplot: Mask visualization
plt.subplot(1, 2, 2)
# Here we display the mask: object pixels in one color, background in another.
plt.scatter(X[:, 0], X[:, 1], c=mask, cmap='gray', edgecolor='k', s=40)
plt.title('Estimated Mask (γ > 0.5)')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(label='Mask Value (True/False)')

plt.tight_layout()
plt.show()

