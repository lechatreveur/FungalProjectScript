#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 14:59:31 2025

@author: user
"""

import numpy as np

# Define the PDFs
def pdf_normal(x, mu, Sigma):
    d = x - mu
    inv_Sigma = np.linalg.inv(Sigma)
    det_Sigma = np.linalg.det(Sigma)
    norm_const = 1.0 / (2 * np.pi * np.sqrt(det_Sigma))
    exponent = -0.5 * np.sum(d @ inv_Sigma * d, axis=1)
    return norm_const * np.exp(exponent)

def pdf_uniform(x, region_min=-5, region_max=5):
    # Check if each point is in the region
    in_region = np.all((x >= region_min) & (x <= region_max), axis=1)
    area = (region_max - region_min)**2  # 100 for [-5,5] x [-5,5]
    pdf_vals = np.zeros(len(x))
    pdf_vals[in_region] = 1.0 / area
    return pdf_vals

# Assume X is the combined dataset (N x 2 array)
# For demonstration, we generate synthetic data as before:
np.random.seed(0)
mu_true = np.array([2.0, 2.0])
Sigma_true = np.array([[1.0, 0.3],
                       [0.3, 1.0]])
N_foreground = 100
X_fg = np.random.multivariate_normal(mu_true, Sigma_true, N_foreground)
N_background = 300
X_bg = np.random.rand(N_background, 2)*10 - 5  # Uniform in [-5,5]x[-5,5]
X = np.vstack((X_fg, X_bg))
N = X.shape[0]

# --- EM Algorithm Initialization ---
p = 0.5
# Initialize mu and Sigma arbitrarily (for instance, using all data)
mu = np.mean(X, axis=0)
Sigma = np.cov(X.T)

# Parameters for EM iterations
max_iter = 100
tol = 1e-4
log_likelihoods = []

for iteration in range(max_iter):
    # E-step: Compute responsibilities gamma_i for each x_i
    f_fg = pdf_normal(X, mu, Sigma)
    f_bg = pdf_uniform(X, -5, 5)
    gamma = (p * f_fg) / (p * f_fg + (1-p) * f_bg)
    
    # Compute the log-likelihood for monitoring convergence
    log_likelihood = np.sum(np.log(p * f_fg + (1-p) * f_bg))
    log_likelihoods.append(log_likelihood)
    
    # M-step: Update parameters using the responsibilities
    p_new = np.mean(gamma)
    mu_new = np.sum(gamma[:, np.newaxis] * X, axis=0) / np.sum(gamma)
    diff = X - mu_new
    Sigma_new = (gamma[:, np.newaxis] * diff).T @ diff / np.sum(gamma)
    
    # Check convergence (parameter changes)
    if np.linalg.norm(mu_new - mu) < tol and np.linalg.norm(Sigma_new - Sigma) < tol:
        print(f'Convergence reached at iteration {iteration}')
        break
    
    # Update parameters for next iteration
    p, mu, Sigma = p_new, mu_new, Sigma_new

print("Estimated mixture weight p:", p)
print("Estimated mean mu:", mu)
print("Estimated covariance Sigma:\n", Sigma)
