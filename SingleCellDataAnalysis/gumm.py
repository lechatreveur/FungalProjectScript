#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:49:23 2025

@author: user
"""

# gumm.py

import numpy as np
from scipy.stats import norm, uniform
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def gaussian_uniform_mixture_log_likelihood(params, data):
    mu, sigma, pi = params
    a, b = np.min(data), np.max(data)

    if sigma <= 0 or not (0 < pi < 1):
        return np.inf

    norm_pdf = norm.pdf(data, mu, sigma)
    unif_pdf = uniform.pdf(data, loc=a, scale=b - a)
    total_likelihood = pi * norm_pdf + (1 - pi) * unif_pdf

    return -np.sum(np.log(total_likelihood))


def fit_gaussian_uniform_mixture(data):
    mu0 = np.mean(data)
    sigma0 = np.std(data)
    pi0 = 0.9

    result = minimize(
        gaussian_uniform_mixture_log_likelihood,
        x0=[mu0, sigma0, pi0],
        args=(data,),
        bounds=[(None, None), (1e-6, None), (1e-3, 1 - 1e-3)],
        method='L-BFGS-B'
    )

    return result.x  # mu, sigma, pi


def plot_gumm(data, title="", xlabel=""):
    mu, sigma, pi = fit_gaussian_uniform_mixture(data)
    a, b = np.min(data), np.max(data)

    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, _ = ax.hist(data, bins=100, density=True, alpha=0.6, edgecolor='black', label="Histogram")

    x = np.linspace(a, b, 1000)
    y_mix = pi * norm.pdf(x, mu, sigma) + (1 - pi) * uniform.pdf(x, loc=a, scale=b - a)
    y_norm = pi * norm.pdf(x, mu, sigma)
    y_unif = (1 - pi) * uniform.pdf(x, loc=a, scale=b - a)

    ax.plot(x, y_mix, 'k-', label='Mixture')
    ax.plot(x, y_norm, 'r--', label='Normal component')
    ax.plot(x, y_unif, 'g--', label='Uniform component')
    ax.axvline(0, color='blue', linestyle='--', label='Zero Growth')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability Density")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"📊 Fitted parameters for {title}:\nμ = {mu:.3f}, σ = {sigma:.3f}, π = {pi:.3f}\n")
    return mu, sigma, pi
