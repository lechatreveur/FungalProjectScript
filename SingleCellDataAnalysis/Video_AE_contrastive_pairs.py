#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_contrastive_pairs.py  — Strategy C, Step 1

Precomputes a phase-invariant, orientation-invariant similarity matrix
across all 419 cells, then saves the top-K positive pairs for contrastive training.

Similarity is defined as:
  max_{Δt, flip} pearson_xcorr( total_pol_A(t), total_pol_B(t + Δt) )

where total_pol = spatially-averaged (Pol1 + Pol2).

This is invariant to:
  - Cell-cycle phase (which frame the cell "starts" at)
  - Which end is Pol1 vs Pol2 (since Pol1+Pol2 is symmetric under flip)

Output:
  polarity_traces.npy      — (N, 101) total polarity traces
  similarity_matrix.npy    — (N, N) max cross-correlation scores
  positive_pairs.npy       — (N, K) top-K most similar cell indices per cell
"""

import os
import sys
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import gaussian_filter1d

BASE_DIR    = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
GAMMA_CACHE = os.path.join(BASE_DIR, "gamma_cache_32x112_padded.npy")
OUT_TRACES  = os.path.join(BASE_DIR, "polarity_total_traces.npy")
OUT_SIMMAT  = os.path.join(BASE_DIR, "similarity_matrix.npy")
OUT_PAIRS   = os.path.join(BASE_DIR, "positive_pairs.npy")

TOP_K = 10   # Number of positive partners to store per cell
SMOOTH_SIGMA = 3.0   # Gaussian smoothing sigma (frames)


def max_normalised_xcorr(a, b):
    """
    Compute the maximum normalised cross-correlation between traces a and b
    at any integer time shift (both directions).

    Returns a scalar in [-1, 1].
    """
    # z-normalise so the result is Pearson-like
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    # Full cross-correlation via FFT
    corr = fftconvolve(a, b[::-1], mode='full')
    # Normalise by length
    return float(corr.max()) / len(a)


def main():
    print("📥 Loading gamma cache (memory-mapped)...")
    gammas = np.load(GAMMA_CACHE, mmap_mode='r')  # (N, 101, 7, 32, 112)
    n_cells = gammas.shape[0]
    print(f"   {n_cells} cells.")

    # --- 1. Extract total polarity trace (Pol1+Pol2, spatial mean) ---
    print("📊 Extracting polarity traces...")
    traces = np.zeros((n_cells, 101), dtype=np.float32)
    for i in range(n_cells):
        pol_total = gammas[i, :, 2, :, :] + gammas[i, :, 3, :, :]  # (101,32,112)
        traces[i] = gaussian_filter1d(pol_total.mean(axis=(1, 2)), sigma=SMOOTH_SIGMA)

    np.save(OUT_TRACES, traces)
    print(f"   Traces shape: {traces.shape} → saved to {OUT_TRACES}")

    # --- 2. Compute N×N similarity matrix ---
    print("🔢 Computing similarity matrix (this takes a few minutes)...")
    sim = np.zeros((n_cells, n_cells), dtype=np.float32)
    for i in range(n_cells):
        if i % 50 == 0:
            print(f"   Row {i}/{n_cells}...")
        for j in range(i + 1, n_cells):
            s = max_normalised_xcorr(traces[i], traces[j])
            sim[i, j] = s
            sim[j, i] = s
        sim[i, i] = 1.0   # self-similarity

    np.save(OUT_SIMMAT, sim)
    print(f"   Similarity matrix saved to {OUT_SIMMAT}")

    # --- 3. Extract top-K positive pair indices per cell ---
    print(f"📋 Extracting top-{TOP_K} positive pairs per cell...")
    # For each cell, argsort by descending similarity, exclude self
    pairs = np.zeros((n_cells, TOP_K), dtype=np.int32)
    for i in range(n_cells):
        row = sim[i].copy()
        row[i] = -999   # exclude self
        top_k_idx = np.argsort(row)[::-1][:TOP_K]
        pairs[i] = top_k_idx

    np.save(OUT_PAIRS, pairs)
    print(f"   Positive pairs saved to {OUT_PAIRS}")

    # --- 4. Report statistics ---
    # Similarity histogram
    off_diag = sim[np.triu_indices(n_cells, k=1)]
    print(f"\n📈 Similarity statistics (N={len(off_diag):,} pairs):")
    print(f"   Mean: {off_diag.mean():.3f}")
    print(f"   Std:  {off_diag.std():.3f}")
    print(f"   P50:  {np.percentile(off_diag, 50):.3f}")
    print(f"   P90:  {np.percentile(off_diag, 90):.3f}")
    print(f"   P99:  {np.percentile(off_diag, 99):.3f}")
    print(f"\n✅ Done.")


if __name__ == "__main__":
    main()
