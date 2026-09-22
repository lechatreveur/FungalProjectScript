#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P16 null calibration for M162's standalone manifold.

M160 was growth-arrested and its polarity manifold came back continuous with no
discrete dynamic modes. M162 is the healthy arm — YES medium, BF doubling
3.93 h, 0.21% model-only frames — so it is the fair test of whether discrete
modes exist at all in this assay.

P16 forbids reading cluster structure off a UMAP scatter or off a bare
silhouette: the floor on an embedding is +0.38-0.43 rather than 0, it is
sample-size and dimension dependent, and the shuffled null can outscore real
data. So the nulls are regenerated here at M162's own n and latent dimension:

  shuffled dims      - each latent dim independently permuted; destroys every
                       joint relationship, preserves every marginal exactly.
                       This is the floor that matters.
  isotropic gaussian - no structure at all.
  3 true blobs       - positive control, genuinely separated.

Run from the repo root with the SSD mounted.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sys.path.insert(0, "/Users/user/Documents/Python_Scripts/FungalProjectScript")
from SingleCellDataAnalysis.FC_AE_3d_train import MultimodalAutoencoder3D
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

BASE = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking/"
            "2026_09_09_M162")
CTRL = Path("/Volumes/X10 Pro/FungalProject_Outputs/umap_control")
REF_FRAC = 15 / 378          # the Sept17 reference fraction, 3.97%


def latents():
    X_traj, X_feat, gids, *_ = load_feature_constrained_data({"M162": str(BASE / "features")})
    sd = torch.load(BASE / "fc_ae_3d_m162.pth", map_location="cpu")
    dim = next(v.shape[0] for k, v in sd.items()
               if k.startswith("encoder_fc") and k.endswith("weight")
               and v.ndim == 2 and v.shape[1] == 128)
    m = MultimodalAutoencoder3D(latent_dim=int(dim))
    m.load_state_dict(sd)
    m.eval()
    with torch.no_grad():
        _, _, z = m(torch.from_numpy(X_traj).float(), torch.from_numpy(X_feat).float())
    return z.numpy()


def embed(X, nb):
    return umap.UMAP(n_components=2, n_neighbors=nb,
                     random_state=42, n_jobs=1).fit_transform(X)


def best_sil(e):
    return max((silhouette_score(e, KMeans(k, n_init=10, random_state=42).fit_predict(e)), k)
               for k in range(2, 9))


def main() -> int:
    Z = latents()
    n, d = Z.shape
    nb = max(2, round(REF_FRAC * n))
    print(f"M162 latents: {Z.shape}; fraction-matched n_neighbors = {nb}\n")

    rng = np.random.default_rng(0)
    Z_shuf = np.column_stack([rng.permutation(Z[:, j]) for j in range(d)])
    Z_gauss = rng.normal(size=(n, d))
    c = rng.normal(size=(3, d)) * 6.0
    Z_blobs = np.vstack([rng.normal(size=(n // 3, d)) + c[i] for i in range(3)])

    sets = [("M162 real", Z), ("shuffled dims", Z_shuf),
            ("isotropic gaussian", Z_gauss), ("3 true blobs", Z_blobs)]
    NBS = (15, nb)
    out = []
    print(f"{'dataset':>20} {'n_nb':>5} {'k':>3} {'silhouette':>11}")
    for label, X in sets:
        for b in NBS:
            e = embed(X, b)
            sil, k = best_sil(e)
            out.append((label, b, e, sil, k))
            print(f"{label:>20} {b:>5} {k:>3} {sil:>+11.3f}", flush=True)

    real = {b: s for l, b, _, s, _ in out if l == "M162 real"}
    shuf = {b: s for l, b, _, s, _ in out if l == "shuffled dims"}
    print("\n" + "=" * 64)
    print("VERDICT (P16: real must clear its own shuffled null)")
    print("=" * 64)
    for b in NBS:
        margin = real[b] - shuf[b]
        verdict = ("clears the null" if margin > 0.05 else
                   "does NOT clear the null")
        print(f"  n_neighbors {b:>4}: real {real[b]:+.3f} vs shuffled "
              f"{shuf[b]:+.3f}  (margin {margin:+.3f})  -> {verdict}")

    fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
    for col, (label, _) in enumerate(sets):
        for row, b in enumerate(NBS):
            _, _, e, sil, k = next(r for r in out if r[0] == label and r[1] == b)
            ax = axes[row, col]
            ax.scatter(e[:, 0], e[:, 1], s=6, c="#16a34a", alpha=0.45, rasterized=True)
            ax.set_title(f"{label}, n_neighbors={b}\nsilhouette {sil:+.3f} at k={k}",
                         fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"M162 (n={n}, {d}D latent): does the healthy arm show discrete modes?",
                 fontsize=13)
    fig.tight_layout()
    CTRL.mkdir(parents=True, exist_ok=True)
    p = CTRL / "m162_null_calibration.png"
    fig.savefig(p, dpi=140)
    print(f"\nplot -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
