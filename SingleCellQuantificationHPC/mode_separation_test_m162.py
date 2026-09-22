#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the monopolar/bipolar gap in M162's manifold real? A conditioned test.

Why this is a different question from the null calibration
----------------------------------------------------------
P16's calibration asks the *unconditioned* question — "are there clusters
anywhere?" — and shows it has almost no power: the silhouette floor on a UMAP
embedding is +0.38-0.43, and M162 clears its shuffled null by only +0.06.

"Do monopolar and bipolar cells occupy distinct regions?" is a *conditioned*
question with a proper null: permute the labels and hold the geometry fixed.
It can return a signal where the unconditioned test cannot, because it knows
what to look for. A visible gap between named categories is exactly the kind of
structure the unconditioned test would miss.

Two independent tests
---------------------
1. **Is the feature itself bimodal?** The Mode classifier splits monopolar from
   bipolar at `pol2_mid` = 2.0 (a threshold inherited from M160). If the gap is
   real the underlying feature should be bimodal on its own, with no reference
   to any manifold. Tested by Gaussian-mixture BIC for 1 vs 2 vs 3 components,
   and by Hartigan's dip statistic against a uniform null.

   This is the stronger evidence if it holds: it needs no autoencoder, no UMAP
   and no threshold.

2. **Do the labels occupy distinct latent regions?** k-nearest-neighbour label
   purity in the 6D latent space — the fraction of each point's k neighbours
   sharing its Mode label — against a null from permuting the labels while
   holding the geometry fixed. Reported as a z-score over 200 permutations.

   Purity is used rather than silhouette because silhouette on a continuous
   manifold is dominated by the elongation of the cloud, which is the failure
   P16 documents.

Run from the repo root with the SSD mounted.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, "/Users/user/Documents/Python_Scripts/FungalProjectScript")
from SingleCellDataAnalysis.FC_AE_3d_train import MultimodalAutoencoder3D
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

BASE = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking/"
            "2026_09_09_M162")
CTRL = Path("/Volumes/X10 Pro/FungalProject_Outputs/umap_control")

# The explorer's Mode thresholds, inherited from M160.
THR = dict(pol1=4.04, pol2=2.0, mono=1.14, bi=1.60, nc=0.0)
LABELS = ["Non-polarized", "Monopolar", "Monopolar Osc", "Bipolar", "Bipolar Osc"]
K_NN = 15
N_PERM = 200


def mode_of(p1, p2, per, nc) -> int:
    """Mirrors getCategory() in the explorer exactly."""
    if not np.isfinite(p1) or p1 < THR["pol1"]:
        return 0
    osc = (nc >= THR["nc"]) if np.isfinite(nc) else True
    if p2 < THR["pol2"]:
        return 2 if (per > THR["mono"] and osc) else 1
    return 4 if (per > THR["bi"] and osc) else 3


def dip_statistic(x: np.ndarray) -> float:
    """Hartigan's dip: max deviation of the ECDF from its closest unimodal fit.

    Approximated by the greatest gap between the ECDF and the best-fitting
    monotone (unimodal) envelope, which is enough to rank against a null.
    """
    x = np.sort(np.asarray(x, float))
    n = len(x)
    ecdf = np.arange(1, n + 1) / n
    # closest unimodal (here: linear on the sorted support) reference
    ref = (x - x[0]) / (x[-1] - x[0] + 1e-12)
    return float(np.max(np.abs(ecdf - ref)))


def test_bimodality(v: np.ndarray, name: str) -> dict:
    v = v[np.isfinite(v)]
    X = v.reshape(-1, 1)
    bic = {}
    for k in (1, 2, 3):
        gm = GaussianMixture(k, n_init=5, random_state=42).fit(X)
        bic[k] = gm.bic(X)
    best = min(bic, key=bic.get)

    rng = np.random.default_rng(0)
    d_obs = dip_statistic(v)
    null = np.array([dip_statistic(rng.normal(v.mean(), v.std(), len(v)))
                     for _ in range(200)])
    p_dip = float((null >= d_obs).mean())

    print(f"\n  {name}  (n={len(v)})")
    print(f"    BIC   1 comp {bic[1]:12.1f}")
    print(f"          2 comp {bic[2]:12.1f}   delta vs 1: {bic[2]-bic[1]:+.1f}")
    print(f"          3 comp {bic[3]:12.1f}   delta vs 1: {bic[3]-bic[1]:+.1f}")
    print(f"    -> BIC prefers {best} component(s)"
          f"{'  (bimodal or richer)' if best > 1 else '  (unimodal)'}")
    print(f"    dip {d_obs:.4f} vs gaussian null {null.mean():.4f}+/-{null.std():.4f}"
          f"   p={p_dip:.3f}")
    return dict(feature=name, best_k=best, bic1=bic[1], bic2=bic[2], bic3=bic[3],
                dip=d_obs, dip_p=p_dip)


def knn_purity(Z: np.ndarray, y: np.ndarray, k: int) -> float:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    idx = nn.kneighbors(Z, return_distance=False)[:, 1:]
    return float((y[idx] == y[:, None]).mean())


def main() -> int:
    feats = pd.read_csv(BASE / "features" / "umap_features_m162.csv")
    X_traj, X_feat, gids, *_ = load_feature_constrained_data(
        {"M162": str(BASE / "features")})
    sd = torch.load(BASE / "fc_ae_3d_m162.pth", map_location="cpu")
    dim = next(v.shape[0] for k, v in sd.items()
               if k.startswith("encoder_fc") and k.endswith("weight")
               and v.ndim == 2 and v.shape[1] == 128)
    m = MultimodalAutoencoder3D(latent_dim=int(dim))
    m.load_state_dict(sd)
    m.eval()
    with torch.no_grad():
        _, _, Z = m(torch.from_numpy(X_traj).float(), torch.from_numpy(X_feat).float())
    Z = Z.numpy()
    print(f"M162: {len(feats)} datapoints, latent {Z.shape}")

    # --- test 1: is the feature bimodal on its own? ----------------------
    print("\n" + "=" * 72)
    print("TEST 1 — is the feature itself bimodal? (no manifold involved)")
    print("=" * 72)
    rows = [test_bimodality(feats[c].to_numpy(float), c)
            for c in ("pol2_mid", "pol1_mid", "d", "Periodicity")]

    # --- test 2: do Mode labels occupy distinct latent regions? ----------
    modes = np.array([mode_of(r.pol1_mid, r.pol2_mid, r.Periodicity, r.NC_score)
                      for r in feats.itertuples()])
    print("\n" + "=" * 72)
    print("TEST 2 — do Mode labels occupy distinct latent regions?")
    print("=" * 72)
    counts = pd.Series(modes).value_counts().sort_index()
    for i, c in counts.items():
        print(f"  {LABELS[i]:<16} {c:>5}  ({100*c/len(modes):5.1f}%)")

    if len(counts) < 2:
        print("\n  only one Mode present — nothing to separate")
        return 0

    obs = knn_purity(Z, modes, K_NN)
    rng = np.random.default_rng(0)
    null = np.array([knn_purity(Z, rng.permutation(modes), K_NN)
                     for _ in range(N_PERM)])
    z = (obs - null.mean()) / (null.std() + 1e-12)
    print(f"\n  kNN label purity (k={K_NN}): {obs:.4f}")
    print(f"  permutation null:            {null.mean():.4f} +/- {null.std():.4f}")
    print(f"  z = {z:+.1f}   (p < {1/N_PERM:.3f} if |z| is large)")

    # the specific claim: monopolar vs bipolar only
    mask = np.isin(modes, (1, 2, 3, 4))
    mb = np.where(np.isin(modes[mask], (1, 2)), 0, 1)   # mono=0, bi=1
    if mb.sum() and (1 - mb).sum():
        obs_mb = knn_purity(Z[mask], mb, K_NN)
        null_mb = np.array([knn_purity(Z[mask], rng.permutation(mb), K_NN)
                            for _ in range(N_PERM)])
        z_mb = (obs_mb - null_mb.mean()) / (null_mb.std() + 1e-12)
        print(f"\n  monopolar vs bipolar only (n={mask.sum()}, "
              f"{(1-mb).sum()} mono / {mb.sum()} bi):")
        print(f"    purity {obs_mb:.4f} vs null {null_mb.mean():.4f}"
              f" +/- {null_mb.std():.4f}   z = {z_mb:+.1f}")

    print("\n" + "=" * 72)
    print("READING")
    print("=" * 72)
    p2 = next(r for r in rows if r["feature"] == "pol2_mid")
    if p2["best_k"] > 1 and z > 3:
        print("  Both tests positive: the feature is multimodal AND the labels")
        print("  occupy distinct latent regions. The gap is real.")
    elif z > 3 and p2["best_k"] == 1:
        print("  Labels separate in latent space, but the underlying feature is")
        print("  unimodal — so the 'gap' is the threshold cutting a continuum,")
        print("  not two populations. A threshold always produces separation.")
    elif p2["best_k"] > 1:
        print("  The feature is multimodal but the labels do not separate in")
        print("  latent space — the autoencoder is not encoding the split.")
    else:
        print("  Neither test positive: no evidence for a real gap.")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].hist(feats["pol2_mid"].dropna(), bins=70, color="#16a34a", alpha=0.75)
    ax[0].axvline(THR["pol2"], color="crimson", ls="--",
                  label=f"mono/bi threshold = {THR['pol2']}")
    ax[0].set_xlabel("pol2_mid")
    ax[0].set_ylabel("datapoints")
    ax[0].set_title("The feature the mono/bi split is made on")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.25)

    ax[1].hist(feats["pol1_mid"].dropna(), bins=70, color="#2563eb", alpha=0.75)
    ax[1].axvline(THR["pol1"], color="crimson", ls="--",
                  label=f"polar threshold = {THR['pol1']}")
    ax[1].set_xlabel("pol1_mid")
    ax[1].set_title("pol1_mid")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.25)

    ax[2].hist(null, bins=30, color="#94a3b8", alpha=0.8, label="permuted labels")
    ax[2].axvline(obs, color="crimson", lw=2, label=f"observed {obs:.3f}")
    ax[2].set_xlabel(f"kNN label purity (k={K_NN})")
    ax[2].set_title(f"Mode labels in latent space (z = {z:+.1f})")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.25)

    fig.suptitle("M162: is the monopolar/bipolar gap real?", fontsize=13)
    fig.tight_layout()
    CTRL.mkdir(parents=True, exist_ok=True)
    p = CTRL / "m162_mode_separation.png"
    fig.savefig(p, dpi=140)
    pd.DataFrame(rows).to_csv(CTRL / "m162_bimodality.csv", index=False)
    print(f"\nplot -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
