"""What do silhouette +0.48 and Hopkins 0.87 actually mean?

Calibrate against data with NO cluster structure, run through the identical
pipeline. If structureless nulls score the same as M160, the metrics were never
measuring clusterability and every number quoted from them is void.

Four datasets, all n=6243 in 6D:
  M160 real          - the cohort itself
  shuffled dims      - each latent dim independently permuted: kills the joint
                       structure, keeps every marginal exactly
  isotropic gaussian - no structure at all
  3 true blobs       - positive control, genuinely separated
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch
import umap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, "/Users/user/Documents/Python_Scripts/FungalProjectScript")
from SingleCellDataAnalysis.FC_AE_3d_train import MultimodalAutoencoder3D
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

BASE = ("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking/"
        "2026_08_28_M160")
CTRL = "/Volumes/X10 Pro/FungalProject_Outputs/umap_control"

X_traj, X_feat, gids, *_ = load_feature_constrained_data({"M160": f"{BASE}/features"})
sd = torch.load(f"{BASE}/fc_ae_3d_m160.pth", map_location="cpu")
dim = next(v.shape[0] for k, v in sd.items()
           if k.startswith("encoder_fc") and k.endswith("weight")
           and v.ndim == 2 and v.shape[1] == 128)
m = MultimodalAutoencoder3D(latent_dim=int(dim))
m.load_state_dict(sd)
m.eval()
with torch.no_grad():
    _, _, Z = m(torch.from_numpy(X_traj).float(), torch.from_numpy(X_feat).float())
Z = Z.numpy()
n, d = Z.shape
print(f"M160 latents: {Z.shape}\n")

rng = np.random.default_rng(0)
Z_shuf = np.column_stack([rng.permutation(Z[:, j]) for j in range(d)])
Z_gauss = rng.normal(size=(n, d))
c = rng.normal(size=(3, d)) * 6.0
Z_blobs = np.vstack([rng.normal(size=(n // 3, d)) + c[i] for i in range(3)])


def hopkins(X, seed=0):
    r = np.random.default_rng(seed)
    k = min(200, len(X) // 4)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    idx = r.choice(len(X), k, replace=False)
    u = nn.kneighbors(X[idx], n_neighbors=2)[0][:, 1]
    q = r.uniform(X.min(0), X.max(0), (k, X.shape[1]))
    w = nn.kneighbors(q, n_neighbors=1)[0][:, 0]
    return w.sum() / (w.sum() + u.sum())


sets = [("M160 real", Z), ("shuffled dims", Z_shuf),
        ("isotropic gaussian", Z_gauss), ("3 true blobs", Z_blobs)]
out = []
print(f"{'dataset':>20} {'n_nb':>5} {'k':>3} {'silhouette':>11} {'Hopkins':>8}")
for label, X in sets:
    for nb in (15, 248):
        e = umap.UMAP(n_components=2, n_neighbors=nb,
                      random_state=42, n_jobs=1).fit_transform(X)
        sil, k = max((silhouette_score(e, KMeans(kk, n_init=10, random_state=42).fit_predict(e)), kk)
                     for kk in range(2, 9))
        h = hopkins(e)
        out.append((label, nb, e, sil, k, h))
        print(f"{label:>20} {nb:>5} {k:>3} {sil:>+11.3f} {h:>8.3f}", flush=True)

fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
for col, (label, _) in enumerate(sets):
    for row, nb in enumerate((15, 248)):
        _, _, e, sil, k, h = next(r for r in out if r[0] == label and r[1] == nb)
        ax = axes[row, col]
        ax.scatter(e[:, 0], e[:, 1], s=2, c="#2563eb", alpha=0.3, rasterized=True)
        ax.set_title(f"{label}, n_neighbors={nb}\nsilhouette {sil:+.3f} at k={k}, "
                     f"Hopkins {h:.3f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle("Do the metrics distinguish M160 from structureless noise?", fontsize=13)
fig.tight_layout()
p = f"{CTRL}/metric_null_calibration.png"
fig.savefig(p, dpi=140)
print(f"\nplot -> {p}")
