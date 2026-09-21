"""How does n_neighbors change the M160 map?

Same latents, same seed, same everything else — only the neighbour count moves.
Reported alongside each map: the fraction of the population it spans, the
silhouette k-means can achieve, and Hopkins.

The point is to see how much of what looks like structure is a choice.
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
NBS = [5, 15, 30, 60, 125, 248, 500, 1000]

X_traj, X_feat, gids, *_ = load_feature_constrained_data({"M160": f"{BASE}/features"})
sd = torch.load(f"{BASE}/fc_ae_3d_m160.pth", map_location="cpu")
dim = next(v.shape[0] for k, v in sd.items()
           if k.startswith("encoder_fc") and k.endswith("weight")
           and v.ndim == 2 and v.shape[1] == 128)
m = MultimodalAutoencoder3D(latent_dim=int(dim)); m.load_state_dict(sd); m.eval()
with torch.no_grad():
    _, _, Z = m(torch.from_numpy(X_traj).float(), torch.from_numpy(X_feat).float())
Z = Z.numpy()
n = len(Z)
print(f"M160 latents: {Z.shape}\n")


def hopkins(X, seed=0):
    rng = np.random.default_rng(seed)
    k = min(200, len(X) // 4)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(len(X), k, replace=False)
    u = nn.kneighbors(X[idx], n_neighbors=2)[0][:, 1]
    r = rng.uniform(X.min(0), X.max(0), (k, X.shape[1]))
    w = nn.kneighbors(r, n_neighbors=1)[0][:, 0]
    return w.sum() / (w.sum() + u.sum())


res = []
print(f"{'n_nb':>6} {'% of n':>8} {'best k':>7} {'silhouette':>11} {'Hopkins':>9}")
for nb in NBS:
    e = umap.UMAP(n_components=2, n_neighbors=nb,
                  random_state=42, n_jobs=1).fit_transform(Z)
    sil, k = max((silhouette_score(e, KMeans(kk, n_init=10, random_state=42).fit_predict(e)), kk)
                 for kk in range(2, 9))
    h = hopkins(e)
    res.append((nb, e, sil, k, h))
    print(f"{nb:>6} {100*nb/n:>7.2f}% {k:>7} {sil:>+11.3f} {h:>9.3f}", flush=True)

fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
for ax, (nb, e, sil, k, h) in zip(axes.ravel(), res):
    ax.scatter(e[:, 0], e[:, 1], s=2, c="#2563eb", alpha=0.3, rasterized=True)
    mark = "  <- current" if nb == 248 else ("  <- library default" if nb == 15 else "")
    ax.set_title(f"n_neighbors = {nb} ({100*nb/n:.2f}% of n){mark}\n"
                 f"silhouette {sil:+.3f} at k={k}, Hopkins {h:.3f}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"M160 full cohort, n={n}: the same latents at eight neighbour counts",
             fontsize=13)
fig.tight_layout()
out = f"{CTRL}/m160_n_neighbors_sweep.png"
fig.savefig(out, dpi=140)
print(f"\nplot -> {out}")
