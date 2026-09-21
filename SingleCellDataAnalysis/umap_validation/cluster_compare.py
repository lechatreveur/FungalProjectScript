"""Does Sept17's clustering survive the M160 pipeline?

Both datasets through IDENTICAL settings: the M160 trainer's architecture at
latent 6, then UMAP with n_neighbors held at the Sept17 reference fraction of
3.97%. If Sept17 clusters here and M160 does not, the pipeline is sound and the
difference is in the data.

UMAP produces apparent structure even on noise, so the visual is backed by
numbers: silhouette over k-means for k = 2..8, and a Hopkins statistic, which
tests clusterability against a uniform null.
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
sys.path.insert(0, "SingleCellQuantificationHPC")
from SingleCellDataAnalysis.FC_AE_3d_train import MultimodalAutoencoder3D
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

CTRL = "/Volumes/X10 Pro/FungalProject_Outputs/umap_control"
M160F = ("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking/"
         "2026_08_28_M160/features")
NEIGHBORS_FRAC = 15 / 378


def latents(features_dir, model_path, film_filter=None):
    X_traj, X_feat, gids, *_ = load_feature_constrained_data({"E": str(features_dir)})
    if film_filter:
        keep = np.array([film_filter in g for g in gids], bool)
        X_traj, X_feat = X_traj[keep], X_feat[keep]
    sd = torch.load(model_path, map_location="cpu")
    dim = next(v.shape[0] for k, v in sd.items()
               if k.startswith("encoder_fc") and k.endswith("weight")
               and v.ndim == 2 and v.shape[1] == 128)
    m = MultimodalAutoencoder3D(latent_dim=int(dim))
    m.load_state_dict(sd)
    m.eval()
    with torch.no_grad():
        _, _, z = m(torch.from_numpy(X_traj).float(), torch.from_numpy(X_feat).float())
    return z.numpy(), dim


def hopkins(X, seed=0):
    """~0.5 = uniform (no clusters); ->1 = strongly clusterable."""
    rng = np.random.default_rng(seed)
    n = min(100, len(X) // 4)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(len(X), n, replace=False)
    u_d = nn.kneighbors(X[idx], n_neighbors=2)[0][:, 1]
    lo, hi = X.min(0), X.max(0)
    rand = rng.uniform(lo, hi, (n, X.shape[1]))
    w_d = nn.kneighbors(rand, n_neighbors=1)[0][:, 0]
    return w_d.sum() / (w_d.sum() + u_d.sum())


def embed_and_score(Z, label):
    n_nb = max(2, min(len(Z) - 1, round(NEIGHBORS_FRAC * len(Z))))
    e2 = umap.UMAP(n_components=2, n_neighbors=n_nb,
                   random_state=42, n_jobs=1).fit_transform(Z)
    print(f"\n{label}: n={len(Z)}, latent dim {Z.shape[1]}, n_neighbors {n_nb}")
    print(f"  Hopkins (latent) {hopkins(Z):.3f}   (embedding) {hopkins(e2):.3f}")
    best = (None, -1)
    for k in range(2, 9):
        s = silhouette_score(e2, KMeans(k, n_init=10, random_state=42).fit_predict(e2))
        if s > best[1]:
            best = (k, s)
        print(f"    k={k}  silhouette {s:+.3f}")
    print(f"  best: k={best[0]}, silhouette {best[1]:+.3f}")
    return e2, best


print("=" * 62)
Z_s, d_s = latents("/Volumes/X10 Pro/Movies/2025_09_17",
                   f"{CTRL}/fc_ae_sept17_via_m160_pipeline.pth")
e_s, best_s = embed_and_score(Z_s, "SEPT17 through the M160 pipeline")

Z_m, d_m = latents(M160F, ("/Volumes/X10 Pro/FungalProject_Outputs/"
                           "model_based_dense_tracking/2026_08_28_M160/fc_ae_3d_m160.pth"))
e_m, best_m = embed_and_score(Z_m, "M160 full cohort")

Z_f, d_f = latents(M160F, ("/Volumes/X10 Pro/FungalProject_Outputs/"
                           "model_based_dense_tracking/2026_08_28_M160/"
                           "fc_ae_3d_m160_FL1.pth"), film_filter="FL1_")
e_f, best_f = embed_and_score(Z_f, "M160 FL1 only")

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
for a, e, best, ttl, n in (
        (ax[0], e_s, best_s, "Sept17 via M160 pipeline", len(Z_s)),
        (ax[1], e_m, best_m, "M160 full cohort", len(Z_m)),
        (ax[2], e_f, best_f, "M160 FL1 only", len(Z_f))):
    a.scatter(e[:, 0], e[:, 1], s=6 if n > 2000 else 14, c="#2563eb",
              alpha=0.45, rasterized=True)
    a.set_title(f"{ttl}\nn={n}, best k={best[0]}, silhouette {best[1]:+.3f}",
                fontsize=10)
    a.set_xlabel("UMAP 1")
    a.grid(alpha=0.25)
ax[0].set_ylabel("UMAP 2")
fig.suptitle("Same pipeline, same UMAP settings — does Sept17's clustering survive?")
fig.tight_layout()
out = f"{CTRL}/cluster_comparison.png"
fig.savefig(out, dpi=150)
print(f"\nplot -> {out}")
