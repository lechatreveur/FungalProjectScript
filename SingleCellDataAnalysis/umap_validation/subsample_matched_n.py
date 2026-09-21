"""Is Sept17's clustering real, or a small-sample effect?

Subsample M160 to Sept17's n=378 and run identically — same latent space, same
UMAP settings, same n_neighbors=15. Ten independent draws, so the answer is a
distribution rather than one lucky split.

If M160 stays continuous at matched n, the difference from Sept17 is real.
If M160 fragments too, then some of Sept17's structure is sparsity.
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

CTRL = "/Volumes/X10 Pro/FungalProject_Outputs/umap_control"
BASE = ("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking/"
        "2026_08_28_M160")
N_REF, NB_REF, DRAWS = 378, 15, 10


def latents(features_dir, model_path, film_filter=None):
    X_traj, X_feat, gids, *_ = load_feature_constrained_data({"E": str(features_dir)})
    if film_filter:
        k = np.array([film_filter in g for g in gids], bool)
        X_traj, X_feat = X_traj[k], X_feat[k]
    sd = torch.load(model_path, map_location="cpu")
    dim = next(v.shape[0] for k, v in sd.items()
               if k.startswith("encoder_fc") and k.endswith("weight")
               and v.ndim == 2 and v.shape[1] == 128)
    m = MultimodalAutoencoder3D(latent_dim=int(dim)); m.load_state_dict(sd); m.eval()
    with torch.no_grad():
        _, _, z = m(torch.from_numpy(X_traj).float(), torch.from_numpy(X_feat).float())
    return z.numpy()


def hopkins(X, seed=0):
    rng = np.random.default_rng(seed)
    n = min(100, len(X) // 4)
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    idx = rng.choice(len(X), n, replace=False)
    u = nn.kneighbors(X[idx], n_neighbors=2)[0][:, 1]
    r = rng.uniform(X.min(0), X.max(0), (n, X.shape[1]))
    w = nn.kneighbors(r, n_neighbors=1)[0][:, 0]
    return w.sum() / (w.sum() + u.sum())


def run(Z, seed):
    e = umap.UMAP(n_components=2, n_neighbors=NB_REF,
                  random_state=42, n_jobs=1).fit_transform(Z)
    best = max((silhouette_score(e, KMeans(k, n_init=10, random_state=42).fit_predict(e)), k)
               for k in range(2, 9))
    return e, best[0], best[1], hopkins(e, seed)


Z_s = latents("/Volumes/X10 Pro/Movies/2025_09_17",
              f"{CTRL}/fc_ae_sept17_via_m160_pipeline.pth")
e_s, sil_s, k_s, hop_s = run(Z_s, 0)
print(f"SEPT17 (n={len(Z_s)}): silhouette {sil_s:+.3f} at k={k_s}, "
      f"Hopkins {hop_s:.3f}")

for label, Z, filt in (("M160 full", latents(f"{BASE}/features",
                                             f"{BASE}/fc_ae_3d_m160.pth"), None),
                       ("M160 FL1", latents(f"{BASE}/features",
                                            f"{BASE}/fc_ae_3d_m160_FL1.pth", "FL1_"), "FL1_")):
    rng = np.random.default_rng(0)
    sils, hops, keep = [], [], []
    for d in range(DRAWS):
        idx = rng.choice(len(Z), N_REF, replace=False)
        e, sil, k, hop = run(Z[idx], d)
        sils.append(sil); hops.append(hop)
        if d < 3:
            keep.append((e, sil, k))
    sils, hops = np.array(sils), np.array(hops)
    print(f"\n{label} subsampled to n={N_REF}, {DRAWS} draws:")
    print(f"  silhouette {sils.mean():+.3f} +/- {sils.std():.3f}  "
          f"(range {sils.min():+.3f} to {sils.max():+.3f})")
    print(f"  Hopkins    {hops.mean():.3f} +/- {hops.std():.3f}")
    z = (sil_s - sils.mean()) / (sils.std() + 1e-9)
    print(f"  Sept17 sits {z:+.1f} sd from this distribution")
    if label == "M160 full":
        M160_keep, M160_sils = keep, sils

fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].scatter(e_s[:, 0], e_s[:, 1], s=14, c="#16a34a", alpha=0.5)
ax[0].set_title(f"Sept17 (n=378)\nsilhouette {sil_s:+.3f}, k={k_s}", fontsize=10)
for i, (e, sil, k) in enumerate(M160_keep):
    ax[i + 1].scatter(e[:, 0], e[:, 1], s=14, c="#2563eb", alpha=0.5)
    ax[i + 1].set_title(f"M160 subsample {i+1} (n=378)\nsilhouette {sil:+.3f}, k={k}",
                        fontsize=10)
for a in ax:
    a.grid(alpha=0.25); a.set_xlabel("UMAP 1")
ax[0].set_ylabel("UMAP 2")
fig.suptitle("Matched sample size: is Sept17's separation real or sparsity?")
fig.tight_layout()
out = f"{CTRL}/subsample_matched_n.png"
fig.savefig(out, dpi=150)
print(f"\nplot -> {out}")
