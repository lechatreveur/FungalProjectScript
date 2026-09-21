"""The n_neighbors sweep and the null calibration, on M160 FL1 only.

FL1 is the laser-stress control: the first fluorescent film of each position,
so the least-exposed cells, with their own autoencoder. It is ~1/7 the size of
the full cohort, which matters here — the whole point of the full-cohort sweep
was that n_neighbors did nothing, and a smaller n is exactly where a sparse
neighbour graph could start to bite.

Per P16 the sweep is reported with its nulls in the same run, because a
silhouette score on a UMAP embedding means nothing without the floor for that
sample size. Two nulls: each latent dimension independently permuted (kills
joint structure, keeps every marginal) and isotropic Gaussian. Plus separated
blobs as a positive control.

Writes two plots to the umap_control folder. Run from the repo root.
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

sys.path.insert(0, "/Users/user/Documents/Python_Scripts/FungalProjectScript")
from SingleCellDataAnalysis.FC_AE_3d_train import MultimodalAutoencoder3D
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

BASE = ("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking/"
        "2026_08_28_M160")
CTRL = "/Volumes/X10 Pro/FungalProject_Outputs/umap_control"
REF_FRAC = 15 / 378          # the Sept17 reference fraction, 3.97%
NBS = [5, 15, 30, 60, 125, 250, 500, 800]


def latents(model_path, film_filter=None):
    X_traj, X_feat, gids, *_ = load_feature_constrained_data({"M160": f"{BASE}/features"})
    if film_filter:
        keep = np.array([film_filter in g for g in gids], bool)
        X_traj, X_feat = X_traj[keep], X_feat[keep]
        print(f"film filter {film_filter!r}: {keep.sum()} of {len(keep)} datapoints")
    sd = torch.load(model_path, map_location="cpu")
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


Z = latents(f"{BASE}/fc_ae_3d_m160_FL1.pth", film_filter="FL1_")
n, d = Z.shape
ref_nb = max(2, round(REF_FRAC * n))
print(f"\nFL1 latents: {Z.shape}  (latent dim {d})")
print(f"fraction-matched n_neighbors at {100*REF_FRAC:.2f}% of n={n}: {ref_nb}\n")

# ---- part 1: the sweep -------------------------------------------------
nbs = [b for b in NBS if b < n]
res = []
print(f"{'n_nb':>6} {'% of n':>8} {'best k':>7} {'silhouette':>11}")
for nb in nbs:
    e = embed(Z, nb)
    sil, k = best_sil(e)
    res.append((nb, e, sil, k))
    print(f"{nb:>6} {100*nb/n:>7.2f}% {k:>7} {sil:>+11.3f}", flush=True)

cols = min(4, len(res))
rows = int(np.ceil(len(res) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.8 * rows), squeeze=False)
for ax, (nb, e, sil, k) in zip(axes.ravel(), res):
    ax.scatter(e[:, 0], e[:, 1], s=6, c="#c2410c", alpha=0.45, rasterized=True)
    mark = ""
    if nb == 15:
        mark = "  <- library default"
    if abs(nb - ref_nb) <= 2:
        mark = "  <- fraction-matched"
    ax.set_title(f"n_neighbors = {nb} ({100*nb/n:.2f}% of n){mark}\n"
                 f"silhouette {sil:+.3f} at k={k}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
for ax in axes.ravel()[len(res):]:
    ax.axis("off")
fig.suptitle(f"M160 FL1 only, n={n}: the same latents at {len(res)} neighbour counts",
             fontsize=13)
fig.tight_layout()
p1 = f"{CTRL}/m160_FL1_n_neighbors_sweep.png"
fig.savefig(p1, dpi=140)
print(f"\nplot -> {p1}")

# ---- part 2: the nulls, at this n --------------------------------------
rng = np.random.default_rng(0)
Z_shuf = np.column_stack([rng.permutation(Z[:, j]) for j in range(d)])
Z_gauss = rng.normal(size=(n, d))
c = rng.normal(size=(3, d)) * 6.0
Z_blobs = np.vstack([rng.normal(size=(n // 3, d)) + c[i] for i in range(3)])

sets = [("FL1 real", Z), ("shuffled dims", Z_shuf),
        ("isotropic gaussian", Z_gauss), ("3 true blobs", Z_blobs)]
NB_NULL = (15, ref_nb)
out = []
print(f"\n{'dataset':>20} {'n_nb':>5} {'k':>3} {'silhouette':>11}")
for label, X in sets:
    for nb in NB_NULL:
        e = embed(X, nb)
        sil, k = best_sil(e)
        out.append((label, nb, e, sil, k))
        print(f"{label:>20} {nb:>5} {k:>3} {sil:>+11.3f}", flush=True)

fig, axes = plt.subplots(2, 4, figsize=(20, 9.5))
for col, (label, _) in enumerate(sets):
    for row, nb in enumerate(NB_NULL):
        _, _, e, sil, k = next(r for r in out if r[0] == label and r[1] == nb)
        ax = axes[row, col]
        ax.scatter(e[:, 0], e[:, 1], s=6, c="#c2410c", alpha=0.45, rasterized=True)
        ax.set_title(f"{label}, n_neighbors={nb}\nsilhouette {sil:+.3f} at k={k}",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
fig.suptitle(f"M160 FL1 (n={n}, {d}D latent): does FL1 clear its own noise floor?",
             fontsize=13)
fig.tight_layout()
p2 = f"{CTRL}/m160_FL1_null_calibration.png"
fig.savefig(p2, dpi=140)
print(f"\nplot -> {p2}")
