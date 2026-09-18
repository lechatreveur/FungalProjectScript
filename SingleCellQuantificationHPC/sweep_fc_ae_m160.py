#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latent-dimension sweep for the M160 autoencoder — find the elbow.

Why this is not `FC_AE_dimension_sweep.py`
------------------------------------------
That script sweeps `FeatureConstrainedAutoencoder`, which takes the trajectory
alone and PREDICTS the eleven features. The model we actually use,
`MultimodalAutoencoder3D`, takes trajectory and features together and
reconstructs both. Sweeping the wrong architecture would tell us the elbow of a
model we do not run. Everything else follows that script: the same dimension
ladder, an 80/20 split, repeats to get an error bar, and the best validation
loss per run.

What the sweep is for
---------------------
The production model compresses 213 numbers — 101 frames x 2 poles, plus 11
features — into 3. UMAP is then fitted on those 3 dimensions to produce 3 and 2
components, which is barely a reduction at all: the manifold structure is almost
entirely the autoencoder's doing. If the elbow sits above 3, a wider latent
gives UMAP a real space to unfold and the map becomes UMAP's work rather than a
re-rendering of the bottleneck.

Reported per dimension: best validation total loss, and its trajectory and
feature terms, averaged over repeats with a standard deviation. The elbow is
also picked automatically by the maximum distance from the chord joining the
first and last point, which is the standard knee criterion; that is a hint, not
a verdict.

Division films are held out exactly as production training holds them out (P10:
the sweep must train the thing we ship).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from SingleCellDataAnalysis.FC_AE_3d_train import MultimodalAutoencoder3D
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

EXP_NAME = "2026_08_28_M160"
_SSD = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking") / EXP_NAME
DEFAULT_FEATURES_DIR = _SSD / "features"
DEFAULT_OUT = _SSD / "latent_sweep"

DIMS = [2, 3, 4, 5, 6, 8, 10, 12, 16]
REPEATS = 3
EPOCHS = 200
PATIENCE = 30          # stop a run that has not improved for this many epochs
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-5
ALPHA = 1.0
SEED = 42


def load(features_dir, keep_division, film_contains=None):
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(
        {"M160": str(features_dir)})
    if not keep_division:
        feats = pd.read_csv(Path(features_dir) / "umap_features_m160.csv")
        if "is_division_film" in feats.columns:
            drop = {f"M160_{r.global_cell_id}_{r.film}"
                    for _, r in feats[feats.is_division_film.fillna(False)].iterrows()}
            keep = np.array([g not in drop for g in gids], bool)
            print(f"excluded division films: {int((~keep).sum())}", flush=True)
            X_traj, X_feat = X_traj[keep], X_feat[keep]
            gids = [g for g, k in zip(gids, keep) if k]
    # Restrict to one acquisition block. The elbow is a property of the dataset,
    # not a constant: a smaller, more homogeneous set may support fewer
    # dimensions before the model starts fitting noise.
    if film_contains:
        keep = np.array([film_contains in g for g in gids], bool)
        print(f"film filter {film_contains!r}: {int(keep.sum())} of {len(gids)}",
              flush=True)
        X_traj, X_feat = X_traj[keep], X_feat[keep]
    return X_traj, X_feat


def one_run(ds, dim, seed, epochs, device):
    """Train at this latent dim, return the best validation losses."""
    g = torch.Generator().manual_seed(seed)
    n_val = len(ds) // 5
    tr, va = random_split(ds, [len(ds) - n_val, n_val], generator=g)
    dl_tr = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True)
    dl_va = DataLoader(va, batch_size=256, shuffle=False)

    torch.manual_seed(seed)
    model = MultimodalAutoencoder3D(latent_dim=dim).to(device)
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()

    best = dict(total=float("inf"), traj=float("inf"), feat=float("inf"), epoch=0)
    since = 0
    for ep in range(epochs):
        model.train()
        for b_traj, b_feat in dl_tr:
            b_traj, b_feat = b_traj.to(device), b_feat.to(device)
            opt.zero_grad()
            r_traj, r_feat, _ = model(b_traj, b_feat)
            (crit(r_traj, b_traj) + ALPHA * crit(r_feat, b_feat)).backward()
            opt.step()

        model.eval()
        tot = tj = ft = 0.0
        with torch.no_grad():
            for v_traj, v_feat in dl_va:
                v_traj, v_feat = v_traj.to(device), v_feat.to(device)
                p_traj, p_feat, _ = model(v_traj, v_feat)
                lt, lf = crit(p_traj, v_traj).item(), crit(p_feat, v_feat).item()
                n = v_traj.size(0)
                tot += (lt + ALPHA * lf) * n
                tj += lt * n
                ft += lf * n
        tot, tj, ft = tot / len(va), tj / len(va), ft / len(va)
        if tot < best["total"] - 1e-6:
            best = dict(total=tot, traj=tj, feat=ft, epoch=ep + 1)
            since = 0
        else:
            since += 1
            if since >= PATIENCE:
                break
    return best


def knee(dims, vals):
    """Maximum distance from the chord joining the first and last point."""
    x = np.asarray(dims, float)
    y = np.asarray(vals, float)
    if len(x) < 3:
        return None
    xn = (x - x[0]) / (x[-1] - x[0])
    yn = (y - y[0]) / (y[-1] - y[0]) if y[-1] != y[0] else y * 0
    # chord runs (0,0) -> (1,1) after normalisation; distance is |yn - xn|
    d = np.abs(yn - xn)
    return int(x[int(np.argmax(d))])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dims", type=int, nargs="+", default=DIMS)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--keep-division-films", action="store_true")
    ap.add_argument("--film-contains", default=None)
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    a.out.mkdir(parents=True, exist_ok=True)

    X_traj, X_feat = load(a.features_dir, a.keep_division_films,
                          a.film_contains)
    print(f"datapoints: {len(X_traj)}   trajectory {X_traj.shape}   "
          f"features {X_feat.shape}", flush=True)
    ds = TensorDataset(torch.tensor(X_traj, dtype=torch.float32),
                       torch.tensor(X_feat, dtype=torch.float32))
    device = torch.device("cpu")
    print(f"device: {device}   dims: {a.dims}   repeats: {a.repeats}   "
          f"epochs<= {a.epochs} (patience {PATIENCE})", flush=True)

    rows, t0 = [], time.time()
    for dim in a.dims:
        for r in range(a.repeats):
            t1 = time.time()
            best = one_run(ds, dim, SEED + r, a.epochs, device)
            rows.append(dict(latent_dim=dim, repeat=r, **best,
                             seconds=round(time.time() - t1, 1)))
            print(f"dim {dim:>3}  repeat {r+1}/{a.repeats}  "
                  f"val {best['total']:.4f} (traj {best['traj']:.4f}, "
                  f"feat {best['feat']:.4f})  best epoch {best['epoch']}  "
                  f"{(time.time()-t1)/60:.1f} min", flush=True)
            pd.DataFrame(rows).to_csv(a.out / "sweep_results.csv", index=False)

    df = pd.DataFrame(rows)
    agg = df.groupby("latent_dim").agg(
        total_mean=("total", "mean"), total_std=("total", "std"),
        traj_mean=("traj", "mean"), feat_mean=("feat", "mean"),
        epochs_mean=("epoch", "mean")).reset_index()
    agg.to_csv(a.out / "sweep_summary.csv", index=False)

    print()
    print(f"{'dim':>4} {'val total':>11} {'±sd':>8} {'traj':>9} {'feat':>9} {'epochs':>7}")
    for _, r in agg.iterrows():
        print(f"{int(r.latent_dim):>4} {r.total_mean:>11.4f} {r.total_std:>8.4f} "
              f"{r.traj_mean:>9.4f} {r.feat_mean:>9.4f} {r.epochs_mean:>7.0f}")

    k_tot = knee(agg.latent_dim.tolist(), agg.total_mean.tolist())
    k_traj = knee(agg.latent_dim.tolist(), agg.traj_mean.tolist())
    print()
    print(f"elbow (total loss)         : {k_tot}")
    print(f"elbow (trajectory term)    : {k_traj}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(agg.latent_dim, agg.total_mean, yerr=agg.total_std.fillna(0),
                    marker="o", color="#0f172a", capsize=4, label="total val loss")
        ax.plot(agg.latent_dim, agg.traj_mean, marker="s", ls="--",
                color="#ef4444", label="trajectory")
        ax.plot(agg.latent_dim, agg.feat_mean, marker="^", ls="-.",
                color="#3b82f6", label="features")
        if k_tot:
            ax.axvline(k_tot, color="#94a3b8", ls=":", lw=1.5)
            ax.annotate(f"elbow = {k_tot}", (k_tot, agg.total_mean.max()),
                        textcoords="offset points", xytext=(6, -6), color="#475569")
        ax.set_xlabel("latent dimensions")
        ax.set_ylabel("best validation MSE")
        ax.set_title(f"M160 MultimodalAutoencoder3D latent sweep "
                     f"({len(X_traj)} datapoints, {a.repeats} repeats)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(a.out / "latent_sweep.png", dpi=150)
        print(f"plot -> {a.out / 'latent_sweep.png'}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")

    (a.out / "_provenance.json").write_text(json.dumps(dict(
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        created_by="SingleCellQuantificationHPC/sweep_fc_ae_m160.py",
        experiment=EXP_NAME, model="MultimodalAutoencoder3D",
        dims=a.dims, repeats=a.repeats, epochs=a.epochs, patience=PATIENCE,
        batch_size=BATCH_SIZE, lr=LR, weight_decay=WEIGHT_DECAY, alpha=ALPHA,
        seed=SEED, n_datapoints=int(len(X_traj)),
        division_films_excluded=bool(not a.keep_division_films),
        film_filter=a.film_contains,
        elbow_total=k_tot, elbow_traj=k_traj,
        total_minutes=round((time.time() - t0) / 60, 1)), indent=2))
    print(f"\ntotal {(time.time() - t0)/60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
