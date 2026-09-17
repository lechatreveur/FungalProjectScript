#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train an M160 multimodal autoencoder (stage 6 input).

The UMAP explorer is fit on autoencoder **latents**, not on the engineered
features — the autoencoder is what folds the 101-frame Pol1/Pol2 trajectory
together with the eleven features into one vector. Fitting UMAP on the features
alone discards the trajectory shape entirely.

The reference model (`fc_ae_3d_final.pth`) is trained on one experiment, Sept17.
Training a separate model on M160 gives a genuinely **standalone** manifold;
projecting M160 through the Sept17 model would instead put it on the reference
manifold, which is the other option and is not what was asked for.

Reuses the canonical model and loader unmodified (P15):
    SingleCellDataAnalysis.FC_AE_3d_train.MultimodalAutoencoder3D
    SingleCellDataAnalysis.FC_AE_data_loader.load_feature_constrained_data

The loader is pointed at the M160 features directory, which carries the layout
it expects: `unaligned_pairs_quant/` holding the stacked traces, the model fits
and the autocorrelation results. Because that stacked file has `global_cell_id`
and `source` columns, the loader builds its ids as
`M160_<global_cell_id>_<film>`, matching the reference's experiment + cell +
source convention.

Reproducibility (P10): seeds fixed, every parameter and the resulting loss
written to a sidecar next to the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
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
from torch.utils.data import TensorDataset, DataLoader

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
for _p in (str(_HERE), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from SingleCellDataAnalysis.FC_AE_3d_train import MultimodalAutoencoder3D
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

EXP_NAME = "2026_08_28_M160"
_SSD_OUT = Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking") / EXP_NAME
DEFAULT_FEATURES_DIR = _SSD_OUT / "features"
DEFAULT_MODEL = _SSD_OUT / "fc_ae_3d_m160.pth"

EPOCHS = 300
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
ALPHA = 1.0          # weight on feature reconstruction, as in the reference
SEED = 42
# Chosen by sweep_fc_ae_m160.py on 6,184 M160 datapoints: the elbow sits at
# 5-6, the trajectory term is flat past 5 (marginal gains 3.5%, 3.0%, then
# noise), and beyond that the extra capacity goes into reconstructing the
# eleven engineered features rather than the dynamics. 6 also gives UMAP a
# real space to reduce, instead of the 3-to-3 near-identity it had before.
LATENT_DIM = 6
MPS_CAP_GIB = 10.0   # P4: cap MPS on an 18 GB workstation, ~75% of the device limit


def pick_device(force_cpu=False):
    """This model is small (a few hundred thousand parameters over ~600 traces),
    so CPU is fast enough and avoids the MPS unified-memory blow-up P4 warns
    about. MPS is used only if asked for, and then capped."""
    if force_cpu or not torch.backends.mps.is_available():
        return torch.device("cpu")
    try:
        rec = torch.mps.recommended_max_memory() / (1024 ** 3)
        frac = min(0.75, MPS_CAP_GIB / rec) if rec > 0 else 0.5
        torch.mps.set_per_process_memory_fraction(float(frac))
        print(f"MPS capped at {frac:.2f} of {rec:.1f} GiB recommended (P4)", flush=True)
    except Exception as exc:
        print(f"could not cap MPS ({exc}); falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device("mps")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--mps", action="store_true", help="use MPS instead of CPU")
    ap.add_argument("--latent-dim", type=int, default=LATENT_DIM)
    ap.add_argument("--keep-division-films", action="store_true",
                    help="train on the dividing films too (default: exclude them)")
    a = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("loading M160 trajectories and features ...", flush=True)
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(
        {"M160": str(a.features_dir)})
    print(f"datapoints: {len(gids)}   trajectory {X_traj.shape}   features {X_feat.shape}",
          flush=True)

    # Drop the film in which the cell divides. Its Pol1/Pol2 traces are
    # dominated by the mother-to-daughter step — area halves, the septum and the
    # second nucleus vanish — so the trajectory describes a tracking transition
    # rather than polarity dynamics, and it is noise for a model meant to learn
    # the latter. The division film is still identified and kept in the feature
    # table; it is excluded only from TRAINING.
    if not a.keep_division_films:
        feats = pd.read_csv(Path(a.features_dir) / "umap_features_m160.csv")
        if "is_division_film" in feats.columns:
            drop = {f"M160_{r.global_cell_id}_{r.film}"
                    for _, r in feats[feats.is_division_film.fillna(False)].iterrows()}
            keep = np.array([g not in drop for g in gids], bool)
            n_drop = int((~keep).sum())
            X_traj, X_feat = X_traj[keep], X_feat[keep]
            gids = [g for g, k in zip(gids, keep) if k]
            labels = [l for l, k in zip(labels, keep) if k]
            print(f"excluded division films: {n_drop}  -> training on {len(gids)}",
                  flush=True)
        else:
            print("  (no is_division_film column; nothing excluded)", flush=True)
    if len(gids) == 0:
        raise SystemExit("no datapoints survived the loader; check trace lengths are 101")

    ds = TensorDataset(torch.tensor(X_traj, dtype=torch.float32),
                       torch.tensor(X_feat, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True)

    device = pick_device(force_cpu=not a.mps)
    print(f"device: {device}", flush=True)

    model = MultimodalAutoencoder3D(latent_dim=a.latent_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    crit = nn.MSELoss()

    history = []
    for epoch in range(a.epochs):
        model.train()
        tot = rec = fea = 0.0
        for b_traj, b_feat in dl:
            b_traj, b_feat = b_traj.to(device), b_feat.to(device)
            opt.zero_grad()
            r_traj, r_feat, _ = model(b_traj, b_feat)
            l_traj = crit(r_traj, b_traj)
            l_feat = crit(r_feat, b_feat)
            loss = l_traj + ALPHA * l_feat
            loss.backward()
            opt.step()
            n = b_traj.size(0)
            tot += loss.item() * n
            rec += l_traj.item() * n
            fea += l_feat.item() * n
        row = dict(epoch=epoch + 1, loss=tot / len(ds),
                   traj=rec / len(ds), feat=fea / len(ds))
        history.append(row)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"epoch {row['epoch']:>4}/{a.epochs}  loss {row['loss']:.4f}  "
                  f"traj {row['traj']:.4f}  feat {row['feat']:.4f}", flush=True)

    a.model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), a.model)

    prov = dict(
        artifact=a.model.name,
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        created_by="SingleCellQuantificationHPC/train_fc_ae_m160.py",
        experiment=EXP_NAME, standalone=True,
        note="trained on M160 only; latents are not comparable with the Sept17 model",
        model=(f"MultimodalAutoencoder3D(seq_len=101, in_channels=2, "
               f"num_features=11, latent_dim={a.latent_dim})"),
        latent_dim=int(a.latent_dim),
        division_films_excluded=bool(not a.keep_division_films),
        features_dir=str(a.features_dir), n_datapoints=int(len(gids)),
        epochs=a.epochs, batch_size=a.batch_size, lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY, alpha=ALPHA, seed=SEED, device=str(device),
        final=history[-1] if history else None)
    (a.model.parent / (a.model.stem + "_provenance.json")).write_text(json.dumps(prov, indent=2))

    print(f"\nsaved {a.model}", flush=True)
    if history:
        print(f"final loss {history[-1]['loss']:.4f} "
              f"(traj {history[-1]['traj']:.4f}, feat {history[-1]['feat']:.4f})", flush=True)


if __name__ == "__main__":
    main()
