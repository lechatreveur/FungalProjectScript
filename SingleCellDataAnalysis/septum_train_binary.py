#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 12:32:20 2026

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binary endpoint presence training (start/end separately) with variable-length crops.

Targets (per sampled window):
  y_start = 1 if start_idx exists AND falls inside crop window; else 0
  y_end   = 1 if end_idx exists AND falls inside crop window;   else 0

Model outputs:
  start_t, end_t: (B,L) per-tile logits (masked)
  start_win, end_win: (B,) pooled window logits (masked max pooling)
"""

import os
import math
import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)



# =========================
# utils: device / seeding
# =========================
def pick_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int = 0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================
# data: manifest split
# =========================
def load_manifests(working_dirs: list[str]) -> pd.DataFrame:
    dfs = []
    for d in working_dirs:
        fp = os.path.join(d, "training_dataset", "manifest.csv")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"manifest.csv not found: {fp}")
        df = pd.read_csv(fp)
        if df.empty:
            raise ValueError(f"manifest.csv is empty: {fp}")
        df["_source_dir"] = d
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic split by film_name + cell_id (same as your previous approach)."""
    film = df["film_name"].astype(str)
    cell = pd.to_numeric(df["cell_id"], errors="coerce").fillna(-1).astype(int)
    key = (film + "__" + cell.astype(str)).apply(lambda s: abs(hash(s)) % 100)
    tr = df[key < 85].reset_index(drop=True)
    va = df[key >= 85].reset_index(drop=True)
    return tr, va


# =========================
# data: dataset with random crops
# =========================
# Median septum duration observed from the manifest — used to clamp partial labels
_DEFAULT_SEP_DURATION = 62

class SeptumWindowDataset(Dataset):
    """
    Loads full strip from NPZ, then samples a random crop with random window length.

    Returns:
      x: (Lw,1,H,W) float32 in [0,1]
      y_start: float {0,1} (start endpoint inside the crop?)
      y_end:   float {0,1} (end endpoint inside the crop?)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        L_min: int = 16,
        L_max: int = 81,
        include_pos_prob: float = 0.7,
        seed: int = 0,
    ):
        self.df = df.reset_index(drop=True)
        self.L_min = int(L_min)
        self.L_max = int(L_max)
        self.include_pos_prob = float(include_pos_prob)
        self.rng = np.random.default_rng(seed)

        # sanity
        if self.L_min <= 0 or self.L_max <= 0 or self.L_min > self.L_max:
            raise ValueError("Require 0 < L_min <= L_max")

    def __len__(self):
        return len(self.df)

    def _load_npz(self, npz_fp: str):
        with np.load(npz_fp, allow_pickle=True) as z:
            strip = np.asarray(z["strip"], dtype=np.uint8)  # (H, H*L)
            H = int(strip.shape[0])
            if strip.ndim != 2 or H <= 0 or (strip.shape[1] % H) != 0:
                raise ValueError(f"Bad strip shape in {npz_fp}: {strip.shape}")
            L = int(strip.shape[1] // H)

            start_idx = int(z["start_idx"][0])
            end_idx = int(z["end_idx"][0])

        tiles = strip.reshape(H, L, H).transpose(1, 0, 2)[:, None, :, :]  # (L,1,H,H)
        x_full = tiles.astype(np.float32) / 255.0
        return x_full, start_idx, end_idx  # x_full: (L,1,H,W)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        npz_fp = os.path.join(str(row["_source_dir"]), str(row["npz_path"]))
        if not os.path.isfile(npz_fp):
            raise FileNotFoundError(f"NPZ not found: {npz_fp}")

        x_full, s_idx, e_idx = self._load_npz(npz_fp)
        Lfull = x_full.shape[0]

        # pick window length
        Lw_hi = min(self.L_max, Lfull)
        Lw = int(self.rng.integers(self.L_min, Lw_hi + 1))

        # helper: choose crop start so that idx is inside window (if feasible)
        def pick_j_include(idx: int) -> int:
            lo = max(0, idx - (Lw - 1))
            hi = min(idx, Lfull - Lw)
            if hi < lo:
                return int(self.rng.integers(0, Lfull - Lw + 1))
            return int(self.rng.integers(lo, hi + 1))

        has_any = (s_idx >= 0) or (e_idx >= 0)

        if has_any and (self.rng.random() < self.include_pos_prob):
            candidates = [idx for idx in (s_idx, e_idx) if idx >= 0]
            target = int(self.rng.choice(candidates))
            j = pick_j_include(target)
        else:
            j = int(self.rng.integers(0, Lfull - Lw + 1))

        x = x_full[j : j + Lw].copy()  # (Lw,1,H,W)

        # --- Augmentations ---
        # Horizontal (left-right) flip
        if self.rng.random() < 0.5:
            x = x[:, :, :, ::-1].copy()
            
        # Vertical (top-bottom) flip
        if self.rng.random() < 0.5:
            x = x[:, :, ::-1, :].copy()
            
        # Contrast jitter ±20%
        if self.rng.random() < 0.5:
            contrast = float(self.rng.uniform(0.8, 1.2))
            mean_px = np.mean(x)
            x = np.clip((x - mean_px) * contrast + mean_px, 0.0, 1.0).astype(np.float32)

        # Brightness jitter ±20%
        jitter = float(self.rng.uniform(-0.2, 0.2))
        x = np.clip(x + jitter, 0.0, 1.0).astype(np.float32)

        # Per-cell polarity inversion for white-septum cells (GUI-labeled).
        # Deterministic (not random) — only inverts cells the user explicitly flagged
        # via the 'i' key in the alignment board GUI. This avoids the model-collapse
        # caused by random 50% inversion on small datasets.
        if bool(row.get("white_septum", False)):
            x = (1.0 - x).astype(np.float32)


        # Fix 1: Create dense state mask with partial-label clamping
        # Cells with only one boundary get clamped to _DEFAULT_SEP_DURATION
        # to prevent garbage signal from unlabeled regions
        y_state_full = np.zeros(Lfull, dtype=np.float32)
        if s_idx >= 0 or e_idx >= 0:
            if s_idx >= 0 and e_idx >= 0:
                # Both bounds known — use them directly
                e_clamp = e_idx
            elif s_idx >= 0:
                # Start only — clamp end
                e_clamp = min(s_idx + _DEFAULT_SEP_DURATION, Lfull - 1)
            else:
                # End only — clamp start
                s_idx = max(0, e_idx - _DEFAULT_SEP_DURATION)
                e_clamp = e_idx
            for ii in range(s_idx, min(e_clamp + 1, Lfull)):
                y_state_full[ii] = 1.0

        y_state = y_state_full[j : j + Lw]

        return {
            "x": torch.from_numpy(x),  # (Lw,1,H,W)
            "y_state": torch.from_numpy(y_state),  # (Lw,)
        }


def collate_pad(batch):
    """Pad variable-length windows to a batch tensor + mask, interpolate spatial H/W differences."""
    import torch.nn.functional as F
    B = len(batch)
    Ls = [b["x"].shape[0] for b in batch]
    Lmax = max(Ls)
    
    Hs = [b["x"].shape[2] for b in batch]
    Ws = [b["x"].shape[3] for b in batch]
    Hmax = max(Hs)
    Wmax = max(Ws)

    x = torch.zeros((B, Lmax, 1, Hmax, Wmax), dtype=torch.float32)
    mask = torch.zeros((B, Lmax), dtype=torch.float32)

    y_state = torch.zeros((B, Lmax), dtype=torch.float32)  # (B, Lmax)

    for i, b in enumerate(batch):
        L = b["x"].shape[0]
        # Resize spatial dimensions if they don't match Hmax/Wmax
        x_i = b["x"]
        if x_i.shape[2] != Hmax or x_i.shape[3] != Wmax:
            x_i = F.interpolate(x_i, size=(Hmax, Wmax), mode="bilinear", align_corners=False)
        x[i, :L] = x_i
        mask[i, :L] = 1.0
        y_state[i, :L] = b["y_state"]

    return {"x": x, "mask": mask, "y_state": y_state}


# =========================
# model: per-tile logits + MIL pooling
# =========================
class TileEncoder(nn.Module):
    def __init__(self, D: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 48x48
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 24x24
            nn.Conv2d(32, D, 3, padding=1),
            nn.BatchNorm2d(D),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1))
        )

    def forward(self, x):
        return self.net(x)[:, :, 0, 0]  # (B, D)


class EndpointMIL(nn.Module):
    """
    Outputs:
      state_t: (B,L) masked logits
    """

    def __init__(self, D: int = 64):
        super().__init__()
        self.enc = TileEncoder(D=D)
        self.temporal = nn.Sequential(
            nn.Conv1d(D, D, 3, padding=1),
            nn.BatchNorm1d(D),
            nn.ReLU(),
            nn.Conv1d(D, D, 3, padding=1),
            nn.BatchNorm1d(D),
            nn.ReLU(),
        )
        self.head_state = nn.Conv1d(D, 1, 1)

    def forward(self, x, mask):
        # x: (B,L,1,H,W), mask: (B,L)
        B, L, _, H, W = x.shape
        emb = self.enc(x.reshape(B * L, 1, H, W)).reshape(B, L, -1)  # (B,L,D)
        feat = self.temporal(emb.transpose(1, 2))  # (B,D,L)

        state_t = self.head_state(feat)[:, 0, :]  # (B,L)

        neg_inf = torch.finfo(state_t.dtype).min
        state_t = state_t.masked_fill(mask == 0, neg_inf)

        return state_t


# =========================
# loss / metrics
# =========================
def compute_pos_weight_from_manifest(train_df: pd.DataFrame) -> tuple[float]:
    """pos_weight = neg/pos for BCEWithLogitsLoss."""
    s = pd.to_numeric(train_df["start_idx"], errors="coerce").fillna(-1).astype(int).values
    e = pd.to_numeric(train_df["end_idx"], errors="coerce").fillna(-1).astype(int).values

    s_pos = int((s >= 0).sum())
    n = len(train_df)
    s_neg = n - s_pos

    s_w = float(s_neg / max(1, s_pos))
    return (s_w,)


@torch.no_grad()
def eval_epoch(model, dl, device, bce_state, thresh=0.5):
    model.eval()
    loss_sum = 0.0
    n = 0
    acc_state = 0

    for batch in tqdm(dl, desc="Validating", ncols=100, leave=False):
        x = batch["x"].to(device)
        mask = batch["mask"].to(device)
        y_state = batch["y_state"].to(device)

        state_t = model(x, mask)
        
        # Flatten state_t and y_state using mask
        valid_idx = (mask == 1.0)
        state_t_flat = state_t[valid_idx]
        y_state_flat = y_state[valid_idx]
        
        loss = bce_state(state_t_flat, y_state_flat)

        ps = (torch.sigmoid(state_t_flat) > thresh).float()

        b = x.size(0)
        loss_sum += float(loss.item()) * b
        n += b
        acc_state += int((ps == y_state_flat).sum().item()) / max(1, len(y_state_flat))

    return loss_sum / max(1, n), acc_state / max(1, n)


# =========================
# training
# =========================
def make_balanced_sampler(train_df: pd.DataFrame, which: str = "any"):
    """
    Balanced sampling between positives/negatives.
    which:
      - "any": positive if (start_idx>=0 OR end_idx>=0)
      - "start": positive if start_idx>=0
      - "end": positive if end_idx>=0
    """
    s = pd.to_numeric(train_df["start_idx"], errors="coerce").fillna(-1).astype(int).values
    e = pd.to_numeric(train_df["end_idx"], errors="coerce").fillna(-1).astype(int).values

    if which == "any":
        is_pos = (s >= 0) | (e >= 0)
    elif which == "start":
        is_pos = (s >= 0)
    elif which == "end":
        is_pos = (e >= 0)
    else:
        raise ValueError("which must be any/start/end")

    n_pos = int(is_pos.sum())
    n_neg = int((~is_pos).sum())
    if n_pos == 0 or n_neg == 0:
        weights = np.ones(len(train_df), dtype=np.float32)
    else:
        w_pos = 0.5 / n_pos
        w_neg = 0.5 / n_neg
        weights = np.where(is_pos, w_pos, w_neg).astype(np.float32)

    return WeightedRandomSampler(torch.from_numpy(weights), num_samples=len(train_df), replacement=True)


def train(
    working_dirs: list[str],
    *,
    D: int = 64,
    batch_size: int = 16,
    epochs: int = 20,
    lr: float = 1e-3,
    num_workers: int = 2,
    device: str | None = None,
    seed: int = 0,
    # variable window training
    L_min: int = 16,
    L_max: int = 81,
    include_pos_prob: float = 0.7,
    # balancing
    balanced_sampling: bool = True,
    resume_from: str = None,
):
    device = pick_device(device)
    seed_everything(seed)

    if isinstance(working_dirs, str):
        working_dirs = [working_dirs]
    df = load_manifests(working_dirs)
    tr_df, va_df = split_train_val(df)

    # datasets
    ds_tr = SeptumWindowDataset(
        tr_df,
        L_min=L_min,
        L_max=L_max,
        include_pos_prob=include_pos_prob,
        seed=seed,
    )
    ds_va = SeptumWindowDataset(
        va_df,
        L_min=L_min,
        L_max=L_max,
        include_pos_prob=0.0,  # unbiased val crops
        seed=seed + 1,
    )

    # loaders
    if balanced_sampling:
        sampler = make_balanced_sampler(tr_df, which="any")
        dl_tr = DataLoader(
            ds_tr,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_pad,
            pin_memory=(device == "cuda"),
        )
    else:
        dl_tr = DataLoader(
            ds_tr,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_pad,
            pin_memory=(device == "cuda"),
        )

    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pad,
        pin_memory=(device == "cuda"),
    )

    # loss weights (computed from TRAIN SPLIT manifest)
    (s_w,) = compute_pos_weight_from_manifest(tr_df)
    logger.info(f"[pos_weight] state={s_w:.3f} (neg/pos on train split)")

    bce_state = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([s_w], device=device))

    # model/opt
    model = EndpointMIL(D=D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    start_epoch = 1
    if resume_from and os.path.isfile(resume_from):
        logger.info(f"Resuming from checkpoint: {resume_from}")
        chkpt = torch.load(resume_from, map_location=device, weights_only=True)
        model.load_state_dict(chkpt["state_dict"])
        import re
        m = re.search(r"model_ep(\d+)\.pt", resume_from)
        if m:
            start_epoch = int(m.group(1)) + 1

    ckpt_dir = os.path.join(working_dirs[0], "training_dataset", "checkpoints_binary")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Fix 4: Track best val_loss for early-stopping checkpoint
    best_val_loss = float("inf")

    for ep in range(start_epoch, epochs + 1):
        model.train()
        running = 0.0
        n = 0

        for batch in tqdm(dl_tr, desc=f"Epoch {ep:03d}/{epochs}", ncols=100):
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y_state = batch["y_state"].to(device)

            state_t = model(x, mask)

            # Flatten state_t and y_state using mask
            valid_idx = (mask == 1.0)
            state_t_flat = state_t[valid_idx]
            y_state_flat = y_state[valid_idx]

            loss = bce_state(state_t_flat, y_state_flat)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            b = x.size(0)
            running += float(loss.item()) * b
            n += b

        tr_loss = running / max(1, n)
        va_loss, va_acc_state = eval_epoch(model, dl_va, device, bce_state)

        # Fix 4: Save best checkpoint whenever val_loss improves
        tag = ""
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(
                {"state_dict": model.state_dict(), "D": D},
                os.path.join(ckpt_dir, "model_best.pt"),
            )
            tag = "  [best]"

        logger.info(
            f"[ep {ep:03d}] train_loss={tr_loss:.4f} "
            f"val_loss={va_loss:.4f} val_acc_state={va_acc_state:.3f}{tag}"
        )

        torch.save(
            {"state_dict": model.state_dict(), "D": D},
            os.path.join(ckpt_dir, f"model_ep{ep:03d}.pt"),
        )

    # Always save final epoch too (for resume)
    torch.save({"state_dict": model.state_dict(), "D": D}, os.path.join(ckpt_dir, "model_latest.pt"))
    logger.info(f"Best val_loss={best_val_loss:.4f} → saved as model_best.pt")
    return model


# =========================
# inference helpers (for sliding-window usage)
# =========================
@torch.no_grad()
def score_full_strip(model: EndpointMIL, x_full: torch.Tensor, device: str):
    """
    Score a full strip in one pass (fast):
      x_full: (L,1,H,W) float32 in [0,1]
    Returns:
      state_prob_t: (L,) per-tile probabilities
    """
    model.eval()
    L = x_full.shape[0]
    x = x_full[None, ...].to(device)  # (1,L,1,H,W)
    mask = torch.ones((1, L), device=device)

    state_t = model(x, mask)
    state_prob_t = torch.sigmoid(state_t[0]).detach().cpu()
    return state_prob_t


@torch.no_grad()
def sliding_window_scores(model: EndpointMIL, x_full: torch.Tensor, window: int = 81, stride: int = 1, device: str = "cpu"):
    """
    Literal sliding window: returns per-window probabilities.
      x_full: (L,1,H,W)
    Returns:
      state_win_probs: (num_windows,)
    """
    model.eval()
    L = x_full.shape[0]
    outs_state = []
    for j in range(0, L - window + 1, stride):
        x = x_full[j:j+window][None, ...].to(device)  # (1,window,1,H,W)
        mask = torch.ones((1, window), device=device)
        state_t = model(x, mask)
        # Using the max state in the window just for legacy sliding window scoring
        outs_state.append(torch.sigmoid(state_t).max().item())
    return np.array(outs_state)


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("working_dirs", nargs="+", help="One or more working directories")
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--L_min", type=int, default=16)
    ap.add_argument("--L_max", type=int, default=81)
    ap.add_argument("--include_pos_prob", type=float, default=0.7)
    ap.add_argument("--no_balanced_sampling", action="store_true")
    ap.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint .pt file to resume from")

    args = ap.parse_args()

    train(
        args.working_dirs,
        D=args.D,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        num_workers=args.num_workers,
        device=args.device,
        seed=args.seed,
        L_min=args.L_min,
        L_max=args.L_max,
        include_pos_prob=args.include_pos_prob,
        balanced_sampling=(not args.no_balanced_sampling),
        resume_from=args.resume_from,
    )