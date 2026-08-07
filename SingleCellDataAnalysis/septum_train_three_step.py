#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Three-Stage training pipeline for Septum AI model:
Step 1: Pretrain 2D TileEncoder to classify BF vs. GFP image tiles.
Step 2: Finetune TileEncoder to classify frame-level septum presence/absence.
Step 3: Train EndpointMIL temporal convolutions with monotonic dynamics constraints.
"""

import os
import math
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "training_three_step.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_LOG_FILE, mode="a", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Model Architecture
# ==============================================================================
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
        return self.net(x)[:, :, 0, 0]


class EndpointMIL(nn.Module):
    def __init__(self, D: int = 64):
        super().__init__()
        self.enc = TileEncoder(D=D)
        self.temporal = nn.Sequential(
            nn.Conv1d(D, D, 3, padding=1, padding_mode="replicate"),
            nn.BatchNorm1d(D),
            nn.ReLU(),
            nn.Conv1d(D, D, 3, padding=1, padding_mode="replicate"),
            nn.BatchNorm1d(D),
            nn.ReLU(),
        )
        self.head_state = nn.Conv1d(D, 1, 1)

    def forward(self, x, mask):
        B, L, _, H, W = x.shape
        emb = self.enc(x.reshape(B * L, 1, H, W)).reshape(B, L, -1)
        feat = self.temporal(emb.transpose(1, 2))

        state_t = self.head_state(feat)[:, 0, :]

        neg_inf = torch.finfo(state_t.dtype).min
        state_t = state_t.masked_fill(mask == 0, neg_inf)

        return state_t


class TileClassifier(nn.Module):
    def __init__(self, encoder, D: int = 64):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(D, 1)

    def forward(self, x):
        feat = self.encoder(x)
        return self.head(feat)[:, 0]


# ==============================================================================
# Helper Functions
# ==============================================================================
def pick_device(device: str | None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def split_train_val(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    film = df["film_name"].astype(str)
    cell = pd.to_numeric(df["cell_id"], errors="coerce").fillna(-1).astype(int)
    key = (film + "__" + cell.astype(str)).apply(lambda s: abs(hash(s)) % 100)
    tr = df[key < 85].reset_index(drop=True)
    va = df[key >= 85].reset_index(drop=True)
    return tr, va


# ==============================================================================
# Step 1: BF vs. GFP Tile Dataset & Training
# ==============================================================================
class BFvsGFPDataset(Dataset):
    def __init__(self, df: pd.DataFrame, working_dir: str, augment: bool = True):
        self.df = df
        self.working_dir = working_dir
        self.augment = augment
        self.samples = []

        logger.info(f"Step 1: Extracting frame-level tiles from {len(df)} tracks...")
        for idx, row in df.iterrows():
            npz_rel = str(row["npz_path"])
            npz_fp = os.path.join(working_dir, npz_rel)
            if not os.path.isfile(npz_fp):
                continue

            film_name = str(row["film_name"])
            is_bf = 1.0 if "BF" in film_name else 0.0

            with np.load(npz_fp, allow_pickle=True) as z:
                strip = np.asarray(z["strip"], dtype=np.uint8)  # (H, H*L)
                H = int(strip.shape[0])
                L = int(strip.shape[1] // H)

            tiles = strip.reshape(H, L, H).transpose(1, 0, 2)  # (L, H, H)
            
            for t in range(L):
                self.samples.append((tiles[t], is_bf))

        logger.info(f"Step 1: Total tiles extracted: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tile, label = self.samples[idx]
        x = tile.astype(np.float32) / 255.0
        x = x[None, :, :]  # (1, H, H)

        if self.augment:
            # Subtle augmentations
            if random.random() < 0.5:
                x = x[:, :, ::-1].copy()
            if random.random() < 0.5:
                x = x[:, ::-1, :].copy()

        return torch.from_numpy(x).float(), torch.tensor(label, dtype=torch.float32)


def pretrain_step1(tr_df, va_df, working_dir, device, D=64, epochs=10, batch_size=64):
    logger.info("--- STEP 1: BF vs. GFP Image Classification Pretraining ---")
    ds_tr = BFvsGFPDataset(tr_df, working_dir, augment=True)
    ds_va = BFvsGFPDataset(va_df, working_dir, augment=False)

    labels = [sample[1] for sample in ds_tr.samples]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    pos_weight = neg_count / max(1, pos_count)
    logger.info(f"Step 1 Class Balance: pos (BF)={pos_count}, neg (GFP)={neg_count}, pos_weight={pos_weight:.3f}")

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=2)

    encoder = TileEncoder(D=D)
    model = TileClassifier(encoder, D=D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_loss = float("inf")
    best_weights = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = bce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
            n_tr += x.size(0)

        # Validation
        model.eval()
        va_loss = 0.0
        n_va = 0
        acc_sum = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = bce(logits, y)
                va_loss += loss.item() * x.size(0)
                n_va += x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                acc_sum += (preds == y).sum().item()

        epoch_tr_loss = tr_loss / n_tr
        epoch_va_loss = va_loss / n_va
        epoch_acc = acc_sum / n_va
        logger.info(f"[Step 1 ep {ep:02d}] tr_loss={epoch_tr_loss:.4f} val_loss={epoch_va_loss:.4f} val_acc={epoch_acc:.2%}")

        if epoch_va_loss < best_loss:
            best_loss = epoch_va_loss
            best_weights = encoder.state_dict()

    logger.info("Step 1 completed successfully.")
    return best_weights


# ==============================================================================
# Step 2: Binary Frame-Level Septum Dataset & Training
# ==============================================================================
class FrameTileDataset(Dataset):
    def __init__(self, df: pd.DataFrame, working_dir: str, augment: bool = True):
        self.df = df
        self.working_dir = working_dir
        self.augment = augment
        self.samples = []

        logger.info(f"Step 2: Extracting septum presence tiles from {len(df)} tracks...")
        for idx, row in df.iterrows():
            npz_rel = str(row["npz_path"])
            npz_fp = os.path.join(working_dir, npz_rel)
            if not os.path.isfile(npz_fp):
                continue

            with np.load(npz_fp, allow_pickle=True) as z:
                strip = np.asarray(z["strip"], dtype=np.uint8)
                H = int(strip.shape[0])
                L = int(strip.shape[1] // H)
                start_idx = int(z["start_idx"][0])
                end_idx = int(z["end_idx"][0])

            tiles = strip.reshape(H, L, H).transpose(1, 0, 2)
            
            for t in range(L):
                is_pos = False
                if start_idx >= 0 or end_idx >= 0:
                    if start_idx >= 0 and end_idx >= 0:
                        is_pos = (start_idx <= t <= end_idx)
                    elif start_idx >= 0:
                        is_pos = (start_idx <= t <= min(start_idx + 62, L - 1))
                    else:
                        is_pos = (max(0, end_idx - 62) <= t <= end_idx)

                label = 1.0 if is_pos else 0.0
                self.samples.append((tiles[t], label))

        logger.info(f"Step 2: Total frame tiles extracted: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tile, label = self.samples[idx]
        x = tile.astype(np.float32) / 255.0
        x = x[None, :, :]

        if self.augment:
            if random.random() < 0.5:
                contrast = random.uniform(0.8, 1.2)
                mean_px = np.mean(x)
                x = np.clip((x - mean_px) * contrast + mean_px, 0.0, 1.0)
            if random.random() < 0.5:
                jitter = random.uniform(-0.2, 0.2)
                x = np.clip(x + jitter, 0.0, 1.0)
            if random.random() < 0.5:
                x = 1.0 - x

        return torch.from_numpy(x).float(), torch.tensor(label, dtype=torch.float32)


def train_step2(tr_df, va_df, working_dir, pretrained_encoder_weights, device, D=64, epochs=10, batch_size=64):
    logger.info("--- STEP 2: Frame-level Septum Classification ---")
    ds_tr = FrameTileDataset(tr_df, working_dir, augment=True)
    ds_va = FrameTileDataset(va_df, working_dir, augment=False)

    labels = [sample[1] for sample in ds_tr.samples]
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    pos_weight = neg_count / max(1, pos_count)
    logger.info(f"Step 2 Class Balance: pos (septum)={pos_count}, neg (no septum)={neg_count}, pos_weight={pos_weight:.3f}")

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=2)

    encoder = TileEncoder(D=D)
    if pretrained_encoder_weights is not None:
        logger.info("Initializing encoder with Step 1 BF vs. GFP pretrained weights...")
        encoder.load_state_dict(pretrained_encoder_weights)

    model = TileClassifier(encoder, D=D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4) # Slightly smaller LR for finetuning
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_loss = float("inf")
    best_weights = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = bce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * x.size(0)
            n_tr += x.size(0)

        # Validation
        model.eval()
        va_loss = 0.0
        n_va = 0
        acc_sum = 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = bce(logits, y)
                va_loss += loss.item() * x.size(0)
                n_va += x.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                acc_sum += (preds == y).sum().item()

        epoch_tr_loss = tr_loss / n_tr
        epoch_va_loss = va_loss / n_va
        epoch_acc = acc_sum / n_va
        logger.info(f"[Step 2 ep {ep:02d}] tr_loss={epoch_tr_loss:.4f} val_loss={epoch_va_loss:.4f} val_acc={epoch_acc:.2%}")

        if epoch_va_loss < best_loss:
            best_loss = epoch_va_loss
            best_weights = encoder.state_dict()

    logger.info("Step 2 completed successfully.")
    return best_weights


# ==============================================================================
# Step 3: Sequence dynamics with Monotonicity Constraint
# ==============================================================================
class SeptumWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, working_dir: str, L_min=16, L_max=81, include_pos_prob=0.7, seed=0, augment=True):
        self.df = df.reset_index(drop=True)
        self.working_dir = working_dir
        self.L_min = L_min
        self.L_max = L_max
        self.include_pos_prob = include_pos_prob
        self.rng = np.random.default_rng(seed)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        npz_rel = str(row["npz_path"])
        npz_fp = os.path.join(self.working_dir, npz_rel)

        with np.load(npz_fp, allow_pickle=True) as z:
            strip = np.asarray(z["strip"], dtype=np.uint8)
            H = int(strip.shape[0])
            L = int(strip.shape[1] // H)
            start_idx = int(z["start_idx"][0])
            end_idx = int(z["end_idx"][0])

        tiles = strip.reshape(H, L, H).transpose(1, 0, 2)[:, None, :, :]
        x_full = tiles.astype(np.float32) / 255.0

        Lfull = x_full.shape[0]
        Lw_hi = min(self.L_max, Lfull)
        Lw = int(self.rng.integers(self.L_min, Lw_hi + 1))

        def pick_j_include(idx: int) -> int:
            lo = max(0, idx - (Lw - 1))
            hi = min(idx, Lfull - Lw)
            if hi < lo:
                return int(self.rng.integers(0, Lfull - Lw + 1))
            return int(self.rng.integers(lo, hi + 1))

        has_any = (start_idx >= 0) or (end_idx >= 0)
        if has_any and (self.rng.random() < self.include_pos_prob):
            candidates = [idx for idx in (start_idx, end_idx) if idx >= 0]
            target = int(self.rng.choice(candidates))
            j = pick_j_include(target)
        else:
            j = int(self.rng.integers(0, Lfull - Lw + 1))

        x = x_full[j : j + Lw].copy()

        if self.augment:
            if self.rng.random() < 0.5:
                x = x[:, :, :, ::-1].copy()
            if self.rng.random() < 0.5:
                x = x[:, :, ::-1, :].copy()
            if self.rng.random() < 0.5:
                contrast = float(self.rng.uniform(0.8, 1.2))
                mean_px = np.mean(x)
                x = np.clip((x - mean_px) * contrast + mean_px, 0.0, 1.0).astype(np.float32)
            jitter = float(self.rng.uniform(-0.2, 0.2))
            x = np.clip(x + jitter, 0.0, 1.0).astype(np.float32)
            if self.rng.random() < 0.5:
                x = (1.0 - x).astype(np.float32)

        # Dense state mask
        y_state_full = np.zeros(Lfull, dtype=np.float32)
        if start_idx >= 0 or end_idx >= 0:
            if start_idx >= 0 and end_idx >= 0:
                e_clamp = end_idx
            elif start_idx >= 0:
                e_clamp = min(start_idx + 62, Lfull - 1)
            else:
                start_idx = max(0, end_idx - 62)
                e_clamp = end_idx
            for ii in range(start_idx, min(e_clamp + 1, Lfull)):
                y_state_full[ii] = 1.0

        y_state = y_state_full[j : j + Lw]

        # Map end_idx to cropped window
        if end_idx >= 0 and (j <= end_idx < j + Lw):
            local_end = end_idx - j
        else:
            local_end = -1

        return {
            "x": torch.from_numpy(x),
            "y_state": torch.from_numpy(y_state),
            "local_end": torch.tensor(local_end, dtype=torch.long),
        }


def collate_pad(batch):
    B = len(batch)
    Ls = [b["x"].shape[0] for b in batch]
    Lmax = max(Ls)

    _, _, H, W = batch[0]["x"].shape
    x = torch.zeros((B, Lmax, 1, H, W), dtype=torch.float32)
    mask = torch.zeros((B, Lmax), dtype=torch.float32)
    y_state = torch.zeros((B, Lmax), dtype=torch.float32)
    local_end = torch.zeros(B, dtype=torch.long)

    for i, b in enumerate(batch):
        L = b["x"].shape[0]
        x[i, :L] = b["x"]
        mask[i, :L] = 1.0
        y_state[i, :L] = b["y_state"]
        local_end[i] = b["local_end"]

    return {"x": x, "mask": mask, "y_state": y_state, "local_end": local_end}


def compute_pos_weight_from_manifest(train_df: pd.DataFrame) -> float:
    pos_frames = 0
    neg_frames = 0
    for idx, row in train_df.iterrows():
        s = int(row.get("start_idx", -1))
        e = int(row.get("end_idx", -1))
        L = int(row.get("L", 101))
        if s >= 0 or e >= 0:
            if s >= 0 and e >= 0:
                e_clamp = e
            elif s >= 0:
                e_clamp = min(s + 62, L - 1)
            else:
                s = max(0, e - 62)
                e_clamp = e
            p_count = max(0, min(e_clamp + 1, L) - s)
            pos_frames += p_count
            neg_frames += (L - p_count)
        else:
            neg_frames += L
    return float(neg_frames / max(1, pos_frames))


def train_step3(tr_df, va_df, working_dir, pretrained_encoder_weights, device, D=64, epochs=15, batch_size=16, lambda_mono=2.0):
    logger.info("--- STEP 3: Sequence-level Dynamics with Monotonic constraints ---")
    ds_tr = SeptumWindowDataset(tr_df, working_dir, augment=True)
    ds_va = SeptumWindowDataset(va_df, working_dir, augment=False)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_pad, num_workers=2)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=collate_pad, num_workers=2)

    model = EndpointMIL(D=D)
    if pretrained_encoder_weights is not None:
        logger.info("Loading pretrained TileEncoder weights from Step 2...")
        model.enc.load_state_dict(pretrained_encoder_weights)

    # Fine-tune spatial encoder with a smaller learning rate during sequence dynamics training
    logger.info("Fine-tuning TileEncoder parameters with a small learning rate...")
    for param in model.enc.parameters():
        param.requires_grad = True

    model = model.to(device)
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": 1e-5},
        {"params": model.temporal.parameters(), "lr": 1e-3},
        {"params": model.head_state.parameters(), "lr": 1e-3},
    ])

    pos_weight = compute_pos_weight_from_manifest(tr_df)
    logger.info(f"Step 3 pos_weight={pos_weight:.3f}, lambda_mono={lambda_mono}")
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device), reduction="none")

    ckpt_dir = os.path.join(working_dir, "training_dataset", "checkpoints_binary")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_loss = float("inf")

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_tr = 0
        for batch in dl_tr:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            y = batch["y_state"].to(device)
            local_end = batch["local_end"]

            state_t = model(x, mask)
            
            # 1. Base BCE Loss
            bce_loss_raw = bce(state_t, y)
            bce_loss = (bce_loss_raw * mask).sum() / mask.sum().clamp(min=1.0)

            # 2. Monotonic constraint loss
            mono_loss_list = []
            probs_batch = torch.sigmoid(state_t) # (B, L)
            
            for b in range(x.size(0)):
                E = int(local_end[b].item())
                if E >= 0:
                    L_seq = int(mask[b].sum().item())
                    probs = probs_batch[b, :L_seq]

                    # Before end: probability should non-decrease (p(t) <= p(t+1))
                    if E > 0:
                        penalty_before = F.relu(probs[:E] - probs[1:E+1])
                        mono_loss_list.append(penalty_before.mean())
                    
                    # After end: probability should non-increase (p(t) >= p(t+1))
                    if E < L_seq - 1:
                        penalty_after = F.relu(probs[E:L_seq-1] - probs[E+1:L_seq])
                        mono_loss_list.append(penalty_after.mean())

            if mono_loss_list:
                mono_loss = torch.stack(mono_loss_list).mean()
            else:
                mono_loss = torch.tensor(0.0, device=device)

            loss = bce_loss + lambda_mono * mono_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += loss.item() * x.size(0)
            n_tr += x.size(0)

        # Validation
        model.eval()
        va_loss = 0.0
        n_va = 0
        acc_sum = 0
        with torch.no_grad():
            for batch in dl_va:
                x = batch["x"].to(device)
                mask = batch["mask"].to(device)
                y = batch["y_state"].to(device)
                local_end = batch["local_end"]

                state_t = model(x, mask)
                bce_loss_raw = bce(state_t, y)
                bce_loss = (bce_loss_raw * mask).sum() / mask.sum().clamp(min=1.0)

                # Monotonic validation loss
                mono_loss_list = []
                probs_batch = torch.sigmoid(state_t)
                for b in range(x.size(0)):
                    E = int(local_end[b].item())
                    if E >= 0:
                        L_seq = int(mask[b].sum().item())
                        probs = probs_batch[b, :L_seq]
                        if E > 0:
                            penalty_before = F.relu(probs[:E] - probs[1:E+1])
                            mono_loss_list.append(penalty_before.mean())
                        if E < L_seq - 1:
                            penalty_after = F.relu(probs[E:L_seq-1] - probs[E+1:L_seq])
                            mono_loss_list.append(penalty_after.mean())

                if mono_loss_list:
                    mono_loss = torch.stack(mono_loss_list).mean()
                else:
                    mono_loss = torch.tensor(0.0, device=device)

                loss = bce_loss + lambda_mono * mono_loss
                va_loss += loss.item() * x.size(0)
                n_va += x.size(0)

                valid_idx = (mask == 1.0)
                preds = (torch.sigmoid(state_t[valid_idx]) >= 0.5).float()
                acc_sum += (preds == y[valid_idx]).sum().item() / max(1, valid_idx.sum().item())

        epoch_tr_loss = tr_loss / n_tr
        epoch_va_loss = va_loss / n_va
        epoch_acc = acc_sum / len(dl_va)
        logger.info(f"[Step 3 ep {ep:02d}] tr_loss={epoch_tr_loss:.4f} val_loss={epoch_va_loss:.4f} val_acc={epoch_acc:.2%}")

        if epoch_va_loss < best_loss:
            best_loss = epoch_va_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_best.pt"))
            logger.info("Saved new best checkpoint to model_best.pt")

        torch.save(model.state_dict(), os.path.join(ckpt_dir, "model_latest.pt"))

    logger.info("Step 3 completed successfully.")


# ==============================================================================
# Main Execution
# ==============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("working_dir")
    ap.add_argument("--epochs-step1", type=int, default=8, help="Epochs for Step 1 pretraining")
    ap.add_argument("--epochs-step2", type=int, default=8, help="Epochs for Step 2 pretraining")
    ap.add_argument("--epochs-step3", type=int, default=12, help="Epochs for Step 3 finetuning")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--lambda-mono", type=float, default=2.0)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    logger.info(f"Using device: {device}")

    # Load manifest
    manifest_fp = os.path.join(args.working_dir, "training_dataset", "manifest.csv")
    if not os.path.isfile(manifest_fp):
        raise FileNotFoundError(f"manifest.csv not found: {manifest_fp}")
    df = pd.read_csv(manifest_fp)

    # Train/Val split
    tr_df, va_df = split_train_val(df)
    logger.info(f"Tracks Split: train={len(tr_df)}, val={len(va_df)}")

    # Step 1
    best_weights_step1 = pretrain_step1(
        tr_df, va_df, args.working_dir, device, D=args.D, epochs=args.epochs_step1, batch_size=args.batch_size * 4
    )

    # Step 2
    best_weights_step2 = train_step2(
        tr_df, va_df, args.working_dir, best_weights_step1, device, D=args.D, epochs=args.epochs_step2, batch_size=args.batch_size * 2
    )

    # Step 3
    train_step3(
        tr_df, va_df, args.working_dir, best_weights_step2, device, D=args.D, epochs=args.epochs_step3, batch_size=args.batch_size, lambda_mono=args.lambda_mono
    )


if __name__ == "__main__":
    main()
