#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_train_contrastive.py — Strategy C

Learns a cycle-aware latent manifold of fungal polarity dynamics.
Uses NT-Xent loss with three types of positive pairs:
  1. Augmentations: Same cell, different time-offsets and flips.
  2. Temporal adjacency: Cell i and Cell i (identity).
  3. Cycle transition: Dividing cell → Newborn cell (the "stitch").
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- Project Paths ---
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
DATA_PATH = os.path.join(BASE_DIR, "video_cache_32x112_padded.npy")
FEATURE_PATH = os.path.join(BASE_DIR, "cycle_stage_features.npy")
TRANSITION_PATH = os.path.join(BASE_DIR, "cycle_transition_pairs.npy")

# --- Hyperparameters ---
BATCH_SIZE = 16
LATENT_DIM = 128
TEMPERATURE = 0.1
LEARNING_RATE = 1e-4
EPOCHS = 200
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ==============================================================================
# 1. Dataset & Augmentations
# ==============================================================================

class FungalContrastiveDataset(Dataset):
    def __init__(self, data_path, transition_path):
        print(f"Loading video data from {data_path}...")
        self.data = np.load(data_path)  # (N, 101, 1, 32, 112)
        self.N = self.data.shape[0]
        
        print(f"Loading transition pairs from {transition_path}...")
        self.transitions = np.load(transition_path) # (M, 2) [div_idx, newborn_idx]
        
    def __len__(self):
        return self.N

    def augment(self, x):
        """
        Apply biological augmentations:
        - Random time shift (±10 frames)
        - Random vertical flip (Pol1/Pol2 swap)
        - Random horizontal flip (reflection)
        """
        # x is (101, 1, 32, 112)
        T = x.shape[0]
        
        # 1. Random Flip (Vertical = Polarity Flip)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2).copy()
            
        # 2. Random Flip (Horizontal = Symmetry)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=3).copy()
            
        # 3. Random Time Shift (roll and pad)
        shift = np.random.randint(-10, 11)
        if shift != 0:
            x = np.roll(x, shift, axis=0)
            if shift > 0:
                x[:shift] = 0
            else:
                x[shift:] = 0
                
        return x

    def __getitem__(self, idx):
        # View 1: Current cell with augmentation
        v1 = self.augment(self.data[idx])
        
        # View 2: Decide between "Self-Augmentation" or "Cycle-Stitch"
        if np.random.rand() > 0.8: # 20% chance to provide a cycle-transition partner
            # Find if this cell is a dividing cell in our list
            trans_matches = self.transitions[self.transitions[:, 0] == idx]
            if len(trans_matches) > 0:
                partner_idx = trans_matches[np.random.randint(len(trans_matches))][1]
                v2 = self.augment(self.data[partner_idx])
            else:
                v2 = self.augment(self.data[idx])
        else:
            v2 = self.augment(self.data[idx])
            
        return torch.from_numpy(v1).float(), torch.from_numpy(v2).float()

# ==============================================================================
# 2. 3D-CNN Encoder Architecture
# ==============================================================================

class PolarityEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # (1, 101, 32, 112)
            nn.Conv3d(1, 32, kernel_size=(3,3,3), stride=(2,1,2), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            nn.Conv3d(128, 256, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten()
        )
        
        # Projection Head (SimCLR style)
        self.projector = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        # x: (B, 101, 1, 32, 112) -> (B, 1, 101, 32, 112)
        x = x.transpose(1, 2)
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

# ==============================================================================
# 3. Contrastive Loss (NT-Xent)
# ==============================================================================

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).to(DEVICE)

    def forward(self, z_i, z_j):
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.mask * torch.exp(similarity_matrix / self.temperature)
        
        loss = -torch.log(nominator / torch.sum(denominator, dim=1))
        return loss.mean()

# ==============================================================================
# 4. Training Loop
# ==============================================================================

def main():
    os.makedirs(os.path.join(BASE_DIR, "checkpoints"), exist_ok=True)
    
    dataset = FungalContrastiveDataset(DATA_PATH, TRANSITION_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = PolarityEncoder(LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = ContrastiveLoss(BATCH_SIZE, TEMPERATURE)
    
    print(f"Starting Training on {DEVICE}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
        for x_i, x_j in pbar:
            x_i, x_j = x_i.to(DEVICE), x_j.to(DEVICE)
            
            optimizer.zero_grad()
            z_i = model(x_i)
            z_j = model(x_j)
            
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
            
        # Save Checkpoint
        if epoch % 20 == 0 or epoch == 1:
            ckpt_path = os.path.join(BASE_DIR, f"video_contrastive_sequential_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved: {ckpt_path}")

    final_path = os.path.join(BASE_DIR, "video_contrastive_sequential_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training Complete. Final model saved to {final_path}")

if __name__ == "__main__":
    main()
