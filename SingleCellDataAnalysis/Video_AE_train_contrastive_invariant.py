#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video_AE_train_contrastive_invariant.py — Strategy D

Size-Invariant Contrastive Learning. 
Learns polarity dynamics by enforcing invariance to absolute cell length
using random spatial scaling and jitter.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.ndimage import zoom

# --- Project Paths ---
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
DATA_PATH = os.path.join(BASE_DIR, "video_cache_32x112_padded.npy")
TRANSITION_PATH = os.path.join(BASE_DIR, "cycle_transition_pairs.npy")

# --- Hyperparameters ---
BATCH_SIZE = 16
LATENT_DIM = 128
TEMPERATURE = 0.1
LEARNING_RATE = 1e-4
EPOCHS = 150
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class InvariantFungalDataset(Dataset):
    def __init__(self, data_path, transition_path):
        print(f"Loading video data from {data_path}...")
        self.data = np.load(data_path)
        self.N = self.data.shape[0]
        self.transitions = np.load(transition_path)
        
    def __len__(self):
        return self.N

    def augment(self, x):
        # x is (101, 1, 32, 112)
        T, C, H, W = x.shape
        
        # 1. Random Polarity Flip (Vertical)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=2).copy()
            
        # 2. Random Symmetry Flip (Horizontal)
        if np.random.rand() > 0.5:
            x = np.flip(x, axis=3).copy()

        # 3. CRITICAL: Random Spatial Scaling (Size Invariance)
        # We simulate the cell being longer or shorter
        scale = np.random.uniform(0.7, 1.3)
        if scale != 1.0:
            # We only scale the width (W=112) as that is the axis of growth
            new_w = int(W * scale)
            # Use simple linear interpolation for speed
            # Note: For speed in training, we use torch.nn.functional.interpolate if we were on GPU
            # But here we'll do a simple numpy slice/pad or use torch on the tensor later.
            pass # We will do this via torch interpolate in the forward pass for speed

        # 4. Random Intensity Jitter
        x = x * np.random.uniform(0.8, 1.2)
                
        return x
    def __getitem__(self, idx):
        # View 1: Current cell
        v1 = self.augment(self.data[idx])
        
        # View 2: Decide between "Self-Augmentation" or "Cycle-Stitch"
        if np.random.rand() > 0.8: 
            trans_matches = self.transitions[self.transitions[:, 0] == idx]
            if len(trans_matches) > 0:
                partner_idx = trans_matches[np.random.randint(len(trans_matches))][1]
                v2 = self.augment(self.data[partner_idx])
            else:
                v2 = self.augment(self.data[idx])
        else:
            v2 = self.augment(self.data[idx])
            
        return torch.from_numpy(v1).float(), torch.from_numpy(v2).float()

class InvariantEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # Standard 3D-CNN Backbone
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3,3,3), stride=(2,1,2), padding=(1,1,1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Flatten()
        )
        self.projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, x):
        # x: (B, 101, 1, 32, 112)
        
        # Apply Size-Invariance Augmentation (Random Resizing in time-space)
        if self.training:
            # Randomly scale the width dimension to simulate different cell lengths
            scale_factor = np.random.uniform(0.8, 1.2)
            # x is (B, T, C, H, W) -> interpolate expects (B, C, T, H, W)
            x = x.transpose(1, 2)
            B, C, T, H, W = x.shape
            new_w = int(W * scale_factor)
            x = F.interpolate(x, size=(T, H, new_w), mode='trilinear', align_corners=False)
            # Crop or Pad back to original W
            if new_w > W:
                start = (new_w - W) // 2
                x = x[:, :, :, :, start:start+W]
            else:
                pad = (W - new_w) // 2
                x = F.pad(x, (pad, W - new_w - pad))
        else:
            x = x.transpose(1, 2)

        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

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

def main():
    dataset = InvariantFungalDataset(DATA_PATH, TRANSITION_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = InvariantEncoder(LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = ContrastiveLoss(BATCH_SIZE, TEMPERATURE)
    
    print(f"🚀 Starting SIZE-INVARIANT Training on {DEVICE}...")
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
            
        if epoch % 20 == 0 or epoch == 1:
            ckpt_path = os.path.join(BASE_DIR, f"video_contrastive_invariant_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)

    final_path = os.path.join(BASE_DIR, "video_contrastive_invariant_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✅ Invariant model saved to {final_path}")

if __name__ == "__main__":
    main()
