#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FC_Contrastive_train.py — Strategy C for Tabular + Time-Series Data

Learns a joint latent manifold of fungal polarity dynamics using:
1. 11 Engineered Features (static vector per cell)
2. Single-Cell Trajectories (101-frame intensity profiles of 2 poles)

Uses NT-Xent loss with lineage-stitch pairs.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Ensure project root is in path
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data

# --- Hyperparameters ---
BATCH_SIZE = 32
LATENT_DIM = 128
TEMPERATURE = 0.1
LEARNING_RATE = 1e-4
EPOCHS = 300
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- Paths ---
BASE_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/fc_contrastive/"
TRANSITION_PATH = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/cycle_transition_pairs.npy"

EXPERIMENTS = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/"
}

# ==============================================================================
# 1. Dataset & Augmentations
# ==============================================================================

class FCTrajectoryContrastiveDataset(Dataset):
    def __init__(self, X_traj, X_feat, transition_path):
        self.X_traj = X_traj  # (N, 101, 2)
        self.X_feat = X_feat  # (N, 11)
        self.N = X_traj.shape[0]
        self.N = X_traj.shape[0]
        
    def __len__(self):
        return self.N

    def augment_traj(self, x):
        """
        - Gaussian noise
        - Random polarity swap (Pol1 <-> Pol2)
        - Random temporal shift
        """
        x = x.copy()
        
        # 1. Polarity Swap (Channels)
        if np.random.rand() > 0.5:
            x = x[:, [1, 0]]
            
        # 2. Gaussian Noise
        x += np.random.normal(0, 0.01, x.shape)
        
        # 3. Temporal Shift
        shift = np.random.randint(-5, 6)
        if shift != 0:
            x = np.roll(x, shift, axis=0)
            if shift > 0:
                x[:shift] = 0
            else:
                x[shift:] = 0
        return x

    def augment_feat(self, f):
        """
        - Gaussian noise
        - Random feature dropout (masking)
        """
        f = f.copy()
        # 1. Gaussian Noise
        f += np.random.normal(0, 0.05, f.shape)
        
        # 2. Feature Dropout (mask 1-2 features)
        if np.random.rand() > 0.5:
            idx = np.random.choice(len(f), size=np.random.randint(1, 3), replace=False)
            f[idx] = 0
        return f

    def __getitem__(self, idx):
        # View 1: Current cell
        t1 = self.augment_traj(self.X_traj[idx])
        f1 = self.augment_feat(self.X_feat[idx])
        
        # View 2: Self-Augment
        t2 = self.augment_traj(self.X_traj[idx])
        f2 = self.augment_feat(self.X_feat[idx])
            
        return (torch.from_numpy(t1).float(), torch.from_numpy(f1).float()), \
               (torch.from_numpy(t2).float(), torch.from_numpy(f2).float())

# ==============================================================================
# 2. Multimodal Encoder
# ==============================================================================

class FCTrajectoryEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        
        # 1. Trajectory Head (1D-CNN)
        self.traj_head = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 2. Feature Head (MLP)
        self.feat_head = nn.Sequential(
            nn.Linear(11, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # 3. Joint Projector
        self.projector = nn.Sequential(
            nn.Linear(64 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

    def forward(self, t, f):
        # t: (B, 101, 2) -> (B, 2, 101)
        t = t.transpose(1, 2)
        h_t = self.traj_head(t)
        h_f = self.feat_head(f)
        
        h_joint = torch.cat([h_t, h_f], dim=1)
        z = self.projector(h_joint)
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
    
    # 1. Load Data
    print("📥 Loading feature-constrained data...")
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS)
    
    dataset = FCTrajectoryContrastiveDataset(X_traj, X_feat, TRANSITION_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    model = FCTrajectoryEncoder(LATENT_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = ContrastiveLoss(BATCH_SIZE, TEMPERATURE)
    
    print(f"🚀 Starting Training on {DEVICE}...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{EPOCHS}")
        for (t_i, f_i), (t_j, f_j) in pbar:
            t_i, f_i = t_i.to(DEVICE), f_i.to(DEVICE)
            t_j, f_j = t_j.to(DEVICE), f_j.to(DEVICE)
            
            optimizer.zero_grad()
            z_i = model(t_i, f_i)
            z_j = model(t_j, f_j)
            
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1)})
            
        if epoch % 50 == 0 or epoch == 1:
            ckpt_path = os.path.join(BASE_DIR, f"checkpoints/fc_contrastive_no_lineage_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved: {ckpt_path}")

    final_path = os.path.join(BASE_DIR, "fc_contrastive_no_lineage_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✅ Training Complete. Final model saved to {final_path}")

if __name__ == "__main__":
    main()
