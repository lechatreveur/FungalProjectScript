#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone inference module for human-in-the-loop GUI.
Provides the PyTorch model without modifying or requiring the training pipeline dependencies.
"""

import os
import torch
from torch import nn
import numpy as np

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
            nn.Conv1d(D, D, 3, padding=1),
            nn.BatchNorm1d(D),
            nn.ReLU(),
            nn.Conv1d(D, D, 3, padding=1),
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


class FungalInferenceCore:
    """
    Lightweight wrapper securely loading the AI weights and processing NumPy strips.
    """
    def __init__(self, chkpt_path: str, D: int = 64, device: str = "cpu"):
        self.device = device
        self.model = EndpointMIL(D=D).to(device)
        self.model.eval()

        # Prefer model_best.pt (best val_loss) over model_latest.pt if available
        best_path = chkpt_path.replace("model_latest.pt", "model_best.pt")
        if os.path.isfile(best_path):
            chkpt_path = best_path

        # Load weights safely
        chkpt = torch.load(chkpt_path, map_location=device, weights_only=True)
        if "state_dict" in chkpt:
            self.model.load_state_dict(chkpt["state_dict"])
        else:
            self.model.load_state_dict(chkpt)
            
    @torch.no_grad()
    def predict_strip(self, strip_uint8: np.ndarray):
        """
        Process a (H, H*L) numpy strip representing the cell across time.
        Returns the top-scoring start and end windows:
        (s_start_idx, s_end_idx, s_score), (e_start_idx, e_end_idx, e_score)
        """
        H = strip_uint8.shape[0]
        if strip_uint8.shape[1] % H != 0:
            return None
            
        L = strip_uint8.shape[1] // H
        if L < 5:
            return None
            
        # Reshape to (L, 1, H, H) and normalize to [0,1]
        tiles = strip_uint8.reshape(H, L, H).transpose(1, 0, 2)[:, None, :, :]
        x_full = tiles.astype(np.float32) / 255.0
        x_full = torch.from_numpy(x_full).to(self.device).float()  # (L, 1, H, W)
        
        # Evaluate the entire sequence at once
        x_batch = x_full[None, ...] # (1, L, 1, H, W)
        mask = torch.ones((1, L), device=self.device)
        
        state_t = self.model(x_batch, mask)
        state_probs = torch.sigmoid(state_t)[0].cpu().numpy()
        return state_probs

    def predict_saliency(self, strip_uint8: np.ndarray):
        """
        Calculates pixel-level saliency and frame-level probabilities.
        Returns (probs, saliency_2d_strip)
        """
        H = strip_uint8.shape[0]
        if strip_uint8.shape[1] % H != 0:
            return None, None
        L = strip_uint8.shape[1] // H
        
        tiles = strip_uint8.reshape(H, L, H).transpose(1, 0, 2)[:, None, :, :]
        x_full = torch.from_numpy(tiles.astype(np.float32) / 255.0).to(self.device).float()
        
        # Enable gradients for saliency map
        x = x_full[None, ...]  # (1, L, 1, H, W)
        x.requires_grad_(True)
        mask = torch.ones((1, L), device=self.device)
        
        # Forward pass
        state_t = self.model(x, mask)
        
        # Backward from the probability peak
        loss = state_t.max()
        loss.backward()
        
        # Extract gradients
        saliency = x.grad.abs().sum(dim=2).squeeze(0)  # (L, H, W)
        saliency = saliency.cpu().numpy()
        
        # Normalize
        s_max = saliency.max()
        if s_max > 0:
            saliency = saliency / s_max
            
        # Assemble back into a 2D strip for overlay (H, L*H)
        saliency_strip = np.zeros_like(strip_uint8, dtype=float)
        for i in range(L):
            saliency_strip[:, i*H:(i+1)*H] = saliency[i]
            
        probs = torch.sigmoid(state_t)[0].detach().cpu().numpy()
        return probs, saliency_strip
