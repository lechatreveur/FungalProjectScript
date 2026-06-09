import os, sys, numpy as np, pandas as pd
import torch
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from SingleCellDataAnalysis.FC_Contrastive_train import FCTrajectoryEncoder, LATENT_DIM
from SingleCellDataAnalysis.FC_Contrastive_umap import DEVICE

EXPERIMENTS_MIX = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/",
    "M133":   "/Volumes/X10 Pro/Movies/2026_04_29_M133/"
}

X_traj, X_feat, gids, labels, _, _ = load_feature_constrained_data(EXPERIMENTS_MIX)
print("X_traj NaNs:", np.isnan(X_traj).sum())
print("X_feat NaNs:", np.isnan(X_feat).sum())
print("X_traj Infs:", np.isinf(X_traj).sum())
print("X_feat Infs:", np.isinf(X_feat).sum())

# Also load M133 by itself to see if the WT model produces NaNs
from SingleCellDataAnalysis.FC_Contrastive_umap_Strategy2 import load_m133_with_wt_scalers, M133_EXPERIMENT, EXPERIMENTS
_, _, _, _, scaler_traj, scaler_feat = load_feature_constrained_data(EXPERIMENTS)
X_traj_M133, X_feat_M133, gids_M133, labels_M133 = load_m133_with_wt_scalers(M133_EXPERIMENT, scaler_traj, scaler_feat)
print("M133 X_traj NaNs:", np.isnan(X_traj_M133).sum())
print("M133 X_feat NaNs:", np.isnan(X_feat_M133).sum())
print("M133 X_traj Infs:", np.isinf(X_traj_M133).sum())
print("M133 X_feat Infs:", np.isinf(X_feat_M133).sum())

# Try forward pass of Strategy 2 WT model on M133
MODEL_PATH = "/Volumes/X10 Pro/FungalProject_Outputs/fc_contrastive/fc_contrastive_final.pth"
model = FCTrajectoryEncoder(LATENT_DIM).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
with torch.no_grad():
    latents_M133 = model(torch.from_numpy(X_traj_M133).float(),
                         torch.from_numpy(X_feat_M133).float()).numpy()
print("M133 latents NaNs:", np.isnan(latents_M133).sum())
