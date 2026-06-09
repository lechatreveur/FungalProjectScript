import os
import sys
import torch

sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.FC_Contrastive_train import (
    FCTrajectoryEncoder, ContrastiveLoss, FCTrajectoryContrastiveDataset,
    BASE_DIR, TRANSITION_PATH, LATENT_DIM, DEVICE, BATCH_SIZE, TEMPERATURE, LEARNING_RATE, EPOCHS
)
from SingleCellDataAnalysis.FC_AE_data_loader import load_feature_constrained_data
from torch.utils.data import DataLoader
from tqdm import tqdm

EXPERIMENTS_MIX = {
    "Sept17": "/Volumes/X10 Pro/Movies/2025_09_17/",
    "M92":    "/Volumes/X10 Pro/Movies/2025_12_31_M92/",
    "M93":    "/Volumes/X10 Pro/Movies/2026_01_08_M93/",
    "June25_20m": "/Volumes/X10 Pro/Movies/2025_06_25/A14_10_20min/",
    "M133":   "/Volumes/X10 Pro/Movies/2026_04_29_M133/"
}

def main():
    print("📥 Loading mixed feature-constrained data (WT + M133)...")
    X_traj, X_feat, gids, labels, s_traj, s_feat = load_feature_constrained_data(EXPERIMENTS_MIX)
    
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
            
    final_path = os.path.join(BASE_DIR, "fc_contrastive_M133_mix_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"✅ Training Complete. Final model saved to {final_path}")

if __name__ == "__main__":
    main()
