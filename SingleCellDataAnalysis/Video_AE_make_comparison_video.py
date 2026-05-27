import os
import torch
import numpy as np
import cv2
import sys
from Video_AE_model import VideoAutoencoder

# Paths
OUTPUT_DIR = "/Volumes/X10 Pro/FungalProject_Outputs/video_ae/"
CHECKPOINT = os.path.join(OUTPUT_DIR, "video_multitask_final.pth")
CACHE_VIDEOS = os.path.join(OUTPUT_DIR, "video_cache.npy")
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "reconstruction_multitask.mp4")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load model
    model = VideoAutoencoder(latent_dim=16).to(device)
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        return
    
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()

    # 2. Load cache
    videos = np.load(CACHE_VIDEOS) # (N, 101, 1, 48, 96)
    idx = np.random.randint(len(videos))
    orig = videos[idx]
    print(f"Loaded cell {idx} from cache.")

    # 3. Generate reconstruction
    with torch.no_grad():
        x = torch.from_numpy(orig).float().unsqueeze(0).to(device) # (1, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4) # (1, C, T, H, W)
        recon, _, _, _ = model(x)
        recon = recon.permute(0, 2, 1, 3, 4).cpu().numpy()[0] # (T, C, H, W)

    # 4. Create Side-by-Side Video
    T, C, H, W = orig.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 10, (W * 2, H), isColor=False)

    for t in range(T):
        # Normalize for display (0-255 uint8)
        o_frame = (orig[t, 0] * 255).astype(np.uint8)
        r_frame = (np.clip(recon[t, 0], 0, 1) * 255).astype(np.uint8)
        
        # Add labels
        combined = np.hstack([o_frame, r_frame])
        # Add text labels using OpenCV
        cv2.putText(combined, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.putText(combined, "Reconstruction", (W + 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        cv2.putText(combined, f"t={t}", (W - 40, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
        
        out.write(combined)

    out.release()
    print(f"Saved comparison video to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
