import torch
import sys
sys.path.append('/Users/user/Documents/Python_Scripts/FungalProjectScript/')
from SingleCellDataAnalysis.Video_AE_model_segmenter import VideoAutoencoderSegmenter

device = torch.device("cpu")
print(f"Device: {device}")

model = VideoAutoencoderSegmenter(latent_dim=16).to(device)
dummy = torch.randn(2, 1, 101, 64, 224).to(device)

print("Forward pass...")
out, z = model(dummy)
print(f"Output shape: {out.shape}")

print("Backward pass...")
loss = out.sum()
loss.backward()
print("Done!")
