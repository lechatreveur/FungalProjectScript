import os
import torch
import numpy as np
from SingleCellDataAnalysis.inference_core import EndpointMIL

working_dir = '/Volumes/X10 Pro/Movies/2025_12_31_M92'
npz_fp = os.path.join(working_dir, 'training_dataset', 'samples', 'A14-YES-1t-FBFBF-2_F0__cell_000005.npz')

with np.load(npz_fp, allow_pickle=True) as z:
    strip = np.asarray(z["strip"], dtype=np.uint8)
    s_idx = int(z["start_idx"][0])
    e_idx = int(z["end_idx"][0])

H = int(strip.shape[0])
L = int(strip.shape[1] // H)

tiles = strip.reshape(H, L, H).transpose(1, 0, 2)[:, None, :, :]
x_full = torch.from_numpy(tiles.astype(np.float32) / 255.0)

model = EndpointMIL(D=64).eval()
chkpt = torch.load(os.path.join(working_dir, 'training_dataset', 'checkpoints_binary', 'model_latest.pt'), map_location='cpu', weights_only=True)
if 'state_dict' in chkpt: model.load_state_dict(chkpt['state_dict'])
else: model.load_state_dict(chkpt)

with torch.no_grad():
    state_t = model(x_full[None, ...], torch.ones((1, L)))
    probs = torch.sigmoid(state_t).numpy()[0]

print(f"Target Septum is at frames: {s_idx} to {e_idx}")
print(f"Average Prob across whole strip: {probs.mean():.4f}")
print(f"Average Prob inside Target Septum: {probs[s_idx:e_idx].mean():.4f}")
print(f"Average Prob outside Target Septum: {np.concatenate([probs[:s_idx], probs[e_idx:]]).mean():.4f}")
