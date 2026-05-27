import matplotlib.pyplot as plt
import os
import re

LOG_FILE = '/Volumes/X10 Pro/FungalProject_Outputs/video_ae/train_segmenter.log'
OUTPUT_DIR = '/Users/user/.gemini/antigravity/brain/8e3e2fd2-945e-4dcf-a62c-5ef369b9b1d7/'

recon_loss = []
gamma_loss = []

with open(LOG_FILE, 'r') as f:
    for line in f:
        # Example: Epoch [ 80/200] Total: 0.0082 (Recon: 0.0077, Gamma: 0.0001) LR: 6.25e-05  ETA: 92.0m
        match = re.search(r'Recon: ([\d\.]+), Gamma: ([\d\.]+)', line)
        if match:
            recon_loss.append(float(match.group(1)))
            gamma_loss.append(float(match.group(2)))

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(recon_loss, label='Recon Loss', color='blue')
plt.title('Video Reconstruction Loss')
plt.xlabel('Epoch'); plt.legend(); plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(gamma_loss, label='Gamma Mask Loss (BCE)', color='purple')
plt.title('Supervised Gamma Mask Loss')
plt.xlabel('Epoch'); plt.legend(); plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "segmenter_training_loss.png"), dpi=150)
