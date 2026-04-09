#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib
# Bypass Mac Qt conflicts
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from SingleCellDataAnalysis.inference_core import EndpointMIL
from SingleCellDataAnalysis.septum_gui_utils import (
    discover_cell_mask_csvs, 
    MaskTableCache, 
    ensure_strip_for_cell,
    build_film_paths
)

def get_saliency_map(model, x_full, device):
    """
    Calculates Pixel-Level Saliency Gradients across the full temporal sequence.
    Given an input (L, 1, H, W), compute gradient of the maximum internal start_t logit
    safely traced backwards down to the individual (H, W) pixels.
    """
    model.eval()
    
    x = x_full[None, ...].to(device)  # (1, L, 1, H, W)
    x.requires_grad_(True)
    mask = torch.ones((1, x.shape[1]), device=device)
    
    # Run a mathematical forward pass
    state_t = model(x, mask)
    
    # Force PyTorch autograd system to route backward through the probability peak
    loss = state_t.max()
    loss.backward()
    
    # Calculate gradient impact across channels
    saliency = x.grad.abs().sum(dim=2).squeeze(0)  # (L, H, W)
    saliency = saliency.cpu().numpy()
    
    state_probs = torch.sigmoid(state_t)[0].detach().cpu().numpy()
    
    # Normalize the saliency map mathematically for visual hot-pixel clarity
    smap_max = saliency.max()
    if smap_max > 0:
        saliency = saliency / smap_max
        
    return saliency, state_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=str, default="/Volumes/X10 Pro/Movies/2025_12_31_M92")
    parser.add_argument("--film_name", type=str, default="A14-YES-1t-FBFBF-2_F0")
    # Grab cell #2 by default since cell it likely has a septum
    parser.add_argument("--cell_id", type=int, default=2, help="Cell ID to probe for visualization")
    parser.add_argument("--weights", type=str, default="SingleCellDataAnalysis/ep_86_bacc_0.906.pt", help="PyTorch Network weights path")
    args = parser.parse_args()

    # Reproduce generic runtime GUI paths natively
    paths = build_film_paths(args.working_dir, args.film_name)
    
    print(f"Loading cell masks tracing database...")
    cids, raw_cell_csv_map = discover_cell_mask_csvs(paths.tracked_dir)
    # Ensure MaskTableCache is indexed by (film_name, cid) to match ensure_strip_for_cell
    cell_csv_map = {(args.film_name, cid): path for cid, path in raw_cell_csv_map.items()}
    masks = MaskTableCache(cell_csv_map, time_col="time_point", mask_col="rle_bf")
    offsets = {} # Optional offset structure
    
    print(f"Fetching raw original cell strip sequence via {args.cell_id}...")
    strip, tp0 = ensure_strip_for_cell(
        cid=args.cell_id,
        film_name=args.film_name,
        frames_dir=paths.frames_dir,
        cache_img_dir=paths.cache_img_dir,
        masks=masks,
        offsets=offsets,
        time_col="time_point",
        mask_col="rle_bf",
        pad=10,
        tile_size=96,
        channel_index=0,
        cache_force=False,
    )
    
    if strip is None or strip.shape[1] == 0:
        print(f"Error: Could not load strip for cell {args.cell_id}. Skipping.")
        return

    H = strip.shape[0]
    L = strip.shape[1] // H
    print(f"Deconstructed biological footprint: Length={L} frames, Pixel Height={H} px.")
    
    # Normalizing cell arrays
    tiles = strip.reshape(H, L, H).transpose(1, 0, 2)[:, None, :, :]
    x_full = torch.from_numpy(tiles.astype(np.float32) / 255.0)
    
    # Load AI model framework
    device = "cpu"
    
    # Calculate external weights path
    if args.weights and os.path.isfile(args.weights):
        ckpt_path = args.weights
        print(f"Booting PyTorch Artificial Intelligence Core from explicit path ({os.path.basename(ckpt_path)})...")
    else:
        ckpt_path = os.path.join(args.working_dir, "training_dataset", "checkpoints_binary", "model_latest.pt")
        # Prefer model_best.pt (best val_loss) over model_latest.pt if available
        best_path = ckpt_path.replace("model_latest.pt", "model_best.pt")
        if os.path.isfile(best_path):
            ckpt_path = best_path
            print(f"Booting PyTorch Artificial Intelligence Core (model_best.pt)...")
        else:
            print(f"Booting PyTorch Artificial Intelligence Core (model_latest.pt)...")
            
    model = EndpointMIL(D=64).to(device)
    chkpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "state_dict" in chkpt:
        model.load_state_dict(chkpt["state_dict"])
    else:
        model.load_state_dict(chkpt)
        
    print("Executing Autograd Trace Mapping...")
    saliency_map, probs = get_saliency_map(model, x_full, device)
    
    saliency_strip = np.zeros_like(strip, dtype=float)
    for i in range(L):
        saliency_strip[:, i*H:(i+1)*H] = saliency_map[i]
        
    s_idx = int(np.argmax(probs))
    score = float(probs[s_idx])
    global_s_idx = tp0 + offsets.get(args.cell_id, 0) + s_idx
    print(f"\n--> AI Target Peak found at spatial offset +{s_idx} : Global Array ID: {global_s_idx}")
    print(f"--> Probability Confidence Score: {score:.1%}\n")

    # Plot Graphical Mathematical Trace
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 6), sharex=False, gridspec_kw={'height_ratios': [3, 1]})
    
    ax1.imshow(strip, cmap="gray", aspect="equal")
    # Overlay heatmap natively utilizing PyTorch partial derivative calculation
    ax1.imshow(saliency_strip, cmap="hot", alpha=0.35, aspect="equal")
    ax1.set_title(f"PyTorch Guided Autograd Backprop Heatmap (Cell {args.cell_id}) | AI Synapse Vision")
    ax1.axis("off")
    
    offsets_x = np.arange(L) * H + (H//2)
    ax2.plot(offsets_x, probs, color="green", linewidth=2.5)
    ax2.axvline(offsets_x[s_idx], color="red", linestyle="--", alpha=0.7)
    ax2.set_xlim(0, L * H)
    ax2.set_title("Temporal Septum Target Probability Threshold")
    ax2.set_ylabel("Probability")
    
    plt.tight_layout()
    # Save the figure dynamically into the current folder so the AI can physically inspect the graph 
    save_path = f"saliency_{args.film_name}_cell_{args.cell_id}.png"
    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"[Saved Image] Successfully authored to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
