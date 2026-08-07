import torch
from pathlib import Path

def main():
    src = Path("/Volumes/X10 Pro/Movies/AI/sam2_logs/sam2.1_hiera_t_fungal_finetune/checkpoints/checkpoint.pt")
    local_dest = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/segment-anything-2/checkpoints/sam2.1_hiera_tiny_fungal_finetuned.pt")
    ssd_dest = Path("/Volumes/X10 Pro/Movies/AI/sam2_checkpoints/sam2.1_hiera_tiny_fungal_finetuned.pt")
    
    if not src.exists():
        print(f"Error: training checkpoint not found at {src}")
        return
        
    print(f"Loading checkpoint from: {src}")
    ckpt = torch.load(src, map_location="cpu")
    
    if "model" not in ckpt:
        print("Error: 'model' key not found in checkpoint dict.")
        return
        
    state_dict = ckpt["model"]
    print(f"Extracted model state dict with {len(state_dict)} keys.")
    
    # Save local copy
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state_dict}, local_dest)
    print(f"Saved local inference checkpoint: {local_dest} ({local_dest.stat().st_size / 1e6:.1f} MB)")
    
    # Save SSD copy
    ssd_dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": state_dict}, ssd_dest)
    print(f"Saved SSD inference checkpoint: {ssd_dest} ({ssd_dest.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
