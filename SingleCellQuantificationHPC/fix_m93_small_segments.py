import json
import pandas as pd
from pathlib import Path
import numpy as np
from skimage.io import imread
from skimage.measure import regionprops
import subprocess
import sys
import shutil

def get_centroid(rle_str, w, h):
    parts = rle_str.split()
    starts = np.array(parts[0::2], dtype=int)
    lengths = np.array(parts[1::2], dtype=int)
    mask = np.zeros(w * h, dtype=np.uint8)
    for s, l in zip(starts, lengths):
        s0 = s - 1 if s > 0 else 0
        mask[s0:s0+l] = 1
    mask = mask.reshape((h, w), order="F")
    props = regionprops(mask)
    if props:
        return int(props[0].centroid[1]), int(props[0].centroid[0]) # x, y
    return 0, 0

def fix_m93_small_segments():
    exp = "2026_01_08_M93"
    BASE = Path("/Volumes/X10 Pro/Movies") / exp
    link_file = BASE / "sequence_linkage.json"
    
    # Backup original linkage
    backup_file = BASE / "sequence_linkage.json.bak"
    if not backup_file.exists():
        shutil.copy(link_file, backup_file)
        
    with open(link_file) as f:
        data = json.load(f)

    changes_made = 0

    for seq, info in data.items():
        films = info["films"]
        for global_id, track in info["global_cells"].items():
            for i in range(len(track) - 1):
                if track[i] != -1 and track[i+1] == -1:
                    fA = films[i]
                    cA = track[i]
                    fB = films[i+1]
                    
                    tracked = BASE / fA / f"TrackedCells_{fA}" / f"cell_{cA}_masks.csv"
                    if not tracked.exists():
                        continue
                    try:
                        df = pd.read_csv(tracked)
                    except Exception:
                        continue
                        
                    rle_col = "rle_bf" if "rle_bf" in df.columns else "rle_gfp"
                    if rle_col not in df.columns:
                        continue
                    df_valid = df.dropna(subset=[rle_col])
                    if len(df_valid) == 0:
                        continue
                    last_row = df_valid.iloc[-1]
                    rle_str = last_row[rle_col]
                    w, h = int(last_row["width"]), int(last_row["height"])
                    
                    cx, cy = get_centroid(rle_str, w, h)
                    
                    mask_fB_path = BASE / fB / f"Masks_{fB}" / f"{fB}_t_000_c_0_seg.tif"
                    if not mask_fB_path.exists():
                        continue
                    mask_fB = imread(mask_fB_path)
                    
                    if cy >= mask_fB.shape[0] or cx >= mask_fB.shape[1]:
                        continue
                        
                    val = mask_fB[cy, cx]
                    if val > 0:
                        area = np.sum(mask_fB == val)
                        if area < 2500:
                            print(f"Broken link in {seq}: {fA} cell {cA} -> {fB}. Found segment {val} with small area {area}. Fixing...")
                            
                            # Perform local quantification
                            quant_script = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/one_cell_quantification_1CH.py")
                            cmd = [
                                sys.executable,
                                str(quant_script),
                                "--experiment_path", str(BASE),
                                "--file_name", fB,
                                "--cell_id", str(val)
                            ]
                            res = subprocess.run(cmd, capture_output=True, text=True)
                            if res.returncode == 0:
                                print(f"  -> Successfully quantified cell {val} locally.")
                                track[i+1] = int(val)
                                changes_made += 1
                            else:
                                print(f"  -> Failed to quantify cell {val}: {res.stderr}")

    if changes_made > 0:
        with open(link_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Fixed {changes_made} small segment broken links in M93. Saved sequence_linkage.json.")
    else:
        print("No small segment broken links found or fixed.")

if __name__ == "__main__":
    fix_m93_small_segments()
