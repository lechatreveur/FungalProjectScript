import os
import json
import pandas as pd
from pathlib import Path

BASE_MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
EXP = "2026_01_08_M93"

def main():
    exp_dir = BASE_MOVIE_ROOT / EXP
    linkage_file = exp_dir / "sequence_linkage.json"
    
    with open(linkage_file, 'r') as f:
        linkage = json.load(f)
        
    samples = []
    
    for seq in ["A14_F1", "A14_F2"]:
        qc_file = exp_dir / seq / f"qc_{seq}.json"
        if not qc_file.exists():
            continue
            
        with open(qc_file, 'r') as f:
            qc = json.load(f)
            
        films = linkage[seq]["films"]
        global_cells = linkage[seq]["global_cells"]
        
        for key, status in qc.items():
            if status.lower() not in ["good", "corrected"]:
                continue
                
            # Check if global or local
            # Format: {seq}_cell_{id} OR {seq}_{film}_cell_{id}
            prefix = f"{seq}_"
            suffix = key[len(prefix):]
            
            if suffix.startswith("cell_"):
                # Global
                local_ids = global_cells.get(key, [-1]*len(films))
                for film, local_id in zip(films, local_ids):
                    if local_id != -1 and "FL" in film: # Only use GFP (FL) films for tracking
                        _add_samples(exp_dir, film, local_id, samples)
            else:
                # Local
                parts = suffix.split("_cell_")
                if len(parts) == 2:
                    film = parts[0]
                    local_id = int(parts[1])
                    if "FL" in film:
                        _add_samples(exp_dir, film, local_id, samples)

    out_csv = exp_dir / "curated_training_samples_gfp.csv"
    df = pd.DataFrame(samples, columns=["film", "cell_id", "t"])
    df.to_csv(out_csv, index=False)
    print(f"Extracted {len(df)} curated frame pairs to {out_csv}")
    
def _add_samples(exp_dir, film, local_id, samples):
    csv_path = exp_dir / film / f"TrackedCells_{film}" / f"cell_{local_id}_masks.csv"
    if not csv_path.exists():
        return
        
    try:
        df = pd.read_csv(csv_path)
        df = df.sort_values("time_point")
        for i in range(len(df) - 1):
            t = int(df.iloc[i]["time_point"])
            t_next = int(df.iloc[i+1]["time_point"])
            # Only consecutive frames
            if t_next == t + 1:
                samples.append((film, local_id, t))
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")

if __name__ == "__main__":
    main()
