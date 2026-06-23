import os
import re
import subprocess
from pathlib import Path
import shutil
import multiprocessing

EXP = "2025_09_17"
MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
HPC_DIR = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC")
PY_EXEC = "python3"

# We only process the Brightfield films for 2025_09_17
BF_FILMS = [
    "A14_1TP1_BF_F1"
]

# Override: only re-track specific cells that failed due to SSD disconnect.
# Set to None to track all cells.
CELL_IDS_OVERRIDE = None

def run_cmd(cmd, cwd=None):
    print(f"Executing: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")

def track_cell_subprocess(args_tuple):
    cid, movie_root, exp, film = args_tuple
    cmd = f"KMP_DUPLICATE_LIB_OK=TRUE {PY_EXEC} one_cell_quantification_1CH.py --cell_id {cid} --experiment_path \"{movie_root / exp}\" --file_name \"{film}\" --no_plot --update_existing"
    res = subprocess.run(cmd, shell=True, cwd=HPC_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        return cid, False, res.stdout
    return cid, True, ""

def main():
    # 3 workers to be absolutely safe on memory while providing a 3x speedup.
    # PyTorch inference with 3 processes takes ~3-4GB of RAM max.
    num_workers = 3 
    
    for film in BF_FILMS:
        print(f"\n==========================================")
        print(f"Processing {film}...")
        film_dir = MOVIE_ROOT / EXP / film
        tracked_dir = film_dir / f"TrackedCells_{film}"
        
        if not tracked_dir.exists():
            print(f"Directory not found: {tracked_dir}. Skipping.")
            continue
            
        # Get all cell IDs
        cell_ids = []
        for f in tracked_dir.iterdir():
            m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
            if m:
                cell_ids.append(int(m.group(1)))
                
        cell_ids.sort()
        if not cell_ids:
            print(f"Found 0 cells to re-track in {film}. Skipping.")
            continue
            
        print(f"Found {len(cell_ids)} cells to re-track in {film}")
        print(f"Tracking in parallel with {num_workers} workers...")
        
        if CELL_IDS_OVERRIDE is not None:
            cell_ids = sorted(set(cell_ids) & set(CELL_IDS_OVERRIDE))
            print(f"  (Override: only tracking {len(cell_ids)} specific cells)")

        args_list = [(cid, MOVIE_ROOT, EXP, film) for cid in cell_ids]
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = []
            for res in pool.imap_unordered(track_cell_subprocess, args_list):
                cid, success, logs = res
                results.append(res)
                if not success:
                    print(f"    [Error] Failed to track cell {cid}:\n{logs}")
                
                # Print progress every 10 cells
                if len(results) % 10 == 0 or len(results) == len(cell_ids):
                    print(f"    Progress: {len(results)}/{len(cell_ids)} cells completed.")
                
        # The user modified one_cell_quantification_1CH.py to directly output to TrackedCells_{film}
        # so there's no need to swap directories anymore.
        print(f"Finished processing {film}.")

    # Finally, regenerate the sequence linkages
    print("\n==========================================")
    print("Regenerating sequence linkages...")
    linkage_script = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis/generate_sequence_linkages.py")
    run_cmd(f"{PY_EXEC} {linkage_script}", cwd=linkage_script.parent)
    print("Done!")

if __name__ == "__main__":
    # Ensure any stray python processes are killed from before
    main()
