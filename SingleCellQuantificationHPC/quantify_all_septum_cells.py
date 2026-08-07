import os
import sys
import json
import subprocess
from pathlib import Path
import multiprocessing
import argparse

def run_quant_subprocess(args_tuple):
    cid, film, track_channel, movie_root, exp, py_exec, hpc_dir = args_tuple
    cmd = f"KMP_DUPLICATE_LIB_OK=TRUE {py_exec} one_cell_quantification_1CH.py --cell_id {cid} --experiment_path \"{movie_root / exp}\" --file_name \"{film}\" --track_channel {track_channel}"
    res = subprocess.run(cmd, shell=True, cwd=hpc_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if res.returncode != 0:
        return film, cid, False, res.stdout
    return film, cid, True, ""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="2025_09_17", help="Experiment folder name")
    parser.add_argument("--films", type=str, default=None, help="Comma-separated film list (optional)")
    args = parser.parse_args()

    exp = args.experiment
    movie_root = Path("/Volumes/X10 Pro/Movies")
    hpc_dir = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC")
    py_exec = sys.executable

    # Determine films
    if args.films:
        films = [f.strip() for f in args.films.split(",") if f.strip()]
    else:
        # Default fallback lists
        if exp == "2025_09_17":
            films = ["A14_1TP1_F1", "A14_1TP1_BF_F1", "A14_1TP2_F1", "A14_1TP2_BF_F1"]
        elif exp == "2026_04_29_M133":
            films = ["YES_Scd1_D_F1", "YES_Scd1_D_1_F1", "YES_Scd1_D_2_F1", "YES_Scd1_D_3_F1", "YES_Scd1_D_4_F1", "YES_Scd1_D_5_F1"]
        else:
            films = [p.name for p in (movie_root / exp).iterdir() if p.is_dir() and not p.name.startswith(".")]

    args_list = []
    for film in films:
        json_path = movie_root / exp / film / f"TrackedCells_{film}" / "cell_plots" / "gui_labels" / "global_septum_alignment.json"
        if not json_path.exists():
            continue
        with open(json_path) as f:
            js = json.load(f)
        
        offsets, intervals = js.get("offsets", {}), js.get("cell_intervals", {})
        if exp == "2026_04_29_M133":
            track_channel = 'bf'
        else:
            track_channel = 'bf' if 'BF' in film else 'gfp'
        
        for cell_id_str, interval in intervals.items():
            if interval.get("has_septum") and interval.get("end_aligned") is not None:
                lid = int(cell_id_str)
                mask_p = movie_root / exp / film / f"TrackedCells_{film}" / f"cell_{lid}_masks.csv"
                data_p = movie_root / exp / film / f"TrackedCells_{film}" / f"cell_{lid}_data.csv"
                
                if mask_p.exists():
                    # We need quantification if data.csv doesn't exist, or if masks.csv is newer
                    if not data_p.exists() or mask_p.stat().st_mtime > data_p.stat().st_mtime:
                        args_list.append((lid, film, track_channel, movie_root, exp, py_exec, hpc_dir))
                        
    print(f"Total septum cells to quantify for {exp}: {len(args_list)}")
    if not args_list:
        print("Nothing to quantify!")
        return
        
    num_workers = 8
    print(f"Quantifying in parallel with {num_workers} workers...")
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = []
        for res in pool.imap_unordered(run_quant_subprocess, args_list):
            film, cid, success, logs = res
            results.append(res)
            if not success:
                print(f"    [Error] Failed to quantify cell {cid} in {film}:\n{logs}")
            
            if len(results) % 50 == 0 or len(results) == len(args_list):
                print(f"    Progress: {len(results)}/{len(args_list)} tasks completed.")
                
    print("Done!")

if __name__ == '__main__':
    main()
