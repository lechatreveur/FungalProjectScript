import os
import re
import json
import sys
import importlib
from pathlib import Path
import multiprocessing

EXP = "2025_09_17"
MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")
HPC_DIR = Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC")

def run_quant_subprocess(args_tuple):
    cid, movie_root, exp, film, track_channel = args_tuple
    import sys
    import importlib
    
    hpc_dir = "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC"
    proj_dir = "/Users/user/Documents/Python_Scripts/FungalProjectScript"
    if hpc_dir not in sys.path: sys.path.append(hpc_dir)
    if proj_dir not in sys.path: sys.path.append(proj_dir)
    
    # Save original argv
    orig_argv = sys.argv
    sys.argv = [
        'one_cell_quantification_1CH.py',
        '--cell_id', str(cid),
        '--experiment_path', str(movie_root / exp),
        '--file_name', str(film),
        '--track_channel', track_channel,
        '--no_plot'
    ]
    
    # Suppress stdout to avoid messy console printouts, but keep stderr/exceptions
    import io
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    
    try:
        if 'one_cell_quantification_1CH' in sys.modules:
            importlib.reload(sys.modules['one_cell_quantification_1CH'])
        else:
            import one_cell_quantification_1CH
        success = True
        err_msg = ""
    except SystemExit as se:
        if se.code == 0:
            success = True
            err_msg = ""
        else:
            success = False
            err_msg = f"SystemExit with code {se.code}"
    except Exception as e:
        success = False
        import traceback
        err_msg = traceback.format_exc()
    finally:
        sys.stdout = old_stdout
        sys.argv = orig_argv
        
    return film, cid, success, err_msg

def main():
    num_workers = 8
    
    # Load F1 qc
    qc_path = MOVIE_ROOT / EXP / "F1" / "qc_F1.json"
    with open(qc_path) as f:
        qc = json.load(f)
        
    # Load sequence linkages
    linkage_path = MOVIE_ROOT / EXP / "sequence_linkage.json"
    with open(linkage_path) as f:
        linkage = json.load(f)
        
    f1_cells = linkage['F1']['global_cells']
    films = linkage['F1']['films']
    
    # Gather tasks
    args_list = []
    for gid, status in qc.items():
        if status in ['good', 'corrected'] and gid in f1_cells:
            local_ids = f1_cells[gid]
            for idx, lid in enumerate(local_ids):
                if lid != -1:
                    film = films[idx]
                    mask_p = MOVIE_ROOT / EXP / film / f"TrackedCells_{film}" / f"cell_{lid}_masks.csv"
                    data_p = MOVIE_ROOT / EXP / film / f"TrackedCells_{film}" / f"cell_{lid}_data.csv"
                    
                    if mask_p.exists():
                        # We need quantification if data.csv doesn't exist, or if masks.csv is newer
                        if not data_p.exists() or mask_p.stat().st_mtime > data_p.stat().st_mtime:
                            track_channel = 'bf' if 'BF' in film else 'gfp'
                            args_list.append((lid, MOVIE_ROOT, EXP, film, track_channel))
                            
    print(f"Total curated tasks to run: {len(args_list)}")
    if not args_list:
        print("Nothing to quantify!")
        return
        
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
