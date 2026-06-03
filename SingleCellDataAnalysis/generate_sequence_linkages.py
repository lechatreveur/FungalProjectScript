import os
import re
import json
import pandas as pd
from pathlib import Path
import sys

# Try importing the map function, if it fails we just use empty mappings
try:
    from map_gfp_bf_id import map_gfp_to_bf_ids
except ImportError:
    map_gfp_to_bf_ids = None

BASE_MOVIE_ROOT = Path("/Volumes/X10 Pro/Movies")

def get_sequence_films(exp_dir):
    films = [d.name for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    
    # Custom grouping logic for M93
    if "M93" in str(exp_dir):
        fields = set()
        for f in films:
            m = re.search(r'_(F\d+)$', f)
            if m:
                fields.add(m.group(1))
                
        seq_groups = {}
        for field in fields:
            group_name = f"A14_{field}"
            ordered = [
                f"A14_FL_1_{field}",
                f"A14_BF_1_{field}",
                f"A14_FL_2_{field}",
                f"A14_BF_2_{field}",
                f"A14_FL_3_{field}",
                f"A14_BF_2b_{field}"
            ]
            actual_ordered = [f for f in ordered if f in films]
            if len(actual_ordered) > 1:
                seq_groups[group_name] = actual_ordered
        return seq_groups
        
    # Custom grouping logic for M133
    if "M133" in str(exp_dir):
        fields = set()
        for f in films:
            m = re.search(r'_(F\d+)$', f)
            if m:
                fields.add(m.group(1))
                
        seq_groups = {}
        for field in fields:
            group_name = f"YES_Scd1_D_{field}"
            ordered = [
                f"YES_Scd1_D_{field}",
                f"YES_Scd1_D_1_{field}",
                f"YES_Scd1_D_2_{field}",
                f"YES_Scd1_D_3_{field}",
                f"YES_Scd1_D_4_{field}",
                f"YES_Scd1_D_5_{field}"
            ]
            actual_ordered = [f for f in ordered if f in films]
            if len(actual_ordered) > 1:
                seq_groups[group_name] = actual_ordered
        return seq_groups
        
    seq_groups = {}
    for f in films:
        m = re.search(r'(.*?)(FL|BF)(\d*)_(F\d+)', f)
        if m:
            base = m.group(1)
            tipo = m.group(2)
            num = m.group(3)
            field = m.group(4)
            group_name = f"{base}{field}"
            seq_groups.setdefault(group_name, []).append(f)
            continue
            
        m2 = re.search(r'(.*?)_(\d+)_(F\d+)', f)
        if m2:
            base = m2.group(1)
            num = m2.group(2)
            field = m2.group(3)
            group_name = f"{base}_{field}"
            seq_groups.setdefault(group_name, []).append(f)
            continue
            
    def sort_key(f):
        m = re.search(r'(FL|BF)(\d*)', f)
        if m:
            tipo = m.group(1)
            num_str = m.group(2)
            num = int(num_str) if num_str else 0
            order = 0 if tipo == 'FL' else 1
            return (num, order)
        
        m2 = re.search(r'_(\d+)_F', f)
        if m2:
            return (int(m2.group(1)), 0)
        return (0, 0)
        
    sorted_groups = {}
    for g, flist in seq_groups.items():
        if len(flist) > 1:
            sorted_groups[g] = sorted(flist, key=sort_key)
            
    return sorted_groups


def generate_linkages(exp_name):
    exp_dir = BASE_MOVIE_ROOT / exp_name
    if not exp_dir.exists():
        return
        
    groups = get_sequence_films(exp_dir)
    if not groups:
        return
        
    output_data = {}
    for group, films in groups.items():
        print(f"Processing group {group}: {films}")
        
        f1 = films[0]
        tracked1 = exp_dir / f1 / f"TrackedCells_{f1}"
        
        if not tracked1.exists():
            continue
            
        c1_ids = []
        for cf in tracked1.iterdir():
            m = re.match(r"^cell_(\d+)_masks\.csv$", cf.name)
            if m:
                c1_ids.append(int(m.group(1)))
                
        c1_ids.sort()
        
        # Initialize global tracking: Every cell in first film gets a sequence
        global_cells = {}
        for c1 in c1_ids:
            gid = f"{group}_cell_{c1}"
            global_cells[gid] = [c1]
            
        for i in range(len(films)-1):
            fA = films[i]
            fB = films[i+1]
            
            # Get all cells in fB
            cB_ids = []
            trackedB = exp_dir / fB / f"TrackedCells_{fB}"
            if trackedB.exists():
                for cf in trackedB.iterdir():
                    m = re.match(r"^cell_(\d+)_masks\.csv$", cf.name)
                    if m:
                        cB_ids.append(int(m.group(1)))
            cB_ids.sort()
            
            rel_A = f"{fA}/TrackedCells_{fA}/"
            rel_B = f"{fB}/TrackedCells_{fB}/"
            
            def get_rle_col(film_name):
                t_dir = exp_dir / film_name / f"TrackedCells_{film_name}"
                for cf in t_dir.iterdir():
                    if cf.name.endswith("_masks.csv"):
                        try:
                            df = pd.read_csv(cf)
                            if 'rle_gfp' in df.columns and df['rle_gfp'].dropna().any():
                                return 'rle_gfp'
                            return 'rle_bf'
                        except Exception:
                            continue
                return 'rle_bf'
                
            rleA = get_rle_col(fA)
            rleB = get_rle_col(fB)
            
            mapping = {}
            if map_gfp_to_bf_ids:
                print(f"  Auto-mapping {fA} ({rleA}) -> {fB} ({rleB})")
                try:
                    sys.path.append(str(Path("/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellDataAnalysis")))
                    from map_gfp_bf_id import map_gfp_to_bf_ids as mgbi
                    m, _, _, _ = mgbi(
                        WORKING_DIR=str(exp_dir),
                        gfp_rel=rel_A,
                        bf_rel=rel_B,
                        gfp_rle_col=rleA,
                        bf_rle_col=rleB,
                        iou_min=0.01,
                        gfp_timepoint="last",
                        bf_timepoint="first",
                        assignment="one_to_one"
                    )
                    mapping = m
                except Exception as e:
                    print(f"  Error mapping: {e}")
                    
            mapped_fB = set()
            for gid, track in list(global_cells.items()):
                cA = track[-1]
                if cA == -1:
                    global_cells[gid].append(-1)
                else:
                    mapped = mapping.get(cA, -1)
                    global_cells[gid].append(mapped)
                    if mapped != -1:
                        mapped_fB.add(mapped)
            
            for cB in cB_ids:
                if cB not in mapped_fB:
                    gid = f"{group}_{fB}_cell_{cB}"
                    track = [-1] * (i + 1) + [cB]
                    global_cells[gid] = track
                
        output_data[group] = {
            "films": films,
            "global_cells": global_cells
        }
        print(f"  Created {len(global_cells)} global tracks (with unmapped links as -1).")
        
    out_file = exp_dir / "sequence_linkage.json"
    with open(out_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved {out_file}")

if __name__ == "__main__":
    generate_linkages("2026_04_23_M130")
    generate_linkages("2026_04_29_M133")
