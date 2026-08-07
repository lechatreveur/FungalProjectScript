import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from .config import ExperimentConfig

logger = logging.getLogger(__name__)

# Run-length decoding for masks
def rle_decode(rle_str: str, shape: tuple[int, int]) -> np.ndarray:
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def load_observed_division_times(config: ExperimentConfig) -> tuple[dict[str, float], dict[str, dict]]:
    """
    Returns a dictionary of gcid -> division_time, and a dictionary of alignments.
    """
    div_times = {}
    alignments = {}
    
    if not config.linkage_json or not os.path.exists(config.linkage_json):
        return div_times, alignments
        
    with open(config.linkage_json, encoding="utf-8") as f:
        linkage = json.load(f)
        
    # Standard source start times and resolutions from source config
    # We will gather film names and scan alignments
    # First, scan which films are in linkage
    all_films = []
    for fov_key, fov_data in linkage.items():
        all_films.extend(fov_data.get("films", []))
    all_films = list(set(all_films))
    
    for film in all_films:
        json_path = config.root / film / f"TrackedCells_{film}" / "cell_plots" / "gui_labels" / "global_septum_alignment.json"
        if json_path.exists():
            with open(json_path, encoding="utf-8") as f:
                alignments[film] = json.load(f)
                
    # Now map each global cell to its division time
    for fov_key, fov_data in linkage.items():
        global_cells = fov_data.get("global_cells", {})
        films = fov_data.get("films", [])
        
        for gcid, local_ids in global_cells.items():
            global_div_time = None
            for k, lcid in enumerate(local_ids):
                if k >= len(films): continue
                film = films[k]
                if lcid == -1 or film not in alignments:
                    continue
                    
                align = alignments[film]
                offsets = align.get("offsets", {})
                intervals = align.get("cell_intervals", {})
                lcid_str = str(lcid)
                
                # Check if this source has configuration for start_time and resolution
                # Linkage ordering maps to sources. Let's find which source this index corresponds to.
                # For M133: GFP1 (idx 0), GFP2 (idx 2), GFP3 (idx 4)
                # Let's resolve the source name
                source_name = None
                if config.name == "Sept17":
                    source_name = "GFP1" if k < 4 else "GFP2" # TP1 (0-3) is GFP1, TP2 (4-7) is GFP2
                else:
                    # Generic mapping from idx k to source
                    # (Usually mapped via SourceConfig where we can query midpoint_min, etc.)
                    # Let's inspect source names
                    src_keys = list(config.sources.keys())
                    if k < len(src_keys):
                        source_name = src_keys[k]
                        
                if not source_name:
                    continue
                    
                src_cfg = config.sources.get(source_name)
                if not src_cfg:
                    continue
                    
                if lcid_str in intervals:
                    interval = intervals[lcid_str]
                    if interval.get("has_septum"):
                        end_val = interval.get("end_aligned")
                        if end_val is not None and end_val != -1 and str(end_val) != "-1":
                            offset = int(offsets.get(lcid_str, 0))
                            local_end = int(end_val) - offset
                            div_time = src_cfg.start_time_min + local_end * src_cfg.time_res_min
                            if global_div_time is None:
                                global_div_time = div_time
            if global_div_time is not None:
                div_times[gcid] = global_div_time
                
    return div_times, alignments

def estimate_missing_division_times(config: ExperimentConfig, 
                                    global_div_times: dict[str, float]) -> dict[str, float]:
    """
    Fits linear area-time growth curve dynamically on cells with division times
    and uses it to estimate missing division times for other cells.
    """
    no_septum_shifts = {}
    if not config.linkage_json or not os.path.exists(config.linkage_json):
        return no_septum_shifts
        
    with open(config.linkage_json, encoding="utf-8") as f:
        linkage = json.load(f)
        
    bf_times = []
    bf_areas = []
    
    # 1. Fit growth model on cells with known division times
    for fov_key, fov_data in linkage.items():
        global_cells = fov_data.get("global_cells", {})
        films = fov_data.get("films", [])
        
        for gcid, local_ids in global_cells.items():
            if gcid not in global_div_times:
                continue
            div_time = global_div_times[gcid]
            
            # Use brightfield film indices to extract areas
            # Sept17: indices 1, 3. M135: 1, 3. M133: 1, 3, 5.
            # We look at any SourceConfig with modality == "BF" or ending in "BF" / containing "BF"
            bf_indices = []
            for src_name, src_cfg in config.sources.items():
                if "BF" in src_name or src_cfg.modality == "BF":
                    # Let's find index in films matching this film folder template or sequence
                    for idx, film in enumerate(films):
                        # Simple match: if film folder name contains the configured name pattern
                        pass
            
            # Fallback hardcoded defaults if no BF indices configured:
            if config.name == "Sept17":
                bf_indices = [1, 3]
            elif config.name == "M133":
                bf_indices = [1, 3, 5]
            else:
                bf_indices = [1, 3]
                
            for k in bf_indices:
                if k >= len(local_ids) or k >= len(films): continue
                lcid = local_ids[k]
                if lcid == -1: continue
                film = films[k]
                
                # Resolve source config to get timing details
                source_name = None
                src_keys = list(config.sources.keys())
                if k < len(src_keys):
                    source_name = src_keys[k]
                if not source_name: continue
                src_cfg = config.sources.get(source_name)
                if not src_cfg: continue
                
                masks_csv = config.root / film / f"TrackedCells_{film}" / f"cell_{lcid}_masks.csv"
                if masks_csv.exists():
                    try:
                        df_m = pd.read_csv(masks_csv)
                        for _, row in df_m.iterrows():
                            t = int(row['time_point'])
                            rle_val = row.get('rle_bf')
                            if isinstance(rle_val, str) and rle_val.strip():
                                h, w = int(row['height']), int(row['width'])
                                mask = rle_decode(rle_val, (h, w))
                                area = int(mask.sum())
                                if area > 0:
                                    gtime = src_cfg.start_time_min + t * src_cfg.time_res_min
                                    rel_t = gtime - div_time
                                    if rel_t <= 0:
                                        bf_times.append(rel_t)
                                        bf_areas.append(area)
                    except Exception:
                        continue
                        
    # Fit line
    if len(bf_times) > 5:
        m_bf, c_bf = np.polyfit(bf_times, bf_areas, 1)
        logger.info(f"Fitted BF timing curve for {config.name}: slope={m_bf:.4f}, intercept={c_bf:.4f}")
    else:
        # Fallback to Sept17 defaults
        m_bf, c_bf = 8.38609, 5925.471
        logger.info(f"Using default BF timing curve parameters for {config.name}")
        
    # 2. Predict missing times
    for fov_key, fov_data in linkage.items():
        global_cells = fov_data.get("global_cells", {})
        films = fov_data.get("films", [])
        
        for gcid, local_ids in global_cells.items():
            if gcid in global_div_times:
                continue
                
            bf_times = []
            bf_areas = []
            
            bf_indices = [1, 3, 5] if config.name == "M133" else [1, 3]
            for k in bf_indices:
                if k >= len(local_ids) or k >= len(films): continue
                lcid = local_ids[k]
                if lcid == -1: continue
                film = films[k]
                
                source_name = None
                src_keys = list(config.sources.keys())
                if k < len(src_keys):
                    source_name = src_keys[k]
                if not source_name: continue
                src_cfg = config.sources.get(source_name)
                if not src_cfg: continue
                
                masks_csv = config.root / film / f"TrackedCells_{film}" / f"cell_{lcid}_masks.csv"
                if masks_csv.exists():
                    try:
                        df_m = pd.read_csv(masks_csv)
                        for _, row in df_m.iterrows():
                            t = int(row['time_point'])
                            rle_val = row.get('rle_bf')
                            if isinstance(rle_val, str) and rle_val.strip():
                                h, w = int(row['height']), int(row['width'])
                                mask = rle_decode(rle_val, (h, w))
                                area = int(mask.sum())
                                if area > 0:
                                    gtime = src_cfg.start_time_min + t * src_cfg.time_res_min
                                    bf_times.append(gtime)
                                    bf_areas.append(area)
                    except Exception:
                        continue
                        
            if len(bf_times) > 0:
                mean_T = np.mean(bf_times)
                mean_A = np.mean(bf_areas)
                tau = mean_T - (mean_A - c_bf) / m_bf
                no_septum_shifts[gcid] = tau
                
    return no_septum_shifts
