import re
import shutil
import tifffile
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from ..config import Config
from ..security import resolve_under_root
from ..schemas import validate_and_decode_rle
from .gt_frames_service import GTFramesService

class GTExportService:
    def __init__(self, config: Config, frames_service: GTFramesService):
        self.config = config
        self.frames_service = frames_service

    def get_training_destination(self, exp: str, custom_subfolder: Optional[str] = None) -> Path:
        """Determine target folder in cellpose_training_data for an experiment."""
        root = self.config.cellpose_training_root
        
        if custom_subfolder:
            dest = root / custom_subfolder
            dest.mkdir(parents=True, exist_ok=True)
            return dest
            
        exp_cfg = self.config.get_experiment_config(exp)
        if exp_cfg and exp_cfg.training_subfolder:
            dest = root / exp_cfg.training_subfolder
            dest.mkdir(parents=True, exist_ok=True)
            return dest

        # Default fallback
        dest = root / "train"
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    def build_frame_instance_mask(self, exp: str, film: str, local_t: int) -> Tuple[np.ndarray, int]:
        """Reconstruct 2D uint16 instance label map from TrackedCells CSVs at local_t."""
        tracked_dir = resolve_under_root(self.config.local_movie_root, exp, film, f"TrackedCells_{film}")
        if not tracked_dir.exists():
            return np.zeros((2000, 2000), dtype=np.uint16), 0

        # Discover dimensions
        H, W = 2000, 2000
        first_csv = next(tracked_dir.glob("cell_*_masks.csv"), None)
        if first_csv:
            try:
                df0 = pd.read_csv(first_csv)
                if not df0.empty and "width" in df0.columns:
                    W = int(df0.iloc[0]["width"])
                    H = int(df0.iloc[0]["height"])
            except Exception:
                pass

        instance_mask = np.zeros((H, W), dtype=np.uint16)
        cell_count = 0

        for csv_file in sorted(tracked_dir.glob("cell_*_masks.csv")):
            if csv_file.name.startswith("."): continue
            m = re.match(r"^cell_(\d+)_masks\.csv$", csv_file.name)
            if not m: continue
            cid = int(m.group(1))

            try:
                df = pd.read_csv(csv_file)
                rows = df[df["time_point"] == local_t]
                if rows.empty: continue

                rle = str(rows.iloc[0].get("rle_bf", ""))
                if not rle or rle.strip() == "" or rle.lower() == "nan":
                    rle = str(rows.iloc[0].get("rle_gfp", ""))
                if not rle or rle.strip() == "" or rle.lower() == "nan":
                    continue

                cell_mask = validate_and_decode_rle(rle, H, W)
                if cell_mask.any():
                    # Assign unique label cid (uint16)
                    instance_mask[cell_mask > 0] = cid
                    cell_count += 1
            except Exception:
                continue

        return instance_mask, cell_count

    def sync_keyframe_to_training(self, exp: str, film: str, local_t: int, custom_subfolder: Optional[str] = None) -> Dict[str, Any]:
        """Save/update instance mask TIFF and source image TIFF in Cellpose training folder."""
        dest_dir = self.get_training_destination(exp, custom_subfolder)
        instance_mask, cell_count = self.build_frame_instance_mask(exp, film, local_t)

        stem = f"{film}_t{local_t:03d}"
        mask_out = dest_dir / f"{stem}_masks.tif"
        raw_out = dest_dir / f"{stem}.tif"

        # 1. Write instance mask
        tifffile.imwrite(str(mask_out), instance_mask, compression="zlib")

        # 2. Copy/write raw image if missing or out of date
        if not raw_out.exists():
            try:
                raw_src = self.frames_service.get_frame_path(exp, film, local_t, channel="bf")
                if raw_src.exists():
                    shutil.copy2(str(raw_src), str(raw_out))
            except Exception:
                try:
                    raw_src = self.frames_service.get_frame_path(exp, film, local_t, channel="gfp")
                    if raw_src.exists():
                        shutil.copy2(str(raw_src), str(raw_out))
                except Exception:
                    pass

        return {
            "status": "success",
            "destination": str(dest_dir),
            "mask_file": mask_out.name,
            "raw_file": raw_out.name,
            "cell_count": cell_count,
            "time_point": local_t,
            "film": film
        }

    def export_all_keyframes(self, exp: str, sequence: Optional[str] = None, custom_subfolder: Optional[str] = None) -> Dict[str, Any]:
        """Export all 3-keyframe ground truth masks across sequence or experiment."""
        from .linkage_service import LinkageService
        from ..repositories.linkage_repository import LinkageRepository
        
        repo = LinkageRepository(self.config.local_movie_root)
        link_svc = LinkageService(repo)
        
        if sequence:
            seq_info = link_svc.get_sequences(exp).get("sequences", {}).get(sequence, {})
            films = seq_info.get("films", [sequence])
        else:
            films = [d.name for d in (self.config.local_movie_root / exp).iterdir() if (d / f"Frames_{d.name}").exists()]

        results = []
        for film in films:
            keyframes = self.frames_service.get_film_keyframes(exp, film)
            for t_val in keyframes:
                res = self.sync_keyframe_to_training(exp, film, t_val, custom_subfolder)
                results.append(res)

        return {
            "status": "success",
            "total_keyframes_exported": len(results),
            "destination": str(self.get_training_destination(exp, custom_subfolder)),
            "details": results
        }
