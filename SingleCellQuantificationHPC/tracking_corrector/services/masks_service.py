import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from ..config import Config
from ..errors import NotFoundError, ValidationError, RevisionConflict
from ..security import resolve_under_root
from ..repositories.mask_repository import MaskRepository
from ..services.audit_service import AuditService
from ..schemas import SaveMasksFramePatch, SaveMasksRequest

class MasksService:
    def __init__(self, mask_repo: MaskRepository, audit_service: AuditService):
        self.mask_repo = mask_repo
        self.audit_service = audit_service

    def get_cell_masks(self, exp: str, film: str, cell_id: int) -> Dict[str, Any]:
        df, revision = self.mask_repo.load_cell_masks(exp, film, cell_id)
        rle_bf = df['rle_bf'].fillna("").tolist() if 'rle_bf' in df.columns else []
        rle_gfp = df['rle_gfp'].fillna("").tolist() if 'rle_gfp' in df.columns else []
        t_vals = df['time_point'].tolist() if 'time_point' in df.columns else list(range(len(df)))
        
        return {
            "cell_id": cell_id,
            "revision": revision,
            "time_points": t_vals,
            "rle_bf": rle_bf,
            "rle_gfp": rle_gfp,
            "num_frames": len(df)
        }

    def get_sequence_masks(self, exp: str, sequence: str, cell_id: str) -> Dict[str, Any]:
        from .linkage_service import LinkageService
        from ..repositories.linkage_repository import LinkageRepository
        
        repo = LinkageRepository(self.mask_repo.base_root)
        linkage_svc = LinkageService(repo, self.audit_service)
        
        seq_res = linkage_svc.get_sequences(exp)
        sequences = seq_res.get("sequences", {})
        
        if sequence not in sequences:
            try:
                cid_int = int(cell_id)
            except ValueError:
                m = re.search(r"(\d+)$", str(cell_id))
                cid_int = int(m.group(1)) if m else 1
                
            df, revision = self.mask_repo.load_cell_masks(exp, sequence, cid_int)
            rle_bf = df['rle_bf'].fillna("").tolist() if 'rle_bf' in df.columns else []
            rle_gfp = df['rle_gfp'].fillna("").tolist() if 'rle_gfp' in df.columns else []
            track_channel = "gfp" if ("FL" in sequence or any(rle_gfp)) else "bf"
            masks = rle_gfp if track_channel == "gfp" else rle_bf
            w = int(df.iloc[0]['width']) if 'width' in df.columns and not df.empty else 512
            h = int(df.iloc[0]['height']) if 'height' in df.columns and not df.empty else 512
            return {
                "masks": masks,
                "num_frames": len(df),
                "width": w,
                "height": h,
                "track_channel": track_channel,
                "local_film": sequence
            }

        seq_info = sequences[sequence]
        films = seq_info.get("films", [sequence])
        global_cells = seq_info.get("global_cells", {})
        
        local_ids = global_cells.get(str(cell_id), [-1] * len(films))
        
        all_masks = []
        boundaries = []
        current_len = 0
        w, h = 512, 512
        track_channel = "bf"
        
        for i, film in enumerate(films):
            boundaries.append(current_len)
            local_id = local_ids[i] if i < len(local_ids) else -1
            
            tracked_dir = self.mask_repo.base_root / exp / film / f"TrackedCells_{film}"
            L = 100
            if tracked_dir.exists():
                csvs = list(tracked_dir.glob("cell_*_masks.csv"))
                if csvs:
                    try:
                        df_tmp = pd.read_csv(csvs[0])
                        L = len(df_tmp)
                    except Exception:
                        pass
                        
            if local_id == -1:
                all_masks.extend([""] * L)
                current_len += L
                continue
                
            try:
                df, _ = self.mask_repo.load_cell_masks(exp, film, local_id)
                if 'width' in df.columns and not df.empty:
                    w = int(df.iloc[0]['width'])
                if 'height' in df.columns and not df.empty:
                    h = int(df.iloc[0]['height'])
                    
                rle_col = 'rle_bf'
                if 'rle_gfp' in df.columns and any(isinstance(x, str) and x.strip() for x in df['rle_gfp'].dropna()):
                    track_channel = 'gfp'
                    rle_col = 'rle_gfp'
                    
                masks = df[rle_col].fillna("").tolist() if rle_col in df.columns else [""] * L
                if len(masks) < L:
                    masks.extend([""] * (L - len(masks)))
                elif len(masks) > L:
                    masks = masks[:L]
                    
                all_masks.extend(masks)
            except Exception:
                all_masks.extend([""] * L)
                
            current_len += L

        return {
            "masks": all_masks,
            "num_frames": len(all_masks),
            "width": w,
            "height": h,
            "track_channel": track_channel,
            "film_boundaries": boundaries,
            "linkage_details": {"films": films, "local_ids": local_ids},
            "local_film": films[0] if films else None
        }

    def save_cell_masks(
        self,
        req: SaveMasksRequest,
        film: str,
        cell_id: int
    ) -> Dict[str, Any]:
        df, old_revision = self.mask_repo.load_cell_masks(req.experiment, film, cell_id)
        if req.expected_revision and req.expected_revision != old_revision:
            raise RevisionConflict(
                f"Conflict saving masks for cell {cell_id} on {film}. Expected revision {req.expected_revision}, current is {old_revision}."
            )
            
        rle_col = f"rle_{req.channel}"
        if rle_col not in df.columns:
            rle_col = "rle_bf" if "rle_bf" in df.columns else "rle_gfp"
            
        changes = req.changes or []
        for action in changes:
            t = action.time_point
            if 0 <= t < len(df):
                df.at[t, rle_col] = action.new_rle
                
        new_revision = self.mask_repo.save_cell_masks(req.experiment, film, cell_id, df)
        
        source_path = self.mask_repo.get_cell_mask_csv_path(req.experiment, film, cell_id)
        self.audit_service.record_revision(
            user=req.user,
            experiment=req.experiment,
            film_or_sequence=film,
            cell_id=str(cell_id),
            operation="save_masks",
            source_file_path=source_path,
            old_checksum=old_revision,
            new_checksum=new_revision,
            changed_frames=[action.time_point for action in changes]
        )
        
        return {
            "status": "success",
            "cell_id": cell_id,
            "revision": new_revision
        }
