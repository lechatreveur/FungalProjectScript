import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..repositories.linkage_repository import LinkageRepository
from .audit_service import AuditService
from ..models import ResolvedFrame
from ..errors import NotFoundError, ValidationError, RevisionConflict
from ..schemas import UpdateLinkageRequest

class LinkageService:
    def __init__(self, linkage_repo: LinkageRepository, audit_service: AuditService):
        self.linkage_repo = linkage_repo
        self.audit_service = audit_service

    def get_sequences(self, exp: str, linkage_filename: str = "sequence_linkage.json") -> Dict[str, Any]:
        data, rev = self.linkage_repo.load_linkage(exp, linkage_filename)
        
        if "sequences" in data and isinstance(data["sequences"], dict):
            raw_seqs = data["sequences"]
        else:
            raw_seqs = {
                k: v for k, v in data.items()
                if k not in ("schema_version", "revision", "metadata") and isinstance(v, dict) and "films" in v
            }
            
        result_seqs = dict(raw_seqs)
        
        # Add isolated films as pseudo-sequences
        exp_dir = self.linkage_repo.base_root / exp
        if exp_dir.exists():
            filmed_in_seqs = set()
            for seq_data in result_seqs.values():
                if isinstance(seq_data, dict):
                    filmed_in_seqs.update(seq_data.get("films", []))
                    
            for item in sorted(exp_dir.iterdir()):
                if item.is_dir() and not item.name.startswith(".") and not item.name.startswith("_"):
                    film_name = item.name
                    if "snap" in film_name.lower() or re.search(r'N1_\d+_F', film_name):
                        continue
                    # Check if film folder has TrackedCells or Frames
                    tracked = item / f"TrackedCells_{film_name}"
                    frames = item / f"Frames_{film_name}"
                    if tracked.is_dir() or frames.is_dir():
                        if film_name not in filmed_in_seqs and film_name not in result_seqs:
                            result_seqs[film_name] = {
                                "films": [film_name],
                                "global_cells": {},
                                "lineage": {}
                            }
                    
        return {
            "sequences": result_seqs,
            "revision": rev
        }

    def resolve_global_t(
        self,
        exp: str,
        sequence: str,
        global_t: int,
        film_frame_counts: Dict[str, int],
        linkage_filename: str = "sequence_linkage.json"
    ) -> ResolvedFrame:
        if global_t < 0:
            raise ValidationError(f"Global time index cannot be negative (got {global_t})")

        seq_res = self.get_sequences(exp, linkage_filename)
        sequences = seq_res["sequences"]

        if sequence not in sequences:
            raise NotFoundError(f"Sequence '{sequence}' not found in experiment '{exp}'")

        seq_info = sequences[sequence]
        films = seq_info.get("films", [])
        if not films:
            raise ValidationError(f"Sequence '{sequence}' has no films")

        curr_t = 0
        for idx, film in enumerate(films):
            f_count = film_frame_counts.get(film, 0)
            if curr_t <= global_t < curr_t + f_count:
                local_t = global_t - curr_t
                return ResolvedFrame(
                    film=film,
                    local_cell_id=None,
                    local_time=local_t,
                    global_time=global_t
                )
            curr_t += f_count

        raise ValidationError(f"Global time point {global_t} out of range for sequence '{sequence}' (max {curr_t - 1})")

    def update_linkage(
        self,
        req: UpdateLinkageRequest,
        user: str = "anonymous",
        linkage_filename: str = "sequence_linkage.json"
    ) -> Dict[str, Any]:
        data, old_rev = self.linkage_repo.load_linkage(req.experiment, linkage_filename)
        
        # Determine container
        if "sequences" in data and isinstance(data["sequences"], dict):
            sequences = data["sequences"]
        else:
            sequences = data
            
        if req.sequence not in sequences:
            sequences[req.sequence] = {
                "films": [req.sequence],
                "global_cells": {},
                "lineage": {}
            }

        seq_data = sequences[req.sequence]
        global_cells = seq_data.setdefault("global_cells", {})

        g_id = req.global_cell_id or req.global_cell
        if req.film_idx is not None and req.new_local_cell is not None and g_id:
            films_count = len(seq_data.get("films", []))
            if g_id in global_cells:
                local_ids = list(global_cells[g_id])
                while len(local_ids) < films_count:
                    local_ids.append(-1)
                if 0 <= req.film_idx < len(local_ids):
                    local_ids[req.film_idx] = req.new_local_cell
                    global_cells[g_id] = local_ids
            else:
                local_ids = [-1] * films_count
                if 0 <= req.film_idx < len(local_ids):
                    local_ids[req.film_idx] = req.new_local_cell
                global_cells[g_id] = local_ids
        
        if req.global_cells:
            for cid_str, l_ids in req.global_cells.items():
                global_cells[cid_str] = l_ids
            
        if req.lineage is not None:
            seq_data["lineage"] = req.lineage

        new_rev = self.linkage_repo.save_linkage(req.experiment, data, expected_revision=req.expected_revision, linkage_filename=linkage_filename)
        source_path = self.linkage_repo.get_linkage_json_path(req.experiment, linkage_filename)

        self.audit_service.record_revision(
            user=user,
            experiment=req.experiment,
            film_or_sequence=req.sequence,
            cell_id="multi",
            operation="update_linkage",
            source_file_path=source_path,
            old_checksum=old_rev,
            new_checksum=new_rev,
            changed_frames=[]
        )

        return {
            "status": "success",
            "sequence": req.sequence,
            "new_revision": new_rev
        }
