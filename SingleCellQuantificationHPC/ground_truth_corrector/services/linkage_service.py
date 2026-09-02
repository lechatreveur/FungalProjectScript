import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..repositories.linkage_repository import LinkageRepository
from ..errors import NotFoundError, RevisionConflict
from ..schemas import UpdateLinkageRequest

class LinkageService:
    def __init__(self, linkage_repo: LinkageRepository):
        self.linkage_repo = linkage_repo

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

    def update_linkage(self, req: UpdateLinkageRequest, linkage_filename: str = "sequence_linkage.json") -> str:
        data, rev = self.linkage_repo.load_linkage(req.experiment, linkage_filename)
        
        if "sequences" in data and isinstance(data["sequences"], dict):
            target = data["sequences"]
        else:
            target = data

        if req.sequence not in target:
            target[req.sequence] = {
                "films": [req.sequence],
                "global_cells": {},
                "lineage": {}
            }

        seq_data = target[req.sequence]
        
        if req.global_cells is not None:
            seq_data["global_cells"] = req.global_cells
        elif req.global_cell and req.film_idx is not None and req.new_local_cell is not None:
            gc_map = seq_data.setdefault("global_cells", {})
            films = seq_data.get("films", [])
            cell_key = req.global_cell
            if cell_key not in gc_map:
                alt = f"{req.sequence}_cell_{req.global_cell}"
                if alt in gc_map:
                    cell_key = alt
                else:
                    alt2 = str(req.global_cell).split("_cell_")[-1]
                    if alt2 in gc_map:
                        cell_key = alt2
            track = gc_map.setdefault(cell_key, [-1] * len(films))
            while len(track) < len(films):
                track.append(-1)
            if 0 <= req.film_idx < len(track):
                track[req.film_idx] = req.new_local_cell

        if req.lineage is not None:
            seq_data["lineage"] = req.lineage

        return self.linkage_repo.save_linkage(req.experiment, data, req.expected_revision, linkage_filename)
