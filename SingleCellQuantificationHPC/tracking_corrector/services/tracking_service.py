import re
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from ..security import resolve_under_root
from ..repositories.mask_repository import atomic_write_dataframe
from ..repositories.linkage_repository import LinkageRepository
from .audit_service import AuditService

class TrackingService:
    def __init__(self, base_movie_root: Path):
        self.base_root = base_movie_root
        self.linkage_repo = LinkageRepository(base_movie_root)

    def list_film_cells(self, exp: str, film: str) -> List[Dict[str, Any]]:
        tracked_dir = resolve_under_root(self.base_root, exp, film, f"TrackedCells_{film}")
        if not tracked_dir.exists():
            return []
            
        cell_ids = []
        for f in tracked_dir.iterdir():
            if f.name.startswith("."):
                continue
            m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
            if m:
                cell_ids.append(int(m.group(1)))
                
        cell_ids.sort()
        return [{"global_id": str(cid), "display_name": f"Cell {cid}"} for cid in cell_ids]

    def list_sequence_cells(self, exp: str, sequence: str) -> Dict[str, Any]:
        from .linkage_service import LinkageService
        audit_svc = AuditService(self.base_root)
        linkage_svc = LinkageService(self.linkage_repo, audit_svc)
        
        seq_res = linkage_svc.get_sequences(exp)
        sequences = seq_res.get("sequences", {})
        
        if sequence not in sequences:
            film_cells = self.list_film_cells(exp, sequence)
            return {"cells": film_cells, "lineage": {}}
            
        seq_info = sequences[sequence]
        films = seq_info.get("films", [])
        global_cells = seq_info.get("global_cells", {})
        lineage = seq_info.get("lineage", {})
        
        if len(films) == 1 and not global_cells:
            film = films[0]
            f_cells = self.list_film_cells(exp, film)
            global_cells = {c["global_id"]: [int(c["global_id"])] for c in f_cells}
            
        def get_sort_key(k: str):
            s = str(k)
            m = re.search(r"(\d+)$", s)
            if m:
                return (0, int(m.group(1)))
            return (1, s)
            
        sorted_keys = sorted(list(global_cells.keys()), key=get_sort_key)
        valid_cells = [c for c in sorted_keys if global_cells[c] and global_cells[c][-1] != -1]
        
        base_names = {}
        name_count = {}
        for c in valid_cells:
            m = re.search(r"(\d+)$", str(c))
            b_name = f"Cell {m.group(1)}" if m else str(c)
            base_names[c] = b_name
            name_count[b_name] = name_count.get(b_name, 0) + 1
            
        def origin_film_hint(global_id, local_ids):
            prefix = f"{sequence}_"
            gid_str = str(global_id)
            if gid_str.startswith(prefix) and "_cell_" in gid_str:
                inner_part = gid_str[len(prefix):gid_str.rfind("_cell_")]
                if inner_part:
                    parts = inner_part.split("_")
                    return "_".join(parts[-2:]) if len(parts) >= 2 else inner_part
            
            for i, lid in enumerate(local_ids):
                if lid != -1 and i < len(films):
                    parts = films[i].split("_")
                    return "_".join(parts[-2:]) if len(parts) >= 2 else films[i]
            return ""
            
        cells_data = []
        for c in valid_cells:
            b_name = base_names[c]
            if name_count[b_name] > 1:
                hint = origin_film_hint(c, global_cells[c])
                disp_name = f"{b_name} ({hint})" if hint else b_name
            else:
                disp_name = b_name
                
            local_ids = global_cells.get(c, [])
            first_film = ""
            first_lid = ""
            for idx, lid in enumerate(local_ids):
                if lid != -1 and idx < len(films):
                    first_film = films[idx]
                    first_lid = str(lid)
                    break

            cells_data.append({
                "global_id": str(c),
                "display_name": disp_name,
                "film": first_film,
                "cell_id": first_lid
            })
            
        return {"cells": cells_data, "lineage": lineage}

