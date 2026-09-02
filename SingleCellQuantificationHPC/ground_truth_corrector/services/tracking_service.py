import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..config import Config
from ..security import resolve_under_root
from .linkage_service import LinkageService
from ..repositories.linkage_repository import LinkageRepository
from ..repositories.mask_repository import MaskRepository
from ..repositories.qc_repository import QCRepository

class TrackingService:
    def __init__(
        self,
        config: Config,
        linkage_repo: LinkageRepository,
        mask_repo: MaskRepository,
        qc_repo: QCRepository
    ):
        self.config = config
        self.linkage_repo = linkage_repo
        self.mask_repo = mask_repo
        self.qc_repo = qc_repo
        self.linkage_service = LinkageService(linkage_repo)

    def list_cells_for_film(self, exp: str, film: str) -> List[Dict[str, Any]]:
        tracked_dir = resolve_under_root(self.config.local_movie_root, exp, film, f"TrackedCells_{film}")
        cells = []
        if tracked_dir.exists():
            for f in sorted(tracked_dir.iterdir()):
                if f.name.startswith("."): continue
                m = re.match(r"^cell_(\d+)_masks\.csv$", f.name)
                if m:
                    cid = int(m.group(1))
                    cells.append({
                        "id": cid,
                        "global_id": str(cid),
                        "display_name": f"Cell {cid}",
                        "film": film
                    })
        cells.sort(key=lambda c: c["id"])
        return cells

    def list_cells_for_sequence(self, exp: str, sequence: str) -> Dict[str, Any]:
        seq_res = self.linkage_service.get_sequences(exp)
        sequences = seq_res.get("sequences", {})
        
        if sequence not in sequences:
            film_cells = self.list_cells_for_film(exp, sequence)
            return {"cells": film_cells, "lineage": {}}
            
        seq_info = sequences[sequence]
        films = seq_info.get("films", [])
        global_cells = seq_info.get("global_cells", {})
        lineage = seq_info.get("lineage", {})
        
        cells_data = []
        if global_cells:
            for gid, track in global_cells.items():
                first_local = -1
                for loc in track:
                    if loc != -1:
                        first_local = loc
                        break
                cells_data.append({
                    "id": first_local if first_local != -1 else 0,
                    "global_id": str(gid),
                    "display_name": f"{gid}",
                    "track": track
                })
        elif len(films) == 1:
            return {"cells": self.list_cells_for_film(exp, films[0]), "lineage": lineage}

        def natural_sort_key(c):
            gid = str(c.get("global_id", ""))
            m = re.search(r"(\d+)$", gid)
            num = int(m.group(1)) if m else 999999
            return (len(gid), num, gid)

        cells_data.sort(key=natural_sort_key)
        return {"cells": cells_data, "lineage": lineage}
