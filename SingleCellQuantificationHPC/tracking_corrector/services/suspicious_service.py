import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
from ..repositories.mask_repository import MaskRepository
from ..security import resolve_under_root

class SuspiciousService:
    def __init__(self, mask_repo: MaskRepository):
        self.mask_repo = mask_repo
        self.base_root = mask_repo.base_root

    def analyze_suspicious_cells(
        self,
        exp: str,
        target: str,
        dist_threshold: float = 30.0
    ) -> Dict[str, Any]:
        target_dir = self.base_root / exp / target
        cache_file = target_dir / f"suspicious_{target}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    disk_data = json.load(f)
                    return {"suspicious": disk_data}
            except Exception:
                pass

        return {"suspicious": {}}
