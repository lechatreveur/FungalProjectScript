import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from ..errors import DataIntegrityError
from .mask_repository import compute_file_hash, atomic_write_text
from ..security import resolve_under_root

class QCRepository:
    def __init__(self, base_movie_root: Path):
        self.base_root = base_movie_root

    def get_qc_json_path(self, exp: str, target: str) -> Path:
        p1 = self.base_root / exp / target / f"qc_{target}.json"
        if p1.exists():
            return p1
        p2 = self.base_root / exp / target / f"TrackedCells_{target}" / "cell_plots" / "gui_labels" / "qc_labels.json"
        if p2.exists():
            return p2
        return p1

    def get_septum_json_path(self, exp: str, film: str) -> Path:
        p1 = self.base_root / exp / film / f"TrackedCells_{film}" / "cell_plots" / "gui_labels" / "global_septum_alignment.json"
        if p1.exists():
            return p1
        p2 = self.base_root / exp / film / "global_septum_alignment.json"
        if p2.exists():
            return p2
        return p1

    def load_qc(self, exp: str, target: str) -> Tuple[Dict[str, Any], str]:
        path = self.get_qc_json_path(exp, target)
        if not path.exists():
            return {}, ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data, compute_file_hash(path)
        except Exception as e:
            raise DataIntegrityError(f"Failed to read QC labels JSON for {exp}/{target}: {e}")

    def save_qc(self, exp: str, target: str, data: Dict[str, Any]) -> str:
        path = self.get_qc_json_path(exp, target)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, indent=2)
        atomic_write_text(path, content)
        return compute_file_hash(path)

    def load_septum(self, exp: str, film: str) -> Tuple[Dict[str, Any], str]:
        path = self.get_septum_json_path(exp, film)
        if not path.exists():
            default_data = {
                "working_dir": str(self.base_root / exp),
                "film_name": film,
                "cell_order": [],
                "offsets": {},
                "global_interval": {"G0": 0, "G1": 55},
                "cell_intervals": {}
            }
            return default_data, ""
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data, compute_file_hash(path)
        except Exception as e:
            raise DataIntegrityError(f"Failed to load septum alignment JSON for {exp}/{film}: {e}")

    def save_septum(self, exp: str, film: str, data: Dict[str, Any]) -> str:
        path = self.get_septum_json_path(exp, film)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, indent=2)
        atomic_write_text(path, content)
        return compute_file_hash(path)

    def get_ai_cache_json_path(self, exp: str, sequence: str) -> Path:
        # Written by an offline batch script (see
        # SingleCellDataAnalysis/predict_m156_septum.py-style drivers), NOT
        # by the live /api/predict_septum endpoint - that one stays
        # deliberately unpersisted (review_only) so it can never be mistaken
        # for a saved label. This file is a separate, clearly-named cache of
        # un-reviewed model suggestions, read-only from the app's side.
        return self.base_root / exp / sequence / f"septum_ai_cache_{sequence}.json"

    def load_ai_cache(self, exp: str, sequence: str) -> Dict[str, Any]:
        path = self.get_ai_cache_json_path(exp, sequence)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # A malformed/partial cache file should never break cell
            # loading - it's a convenience layer, not source of truth.
            return {}
