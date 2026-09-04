import re
import json
import threading
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from ..errors import DataIntegrityError
from .mask_repository import compute_file_hash, atomic_write_text
from ..security import resolve_under_root
from ..qc_schema import validate_global_qc_status, validate_local_qc_status, InvalidQCStatusError

class QCRepository:
    def __init__(self, base_movie_root: Path):
        self.base_root = base_movie_root
        self._lock = threading.Lock()

    @staticmethod
    def get_status_for_cell(qc_data: Dict[str, Any], cell_id: Any) -> Optional[str]:
        """
        Shared lookup method to resolve QC status for a cell.
        Handles exact key matches first, then falls back to fuzzy endswith matching
        (e.g. matching 'M156_2130' with '3_F0_cell_2130' or '2130').
        """
        if not isinstance(qc_data, dict) or cell_id is None:
            return None

        cid_str = str(cell_id).strip()
        if not cid_str:
            return None

        # Extract value if stored as dict record
        def extract_val(val: Any) -> Optional[str]:
            if isinstance(val, dict):
                return val.get("status")
            return str(val) if val is not None else None

        # 1. Exact match check
        if cid_str in qc_data:
            return extract_val(qc_data[cid_str])

        # Extract numeric suffix if available (e.g. "3_F0_cell_2130" -> "2130", "M156_2130" -> "2130")
        match = re.search(r"(\d+)$", cid_str)
        numeric_id = match.group(1) if match else cid_str

        # 2. Fuzzy matching fallback
        for k, v in qc_data.items():
            if k == cid_str or k.endswith(f"_cell_{numeric_id}") or k.endswith(f"_{numeric_id}"):
                return extract_val(v)
            if cid_str.endswith(f"_cell_{k}") or cid_str.endswith(f"_{k}"):
                return extract_val(v)

        return None

    def _resolve_exp_dir(self, exp: str) -> Path:
        if self.base_root.exists():
            # Check full date-prefixed folder ending in _exp (e.g. 2026_07_16_M156 for M156)
            for child in self.base_root.iterdir():
                if child.is_dir() and child.name.endswith(f"_{exp}") and not child.name.startswith("."):
                    return child
            # Direct match
            for child in self.base_root.iterdir():
                if child.is_dir() and child.name == exp:
                    return child
        return self.base_root / exp

    def get_qc_json_path(self, exp: str, target: str) -> Path:
        exp_dir = self._resolve_exp_dir(exp)
        p1 = exp_dir / target / f"qc_{target}.json"
        if p1.exists():
            return p1
        p2 = exp_dir / target / f"TrackedCells_{target}" / "cell_plots" / "gui_labels" / "qc_labels.json"
        if p2.exists():
            return p2
        return p1

    def get_review_state_json_path(self, exp: str, target: str) -> Path:
        qc_path = self.get_qc_json_path(exp, target)
        return qc_path.parent / f"mistrack_review_state_{target}.json"

    def get_septum_json_path(self, exp: str, film: str) -> Path:
        exp_dir = self._resolve_exp_dir(exp)
        p1 = exp_dir / film / f"TrackedCells_{film}" / "cell_plots" / "gui_labels" / "global_septum_alignment.json"
        if p1.exists():
            return p1
        p2 = exp_dir / film / "global_septum_alignment.json"
        if p2.exists():
            return p2
        return p1

    def load_qc(self, exp: str, target: str) -> Tuple[Dict[str, Any], str]:
        path = self.get_qc_json_path(exp, target)
        if not path.exists():
            return {}, ""
        with self._lock:
            text = ""
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                data = json.loads(text)
                return data, compute_file_hash(path)
            except Exception as e:
                if text:
                    try:
                        idx = text.rfind("}")
                        while idx > 0:
                            sub = text[:idx + 1]
                            try:
                                data = json.loads(sub)
                                content = json.dumps(data, indent=2)
                                atomic_write_text(path, content)
                                return data, compute_file_hash(path)
                            except Exception:
                                idx = text.rfind("}", 0, idx)
                    except Exception:
                        pass
                raise DataIntegrityError(f"Failed to read QC labels JSON for {exp}/{target}: {e}")

    def save_qc(self, exp: str, target: str, data: Dict[str, Any]) -> str:
        """Saves Global Level QC dictionary after validating all status strings."""
        if not isinstance(data, dict):
            raise InvalidQCStatusError("Global QC data must be a dictionary")
        for k, v in data.items():
            status_val = v.get("status") if isinstance(v, dict) else v
            validate_global_qc_status(str(status_val))
        path = self.get_qc_json_path(exp, target)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, indent=2)
        with self._lock:
            atomic_write_text(path, content)
            return compute_file_hash(path)

    def load_review_state(self, exp: str, target: str) -> Dict[str, Any]:
        path = self.get_review_state_json_path(exp, target)
        if not path.exists():
            return {}
        with self._lock:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}

    def save_review_state(self, exp: str, target: str, state_data: Dict[str, Any]) -> None:
        """Saves Local Level QC review state dictionary after validating statuses."""
        if not isinstance(state_data, dict):
            raise InvalidQCStatusError("Review state data must be a dictionary")
        for k, v in state_data.items():
            if isinstance(v, dict) and "status" in v:
                # Historical "exhausted" status will be migrated to "pending" with reviewed=True,
                # but during runtime writes we enforce valid local QC statuses.
                st = v["status"]
                if st == "exhausted":
                    v["status"] = "pending"
                    v["reviewed"] = True
                else:
                    validate_local_qc_status(st)
        path = self.get_review_state_json_path(exp, target)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(state_data, indent=2)
        with self._lock:
            atomic_write_text(path, content)


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
