import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
from ..security import resolve_under_root
from ..errors import NotFoundError

class QCRepository:
    def __init__(self, base_movie_root: Path):
        self.base_root = base_movie_root

    def get_qc_csv_path(self, exp: str, film: str) -> Path:
        return resolve_under_root(self.base_root, exp, film, f"TrackedCells_{film}", "qc.csv")

    def load_qc(self, exp: str, film: str) -> Dict[str, Any]:
        """Load QC data for a film, keyed by string cell_id."""
        path = self.get_qc_csv_path(exp, film)
        if not path.exists():
            return {}
        try:
            df = pd.read_csv(path)
            res = {}
            for _, row in df.iterrows():
                cid = str(int(row["cell_id"])) if pd.notna(row["cell_id"]) else ""
                if cid:
                    res[cid] = {
                        "status": row.get("status", "unreviewed"),
                        "reasons": str(row.get("reasons", "")).split(";") if pd.notna(row.get("reasons")) and str(row.get("reasons")).strip() else [],
                        "note": str(row.get("note", "")) if pd.notna(row.get("note")) else "",
                        "reviewer": str(row.get("reviewer", "")) if pd.notna(row.get("reviewer")) else ""
                    }
            return res
        except Exception:
            return {}

    def save_qc_entry(self, exp: str, film: str, cell_id: str, status: str, reasons: Optional[list] = None, note: str = "", reviewer: str = "anonymous") -> None:
        path = self.get_qc_csv_path(exp, film)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        qc_data = {}
        if path.exists():
            try:
                df = pd.read_csv(path)
                for _, row in df.iterrows():
                    cid = str(int(row["cell_id"])) if pd.notna(row["cell_id"]) else ""
                    if cid:
                        qc_data[cid] = row.to_dict()
            except Exception:
                pass

        try:
            cid_int = int(cell_id)
        except ValueError:
            cid_int = cell_id

        qc_data[str(cell_id)] = {
            "cell_id": cid_int,
            "status": status,
            "reasons": ";".join(reasons) if reasons else "",
            "note": note,
            "reviewer": reviewer
        }
        
        rows = list(qc_data.values())
        out_df = pd.DataFrame(rows)
        out_df.to_csv(path, index=False)

    def get_sequence_qc_path(self, exp: str, sequence: str) -> Path:
        return resolve_under_root(self.base_root, exp, f"qc_{sequence}.json")

    def load_sequence_qc(self, exp: str, sequence: str) -> Dict[str, Any]:
        path = self.get_sequence_qc_path(exp, sequence)
        if not path.exists():
            return {}
        try:
            import json
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def save_sequence_qc_entry(self, exp: str, sequence: str, global_id: str, status: str, reasons: Optional[list] = None, note: str = "", reviewer: str = "anonymous") -> None:
        path = self.get_sequence_qc_path(exp, sequence)
        import json
        qc_data = self.load_sequence_qc(exp, sequence)
        qc_data[str(global_id)] = {
            "status": status,
            "reasons": reasons if reasons else [],
            "note": note,
            "reviewer": reviewer
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(qc_data, f, indent=2)
