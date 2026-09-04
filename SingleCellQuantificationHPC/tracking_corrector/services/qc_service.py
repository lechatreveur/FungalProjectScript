import re
from typing import Dict, Any, List, Optional
from ..repositories.qc_repository import QCRepository
from .audit_service import AuditService
from ..schemas import SaveQCRequest
from ..models import QCRecord
from ..qc_schema import (
    GlobalCellQC,
    LocalCellQC,
    validate_global_qc_status,
    validate_local_qc_status,
    InvalidQCStatusError,
)

class QCService:
    def __init__(self, qc_repo: QCRepository, audit_service: AuditService):
        self.qc_repo = qc_repo
        self.audit_service = audit_service

    def get_qc_records(self, exp: str, target: str, level: str = "global") -> Dict[str, Any]:
        if level == "local":
            review_state = self.qc_repo.load_review_state(exp, target)
            records = {}
            for k, v in review_state.items():
                if isinstance(v, dict):
                    records[k] = v.get("status", LocalCellQC.PENDING.value)
                else:
                    records[k] = str(v)
            return {"qc_records": records, "revision": ""}
        else:
            data, rev = self.qc_repo.load_qc(exp, target)
            records = data.get("records", data) if isinstance(data, dict) else {}
            
            resolved_records = dict(records)
            for k, v in list(records.items()):
                match = re.search(r"(\d+)$", str(k))
                if match:
                    num = match.group(1)
                    resolved_records.setdefault(f"{target}_cell_{num}", v)
                    resolved_records.setdefault(f"cell_{num}", v)
                    resolved_records.setdefault(num, v)

            return {
                "qc_records": resolved_records,
                "revision": rev
            }

    def save_qc_record_simple(
        self, exp: str, target: str, cell_id: str, status: str, level: str = "global"
    ) -> Dict[str, Any]:
        cid_str = str(cell_id).strip()
        if level == "local":
            valid_status = validate_local_qc_status(status)
            review_state = self.qc_repo.load_review_state(exp, target)
            cell_info = review_state.get(cid_str)
            if cell_info is None:
                cell_info = {"total_windows": 1, "shown_windows": [], "status": valid_status}
            else:
                cell_info["status"] = valid_status
            review_state[cid_str] = cell_info
            self.qc_repo.save_review_state(exp, target, review_state)
            return {"status": "success", "level": "local", "qc": review_state}

        # Global Level write
        valid_status = validate_global_qc_status(status)
        data, old_rev = self.qc_repo.load_qc(exp, target)
        if not isinstance(data, dict):
            data = {}

        match = re.search(r"(\d+)$", cid_str)
        num_id = match.group(1) if match else cid_str

        # Find any matching keys (canonical or bare stem) to update/delete in sync
        matching_keys = set([
            k for k in data.keys()
            if k == cid_str or k.endswith(f"_cell_{num_id}") or k.endswith(f"_{num_id}") or cid_str.endswith(f"_{k}")
        ])

        if valid_status == GlobalCellQC.PENDING.value:
            for k in matching_keys:
                del data[k]
            if cid_str in data:
                del data[cid_str]
        else:
            for k in matching_keys:
                data[k] = valid_status
            data[cell_id] = valid_status
            
        new_rev = self.qc_repo.save_qc(exp, target, data)
        return {"status": "success", "level": "global", "qc": data, "new_revision": new_rev}

    def save_qc_record(self, req: SaveQCRequest, film: str) -> Dict[str, Any]:
        return self.save_qc_record_simple(req.experiment, film, req.cell_id, req.status)

