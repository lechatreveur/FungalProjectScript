from typing import Dict, Any, List, Optional
from ..repositories.qc_repository import QCRepository
from .audit_service import AuditService
from ..schemas import SaveQCRequest
from ..models import QCRecord

class QCService:
    def __init__(self, qc_repo: QCRepository, audit_service: AuditService):
        self.qc_repo = qc_repo
        self.audit_service = audit_service

    def get_qc_records(self, exp: str, target: str) -> Dict[str, Any]:
        data, rev = self.qc_repo.load_qc(exp, target)
        records = data.get("records", data) if isinstance(data, dict) else {}
        return {
            "qc_records": records,
            "revision": rev
        }

    def save_qc_record_simple(self, exp: str, target: str, cell_id: str, status: str) -> Dict[str, Any]:
        data, old_rev = self.qc_repo.load_qc(exp, target)
        if not isinstance(data, dict):
            data = {}
            
        if status == "pending":
            if cell_id in data:
                del data[cell_id]
        else:
            data[cell_id] = status
            
        new_rev = self.qc_repo.save_qc(exp, target, data)
        return {"status": "success", "qc": data, "new_revision": new_rev}

    def save_qc_record(self, req: SaveQCRequest, film: str) -> Dict[str, Any]:
        return self.save_qc_record_simple(req.experiment, film, req.cell_id, req.status)
