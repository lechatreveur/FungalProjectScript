from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

@dataclass(frozen=True)
class CellRef:
    """Typed reference to a cell within a specific experiment and film."""
    experiment: str
    film: str
    local_cell_id: int

    def __str__(self) -> str:
        return f"{self.experiment}/{self.film}/cell_{self.local_cell_id}"

@dataclass(frozen=True)
class GlobalCellRef:
    """Typed reference to a global sequence cell."""
    experiment: str
    sequence: str
    global_cell_id: str

@dataclass(frozen=True)
class ResolvedFrame:
    """Structured resolution of a global sequence timepoint to local film timepoint."""
    film: str
    local_cell_id: Optional[int]
    local_time: int
    global_time: int

@dataclass
class QCRecord:
    """Quality control review record for a cell."""
    cell_id: str
    status: Literal["good", "bad", "unreviewed"] = "unreviewed"
    reasons: List[str] = field(default_factory=list)
    note: str = ""
    updated_by: str = "anonymous"
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    revision: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reasons": self.reasons,
            "note": self.note,
            "updated_by": self.updated_by,
            "updated_at": self.updated_at,
            "revision": self.revision
        }

@dataclass
class AuditEntry:
    """Audit log record for scientific data modifications."""
    revision_id: str
    timestamp: str
    user: str
    experiment: str
    film_or_sequence: str
    cell_id: str
    operation: str
    source_file: str
    old_checksum: str
    new_checksum: str
    changed_frames: List[int]
    provenance: str = "manual"
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "revision_id": self.revision_id,
            "timestamp": self.timestamp,
            "user": self.user,
            "experiment": self.experiment,
            "film_or_sequence": self.film_or_sequence,
            "cell_id": self.cell_id,
            "operation": self.operation,
            "source_file": self.source_file,
            "old_checksum": self.old_checksum,
            "new_checksum": self.new_checksum,
            "changed_frames": self.changed_frames,
            "provenance": self.provenance,
            "reason": self.reason
        }
