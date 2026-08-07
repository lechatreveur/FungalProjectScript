import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from ..models import AuditEntry
from ..repositories.mask_repository import compute_file_hash, atomic_write_text

class AuditService:
    def __init__(self, base_movie_root: Path):
        self.base_root = base_movie_root
        self.audit_dir = base_movie_root / ".tracking_corrector"
        self.revisions_dir = self.audit_dir / "revisions"
        self.audit_log_path = self.audit_dir / "audit.jsonl"
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.revisions_dir.mkdir(parents=True, exist_ok=True)

    def record_revision(
        self,
        user: str,
        experiment: str,
        film_or_sequence: str,
        cell_id: str,
        operation: str,
        source_file_path: Path,
        old_checksum: str,
        new_checksum: str,
        changed_frames: List[int],
        provenance: str = "manual",
        reason: str = ""
    ) -> AuditEntry:
        revision_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Save copy of source file if it exists
        if source_file_path.exists():
            category = source_file_path.name.replace(".csv", "").replace(".json", "")
            backup_path = self.revisions_dir / f"{revision_id}_{category}{source_file_path.suffix}"
            try:
                content = source_file_path.read_text(encoding="utf-8")
                atomic_write_text(backup_path, content)
            except Exception:
                pass

        entry = AuditEntry(
            revision_id=revision_id,
            timestamp=timestamp,
            user=user,
            experiment=experiment,
            film_or_sequence=film_or_sequence,
            cell_id=cell_id,
            operation=operation,
            source_file=str(source_file_path.relative_to(self.base_root) if source_file_path.is_relative_to(self.base_root) else source_file_path),
            old_checksum=old_checksum,
            new_checksum=new_checksum,
            changed_frames=changed_frames,
            provenance=provenance,
            reason=reason
        )

        # Append to audit.jsonl
        line = json.dumps(entry.to_dict()) + "\n"
        with open(self.audit_log_path, "a", encoding="utf-8") as f:
            f.write(line)

        return entry
