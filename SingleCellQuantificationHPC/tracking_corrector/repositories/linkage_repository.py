import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from ..errors import NotFoundError, RevisionConflict, DataIntegrityError
from .mask_repository import compute_file_hash, atomic_write_text
from ..security import resolve_under_root

class LinkageRepository:
    def __init__(self, base_movie_root: Path):
        self.base_root = base_movie_root

    def get_linkage_json_path(self, exp: str, linkage_filename: str = "sequence_linkage.json") -> Path:
        return resolve_under_root(self.base_root, exp, linkage_filename)

    def load_linkage(self, exp: str, linkage_filename: str = "sequence_linkage.json") -> Tuple[Dict[str, Any], str]:
        path = self.get_linkage_json_path(exp, linkage_filename)
        if not path.exists():
            return {"schema_version": 1, "sequences": {}}, ""
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise DataIntegrityError(f"Failed to parse linkage JSON for {exp}: {e}")
            
        revision = compute_file_hash(path)
        return data, revision

    def save_linkage(self, exp: str, data: Dict[str, Any], expected_revision: Optional[str] = None, linkage_filename: str = "sequence_linkage.json") -> str:
        path = self.get_linkage_json_path(exp, linkage_filename)
        if expected_revision and path.exists():
            current_rev = compute_file_hash(path)
            if current_rev != expected_revision:
                raise RevisionConflict(f"Sequence linkage for {exp} was modified by another session.")
                
        content = json.dumps(data, indent=2)
        atomic_write_text(path, content)
        return compute_file_hash(path)
