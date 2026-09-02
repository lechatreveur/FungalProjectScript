import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from ..errors import RevisionConflict, NotFoundError, DataIntegrityError
from ..security import resolve_under_root

def compute_file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class LinkageRepository:
    def __init__(self, base_movie_root: Path):
        self.base_root = base_movie_root

    def get_linkage_json_path(self, exp: str, linkage_filename: str = "sequence_linkage.json") -> Path:
        return resolve_under_root(self.base_root, exp, linkage_filename)

    def load_linkage(self, exp: str, linkage_filename: str = "sequence_linkage.json") -> Tuple[Dict[str, Any], str]:
        path = self.get_linkage_json_path(exp, linkage_filename)
        if not path.exists():
            return {}, ""
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            raise DataIntegrityError(f"Failed to parse sequence linkage file {path}: {e}")
            
        revision = compute_file_hash(path)
        return data, revision

    def save_linkage(self, exp: str, data: Dict[str, Any], expected_revision: Optional[str] = None, linkage_filename: str = "sequence_linkage.json") -> str:
        path = self.get_linkage_json_path(exp, linkage_filename)
        if expected_revision and path.exists():
            current_revision = compute_file_hash(path)
            if current_revision != expected_revision:
                raise RevisionConflict(
                    f"Linkage file for {exp} modified by another user. Expected {expected_revision[:8]}, current {current_revision[:8]}"
                )
                
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
        return compute_file_hash(path)
