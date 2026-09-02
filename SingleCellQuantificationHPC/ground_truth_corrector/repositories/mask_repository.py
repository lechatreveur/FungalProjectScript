import os
import hashlib
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from ..errors import RevisionConflict, NotFoundError, DataIntegrityError
from ..security import resolve_under_root

def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_dataframe(path: Path, df: pd.DataFrame) -> str:
    """Atomically save pandas DataFrame to CSV and return its new SHA-256 checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_csv(tmp_path, index=False)
        try:
            with open(tmp_path, "rb") as f:
                os.fsync(f.fileno())
        except OSError:
            pass
        os.replace(tmp_path, path)
    except OSError:
        df.to_csv(path, index=False)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
    return compute_file_hash(path)


class MaskRepository:
    def __init__(self, base_movie_root: Path):
        self.base_root = base_movie_root

    def get_cell_mask_csv_path(self, exp: str, film: str, cell_id: int) -> Path:
        return resolve_under_root(self.base_root, exp, film, f"TrackedCells_{film}", f"cell_{cell_id}_masks.csv")

    def load_cell_masks(self, exp: str, film: str, cell_id: int) -> Tuple[pd.DataFrame, str]:
        """Load cell masks CSV as DataFrame and return current revision SHA-256 checksum."""
        path = self.get_cell_mask_csv_path(exp, film, cell_id)
        if not path.exists():
            raise NotFoundError(f"Mask CSV for cell {cell_id} not found in film {film}")
            
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise DataIntegrityError(f"Failed to parse mask CSV for cell {cell_id}: {e}")
            
        if "time_point" not in df.columns:
            raise DataIntegrityError(f"Mask CSV for cell {cell_id} missing 'time_point' column")
            
        df = df.sort_values("time_point").reset_index(drop=True)
        revision = compute_file_hash(path)
        return df, revision

    def save_cell_masks(self, exp: str, film: str, cell_id: int, df: pd.DataFrame, expected_revision: Optional[str] = None) -> str:
        """Save cell masks DataFrame atomically, verifying expected revision if provided."""
        path = self.get_cell_mask_csv_path(exp, film, cell_id)
        if expected_revision and path.exists():
            current_revision = compute_file_hash(path)
            if current_revision != expected_revision:
                raise RevisionConflict(
                    f"Cell {cell_id} in {film} was modified by another user/process. Expected {expected_revision[:8]}, current {current_revision[:8]}"
                )
                
        return atomic_write_dataframe(path, df)
