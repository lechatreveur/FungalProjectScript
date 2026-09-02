from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
import numpy as np

def validate_and_decode_rle(rle: str, height: int, width: int) -> np.ndarray:
    """Validate RLE string format and decode to binary mask array of shape (height, width) in Fortran order."""
    if not isinstance(rle, str):
        raise ValueError("RLE must be a string")
        
    s = rle.strip()
    if not s or s.lower() == "nan":
        return np.zeros((height, width), dtype=np.uint8)
        
    try:
        nums = np.fromstring(s, dtype=int, sep=' ')
    except Exception as e:
        raise ValueError(f"Failed to parse RLE numbers: {e}")
        
    if len(nums) % 2 != 0:
        raise ValueError("RLE must contain an even number of integers (start length pairs)")
        
    starts = nums[0::2] - 1
    lengths = nums[1::2]
    ends = starts + lengths
    
    if (starts < 0).any() or (ends > height * width).any():
        raise ValueError("RLE indices out of bounds")
        
    flat = np.zeros(height * width, dtype=np.uint8)
    for st, en in zip(starts, ends):
        flat[st:en] = 1
        
    return flat.reshape((height, width), order='F')


def encode_mask_to_rle(mask: np.ndarray) -> str:
    """Encode binary mask array (H, W) to Fortran-order RLE string."""
    flat = mask.flatten(order='F')
    diffs = np.diff(np.pad(flat.astype(int), (1, 1), 'constant'))
    starts = np.where(diffs == 1)[0] + 1
    ends = np.where(diffs == -1)[0] + 1
    lengths = ends - starts
    
    pairs = []
    for s, l in zip(starts, lengths):
        pairs.extend([str(s), str(l)])
    return " ".join(pairs)


class SaveMasksFramePatch(BaseModel):
    time_point: int
    old_rle_hash: Optional[str] = None
    new_rle: str


class SaveMasksRequest(BaseModel):
    experiment: str
    film: Optional[str] = None
    sequence: Optional[str] = None
    cell_id: str
    channel: Literal["bf", "gfp"] = "bf"
    masks: Optional[List[str]] = None
    changes: Optional[List[SaveMasksFramePatch]] = None
    expected_revision: Optional[str] = None
    provenance: str = "manual_gt"

    @validator("cell_id")
    def validate_cell_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("cell_id is required")
        return v.strip()


class UpdateLinkageRequest(BaseModel):
    experiment: str
    sequence: str
    global_cell: Optional[str] = None
    global_cell_id: Optional[str] = None
    film_idx: Optional[int] = None
    new_local_cell: Optional[int] = None
    global_cells: Optional[Dict[str, List[int]]] = None
    lineage: Optional[Dict[str, Any]] = None
    expected_revision: Optional[str] = None
    user: str = "anonymous"


class SaveQCRequest(BaseModel):
    experiment: str
    film: Optional[str] = None
    sequence: Optional[str] = None
    cell_id: str
    status: Literal["good", "bad", "corrected", "mistracked", "unreviewed"]
    reasons: List[str] = Field(default_factory=list)
    note: str = ""
    reviewer: str = "anonymous"


class ExportTrainingDataRequest(BaseModel):
    experiment: str
    sequence: Optional[str] = None
    film: Optional[str] = None
    subfolder: Optional[str] = None
    overwrite_all: bool = False
