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
    provenance: str = "manual"

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
    status: Literal["good", "bad", "unreviewed"]
    reasons: List[str] = Field(default_factory=list)
    note: str = ""
    reviewer: str = "anonymous"


class SaveSeptumRequest(BaseModel):
    experiment: str
    film: str
    cell_id: str
    has_septum: bool
    # Explicit local-film frame coordinates. New clients should send these.
    # The aligned fields remain supported for compatibility with existing data.
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    start_aligned: Optional[int] = None
    end_aligned: Optional[int] = None
    has_septum_2: Optional[bool] = None
    start_frame_2: Optional[int] = None
    end_frame_2: Optional[int] = None
    start_aligned_2: Optional[int] = None
    end_aligned_2: Optional[int] = None
    is_white_septum: Optional[bool] = None
    is_white_septum_2: Optional[bool] = None
    offset: int = 0
    sequence: Optional[str] = None
    global_cell_id: Optional[str] = None
    label_source: Optional[str] = "cell"
    pattern_center_row: Optional[float] = None
    pattern_center_col: Optional[float] = None
    note: str = ""
    annotator: str = "anonymous"


    @validator("cell_id", pre=True)
    def coerce_cell_id_to_str(cls, v: Any) -> str:
        if v is None:
            raise ValueError("cell_id is required")
        return str(v).strip()


class MaskEditAction(BaseModel):
    time_point: int
    rle: str
