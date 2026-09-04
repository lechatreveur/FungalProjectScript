"""
Fungal Cell Tracking Corrector Package (`tracking_corrector`).
Scientific data editing suite for fungal cell segmentation, tracking, septum alignment, and QC annotation.
"""

__version__ = "2.0.0"

from .qc_schema import (
    GlobalCellQC,
    LocalCellQC,
    InvalidQCStatusError,
    validate_global_qc_status,
    validate_local_qc_status,
    USABLE_GLOBAL_STATUSES,
    EXCLUDED_GLOBAL_STATUSES,
)

