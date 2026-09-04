"""
qc_schema.py

Formal schema and validation module for the two-level QC status system:
1. Global Cell Level (identity: global_cell_id, e.g. '3_F0_cell_1')
   Allowed statuses: 'good', 'bad', 'pending', 'corrected'
2. Local Cell Level (identity: per-source track, e.g. '3_F0_3_BF1_F0_cell_104' or raw 'M156_####')
   Allowed statuses: 'pending', 'mistracked'
"""

from enum import Enum
from typing import Set, Union


class GlobalCellQC(str, Enum):
    GOOD = "good"
    BAD = "bad"
    PENDING = "pending"
    CORRECTED = "corrected"

    @classmethod
    def valid_statuses(cls) -> Set[str]:
        return {item.value for item in cls}


class LocalCellQC(str, Enum):
    PENDING = "pending"
    MISTRACKED = "mistracked"

    @classmethod
    def valid_statuses(cls) -> Set[str]:
        return {item.value for item in cls}


class InvalidQCStatusError(ValueError):
    """Raised when a QC status string is invalid for a given level."""
    pass


def validate_global_qc_status(status: Union[str, GlobalCellQC]) -> str:
    if isinstance(status, GlobalCellQC):
        return status.value
    if not isinstance(status, str):
        raise InvalidQCStatusError(f"Global QC status must be a string, got {type(status).__name__}")
    normalized = status.strip().lower()
    if normalized not in GlobalCellQC.valid_statuses():
        raise InvalidQCStatusError(
            f"Invalid status '{status}' for Global Cell QC level. Allowed statuses: {sorted(list(GlobalCellQC.valid_statuses()))}"
        )
    return normalized


def validate_local_qc_status(status: Union[str, LocalCellQC]) -> str:
    if isinstance(status, LocalCellQC):
        return status.value
    if not isinstance(status, str):
        raise InvalidQCStatusError(f"Local QC status must be a string, got {type(status).__name__}")
    normalized = status.strip().lower()
    if normalized not in LocalCellQC.valid_statuses():
        raise InvalidQCStatusError(
            f"Invalid status '{status}' for Local Cell QC level. Allowed statuses: {sorted(list(LocalCellQC.valid_statuses()))}"
        )
    return normalized


# Canonical sets for filtering logic in downstream analysis & pipelines
USABLE_GLOBAL_STATUSES = {GlobalCellQC.GOOD.value, GlobalCellQC.CORRECTED.value}
EXCLUDED_GLOBAL_STATUSES = {GlobalCellQC.BAD.value}
