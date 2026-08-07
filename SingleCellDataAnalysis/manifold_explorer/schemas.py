import pandas as pd
import logging

logger = logging.getLogger(__name__)

REQUIRED_STACKED_COLUMNS = {
    "cell_id",
    "time_point",
    "pol1_int_corr",
    "pol2_int_corr",
}

REQUIRED_ID_MAP_COLUMNS = {
    "new_cell_id",
    "field",
    "source"
}

def validate_stacked_schema(df: pd.DataFrame, filepath: str, experiment_name: str) -> None:
    missing = REQUIRED_STACKED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Schema error in {experiment_name} stacked CSV ({filepath}): "
            f"Missing required columns: {sorted(list(missing))}"
        )

def validate_id_map_schema(df: pd.DataFrame, filepath: str, experiment_name: str) -> None:
    missing = REQUIRED_ID_MAP_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Schema error in {experiment_name} ID map CSV ({filepath}): "
            f"Missing required columns: {sorted(list(missing))}"
        )

def validate_trajectory(grp: pd.DataFrame, gid: str, expected_frames: int | None) -> str | None:
    if expected_frames is not None and len(grp) != expected_frames:
        return f"Trajectory length is {len(grp)} instead of {expected_frames}"
        
    p1 = grp["pol1_int_corr"].values
    p2 = grp["pol2_int_corr"].values
    
    if not (pd.Series(p1).notna().all() and pd.Series(p2).notna().all()):
        return "Trajectory contains NaN values"
        
    if p1.var() < 1e-8 and p2.var() < 1e-8:
        return "Trajectory has zero variance (constant signal)"
        
    return None
