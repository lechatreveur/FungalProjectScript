import pandas as pd
import logging

logger = logging.getLogger(__name__)

def make_qc_report(cells: pd.DataFrame) -> dict:
    report = {
        "loaded_cells": len(cells),
        "included_cells": int(cells["included"].sum()),
        "excluded_cells": int((~cells["included"]).sum()),
        "exclusion_reasons": cells["exclusion_reason"].value_counts().to_dict(),
        "missing_trajectory": int(cells["trajectory_p1"].isna().sum()),
        "missing_division_time": int(cells["time_to_division"].isna().sum())
    }
    return report

def apply_qc_rules(cells: pd.DataFrame) -> pd.DataFrame:
    cells = cells.copy()
    
    # 1. Trajectory length or missing validation
    mask_no_traj = cells["trajectory_p1"].isna()
    cells.loc[mask_no_traj, "included"] = False
    cells.loc[mask_no_traj, "exclusion_reason"] = "Missing trajectory data"
    
    # 2. QC status from JSON
    # If the curation json classifies it as bad, exclude it.
    mask_bad_qc = cells["qc_status"] == "bad"
    cells.loc[mask_bad_qc, "included"] = False
    cells.loc[mask_bad_qc, "exclusion_reason"] = "Excluded by manual QC curation"
    
    # E.g. exclude specific fields or conditions if needed
    # (By default, other experiments are included unless they fail QC or lack trajectories).
    
    return cells
