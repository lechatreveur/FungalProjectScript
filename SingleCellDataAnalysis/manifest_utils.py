#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:27:58 2026

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import pandas as pd

FIELD_RE = re.compile(r"_F(\d+)$")  # matches ..._F0 / ..._F1 / ..._F2

def load_manifest(working_dir: str,
                  relpath: str = "training_dataset/manifest.csv") -> pd.DataFrame:
    path = os.path.join(working_dir, relpath)
    m = pd.read_csv(path)

    # types
    m["cell_id"] = pd.to_numeric(m["cell_id"], errors="coerce").astype("Int64")
    m["has"] = pd.to_numeric(m["has"], errors="coerce").fillna(0).astype(int)
    m["offset"] = pd.to_numeric(m["offset"], errors="coerce")

    # Only has==1 is meaningful
    m.loc[m["has"] != 1, "offset"] = np.nan

    # derive field from film_name
    def _field(name):
        mo = FIELD_RE.search(str(name))
        return f"F{mo.group(1)}" if mo else None
    m["field"] = m["film_name"].map(_field)

    return m