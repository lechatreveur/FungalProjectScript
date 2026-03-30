#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 17:17:04 2025

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_3d_periods.py - 3D scatterplot of pooled period features (pol1 and pol2)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from SingleCellDataAnalysis.config import WORKING_DIR

INPUT_CSV = os.path.join(WORKING_DIR, "gp_summary_features.csv")

print(f"📥 Loading GP summary features from {INPUT_CSV}")
df = pd.read_csv(INPUT_CSV)

# Extract period features
periods_pol1 = df[["pol1_p1", "pol1_p2", "pol1_p3"]].copy()
periods_pol2 = df[["pol2_p1", "pol2_p2", "pol2_p3"]].copy()

# Rename for consistency before pooling
periods_pol1.columns = ["p1", "p2", "p3"]
periods_pol2.columns = ["p1", "p2", "p3"]

# Pool both pol1 and pol2 periods together
pooled_periods = pd.concat([periods_pol1, periods_pol2], ignore_index=True)

# ==== 3D Scatterplot ====
print("🧭 Plotting 3D scatter of pooled periods...")
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pooled_periods["p1"], pooled_periods["p2"], pooled_periods["p3"], alpha=0.6)

ax.set_xlabel("Period 1")
ax.set_ylabel("Period 2")
ax.set_zlabel("Period 3")
ax.set_title("3D Scatter Plot of Pooled Periods (pol1 + pol2)")

plt.tight_layout()
plt.show()
