import os, glob
import pandas as pd

working_dir = "/Volumes/X10 Pro/Movies/2026_04_29_M133"
films = ["YES_Scd1_D_2", "YES_Scd1_D_4"]
fields = ["F0", "F1", "F2"]

for film in films:
    for field in fields:
        d = os.path.join(working_dir, f"{film}_{field}", f"TrackedCells_{film}_{field}")
        if not os.path.exists(d): continue
        csvs = glob.glob(os.path.join(d, "cell_*_data.csv"))
        dfs = []
        for c in csvs:
            try:
                dfs.append(pd.read_csv(c))
            except pd.errors.EmptyDataError:
                pass
        if dfs:
            all_cells = pd.concat(dfs, ignore_index=True)
            all_cells.to_csv(os.path.join(d, "all_cells_time_series.csv"), index=False)
            print(f"Generated all_cells_time_series.csv for {film}_{field} with {len(all_cells)} rows")
