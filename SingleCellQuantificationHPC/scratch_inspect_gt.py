#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Cell_tracking_functions import rle_decode
from ai_tracking_inference import get_centroid

def main():
    csv_path = "/Volumes/X10 Pro/Movies/2026_01_16_M96/A14_BF_2_F3/TrackedCells_A14_BF_2_F3/cell_38_masks.csv"
    df = pd.read_csv(csv_path)
    H, W = int(df.iloc[0]['height']), int(df.iloc[0]['width'])
    
    print("Time point, Area, Centroid, Touches border, Source")
    for idx, row in df.iterrows():
        t = row['time_point']
        rle = row['rle_gfp']
        if not isinstance(rle, str) or not rle or rle == "nan":
            print(f"{t}: Empty (NaN)")
            continue
        mask = rle_decode(rle, (H, W)).astype(bool)
        area = mask.sum()
        centroid = get_centroid(mask)
        print(f"{t}: area={area}, centroid=({centroid[0]:.2f}, {centroid[1]:.2f}), source={row.get('source_gfp', 'N/A')}")

if __name__ == "__main__":
    main()
