import sys
import numpy as np
from pathlib import Path
import pandas as pd

def rle_decode(rle_str, shape):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

exp = "2026_01_08_M93"
BASE = Path(f"/Volumes/X10 Pro/Movies/{exp}")

fA = "A14_FL_2_F1"
fB = "A14_BF_2_F1"
cB = 6

# Load cell 6 from fB
maskB_path = BASE / fB / f"TrackedCells_{fB}" / f"cell_{cB}_masks.csv"
if not maskB_path.exists():
    print(f"Cell {cB} masks not found in {fB}")
    sys.exit(1)

dfB = pd.read_csv(maskB_path)
rleB = dfB.iloc[0]['rle_bf']
wB = int(dfB.iloc[0]['width'])
hB = int(dfB.iloc[0]['height'])
maskB = rle_decode(rleB, (hB, wB))

fA_dir = BASE / fA / f"TrackedCells_{fA}"
max_overlap = 0
best_cA = -1
overlaps = {}

for cf in fA_dir.glob("cell_*_masks.csv"):
    cA = int(cf.stem.split("_")[1])
    dfA = pd.read_csv(cf)
    
    rle_col = 'rle_bf'
    if 'rle_gfp' in dfA.columns and dfA['rle_gfp'].dropna().any():
        rle_col = 'rle_gfp'
        
    rleA = dfA.iloc[-1][rle_col]
    if pd.isna(rleA) or not str(rleA).strip():
        continue
        
    wA = int(dfA.iloc[-1]['width'])
    hA = int(dfA.iloc[-1]['height'])
    maskA = rle_decode(rleA, (hA, wA))
    
    intersection = np.sum(maskA & maskB)
    areaA = np.sum(maskA)
    areaB = np.sum(maskB)
    
    if intersection > 0:
        iou = intersection / (areaA + areaB - intersection)
        overlapB = intersection / areaB
        overlaps[cA] = {"intersection": intersection, "iou": iou, "overlap_of_B": overlapB, "areaA": areaA, "areaB": areaB}
        if intersection > max_overlap:
            max_overlap = intersection
            best_cA = cA

print(f"Overlaps with A14_BF_2_F1 cell 6:")
for cA, data in sorted(overlaps.items(), key=lambda x: x[1]['intersection'], reverse=True):
    print(f"Cell {cA}: Intersection={data['intersection']}, IoU={data['iou']:.3f}, Overlap%={data['overlap_of_B']:.3f}, AreaA={data['areaA']}, AreaB={data['areaB']}")

