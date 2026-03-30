#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 14 13:26:55 2026

@author: user
"""

import numpy as np

path = "/Volumes/Movies/2025_12_31_M92/training_dataset/samples/A14-YES-1t-FBFBF-2_F0__cell_000001.npz"
data = np.load(path, allow_pickle=True)  # keep False unless you *know* it contains pickled objects

print("Keys:", data.files)
for k in data.files:
    arr = data[k]
    print(k, arr.shape, arr.dtype, f"min={arr.min() if arr.size else 'NA'} max={arr.max() if arr.size else 'NA'}")
#%%
strip = data["strip"]  # shape (96, 7776)
plt.figure(figsize=(12, 2))
plt.imshow(strip, cmap="gray", aspect="auto")
plt.colorbar()
plt.title("strip")
plt.show()