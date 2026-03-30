#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 17:24:27 2025

@author: user
"""

import os
import numpy as np
from imaris_ims_file_reader.ims import ims
from skimage import io

def export_ims_slices(input_ims, output_dir, channel=0, time_point=0, format='png'):
    os.makedirs(output_dir, exist_ok=True)
    img = ims(input_ims)  # open highest resolution by default

    # Get shape: (T, C, Z, Y, X)
    _, _, Z, _, _ = img.shape

    for z in range(Z):
        slice_2d = img[time_point, channel, z, :, :]
        if slice_2d.dtype == np.uint16:
            io.imsave(os.path.join(output_dir, f'z{z:04d}.{format}'), slice_2d)
        else:
            io.imsave(os.path.join(output_dir, f'z{z:04d}.{format}'), slice_2d.astype(np.uint16))

    print(f"Export completed: {Z} slices in '{output_dir}'")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description="Export IMS Z‑slices as images")
    p.add_argument('ims_file', help="Path to .ims file")
    p.add_argument('out_dir', help="Directory to save images")
    p.add_argument('-ch', '--channel', type=int, default=0)
    p.add_argument('-t', '--time', type=int, default=0)
    p.add_argument('-f', '--format', default='png', choices=['png','tif','tiff'])
    args = p.parse_args()

    export_ims_slices(args.ims_file, args.out_dir,
                      channel=args.channel,
                      time_point=args.time,
                      format=args.format)
