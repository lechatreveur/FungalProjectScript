#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import fnmatch
import numpy as np
from tifffile import imread, imwrite
from tqdm import tqdm
import cv2
from imaris_ims_file_reader.ims import ims  # be explicit about the submodule

# ---------- Settings ----------
working_dir = "/Volumes/Movies/2025_09_17/"
z_index = 0          # used only if you want a specific Z slice
fps = 10
do_max_projection = True  # True: Z-max projection; False: single Z slice

# ---------- Utilities ----------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def scale_to_u8(img, vmin, vmax):
    """Clip to [vmin,vmax] and scale to uint8."""
    img = np.asarray(img)
    img = np.squeeze(img)
    if img.ndim != 2:
        raise ValueError(f"Expected 2D frame, got {img.shape}")
    if vmax <= vmin:
        vmax = vmin + 1
    img = np.clip((img - vmin) * (255.0 / (vmax - vmin)), 0, 255)
    return img.astype(np.uint8)

def find_frames_and_range(frame_dir, pattern):
    frames = sorted([f for f in os.listdir(frame_dir)
                     if f.endswith(".tif") and fnmatch.fnmatch(f, pattern)])
    if not frames:
        return [], None, None
    # sample pixels sparsely for global brightness range
    samples = []
    for fname in frames:
        arr = imread(os.path.join(frame_dir, fname))
        arr = np.squeeze(arr)  # ensure 2D for percentile
        samples.extend(arr.ravel()[::10])
    samples = np.asarray(samples)
    gmin = float(np.percentile(samples, 1.0))
    gmax = float(np.percentile(samples, 99.5))
    return frames, gmin, gmax

def write_video_from_frames(frame_dir, frame_names, out_path, fps, gmin, gmax):
    # Probe first frame to get size
    first = imread(os.path.join(frame_dir, frame_names[0]))
    gray = np.squeeze(first)
    if gray.ndim != 2:
        raise ValueError(f"Movie first frame not 2D after squeeze: {gray.shape}")
    H, W = gray.shape

    # Prefer mp4v -> fall back to MJPG/AVI if needed
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H), True)
    if not writer.isOpened():
        print("mp4v failed; falling back to MJPG/AVI")
        out_path = out_path.rsplit(".", 1)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H), True)
        if not writer.isOpened():
            raise RuntimeError("VideoWriter could not be opened with either mp4v or MJPG")

    for fname in frame_names:
        arr = imread(os.path.join(frame_dir, fname))
        gray = np.squeeze(arr)
        gray8 = scale_to_u8(gray, gmin, gmax)
        bgr = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)  # portable with most codecs
        writer.write(bgr)

    writer.release()
    print(f"Saved movie: {out_path}")

# ---------- Main IMS processing ----------
def process_ims_file(ims_path):
    file_name = os.path.splitext(os.path.basename(ims_path))[0]
    print(f"Processing {file_name}")

    # Output dirs
    frames_dir = os.path.join(working_dir, f"{file_name}/Frames_{file_name}")
    masks_dir = os.path.join(working_dir, f"{file_name}/Masks_{file_name}")
    tracked_dir = os.path.join(working_dir, f"{file_name}/TrackedCells_{file_name}")
    gfp_seg_dir = os.path.join(masks_dir, "GFP_seg")
    bf_seg_dir = os.path.join(masks_dir, "brightfield_seg")
    for d in [frames_dir, masks_dir, tracked_dir, gfp_seg_dir, bf_seg_dir]:
        ensure_dir(d)

    # Open IMS with predictable shapes (highest res)
    data = ims(ims_path, squeeze_output=False, ResolutionLevelLock=0)
    T, C, Z, H, W = data.shape  # (time, channel, z, y, x)

    # ---- Export per-time 2D frames for channel 0 ----
    # Choose Z handling
    z_axis = 0  # in (Z, Y, X)
    with tqdm(total=T, desc=f"Exporting frames {file_name}") as pbar:
        for t in range(T):
            if do_max_projection and Z > 1:
                vol = data[t, 0, slice(None), :, :]   # (Z, H, W)
                frame2d = np.max(vol, axis=z_axis)
            else:
                z = min(z_index, max(Z - 1, 0))
                frame2d = data[t, 0, z, :, :]         # (H, W)

            # Ensure 2D, save as uint16 TIFF
            frame2d = np.squeeze(np.asarray(frame2d))
            if frame2d.ndim != 2:
                raise RuntimeError(f"Exported frame not 2D: {frame2d.shape}")

            out_path = os.path.join(frames_dir, f"{file_name}_t_{t:03d}_c_0.tif")
            imwrite(out_path, frame2d.astype(np.uint16))
            pbar.update(1)

    # Free IMS handle ASAP (macOS ffmpeg can be picky with open files)
    del data
    gc.collect()

    # ---- Build movie from exported frames (channel 0) ----
    pattern = f"{file_name}_t_???_c_0.tif"
    frame_names, gmin, gmax = find_frames_and_range(frames_dir, pattern)
    if not frame_names:
        print("No frames found to make a movie; skipping.")
        return
    print(f"Global intensity range: min={gmin}, max={gmax}")

    movie_path = os.path.join(tracked_dir, f"{file_name}_c_0.mp4")
    write_video_from_frames(frames_dir, frame_names, movie_path, fps, gmin, gmax)

# ---------- Batch ----------
if __name__ == "__main__":
    ims_files = [f for f in os.listdir(working_dir) if f.endswith(".ims")]
    for f in ims_files:
        process_ims_file(os.path.join(working_dir, f))
