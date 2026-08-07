#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sam2_tracker.py
---------------
Utility script to run Meta SAM2 tracking on cell sequences.
"""

import os
import sys
import tempfile
import numpy as np
import torch
from pathlib import Path
from skimage.io import imread
from PIL import Image

# Ensure the cloned repo is in the import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "segment-anything-2")))

from sam2.build_sam import build_sam2_video_predictor

def normalize_frame(img):
    """Normalize image to standard uint8 RGB (expected by SAM2)."""
    img_f = img.astype(np.float32)
    img_min, img_max = img_f.min(), img_f.max()
    if img_max > img_min:
        img_f = (img_f - img_min) / (img_max - img_min) * 255.0
    else:
        img_f = np.zeros_like(img_f)
    img_u8 = img_f.astype(np.uint8)
    
    if img_u8.ndim == 2:
        img_rgb = np.stack([img_u8, img_u8, img_u8], axis=-1)
    elif img_u8.ndim == 3:
        if img_u8.shape[2] == 4:
            img_rgb = img_u8[:, :, :3]
        else:
            img_rgb = img_u8
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")
        
    return img_rgb

def track_cell_with_sam2(
    frame_paths,
    initial_mask,
    checkpoint_path,
    model_cfg,
    device="mps",
    start_frame_idx=0
):
    """
    Tracks a single cell mask across a list of frames using SAM2.
    
    Parameters
    ----------
    frame_paths : list of Path/str
        Sorted paths to the TIFF frame files.
    initial_mask : np.ndarray (bool)
        Mask of the cell at start_frame_idx.
    checkpoint_path : str
        Path to SAM2 checkpoint (.pt file).
    model_cfg : str
        Path or key to the SAM2 model configuration (.yaml file).
    device : str
        Execution device ('mps', 'cuda', 'cpu').
    start_frame_idx : int
        The frame index where the initial mask is defined.
        
    Returns
    -------
    dict
        Dictionary mapping frame_idx (int) -> mask (np.ndarray of bool).
    """
    # Create temporary directory for JPEG frames
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Convert TIFF frames to JPEG and save them in the temp directory
        for t, fpath in enumerate(frame_paths):
            img = imread(str(fpath))
            img_rgb = normalize_frame(img)
            # Save using PIL to ensure JPEG compatibility and fast writing
            img_pil = Image.fromarray(img_rgb)
            img_pil.save(temp_path / f"{t:05d}.jpg", quality=95)
            
        # 2. Build the predictor
        predictor = build_sam2_video_predictor(model_cfg, checkpoint_path, device=device)
        
        # 3. Initialize state with the temp directory
        inference_state = predictor.init_state(video_path=str(temp_path))
        
        # 4. Add the initial mask prompt at start_frame_idx
        # Convert mask to boolean torch tensor on the correct device
        mask_tensor = torch.tensor(initial_mask, dtype=torch.bool, device=device)
        
        predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=start_frame_idx,
            obj_id=1,
            mask=mask_tensor
        )
        
        # 5. Propagate forward
        tracked_results = {}
        # propagate_in_video yields for all frames.
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=start_frame_idx
        ):
            # Out mask logits shape: (num_objects, 1, H, W)
            # Threshold logits at 0.0 to get binary mask
            mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy().astype(bool)
            tracked_results[out_frame_idx] = mask
            
        # Clean up predictor state
        predictor.reset_state(inference_state)
        
        return tracked_results
