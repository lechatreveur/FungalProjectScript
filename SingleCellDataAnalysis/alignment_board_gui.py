#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Septum alignment board GUI (Matplotlib).
Supports multi-film sessions for batch labeling across experiments.
"""

from __future__ import annotations

import os
import math
import json
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, SpanSelector
from matplotlib.patches import Rectangle

from .septum_gui_utils import (
    build_film_paths,
    discover_cell_mask_csvs,
    MaskTableCache,
    compose_sheet_from_strips,
    load_multi_state,
    save_multi_state,
    FilmPaths,
    ensure_strip_for_cell,
)

try:
    from .inference_core import FungalInferenceCore
except ImportError:
    FungalInferenceCore = None

from .septum_training_utils import export_film_training_dataset

# Keep widgets alive across function scope
_LIVE_WIDGETS = []

def review_septum_alignment_board_gui(
    WORKING_DIR: str,
    FILM_NAME: str | List[str],
    frame_channel_index: int = 0,
    mask_col: str = "rle_bf",
    time_col: str = "time_point",
    pad: int = 10,
    tile_size: int = 96,
    n_rows: int = 12,
    n_cols: int = 21,
    row_stride_keys: Tuple[int, int] = (1, 5),
    cache_force: bool = False,
    model_path: Optional[str] = None,
    block: bool = False,
):
    if isinstance(FILM_NAME, str):
        FILM_NAMES = [FILM_NAME]
    else:
        FILM_NAMES = FILM_NAME

    film_paths_map: Dict[str, FilmPaths] = {}
    film_cells_map: Dict[str, List[int]] = {}
    cell_csv_map: Dict[Tuple[str, int], str] = {}
    ordered_keys: List[Tuple[str, int]] = []

    for fname in FILM_NAMES:
        paths = build_film_paths(WORKING_DIR, fname)
        if not os.path.isdir(paths.frames_dir) or not os.path.isdir(paths.tracked_dir):
            print(f"[warn] Skipping {fname}: frames or tracked dir missing.")
            continue
        
        film_paths_map[fname] = paths
        os.makedirs(paths.cache_img_dir, exist_ok=True)
        os.makedirs(paths.label_dir, exist_ok=True)

        try:
            f_cids, f_csv_map = discover_cell_mask_csvs(paths.tracked_dir)
            film_cells_map[fname] = f_cids
            for cid, csv_p in f_csv_map.items():
                cell_csv_map[(fname, cid)] = csv_p
            
            for cid in f_cids:
                ordered_keys.append((fname, cid))
        except Exception as e:
            print(f"[warn] Error discovering cells in {fname}: {e}")

    if not ordered_keys:
        print("[error] No valid films found to display.")
        return

    n_cells_total = len(ordered_keys)
    masks = MaskTableCache(cell_csv_map=cell_csv_map, time_col=time_col, mask_col=mask_col)

    # Load multi-film state
    offsets, a_left_map, cell_intervals, global_intervals_map = load_multi_state(film_paths_map, film_cells_map)

    inference_runner = None
    if FungalInferenceCore is not None:
        checkpoints = []
        if model_path: checkpoints.append(model_path)
        checkpoints.append(os.path.join(WORKING_DIR, "training_dataset", "checkpoints_binary", "model_latest.pt"))
        checkpoints.append("/Volumes/X10 Pro/Movies/2025_12_31_M92/training_dataset/checkpoints_binary/model_latest.pt")

        found_ckpt = None
        for cp in checkpoints:
            if cp and os.path.exists(cp):
                found_ckpt = cp
                break
        
        if found_ckpt:
            try:
                inference_runner = FungalInferenceCore(found_ckpt, device="cpu")
                print(f"Loaded AI Model: {os.path.basename(found_ckpt)}")
            except Exception as e:
                print(f"[warn] Could not load AI Model: {e}")

    # ---------- GUI Setup ----------
    plt.ion()
    fig = plt.figure(figsize=(14, 9))
    plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.22)

    ax_sheet = fig.add_axes([0.05, 0.30, 0.90, 0.64])
    ax_time  = fig.add_axes([0.05, 0.22, 0.90, 0.05])
    ax_s_h   = fig.add_axes([0.05, 0.12, 0.70, 0.04])
    ax_s_v   = fig.add_axes([0.96, 0.30, 0.02, 0.64])
    
    ax_btn_save = fig.add_axes([0.77, 0.12, 0.08, 0.05])
    ax_btn_quit = fig.add_axes([0.87, 0.12, 0.08, 0.05])
    ax_btn_export = fig.add_axes([0.87, 0.05, 0.08, 0.05])

    btn_save = Button(ax_btn_save, "Save")
    btn_quit = Button(ax_btn_quit, "Quit")
    btn_export = Button(ax_btn_export, "Export")

    # Sliders
    initial_a_left = a_left_map.get(FILM_NAMES[0], 0)
    s_h = Slider(ax_s_h, "Aligned T", -1000, 1000, valinit=initial_a_left, valstep=1)
    s_v = Slider(ax_s_v, "", 0, max(0, n_cells_total - n_rows), valinit=0, valstep=1, orientation="vertical")

    # Overlays
    tile_gap = 2
    row_highlight = Rectangle((0, 0), 1, 1, fill=False, edgecolor="yellow", linewidth=3, zorder=10)
    ax_sheet.add_patch(row_highlight)
    
    interval_span = None
    interval_text = ax_time.text(0.01, 0.6, "", transform=ax_time.transAxes, fontweight='bold', color='red')
    
    pin_text = fig.text(0.05, 0.02, "Pinned: (none)")
    selection_text = fig.text(0.20, 0.02, "Selected: 0")
    mode_text = fig.text(0.35, 0.02, "Mode: GLOBAL (g/l)")
    white_sep_text = fig.text(0.55, 0.02, "White septum: OFF (i)", color="gray", fontweight='bold')
    hint_text = fig.text(0.75, 0.02, "Sync All: Shift+S | Range: Shift+Click", color="blue", fontsize=8)

    hide_text = fig.text(0.55, 0.05, "Hide No Septum: OFF (h)", color="gray", fontweight='bold')
    no_sep_status_text = fig.text(0.75, 0.05, "No Septum: OFF (n)", color="gray", fontweight='bold')
    saliency_text = fig.text(0.35, 0.05, "AI Saliency: OFF (V)", color="gray", fontweight='bold')

    selection_patches = [Rectangle((0,0),1,1, facecolor=(0.2,0.8,1,0.2), visible=False, zorder=9) for _ in range(n_rows)]
    white_sep_patches = [Rectangle((0,0),1,1, facecolor=(1,0.9,0,0.2), visible=False, zorder=8) for _ in range(n_rows)]
    no_sep_patches = [Rectangle((0,0),1,1, facecolor=(1,0,0,0.2), visible=False, zorder=8) for _ in range(n_rows)]
    for p in selection_patches: ax_sheet.add_patch(p)
    for p in white_sep_patches: ax_sheet.add_patch(p)
    for p in no_sep_patches: ax_sheet.add_patch(p)

    # State
    label_mode = "global"
    active_row_idx = 0
    active_cell_key = ordered_keys[0]
    selected_cell_keys: Set[Tuple[str, int]] = set()
    pinned_global_idx = None
    pinned_cell_key = None
    hide_no_septum = False
    show_saliency = False
    
    sheet_artist = None
    saliency_artist = None
    render_meta = {"row_keys": []}
    last_row_start = -1
    last_visible_keys = []

    # ---------- Helpers ----------
    def _get_viewable_keys():
        if hide_no_septum:
            return [k for k in ordered_keys if cell_intervals.get(k, {}).get("has_septum", True) is not False]
        return ordered_keys

    def _update_white_septum_status():
        if active_cell_key is None:
            white_sep_text.set_text("White septum: OFF (i)")
            white_sep_text.set_color("gray")
        else:
            is_white = bool(cell_intervals.get(active_cell_key, {}).get("white_septum", False))
            white_sep_text.set_text(f"White septum: {'ON' if is_white else 'OFF'} (i)")
            white_sep_text.set_color("goldenrod" if is_white else "gray")

    def _update_no_septum_status():
        if active_cell_key is None:
            no_sep_status_text.set_text("No Septum: OFF (n)")
            no_sep_status_text.set_color("gray")
        else:
            has_sep = cell_intervals.get(active_cell_key, {}).get("has_septum", True)
            is_no_sep = (has_sep is False)
            no_sep_status_text.set_text(f"No Septum: {'ON' if is_no_sep else 'OFF'} (n)")
            no_sep_status_text.set_color("red" if is_no_sep else "gray")

    def _toggle_white_septum():
        if active_cell_key:
            if active_cell_key not in cell_intervals:
                cell_intervals[active_cell_key] = {"has_septum": True, "white_septum": False}
            cell_intervals[active_cell_key]["white_septum"] = not bool(cell_intervals[active_cell_key].get("white_septum", False))
            _update_white_septum_status()
            _render_light()

    def _set_mode(m: str):
        nonlocal label_mode
        label_mode = m
        mode_text.set_text(f"Mode: {label_mode.upper()} (g/l)")
        _update_interval_artist()
        fig.canvas.draw_idle()

    def _set_active_row_by_index(r: int):
        nonlocal active_row_idx, active_cell_key
        vis = render_meta.get("row_keys", [])
        if not vis: return
        active_row_idx = max(0, min(r, len(vis) - 1))
        active_cell_key = vis[active_row_idx]
        
        # Sync time slider to current film's global shift
        s_h.eventson = False
        s_h.set_val(a_left_map.get(active_cell_key[0], 0))
        s_h.eventson = True
        
        _render_light()

    def _render_light():
        vis = render_meta.get("row_keys", [])
        stride = tile_size + tile_gap
        for r, p in enumerate(selection_patches):
            in_view = r < len(vis)
            p.set_visible(in_view and vis[r] in selected_cell_keys)
            white_sep_patches[r].set_visible(in_view and bool(cell_intervals.get(vis[r], {}).get("white_septum")))
            no_sep_patches[r].set_visible(in_view and (cell_intervals.get(vis[r], {}).get("has_septum", True) is False))
            if in_view:
                y0 = r * stride
                p.set_xy((0, y0))
                p.set_width(ax_sheet.get_xlim()[1])
                p.set_height(tile_size - 1)
                white_sep_patches[r].set_xy((0, y0))
                white_sep_patches[r].set_width(ax_sheet.get_xlim()[1])
                white_sep_patches[r].set_height(tile_size - 1)
                no_sep_patches[r].set_xy((0, y0))
                no_sep_patches[r].set_width(ax_sheet.get_xlim()[1])
                no_sep_patches[r].set_height(tile_size - 1)

        row_highlight.set_y(active_row_idx * stride)
        row_highlight.set_width(ax_sheet.get_xlim()[1])
        row_highlight.set_height(tile_size - 1)
        
        _update_white_septum_status()
        _update_interval_artist()
        _update_title()
        _update_prediction_overlay()
        fig.canvas.draw_idle()

    def _render_heavy(force=False):
        nonlocal render_meta, sheet_artist, last_row_start, last_visible_keys, active_cell_key, active_row_idx
        
        vk = _get_viewable_keys()
        v_max = max(0, len(vk) - n_rows)
        if s_v.valmax != v_max:
            s_v.valmax = v_max
            s_v.ax.set_ylim(0, max(1, v_max))
            
        row_start = max(0, min(int(s_v.val), v_max))
        vis = vk[row_start: row_start + n_rows]
        
        if len(vis) == 0:
            active_cell_key = None
            active_row_idx = 0
        else:
            if active_row_idx >= len(vis):
                active_row_idx = 0
            active_cell_key = vis[active_row_idx]
            
        if not force and last_row_start == row_start and last_visible_keys == vis:
            _render_light()
            return
        
        sheet, meta = compose_sheet_from_strips(
            visible_keys=vis, 
            a_left_map=a_left_map, 
            n_cols=n_cols, 
            tile_size=tile_size, 
            tile_gap=tile_gap, 
            film_paths_map=film_paths_map, 
            masks=masks, 
            offsets=offsets, 
            time_col=time_col, 
            mask_col=mask_col, 
            pad=pad, 
            channel_index=frame_channel_index, 
            cache_force=cache_force
        )
        last_row_start, last_visible_keys, render_meta = row_start, list(vis), meta
        
        if sheet_artist is None:
            sheet_artist = ax_sheet.imshow(sheet, cmap="gray", origin="upper", aspect="equal")
            # Saliency overlay artist (placeholder size)
            shape4 = (sheet.shape[0], sheet.shape[1], 4)
            saliency_artist = ax_sheet.imshow(np.zeros(shape4), origin="upper", aspect="equal", zorder=15, visible=False)
            ax_sheet.axis("off")
        else:
            sheet_artist.set_data(sheet)
            sheet_artist.set_extent((0, sheet.shape[1], sheet.shape[0], 0))
        
        _render_light()

    def _get_active_interval():
        if label_mode == "cell" and active_cell_key:
            v = cell_intervals.get(active_cell_key, {})
            s, e = v.get("start_aligned"), v.get("end_aligned")
            if s is not None and e is not None: return (int(round(float(s))), int(round(float(e))))
            
        if active_cell_key:
            fname = active_cell_key[0]
            return global_intervals_map.get(fname, None)
        return None

    def _update_interval_artist():
        nonlocal interval_span
        if interval_span: interval_span.remove()
        interval_span = None

        if active_cell_key:
            a_left = int(a_left_map.get(active_cell_key[0], 0))
        else:
            a_left = int(s_h.val)
            
        dt = (tile_size / 2.0) / (tile_size + tile_gap)
        ax_time.set_xlim(a_left - dt, a_left + n_cols - 1 + dt)

        gi = _get_active_interval()
        if gi is not None:
            G0, G1 = gi
            interval_span = ax_time.axvspan(G0, G1, alpha=0.2, color="red")
            interval_text.set_text(f"{label_mode.capitalize()} Interval: [{G0}, {G1}]")
        else:
            interval_text.set_text(f"{label_mode.capitalize()} Interval: None")

    def _update_title():
        row_start = int(s_v.val)
        vis = render_meta.get("row_keys", [])
        fname, cid = active_cell_key if active_cell_key else ("-", "-")
        ax_sheet.set_title(f"Multi-Film Session | Rows {row_start}-{row_start+len(vis)-1} / {n_cells_total} | Active: {fname}:{cid}")

    def _on_span(xmin, xmax):
        nonlocal cell_intervals, global_intervals_map
        lo, hi = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        G0, G1 = int(math.floor(lo)), int(math.ceil(hi))
        if label_mode == "cell" and active_cell_key:
            cell_intervals[active_cell_key] = {"has_septum": True, "start_aligned": float(G0), "end_aligned": float(G1)}
        elif label_mode == "global" and active_cell_key:
            fname = active_cell_key[0]
            global_intervals_map[fname] = (G0, G1)

        _update_interval_artist()
        fig.canvas.draw_idle()

    span = SpanSelector(ax_time, _on_span, direction="horizontal", useblit=False, interactive=True)

    def on_click(event):
        if event.inaxes != ax_sheet or event.ydata is None: return
        row_idx = int(event.ydata // (tile_size + tile_gap))
        vis = render_meta.get("row_keys", [])
        if 0 <= row_idx < len(vis):
            if event.button == 3: # Right click toggle select
                key = vis[row_idx]
                if key in selected_cell_keys: selected_cell_keys.remove(key)
                else: selected_cell_keys.add(key)
                selection_text.set_text(f"Selected: {len(selected_cell_keys)}")
                _render_light()
            else:
                _set_active_row_by_index(row_idx)

    def on_key(event):
        nonlocal a_left_map, hide_no_septum, active_row_idx, show_saliency
        if not event.key: return
        k = event.key.lower()
        
        # Navigation
        if k in ("up", "k"):
            if active_row_idx == 0 and s_v.val > 0:
                s_v.set_val(max(0, s_v.val - 1))
            else:
                _set_active_row_by_index(active_row_idx - 1)
        elif k in ("down", "j"):
            if active_row_idx == n_rows - 1 and s_v.val < s_v.valmax:
                s_v.set_val(min(s_v.valmax, s_v.val + 1))
            else:
                _set_active_row_by_index(active_row_idx + 1)
        elif k == "pageup":
            s_v.set_val(max(0, s_v.val - n_rows))
        elif k == "pagedown":
            s_v.set_val(min(s_v.valmax, s_v.val + n_rows))
            
        elif "left" in k or "right" in k:
            if active_cell_key:
                fname = active_cell_key[0]
                delta = -1 if "left" in k else 1
                if "shift" in k or "alt" in k or "cmd" in k or "ctrl" in k: 
                    delta *= 10
                a_left_map[fname] = a_left_map.get(fname, 0) + delta
                s_h.eventson = False
                s_h.set_val(a_left_map[fname])
                s_h.eventson = True
                _render_heavy(force=True)

        elif k == "s": # Sync All Shifts or Save
            if "shift" in event.key:
                if active_cell_key:
                    ref = a_left_map.get(active_cell_key[0], 0)
                    for f in a_left_map: a_left_map[f] = ref
                    print(f"Synced all films to shift {ref}")
                    _render_heavy(force=True)
            else:
                on_save(None)

        elif k == "w":
            vk = _get_viewable_keys()
            i_vk = int(s_v.val) + active_row_idx
            j_vk = i_vk - 1
            if 0 <= j_vk < len(vk):
                i = ordered_keys.index(vk[i_vk])
                j = ordered_keys.index(vk[j_vk])
                ordered_keys[i], ordered_keys[j] = ordered_keys[j], ordered_keys[i]
                if active_row_idx > 0:
                    _set_active_row_by_index(active_row_idx - 1)
                else:
                    s_v.set_val(max(0, s_v.val - 1))
                    _set_active_row_by_index(0)
                _render_heavy(force=True)

        elif k == "x":
            vk = _get_viewable_keys()
            i_vk = int(s_v.val) + active_row_idx
            j_vk = i_vk + 1
            if 0 <= j_vk < len(vk):
                i = ordered_keys.index(vk[i_vk])
                j = ordered_keys.index(vk[j_vk])
                ordered_keys[i], ordered_keys[j] = ordered_keys[j], ordered_keys[i]
                if active_row_idx < min(n_rows - 1, len(render_meta.get("row_keys", [])) - 1):
                    _set_active_row_by_index(active_row_idx + 1)
                else:
                    s_v.set_val(min(s_v.valmax, s_v.val + 1))
                    _set_active_row_by_index(active_row_idx)
                _render_heavy(force=True)

        elif k == "g": _set_mode("global")
        elif k == "l": _set_mode("cell")
        elif k == "i": _toggle_white_septum()
        elif k == "n":
            if active_cell_key:
                v = cell_intervals.get(active_cell_key)
                if v is None:
                    cell_intervals[active_cell_key] = {"has_septum": False}
                else:
                    v["has_septum"] = not v.get("has_septum", True)
                _update_no_septum_status()
                if hide_no_septum:
                    _render_heavy(force=True)
                    _set_active_row_by_index(active_row_idx)
                else:
                    _render_light()
        elif k == "h":
            hide_no_septum = not hide_no_septum
            hide_text.set_text(f"Hide No Septum: {'ON' if hide_no_septum else 'OFF'} (h)")
            hide_text.set_color("red" if hide_no_septum else "gray")
            
            vk = _get_viewable_keys()
            v_max = max(0, len(vk) - n_rows)
            if s_v.valmax != v_max:
                s_v.valmax = v_max
                s_v.ax.set_ylim(0, max(1, v_max))
            s_v.set_val(min(s_v.val, v_max))
            
            active_row_idx = min(active_row_idx, max(0, min(n_rows - 1, len(vk) - 1)))
            _render_heavy(force=True)
            _set_active_row_by_index(active_row_idx)

        elif k == "v":
            show_saliency = not show_saliency
            saliency_text.set_text(f"AI Saliency: {'ON' if show_saliency else 'OFF'} (V)")
            saliency_text.set_color("red" if show_saliency else "gray")
            _render_light()

        elif k in (",", ".", "[", "]"):
            if active_cell_key:
                d = {",": -row_stride_keys[0], ".": row_stride_keys[0], "[": -row_stride_keys[1], "]": row_stride_keys[1]}[k]
                targets = selected_cell_keys if selected_cell_keys else {active_cell_key}
                for t in targets: offsets[t] = offsets.get(t, 0) + d
                _render_heavy(force=True)
        elif k == "e": on_export(None)
        elif k == "p": 
            nonlocal pinned_global_idx, pinned_cell_key
            idx = int(s_v.val) + active_row_idx
            if pinned_global_idx is None:
                pinned_global_idx, pinned_cell_key = idx, ordered_keys[idx]
            else:
                ordered_keys[idx], ordered_keys[pinned_global_idx] = ordered_keys[pinned_global_idx], ordered_keys[idx]
                pinned_global_idx, pinned_cell_key = None, None
                _render_heavy(force=True)
            pin_text.set_text(f"Pinned: {pinned_cell_key if pinned_cell_key else '(none)'}")
        elif k == "backspace":
            if label_mode == "cell" and active_cell_key in cell_intervals:
                del cell_intervals[active_cell_key]
                _update_no_septum_status()
                _update_white_septum_status()
                _render_light()
            elif label_mode == "global" and active_cell_key:
                fname = active_cell_key[0]
                if fname in global_intervals_map:
                    del global_intervals_map[fname]
                _update_interval_artist()
                fig.canvas.draw_idle()

    def on_save(event):
        save_multi_state(film_paths_map, film_cells_map, offsets, a_left_map, cell_intervals, ordered_keys, global_intervals_map)
        print("Progress saved.")

    def on_export(event):
        try:
            on_save(None)
        except Exception as e:
            print(f"Failed to save before export: {e}")
            
        active_films = sorted(list(film_paths_map.keys()))
        last_global_interval = None
        for fname in active_films:
            paths = film_paths_map[fname]
            if fname in global_intervals_map:
                last_global_interval = global_intervals_map[fname]
            elif last_global_interval:
                global_intervals_map[fname] = last_global_interval
                
            print(f"Exporting dataset for {fname}...")
            export_film_training_dataset(
                working_dir=WORKING_DIR,
                film_name=fname,
                all_cell_ids=film_cells_map[fname],
                offsets=offsets,
                global_interval=global_intervals_map.get(fname, None),
                cell_intervals=cell_intervals,
                frames_dir=paths.frames_dir,
                cache_img_dir=paths.cache_img_dir,
                masks=masks,
                time_col=time_col,
                mask_col=mask_col,
                pad=pad,
                tile_size=tile_size,
                channel_index=frame_channel_index,
                cache_force=cache_force,
            )
        print("All experiments exported.")

    def on_quit(event):
        try:
            on_save(None)
        except Exception as e:
            print(f"[warn] Save failed on quit: {e}")
        plt.close(fig)

    # AI Prediction Overlay Stub
    pred_span = None
    pred_text = ax_time.text(0.65, 0.6, "", transform=ax_time.transAxes, color="green", weight="bold")

    def _update_prediction_overlay():
        nonlocal pred_span
        if pred_span is not None:
            for item in pred_span: item.remove()
            pred_span = None
        
        if saliency_artist:
            saliency_artist.set_visible(False)
            
        if inference_runner is None or active_cell_key is None: return
        
        fname, cid = active_cell_key
        paths = film_paths_map[fname]
        strip, tp0 = ensure_strip_for_cell(
            cid=cid,
            film_name=fname,
            frames_dir=paths.frames_dir,
            cache_img_dir=paths.cache_img_dir,
            masks=masks,
            offsets=offsets,
            time_col=time_col,
            mask_col=mask_col,
            pad=pad,
            tile_size=tile_size,
            channel_index=frame_channel_index,
            cache_force=cache_force
        )
        
        if strip is not None and strip.shape[1] > 0:
            if show_saliency:
                probs, smap = inference_runner.predict_saliency(strip)
            else:
                probs = inference_runner.predict_strip(strip)
                smap = None
                
            if probs is not None:
                off = offsets.get(active_cell_key, 0)
                x = np.arange(len(probs)) + tp0 + off
                pred_span = ax_time.plot(x, probs, color="green", alpha=0.6, linewidth=1)
                best_idx = np.argmax(probs)
                pred_text.set_text(f"AI Peak: {int(x[best_idx])} ({probs[best_idx]:.1%})")
                
                # Saliency Overlay
                if show_saliency and smap is not None and saliency_artist is not None and sheet_artist is not None:
                    stride = tile_size + tile_gap
                    y0 = active_row_idx * stride
                    
                    # Create full-sheet sized overlay grid precisely aligned with current view
                    smap_full = np.zeros(sheet_artist.get_array().shape[:2], dtype=float)
                    
                    a_left = int(a_left_map.get(active_cell_key[0], 0))
                    for j in range(n_cols):
                        tp = int((a_left + j) - off)
                        idx = tp - tp0
                        if 0 <= idx < len(probs):
                            x_src = idx * tile_size
                            x_dst = j * (tile_size + tile_gap)
                            # Slide target cell frame onto global graphic layout sheet
                            smap_full[y0:y0+tile_size, x_dst:x_dst+tile_size] = smap[:, x_src:x_src+tile_size]
                            
                    # Convert to RGBA for pure overlay transparency
                    import matplotlib.cm as cm
                    rgba = plt.cm.hot(smap_full)
                    alpha_mask = smap_full * 0.6  # 60% opacity max at peaks
                    alpha_mask[smap_full < 0.15] = 0.0 # Make noise invisible
                    rgba[..., 3] = alpha_mask

                    saliency_artist.set_data(rgba)
                    saliency_artist.set_visible(True)
            else:
                pred_text.set_text("AI Error")
        else:
            pred_text.set_text("No Strip")

    s_h.on_changed(lambda v: (a_left_map.update({active_cell_key[0]: int(v)}) or _render_heavy(force=True)) if active_cell_key else None)
    s_v.on_changed(lambda v: _render_heavy(force=True))
    btn_save.on_clicked(on_save)
    btn_quit.on_clicked(on_quit)
    btn_export.on_clicked(on_export)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    
    def on_close(event):
        try: on_save(None)
        except Exception: pass
    fig.canvas.mpl_connect("close_event", on_close)

    _LIVE_WIDGETS.append((fig, btn_save, btn_quit, btn_export, s_h, s_v, span, row_highlight))
    _render_heavy(force=True)
    if block:
        plt.show(block=True)
    else:
        plt.show()

