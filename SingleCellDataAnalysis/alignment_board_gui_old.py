#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 11:46:17 2026

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Septum alignment board GUI (Matplotlib).

This file should stay small and only contain GUI logic.
All file formats / decoding / caching are in septum_gui_utils.py.
"""

from __future__ import annotations

import os
import math
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, SpanSelector
from matplotlib.patches import Rectangle

from .septum_gui_utils import (
    build_film_paths,
    discover_cell_mask_csvs,
    MaskTableCache,
    compose_sheet_from_strips,
    load_state,
    save_state_and_labels,
)


# Keep widgets alive across function scope (prevents GC)
try:
    _LIVE_WIDGETS
except NameError:
    _LIVE_WIDGETS = []


def review_septum_alignment_board_gui(
    WORKING_DIR: str,
    FILM_NAME: str,
    frame_channel_index: int = 0,
    mask_col: str = "rle_gfp",
    time_col: str = "time_point",
    pad: int = 10,
    tile_size: int = 96,
    n_rows: int = 12,
    n_cols: int = 21,
    row_stride_keys: Tuple[int, int] = (1, 5),  # (small,big)
    cache_force: bool = False,
    block: bool = False,
):
    paths = build_film_paths(WORKING_DIR, FILM_NAME)

    if not os.path.isdir(paths.frames_dir):
        raise FileNotFoundError(f"Frames folder not found: {paths.frames_dir}")
    if not os.path.isdir(paths.tracked_dir):
        raise FileNotFoundError(f"TrackedCells folder not found: {paths.tracked_dir}")

    os.makedirs(paths.cache_img_dir, exist_ok=True)
    os.makedirs(paths.label_dir, exist_ok=True)

    all_cell_ids, cell_csv_map = discover_cell_mask_csvs(paths.tracked_dir)
    n_cells_total = len(all_cell_ids)

    masks = MaskTableCache(cell_csv_map=cell_csv_map, time_col=time_col, mask_col=mask_col)

    offsets, global_interval, last_row_start, last_a_left, cell_order = load_state(
        paths.json_path, FILM_NAME, all_cell_ids
    )
    ordered_ids = cell_order  # this is what you render


    # reasonable default a_left if resume is empty
    for cid in all_cell_ids[:min(10, n_cells_total)]:
        masks.load(cid)
    mins = []
    for cid in all_cell_ids:
        tp_min, _tp_max = masks.minmax(cid)
        if tp_min is not None:
            mins.append(int(tp_min) + int(offsets.get(cid, 0)))
    default_a_left = int(min(mins)) if mins else 0

    row_start0 = max(0, min(int(last_row_start), max(0, n_cells_total - n_rows)))
    a_left0 = int(last_a_left) if isinstance(last_a_left, int) else default_a_left

    # ---------- GUI wiring ----------
    plt.ion()
    fig = plt.figure(figsize=(14, 9))
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.22)

    ax_sheet = fig.add_axes([0.05, 0.30, 0.90, 0.64])
    ax_time  = fig.add_axes([0.05, 0.22, 0.90, 0.06])

    ax_s_h   = fig.add_axes([0.05, 0.12, 0.74, 0.05])
    ax_btn_save = fig.add_axes([0.81, 0.115, 0.08, 0.06])
    ax_btn_quit = fig.add_axes([0.90, 0.115, 0.08, 0.06])
    ax_s_v   = fig.add_axes([0.96, 0.30, 0.02, 0.64])

    ax_sheet.set_title("Alignment board (click a row; shift with , . [ ] ; drag interval below)")
    ax_sheet.axis("off")

    ax_time.set_xlabel("Aligned time")
    ax_time.set_yticks([])
    ax_time.set_ylim(0, 1)

    # toolbar sanity (critical for click/drag widgets)
    try:
        tb = fig.canvas.manager.toolbar
        if tb is not None:
            if hasattr(tb, "mode"): tb.mode = ""
            if hasattr(tb, "pan"):  tb.pan();  tb.pan()
            if hasattr(tb, "zoom"): tb.zoom(); tb.zoom()
    except Exception:
        pass

    # bring to front (Qt)
    try:
        w = fig.canvas.manager.window
        w.raise_(); w.activateWindow()
    except Exception:
        pass

    btn_save = Button(ax_btn_save, "Save")
    btn_quit = Button(ax_btn_quit, "Quit")

    s_h = Slider(ax_s_h, "Aligned time (left)", valmin=-5000, valmax=5000, valinit=a_left0, valstep=1)

    v_max = max(0, n_cells_total - n_rows)
    s_v = Slider(ax_s_v, "Cells", valmin=0, valmax=v_max, valinit=row_start0, valstep=1, orientation="vertical")

    # highlight
    tile_gap = 2
    active_row_idx = 0
    active_cell_id = all_cell_ids[row_start0] if all_cell_ids else None
    row_highlight = Rectangle((0, 0), 1, 1, fill=False, linewidth=2)
    ax_sheet.add_patch(row_highlight)

    # interval artists (do NOT cla ax_time)
    interval_span = None
    interval_text = ax_time.text(0.01, 0.6, "", transform=ax_time.transAxes)

    render_meta = {"row_cell_ids": [], "a_left": int(s_h.val), "aligned_cols": []}
    
    # pin and swap
    pinned_global_idx = None      # int or None
    pinned_cell_id = None         # for display
    pin_text = fig.text(0.05, 0.02, "Pinned: (none)", fontsize=10)  # bottom-left
    
    # multi-row selection state 
    selected_cell_ids = set()   # selected across the whole film (cell_ids)
    selection_text = fig.text(0.30, 0.02, "Selected: 0", fontsize=10)
    
    # selection overlay artists (recreated each render) 
    # One overlay rect per visible row (reused; toggled visible/invisible)
    selection_patches = []
    for _ in range(n_rows):
        p = Rectangle(
            (0, 0), 1, 1,
            fill=True,
            facecolor=(0.2, 0.8, 1.0, 0.18),
            edgecolor=(0.2, 0.8, 1.0, 0.9),
            linewidth=2.0,
            visible=False,
        )
        p.set_zorder(9)   # below active highlight if highlight is 10
        ax_sheet.add_patch(p)
        selection_patches.append(p)
    
    row_highlight.set_zorder(10)


    # 
    sheet_artist = None  # imshow artist (created once)
    last_row_start = None
    last_a_left = None
    last_visible_cids = None
    



    def _visible_cell_ids():
        row_start = int(s_v.val)
        return ordered_ids[row_start: row_start + n_rows]
    
    def _update_selection_label():
        selection_text.set_text(f"Selected: {len(selected_cell_ids)}")
        
    
    def _toggle_select_cell(cid: int):
        if cid in selected_cell_ids:
            selected_cell_ids.remove(cid)
        else:
            selected_cell_ids.add(cid)
        _update_selection_label()
        _render_light()
    
    def _clear_selection():
        selected_cell_ids.clear()
        _update_selection_label()
        _render_light()
    
    def _select_range_visible(i0: int, i1: int):
        vis = _visible_cell_ids()
        lo, hi = sorted((i0, i1))
        lo = max(0, lo); hi = min(len(vis) - 1, hi)
        for r in range(lo, hi + 1):
            selected_cell_ids.add(vis[r])
        _update_selection_label()
        _render_light()

    def _set_pin_display():
        if pinned_global_idx is None:
            pin_text.set_text("Pinned: (none)")
        else:
            pin_text.set_text(f"Pinned: cell {pinned_cell_id} (row {pinned_global_idx})")
        
    
    def _pin_or_swap():
        nonlocal pinned_global_idx, pinned_cell_id, ordered_ids, active_cell_id, active_row_idx
    
        row_start = int(s_v.val)
        active_global_idx = row_start + active_row_idx
        if active_global_idx < 0 or active_global_idx >= len(ordered_ids):
            return
    
        if pinned_global_idx is None:
            # pin current active
            pinned_global_idx = active_global_idx
            pinned_cell_id = ordered_ids[pinned_global_idx]
            _set_pin_display()
            return
    
        # swap pinned with active
        j = pinned_global_idx
        i = active_global_idx
        if i == j:
            # pressing p on same row just unpins
            pinned_global_idx = None
            pinned_cell_id = None
            _set_pin_display()
            return
    
        ordered_ids[i], ordered_ids[j] = ordered_ids[j], ordered_ids[i]
    
        # keep active selection on the same *cell* after swap:
        # (the active cell becomes what moved into active position)
        active_cell_id = ordered_ids[i]
        pinned_global_idx = None
        pinned_cell_id = None
        _set_pin_display()
        _render_heavy(force=True)
    
    def _unpin():
        nonlocal pinned_global_idx, pinned_cell_id
        pinned_global_idx = None
        pinned_cell_id = None
        _set_pin_display()


    def _swap_rows(delta: int):
        """
        Swap active row with row above (delta=-1) or below (delta=+1)
        in the GLOBAL ordered list.
        """
        nonlocal ordered_ids, active_row_idx, active_cell_id
    
        row_start = int(s_v.val)
        i_global = row_start + active_row_idx
        j_global = i_global + int(delta)
    
        if i_global < 0 or i_global >= len(ordered_ids):
            return
        if j_global < 0 or j_global >= len(ordered_ids):
            return
    
        ordered_ids[i_global], ordered_ids[j_global] = ordered_ids[j_global], ordered_ids[i_global]
    
        # keep the same cell selected, but it moved one row
        active_row_idx = active_row_idx + int(delta)
        active_cell_id = ordered_ids[row_start + active_row_idx]
        _render_heavy(force=True)

    def _really_close_figure(fig_):
        try:
            mgr = fig_.canvas.manager
            win = getattr(mgr, "window", None)
            if win is not None:
                win.close()
                return
        except Exception:
            pass
        plt.close(fig_)

    def _save_now():
        save_state_and_labels(
            working_dir=WORKING_DIR,
            film_name=FILM_NAME,
            json_path=paths.json_path,
            csv_path=paths.csv_path,
            offsets=offsets,
            all_cell_ids=all_cell_ids,
            global_interval=global_interval,
            view_row_start=int(s_v.val),
            view_a_left=int(s_h.val),
            masks=masks,
            cell_order=ordered_ids,
        )

    def _update_interval_artist():
        nonlocal interval_span
        a_left = int(s_h.val)
        a_right = a_left + n_cols - 1
        ax_time.set_xlim(a_left, a_right)

        if interval_span is not None:
            try:
                interval_span.remove()
            except Exception:
                pass
            interval_span = None

        if global_interval is not None:
            G0, G1 = global_interval
            interval_span = ax_time.axvspan(G0, G1, alpha=0.2)
            interval_text.set_text(f"Global interval: [{G0}, {G1}] (inclusive)")
        else:
            interval_text.set_text("Global interval: (not set) — drag to select")

        

    def _update_highlight():
        R = len(render_meta["row_cell_ids"])
        if R == 0:
            return
        Ht = tile_size
        stride = Ht + tile_gap
        y0 = active_row_idx * stride
    
        sheet_w = n_cols * tile_size + (n_cols - 1) * tile_gap
        row_highlight.set_x(0)
        row_highlight.set_y(y0)
        row_highlight.set_width(sheet_w - 1)
        row_highlight.set_height(Ht - 1)
    
        # make active row visually obvious
        row_highlight.set_edgecolor("yellow")
        row_highlight.set_linewidth(3.0)
        row_highlight.set_linestyle("-")
    
        

    def _update_selection_overlays():
        """Fast: reuse a fixed pool of rectangles, just move + toggle visibility."""
        vis = render_meta.get("row_cell_ids", [])
        if vis is None:
            vis = []
    
        Ht = tile_size
        stride = Ht + tile_gap
        sheet_w = n_cols * tile_size + (n_cols - 1) * tile_gap
    
        # Update one patch per visible row; hide unused patches
        for r in range(len(selection_patches)):
            patch = selection_patches[r]
    
            if r >= len(vis):
                if patch.get_visible():
                    patch.set_visible(False)
                continue
    
            cid = vis[r]
            if cid in selected_cell_ids:
                y0 = r * stride
                patch.set_x(0)
                patch.set_y(y0)
                patch.set_width(sheet_w - 1)
                patch.set_height(Ht - 1)
                if not patch.get_visible():
                    patch.set_visible(True)
            else:
                if patch.get_visible():
                    patch.set_visible(False)



    def _set_active_row_by_index(r: int):
        nonlocal active_row_idx, active_cell_id
        r = int(max(0, min(r, len(render_meta["row_cell_ids"]) - 1)))
        active_row_idx = r
        active_cell_id = render_meta["row_cell_ids"][r] if render_meta["row_cell_ids"] else None
    
        # only update overlays/title; do NOT rebuild the sheet here
        _update_selection_overlays()
        _update_highlight()
        ax_sheet.set_title(
            f"{FILM_NAME} | rows {int(s_v.val)}–{int(s_v.val) + len(render_meta['row_cell_ids']) - 1} / {n_cells_total} "
            f"| active: {active_cell_id} | selected: {len(selected_cell_ids)} | pinned: {pinned_cell_id}"
        )
        fig.canvas.draw_idle()


    def _update_title():
        row_start = int(s_v.val)
        vis = render_meta.get("row_cell_ids", [])
        ax_sheet.set_title(
            f"{FILM_NAME} | rows {row_start}–{row_start + len(vis) - 1} / {n_cells_total} "
            f"| active: {active_cell_id} | selected: {len(selected_cell_ids)} | pinned: {pinned_cell_id}"
        )
    
    def _render_light():
        """Fast: overlays + title only (no sheet recompute)."""
        _update_selection_overlays()
        _update_interval_artist()
        _update_highlight()
        _update_title()
        fig.canvas.draw_idle()
    pending_timer = None
    
    def _render_heavy_debounced(delay=0.05):
        """Schedule a heavy render soon; coalesce repeated calls."""
        nonlocal pending_timer
        if pending_timer is not None:
            try:
                pending_timer.stop()
            except Exception:
                pass
            pending_timer = None
    
        pending_timer = fig.canvas.new_timer(interval=int(delay * 1000))
        pending_timer.single_shot = True
        pending_timer.add_callback(lambda: _render_heavy(force=True))
        pending_timer.start()

    def _render_heavy(force: bool = False):
        """Slow: recompute sheet then update imshow via set_data (no cla)."""
        nonlocal render_meta, active_cell_id, active_row_idx, sheet_artist
        nonlocal last_row_start, last_a_left, last_visible_cids
    
        row_start = int(s_v.val)
        a_left = int(s_h.val)
        visible_cids = ordered_ids[row_start: row_start + n_rows]
    
        # Skip recompute if nothing changed (optional but helps)
        if (not force and
            last_row_start == row_start and
            last_a_left == a_left and
            last_visible_cids == visible_cids):
            _render_light()
            return
    
        sheet, meta = compose_sheet_from_strips(
            visible_cids=visible_cids,
            a_left=a_left,
            n_cols=n_cols,
            tile_size=tile_size,
            tile_gap=tile_gap,
            film_name=FILM_NAME,
            frames_dir=paths.frames_dir,
            cache_img_dir=paths.cache_img_dir,
            masks=masks,
            offsets=offsets,
            time_col=time_col,
            mask_col=mask_col,
            pad=pad,
            channel_index=frame_channel_index,
            cache_force=cache_force,
        )
        render_meta = meta
    
        # Create imshow once; afterwards only update pixels
        if sheet_artist is None:
            sheet_artist = ax_sheet.imshow(
                sheet, cmap="gray", interpolation="nearest", aspect="auto"
            )
        else:
            sheet_artist.set_data(sheet)
    
        # Ensure active row/cell valid
        if len(visible_cids) == 0:
            active_cell_id = None
            active_row_idx = 0
        else:
            if active_row_idx >= len(visible_cids):
                active_row_idx = 0
            active_cell_id = visible_cids[active_row_idx]
    
        # Important: keep patches on top
        row_highlight.set_zorder(10)
    
        # Remember last state
        last_row_start = row_start
        last_a_left = a_left
        last_visible_cids = list(visible_cids)
    
        _render_light()


    _render_heavy(force=True)

    # ensure first active row is set after first render_meta exists
    if render_meta["row_cell_ids"]:
        active_row_idx = 0
        active_cell_id = render_meta["row_cell_ids"][0]
        _update_selection_overlays()
        _update_highlight()
        fig.canvas.draw_idle()

    # span selector
    def _on_span(xmin, xmax):
        nonlocal global_interval
        lo, hi = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
        G0 = int(math.floor(lo))
        G1 = int(math.ceil(hi))
        if G1 < G0:
            G0, G1 = G1, G0
        global_interval = (G0, G1)
        _update_interval_artist()

    span = SpanSelector(ax_time, _on_span, direction="horizontal",
                        useblit=False, interactive=True, grab_range=8)

    # slider callbacks
    def _on_slider(_):
        _render_heavy_debounced(delay=0.08)


    s_h.on_changed(_on_slider)
    s_v.on_changed(_on_slider)

    # click selects row
    def _on_click(evt):
        nonlocal active_row_idx, active_cell_id
        if evt.inaxes != ax_sheet or evt.ydata is None:
            return
    
        y = float(evt.ydata)
        stride = tile_size + tile_gap
        r = int(y // stride)
    
        vis = _visible_cell_ids()
        if r < 0 or r >= len(vis):
            return
    
        clicked_cid = vis[r]
    
        # modifier logic (matplotlib puts modifier keys in evt.key sometimes)
        key = (evt.key or "").lower()
    
        if "shift" in key:
            # range select from current active row to clicked row
            _select_range_visible(active_row_idx, r)
            _set_active_row_by_index(r)
        elif ("control" in key) or ("ctrl" in key) or ("cmd" in key) or ("super" in key):
            # toggle selection
            _toggle_select_cell(clicked_cid)
            _set_active_row_by_index(r)
        else:
            # plain click: just move active row
            _set_active_row_by_index(r)



    # keys
    def _shift_active(delta: int):
        nonlocal offsets
        if active_cell_id is None:
            return
    
        targets = selected_cell_ids if selected_cell_ids else {active_cell_id}
        for cid in targets:
            offsets[cid] = int(offsets.get(cid, 0)) + int(delta)
        
        _update_title()
        fig.canvas.draw_idle()
        _render_heavy_debounced()



    def _scroll_time(delta: int):
        s_h.set_val(int(s_h.val) + int(delta))

    def _scroll_cells(delta: int):
        v = int(s_v.val) + int(delta)
        v = max(0, min(v, v_max))
        s_v.set_val(v)

    def _clear_interval():
        nonlocal global_interval
        global_interval = None
        _update_interval_artist()

    def _on_key(evt):
        if evt.key is None:
            return
        k = evt.key.lower()
        small, big = row_stride_keys

        if k == ",":
            _shift_active(-small)
        elif k == ".":
            _shift_active(+small)
        elif k == "[":
            _shift_active(-big)
        elif k == "]":
            _shift_active(+big)
        elif k == "0":
            if active_cell_id is None:
                return
            targets = selected_cell_ids if len(selected_cell_ids) > 0 else {active_cell_id}
            for cid in targets:
                offsets[cid] = 0
            _render_heavy()


        elif k == "left":
            _scroll_time(-1)
        elif k == "right":
            _scroll_time(+1)
        elif k == "cmd+left":
            _scroll_time(-10)
        elif k == "cmd+right":
            _scroll_time(+10)
        elif k == "up":
            _scroll_cells(-1)
        elif k == "down":
            _scroll_cells(+1)
        elif k == "pageup":
            _scroll_cells(-n_rows)
        elif k == "pagedown":
            _scroll_cells(+n_rows)

        elif k in ("backspace", "delete"):
            _clear_interval()

        elif k in ("ctrl+s", "cmd+s", "s"):
            _save_now()
        elif k in ("q", "escape"):
            _save_now()
            _really_close_figure(fig)
        elif k == "w":
            _swap_rows(-1)   # swap up
        elif k == "x":
            _swap_rows(+1)   # swap down
        elif k == "p":
            _pin_or_swap()
        elif k == "u":
            _unpin()
        elif k == " ":
            # space toggles selection of active cell
            if active_cell_id is not None:
                _toggle_select_cell(active_cell_id)
        
        elif k == "c":
            # clear selection
            _clear_selection()
        
        elif k == "a":
            # select all visible rows
            for cid in _visible_cell_ids():
                selected_cell_ids.add(cid)
            _update_selection_label()



    fig.canvas.mpl_connect("key_press_event", _on_key)
    fig.canvas.mpl_connect("button_press_event", _on_click)


    # buttons
    def _on_save(_evt=None):
        _save_now()

    def _on_quit(_evt=None):
        _save_now()
        _really_close_figure(fig)

    btn_save.on_clicked(_on_save)
    btn_quit.on_clicked(_on_quit)

    # save on close
    def _on_close(_evt):
        try:
            _save_now()
        except Exception as e:
            print(f"[warn] save failed on close: {e}")

    fig.canvas.mpl_connect("close_event", _on_close)

    # keep widgets alive
    fig._btn_save = btn_save
    fig._btn_quit = btn_quit
    fig._s_h = s_h
    fig._s_v = s_v
    fig._span = span
    fig._row_highlight = row_highlight
    _LIVE_WIDGETS.append((fig, btn_save, btn_quit, s_h, s_v, span, row_highlight))

    plt.show(block=block)

    return {
        "film_name": FILM_NAME,
        "json_path": paths.json_path,
        "csv_path": paths.csv_path,
        "n_cells": n_cells_total,
        "n_rows_visible": n_rows,
        "n_cols_visible": n_cols,
        "global_interval": global_interval,
        "offsets": offsets,
        "cache_dir": paths.cache_img_dir,
        "label_dir": paths.label_dir,
    }
