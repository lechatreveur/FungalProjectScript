#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 19:37:54 2026

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

try:
    import cv2
except Exception as e:
    raise RuntimeError("OpenCV required: pip install opencv-python") from e

# keep GUI objects alive (Spyder/Qt sometimes garbage-collects widgets)
_LIVE_WIDGETS = []


def _invert_local_to_global(local_to_global: Dict[int, int]) -> Dict[int, int]:
    """Return global_id -> local_id (keeps first local_id if collisions)."""
    out = {}
    for local_id, gid in local_to_global.items():
        if gid is None:
            continue
        if gid not in out:
            out[gid] = local_id
    return out


def _open_video(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open mp4: {path}")
    return cap


def _read_frame_at(cap: cv2.VideoCapture, idx: int) -> Optional[np.ndarray]:
    """Random access by frame index; returns RGB image for matplotlib."""
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        return None
    idx = int(np.clip(idx, 0, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def review_summary15_by_global_id(
    working_dir: str,
    field_seq: List[Tuple[str, str]],                 # [("gfp",film1), ("bf",film2), ...]
    global_maps_by_film: Dict[str, Dict[int, int]],   # film_name -> {local_id -> global_id}
    field_label: str = "F0",
    summary_mp4: Optional[str] = None,               # default: <pipeline_outputs>/combined_movies/<field>__summary15.mp4
    summary_index_csv: Optional[str] = None,         # default: same dir, <field>__summary15.index.csv
    out_qc_csv: Optional[str] = None,                # default: <pipeline_outputs>/QC__<field>.csv
    start_global_id: Optional[int] = None,
    block: bool = False,
):
    """
    Review only 15 stills (3 per film) via a summary mp4 + index csv.
    Shows ONE still at a time. No playback.

    Hotkeys
    -------
    Global ID:
      n / right : next ID
      p / left  : prev ID

    Film:
      ] / down  : next film
      [ / up    : prev film

    Tag:
      t : cycle tag begin -> middle -> end

    Label:
      g/b/u : good / bad / unsure

    Save/Quit:
      s : save
      esc : quit (saves)
    """

    wd = Path(working_dir)
    films = [film for _, film in field_seq]

    # defaults for summary paths
    if summary_mp4 is None or summary_index_csv is None:
        combined_dir = wd / "pipeline_outputs" / "combined_movies"
        if summary_mp4 is None:
            summary_mp4 = str(combined_dir / f"{field_label}__summary15.mp4")
        if summary_index_csv is None:
            summary_index_csv = str(combined_dir / f"{field_label}__summary15.index.csv")

    if not os.path.isfile(summary_mp4):
        raise FileNotFoundError(f"Missing summary mp4: {summary_mp4}")
    if not os.path.isfile(summary_index_csv):
        raise FileNotFoundError(f"Missing summary index csv: {summary_index_csv}")

    # QC output
    if out_qc_csv is None:
        out_qc_csv = str(wd / "pipeline_outputs" / f"QC__{field_label}.csv")
    os.makedirs(os.path.dirname(out_qc_csv), exist_ok=True)

    # load existing QC
    qc: Dict[int, str] = {}
    if os.path.isfile(out_qc_csv):
        try:
            with open(out_qc_csv, "r", newline="") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    try:
                        gid = int(row["global_id"])
                        qc[gid] = row.get("label", "")
                    except Exception:
                        pass
        except Exception:
            pass

    # invert global map -> local id per film for display
    global_to_local: Dict[str, Dict[int, int]] = {}
    for film, l2g in global_maps_by_film.items():
        global_to_local[film] = _invert_local_to_global(l2g)

    # list of gids to review
    all_gids = set()
    for film in films:
        all_gids.update(global_to_local.get(film, {}).keys())
    gids = sorted(all_gids)
    if not gids:
        raise RuntimeError("No global IDs found in global_maps_by_film.")

    # starting gid index
    if start_global_id is not None and start_global_id in gids:
        gid_idx = gids.index(start_global_id)
    else:
        gid_idx = 0

    # read summary index csv to map (film, tag) -> out_start_frame (+ meta)
    # We assume your make_15frame_summary_mp4 writes rows with:
    # film, tag, src_frame, src_n_frames, out_start_frame, out_n_frames_held, ...
    seg = {}  # (film, tag) -> dict
    with open(summary_index_csv, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            key = (row["film"], row["tag"])
            seg[key] = {
                "src_frame": int(float(row["src_frame"])),
                "src_n_frames": int(float(row["src_n_frames"])),
                "out_start_frame": int(float(row["out_start_frame"])),
                "out_n_frames_held": int(float(row["out_n_frames_held"])),
            }

    tags = ["begin", "middle", "end"]
    film_idx = 0
    tag_idx = 1  # default "middle"

    cap = _open_video(summary_mp4)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- UI layout ---
    fig = plt.figure(figsize=(11.5, 7.5))
    ax_img = fig.add_axes([0.05, 0.18, 0.70, 0.77])
    ax_img.set_axis_off()

    ax_info = fig.add_axes([0.77, 0.18, 0.22, 0.77])
    ax_info.set_axis_off()

    # Buttons row
    ax_prevgid = fig.add_axes([0.05, 0.08, 0.10, 0.06])
    ax_nextgid = fig.add_axes([0.16, 0.08, 0.10, 0.06])
    ax_prevfilm = fig.add_axes([0.31, 0.08, 0.10, 0.06])
    ax_nextfilm = fig.add_axes([0.42, 0.08, 0.10, 0.06])

    ax_good   = fig.add_axes([0.56, 0.08, 0.07, 0.06])
    ax_bad    = fig.add_axes([0.64, 0.08, 0.07, 0.06])
    ax_unsure = fig.add_axes([0.72, 0.08, 0.07, 0.06])
    ax_save   = fig.add_axes([0.80, 0.08, 0.08, 0.06])
    ax_quit   = fig.add_axes([0.89, 0.08, 0.08, 0.06])

    btn_prevgid = Button(ax_prevgid, "PrevID")
    btn_nextgid = Button(ax_nextgid, "NextID")
    btn_prevfilm = Button(ax_prevfilm, "PrevFilm")
    btn_nextfilm = Button(ax_nextfilm, "NextFilm")
    btn_good   = Button(ax_good, "Good")
    btn_bad    = Button(ax_bad, "Bad")
    btn_unsure = Button(ax_unsure, "Unsure")
    btn_save   = Button(ax_save, "Save")
    btn_quit   = Button(ax_quit, "Quit")

    im_artist = ax_img.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
    info_text = ax_info.text(0.0, 1.0, "", va="top", ha="left", fontsize=11, family="monospace")

    def _current_gid() -> int:
        return gids[gid_idx]

    def _current_film() -> str:
        return films[film_idx]

    def _current_tag() -> str:
        return tags[tag_idx]

    def _save():
        rows = [{"global_id": gid, "label": qc.get(gid, "")} for gid in gids]
        with open(out_qc_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["global_id", "label"])
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {out_qc_csv}")

    def _quit(evt=None):
        try:
            _save()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        plt.close(fig)

    def _set_label(lbl: str):
        qc[_current_gid()] = lbl
        _render()

    def _next_gid(step: int):
        nonlocal gid_idx
        gid_idx = int(np.clip(gid_idx + step, 0, len(gids) - 1))
        _render()

    def _next_film(step: int):
        nonlocal film_idx
        film_idx = (film_idx + step) % len(films)
        _render()

    def _next_tag():
        nonlocal tag_idx
        tag_idx = (tag_idx + 1) % len(tags)
        _render()

    def _render():
        gid = _current_gid()
        film = _current_film()
        tag = _current_tag()

        key = (film, tag)
        if key not in seg:
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            title = f"{field_label} | film={film} | tag={tag} | (missing in index)"
            meta_line = "src_frame=?  src_n=?"
        else:
            s = seg[key]
            out_start = s["out_start_frame"]
            out_hold = max(1, s["out_n_frames_held"])
            # choose the first held frame (stable)
            out_idx = int(np.clip(out_start, 0, max(0, n_total - 1)))
            frame = _read_frame_at(cap, out_idx)
            if frame is None:
                frame = np.zeros((10, 10, 3), dtype=np.uint8)
            title = f"{field_label} | film={film} | tag={tag} | summary_frame={out_idx}/{max(0, n_total-1)}"
            meta_line = f"src_frame={s['src_frame']}  src_n={s['src_n_frames']}  held={out_hold}"

        im_artist.set_data(frame)
        ax_img.set_title(title, fontsize=12)

        lines = []
        lines.append(f"GLOBAL ID: {gid}")
        lines.append(f"Label: {qc.get(gid, '')}")
        lines.append("")
        lines.append(f"Viewing: {film}  |  tag={tag}")
        lines.append(meta_line)
        lines.append("")
        lines.append("Local IDs per film (burned-in):")
        for k, f in field_seq:
            local = global_to_local.get(f, {}).get(gid, None)
            lines.append(f"  {k:>3}  {f}:  {'' if local is None else local}")
        lines.append("")
        lines.append("Hotkeys:")
        lines.append("  n/p or →/← : next/prev global id")
        lines.append("  ]/[ or ↓/↑ : next/prev film")
        lines.append("  t : cycle tag begin/middle/end")
        lines.append("  g/b/u : label good/bad/unsure")
        lines.append("  s : save   esc : quit")

        info_text.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    # button bindings
    btn_prevgid.on_clicked(lambda evt: _next_gid(-1))
    btn_nextgid.on_clicked(lambda evt: _next_gid(+1))
    btn_prevfilm.on_clicked(lambda evt: _next_film(-1))
    btn_nextfilm.on_clicked(lambda evt: _next_film(+1))
    btn_good.on_clicked(lambda evt: _set_label("good"))
    btn_bad.on_clicked(lambda evt: _set_label("bad"))
    btn_unsure.on_clicked(lambda evt: _set_label("unsure"))
    btn_save.on_clicked(lambda evt: _save())
    btn_quit.on_clicked(_quit)

    def _on_key(event):
        k = event.key
        if k in ("n", "right"):
            _next_gid(+1)
        elif k in ("p", "left"):
            _next_gid(-1)

        elif k in ("]", "down"):
            _next_film(+1)
        elif k in ("[", "up"):
            _next_film(-1)

        elif k == "t":
            _next_tag()

        elif k == "g":
            _set_label("good")
        elif k == "b":
            _set_label("bad")
        elif k == "u":
            _set_label("unsure")

        elif k == "s":
            _save()

        elif k in ("escape",):
            _quit()

    fig.canvas.mpl_connect("key_press_event", _on_key)

    # Keep alive to avoid widget GC issues in Spyder
    fig._btns = (btn_prevgid, btn_nextgid, btn_prevfilm, btn_nextfilm, btn_good, btn_bad, btn_unsure, btn_save, btn_quit)
    fig._cap = cap
    fig._im = im_artist
    fig._info = info_text
    _LIVE_WIDGETS.append(fig)

    _render()
    plt.show(block=block)
    return {"qc": qc, "qc_csv": out_qc_csv, "global_ids": gids, "summary_mp4": summary_mp4, "summary_index_csv": summary_index_csv}