#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, TextBox

try:
    import cv2
except Exception as e:
    raise RuntimeError("OpenCV required: pip install opencv-python") from e

# keep GUI objects alive
_LIVE_WIDGETS = []
def build_combined_population_mp4(
    working_dir: str,
    films: List[str],
    out_mp4: str,
    out_index_csv: str,
    force: bool = False,
    target_fps: Optional[float] = None,
    resize_to_first: bool = True,
):
    """
    Concatenate <film>_population.mp4 files into one mp4, and write an index CSV:
      film,start_frame,n_frames,fps,w,h

    - If resize_to_first=True: all frames are resized to the first movie's WxH
      (this avoids VideoWriter errors when sizes differ).
    - If target_fps is None: use fps from the first movie.
    """
    wd = Path(working_dir)
    out_mp4 = str(out_mp4)
    out_index_csv = str(out_index_csv)

    if (not force) and os.path.isfile(out_mp4) and os.path.isfile(out_index_csv):
        print(f"[combined] cache exists: {out_mp4}")
        return out_mp4, out_index_csv

    mp4_paths = []
    for film in films:
        p = wd / f"{film}_population.mp4"
        if not p.is_file():
            raise FileNotFoundError(f"Missing mp4: {p}")
        mp4_paths.append(str(p))

    # Open first to define writer params
    cap0 = cv2.VideoCapture(mp4_paths[0])
    if not cap0.isOpened():
        raise FileNotFoundError(f"Cannot open mp4: {mp4_paths[0]}")

    fps0 = cap0.get(cv2.CAP_PROP_FPS)
    w0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    fps_out = float(target_fps) if target_fps is not None else float(fps0 if fps0 and fps0 > 1e-6 else 10.0)
    W = w0
    H = h0

    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, fps_out, (W, H), True)
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_mp4}")

    rows = []
    global_frame_cursor = 0

    for film, path in zip(films, mp4_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            vw.release()
            raise FileNotFoundError(f"Cannot open mp4: {path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = global_frame_cursor
        written = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Ensure BGR 3ch
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            if resize_to_first and (frame.shape[1] != W or frame.shape[0] != H):
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)

            vw.write(frame)
            written += 1
            global_frame_cursor += 1

        cap.release()

        rows.append({
            "film": film,
            "start_frame": start_frame,
            "n_frames": written,
            "fps": fps,
            "w": w,
            "h": h,
        })
        print(f"[combined] {film}: wrote {written} frames (start={start_frame})")

    vw.release()

    # write index csv
    os.makedirs(os.path.dirname(out_index_csv), exist_ok=True)
    with open(out_index_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["film", "start_frame", "n_frames", "fps", "w", "h"])
        w.writeheader()
        w.writerows(rows)

    print(f"[combined] wrote: {out_mp4}")
    print(f"[combined] index: {out_index_csv}")
    return out_mp4, out_index_csv


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
    """Random-access by frame index."""
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if idx < 0 or idx >= n:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok:
        return None
    # BGR -> RGB for matplotlib
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def review_population_mp4s_by_global_id_singlefilm(
    working_dir: str,
    field_seq: List[Tuple[str, str]],                 # [("gfp",film1), ("bf",film2), ...]
    global_maps_by_film: Dict[str, Dict[int, int]],   # film_name -> {local_id -> global_id}
    field_label: str = "F0",
    out_qc_csv: Optional[str] = None,
    start_global_id: Optional[int] = None,
    default_frame: str = "middle",  # "first"|"middle"|"last"
    block: bool = False,
):
    """
    Show ONE film at a time (big), let user switch films.
    For each global_id, show the corresponding local IDs in each film.
    User labels that global_id as good/bad/unsure.

    Controls
    --------
    - Film switching:
        [ / ] : prev/next film
        1..5  : jump to film index (if <=9 films, 1-based)
        Buttons: PrevFilm / NextFilm
        RadioButtons: select film

    - Global ID navigation:
        n / p : next / previous global id
        Buttons: NextID / PrevID

    - Label current global id:
        g = good, b = bad, u = unsure
        Buttons: Good / Bad / Unsure

    - Playback (optional):
        space: toggle play/pause
        , / . : step -1 / +1 frame
        Slider-ish is omitted to keep it simple/robust (you can add later)

    - Save:
        s : save CSV
        Button: Save
    """

    wd = Path(working_dir)

    # Resolve mp4 paths in the working dir: "<film_name>_population.mp4"
    films = [film for _, film in field_seq]
    
    # --- build / load combined movie cache ---
    combined_dir = wd / "pipeline_outputs" / "combined_movies"
    combined_mp4 = combined_dir / f"{field_label}__combined_population.mp4"
    combined_idx = combined_dir / f"{field_label}__combined_index.csv"
    
    build_combined_population_mp4(
        working_dir=str(wd),
        films=films,
        out_mp4=str(combined_mp4),
        out_index_csv=str(combined_idx),
        force=False,
        target_fps=None,
        resize_to_first=True,
    )
    
    # read index
    film_start = {}
    film_n = {}
    with open(combined_idx, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            film_start[row["film"]] = int(row["start_frame"])
            film_n[row["film"]] = int(row["n_frames"])
    
    cap_combined = _open_video(str(combined_mp4))
    n_total = int(cap_combined.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # mp4_paths = {}
    # for film in films:
    #     p = wd / f"{film}_population.mp4"
    #     if not p.is_file():
    #         raise FileNotFoundError(f"Missing mp4: {p}")
    #     mp4_paths[film] = str(p)

    # Invert maps so we can show local id for a global id in each film
    global_to_local = {}
    for film, l2g in global_maps_by_film.items():
        global_to_local[film] = _invert_local_to_global(l2g)

    # Build the set of global IDs to review (union over all films)
    all_gids = set()
    for film in films:
        all_gids.update(global_to_local.get(film, {}).keys())
    gids = sorted(all_gids)
    if not gids:
        raise RuntimeError("No global IDs found in global_maps_by_film.")

    # Start position
    if start_global_id is not None and start_global_id in gids:
        gid_idx = gids.index(start_global_id)
    else:
        gid_idx = 0

    # Default qc output
    if out_qc_csv is None:
        out_qc_csv = str(wd / "pipeline_outputs" / f"QC__{field_label}.csv")
    os.makedirs(os.path.dirname(out_qc_csv), exist_ok=True)

    # Load existing QC if present
    qc = {}  # gid -> label str
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

    # # Open all videos once (fast switching)
    # caps = {film: _open_video(mp4_paths[film]) for film in films}
    # meta = {}
    # for film, cap in caps.items():
    #     n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     meta[film] = {"n": n, "w": w, "h": h}

    # film index
    film_idx = 0
    current_film = films[film_idx]

    # current local frame index per film (keeps your position per film)
    film_frame = {}
    for film in films:
        n = film_n.get(film, 0)
        if n <= 0:
            film_frame[film] = 0
            continue
        if default_frame == "first":
            film_frame[film] = 0
        elif default_frame == "last":
            film_frame[film] = max(0, n - 1)
        else:
            film_frame[film] = max(0, n // 2)

    playing = False

    # --- Figure layout ---
    fig = plt.figure(figsize=(11.5, 7.5))
    ax_img = fig.add_axes([0.05, 0.18, 0.70, 0.77])   # big image area
    ax_img.set_axis_off()

    ax_info = fig.add_axes([0.77, 0.18, 0.22, 0.77])
    ax_info.set_axis_off()

    # Buttons row
    ax_prevfilm = fig.add_axes([0.05, 0.08, 0.10, 0.06])
    ax_nextfilm = fig.add_axes([0.16, 0.08, 0.10, 0.06])
    ax_previd   = fig.add_axes([0.31, 0.08, 0.10, 0.06])
    ax_nextid   = fig.add_axes([0.42, 0.08, 0.10, 0.06])

    ax_good     = fig.add_axes([0.56, 0.08, 0.07, 0.06])
    ax_bad      = fig.add_axes([0.64, 0.08, 0.07, 0.06])
    ax_unsure   = fig.add_axes([0.72, 0.08, 0.07, 0.06])
    ax_save     = fig.add_axes([0.80, 0.08, 0.08, 0.06])
    ax_quit     = fig.add_axes([0.89, 0.08, 0.08, 0.06])

    btn_prevfilm = Button(ax_prevfilm, "PrevFilm")
    btn_nextfilm = Button(ax_nextfilm, "NextFilm")
    btn_previd   = Button(ax_previd,   "PrevID")
    btn_nextid   = Button(ax_nextid,   "NextID")
    btn_good     = Button(ax_good,     "Good")
    btn_bad      = Button(ax_bad,      "Bad")
    btn_unsure   = Button(ax_unsure,   "Unsure")
    btn_save     = Button(ax_save,     "Save")
    btn_quit     = Button(ax_quit,     "Quit")

    # Film selector (radio)
    ax_radio = fig.add_axes([0.77, 0.02, 0.22, 0.14])
    radio = RadioButtons(ax_radio, films, active=0)

    # Image artist placeholder
    im_artist = ax_img.imshow(np.zeros((10, 10, 3), dtype=np.uint8))

    # Info text
    info_text = ax_info.text(0.0, 1.0, "", va="top", ha="left", fontsize=11, family="monospace")

    def _current_gid() -> int:
        return gids[gid_idx]

    def _set_film_by_index(new_idx: int):
        nonlocal film_idx, current_film
        film_idx = int(new_idx) % len(films)
        current_film = films[film_idx]
        # sync radio highlight
        try:
            radio.set_active(film_idx)
        except Exception:
            pass

    def _render():
        """Redraw frame + info panel."""
        gid = _current_gid()
        film = current_film

        # local frame index within this film segment
        local_fidx = int(film_frame.get(film, 0))
        n_local = int(film_n.get(film, 0))
        if n_local <= 0:
            local_fidx = 0
            global_fidx = int(film_start.get(film, 0))
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
        else:
            local_fidx = int(np.clip(local_fidx, 0, n_local - 1))
            film_frame[film] = local_fidx
            global_fidx = int(film_start[film]) + local_fidx

            frame = _read_frame_at(cap_combined, global_fidx)
            if frame is None:
                frame = np.zeros((10, 10, 3), dtype=np.uint8)

        im_artist.set_data(frame)
        ax_img.set_title(
            f"{field_label} | film: {film} | local {local_fidx}/{max(0, n_local-1)} | global {global_fidx}/{max(0, n_total-1)}",
            fontsize=12
        )

        # Info panel
        lines = []
        lines.append(f"GLOBAL ID: {gid}")
        lines.append(f"Label: {qc.get(gid, '')}")
        lines.append("")
        lines.append("Local IDs per film:")
        for k, f in field_seq:
            local = global_to_local.get(f, {}).get(gid, None)
            lines.append(f"  {k:>3}  {f}:  {'' if local is None else local}")
        lines.append("")
        lines.append("Hotkeys:")
        lines.append("  n/p next/prev ID")
        lines.append("  g/b/u label good/bad/unsure")
        lines.append("  [/ ] prev/next film")
        lines.append("  space play/pause  ,/. step")
        lines.append("  s save")

        info_text.set_text("\n".join(lines))
        fig.canvas.draw_idle()

    def _save():
        rows = []
        for gid in gids:
            rows.append({"global_id": gid, "label": qc.get(gid, "")})
        with open(out_qc_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["global_id", "label"])
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {out_qc_csv}")

    def _next_id(step: int):
        nonlocal gid_idx
        gid_idx = int(np.clip(gid_idx + step, 0, len(gids) - 1))
        _render()

    def _next_film(step: int):
        _set_film_by_index(film_idx + step)
        _render()

    def _set_label(lbl: str):
        qc[_current_gid()] = lbl
        _render()

    # Button callbacks
    btn_prevfilm.on_clicked(lambda evt: _next_film(-1))
    btn_nextfilm.on_clicked(lambda evt: _next_film(+1))
    btn_previd.on_clicked(lambda evt: _next_id(-1))
    btn_nextid.on_clicked(lambda evt: _next_id(+1))
    btn_good.on_clicked(lambda evt: _set_label("good"))
    btn_bad.on_clicked(lambda evt: _set_label("bad"))
    btn_unsure.on_clicked(lambda evt: _set_label("unsure"))
    btn_save.on_clicked(lambda evt: _save())

    def _quit(evt=None):
        try:
            _save()
        except Exception:
            pass
        try:
            cap_combined.release()
        except Exception:
            pass
        plt.close(fig)

    btn_quit.on_clicked(_quit)

    def _radio(label):
        # label is the film name
        idx = films.index(label)
        _set_film_by_index(idx)
        _render()

    radio.on_clicked(_radio)

    # Keybindings + playback timer
    timer = fig.canvas.new_timer(interval=40)  # ~25 fps display

    def _on_timer():
        nonlocal playing
        if not playing:
            return
        film = current_film
        n = int(film_n.get(film, 0))
        if n <= 0:
            return
        film_frame[film] = (int(film_frame.get(film, 0)) + 1) % n
        _render()

    timer.add_callback(_on_timer)

    def _on_key(event):
        nonlocal playing
        k = event.key

        if k in ("n", "right"):
            _next_id(+1)
        elif k in ("p", "left"):
            _next_id(-1)

        elif k == "g":
            _set_label("good")
        elif k == "b":
            _set_label("bad")
        elif k == "u":
            _set_label("unsure")

        elif k == "]":
            _next_film(+1)
        elif k == "[":
            _next_film(-1)

        elif k == " ":
            playing = not playing
            if playing:
                timer.start()
            else:
                timer.stop()

        elif k == ".":
            playing = False
            timer.stop()
            n = int(film_n.get(current_film, 0))
            if n > 0:
                film_frame[current_film] = min(n - 1, int(film_frame.get(current_film, 0)) + 1)
            _render()

        elif k == ",":
            playing = False
            timer.stop()
            film_frame[current_film] = max(0, int(film_frame.get(current_film, 0)) - 1)
            _render()

        elif k == "s":
            _save()

        # quick jump film by number keys (1..9)
        elif k is not None and len(k) == 1 and k.isdigit():
            idx = int(k) - 1
            if 0 <= idx < len(films):
                _set_film_by_index(idx)
                _render()

        elif k in ("escape",):
            _quit()

    fig.canvas.mpl_connect("key_press_event", _on_key)

    # ---- KEEP ALIVE (your style) ----
    fig._btn_prevfilm = btn_prevfilm
    fig._btn_nextfilm = btn_nextfilm
    fig._btn_previd = btn_previd
    fig._btn_nextid = btn_nextid
    fig._btn_good = btn_good
    fig._btn_bad = btn_bad
    fig._btn_unsure = btn_unsure
    fig._btn_save = btn_save
    fig._btn_quit = btn_quit
    fig._radio = radio
    fig._timer = timer
    fig._im_artist = im_artist
    fig._info_text = info_text
    _LIVE_WIDGETS.append((fig, btn_prevfilm, btn_nextfilm, btn_previd, btn_nextid,
                          btn_good, btn_bad, btn_unsure, btn_save, btn_quit,
                          radio, timer, im_artist, info_text))

    _render()
    plt.show(block=block)

    return {"qc": qc, "qc_csv": out_qc_csv, "global_ids": gids}


def make_15frame_summary_mp4(
    working_dir: str,
    films: List[str],
    out_mp4: str,
    out_index_csv: Optional[str] = None,
    seconds_per_frame: float = 2.0,
    fps_out: Optional[float] = None,
    resize_to_first: bool = True,
    force: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Build a short summary mp4:
      for each film in `films`, take 3 frames: first, middle, last.
      Total frames = 3 * len(films). Each frame is held for seconds_per_frame.

    Inputs:
      - working_dir: directory containing "<film>_population.mp4"
      - films: list of film names (length 5 in your case)
      - out_mp4: output mp4 path
      - out_index_csv: optional CSV recording mapping (recommended)
      - seconds_per_frame: hold duration per sampled frame
      - fps_out: output fps. If None -> choose 10 fps (nice round) so each still is 20 frames at 2 sec.
      - resize_to_first: resize all frames to first movie size to avoid writer mismatch
      - force: rebuild even if out files exist

    Returns:
      (out_mp4, out_index_csv)
    """
    wd = Path(working_dir)
    out_mp4 = str(out_mp4)
    if out_index_csv is None:
        out_index_csv = str(Path(out_mp4).with_suffix(".index.csv"))

    if (not force) and os.path.isfile(out_mp4) and os.path.isfile(out_index_csv):
        print(f"[summary15] cache exists: {out_mp4}")
        return out_mp4, out_index_csv

    mp4_paths = []
    for film in films:
        p = wd / f"{film}_population.mp4"
        if not p.is_file():
            raise FileNotFoundError(f"Missing mp4: {p}")
        mp4_paths.append(str(p))

    # pick output fps
    fps = float(fps_out) if fps_out is not None else 10.0
    hold = int(round(seconds_per_frame * fps))
    hold = max(1, hold)

    # determine output size from first movie
    cap0 = cv2.VideoCapture(mp4_paths[0])
    if not cap0.isOpened():
        raise FileNotFoundError(f"Cannot open mp4: {mp4_paths[0]}")
    W0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap0.release()

    os.makedirs(os.path.dirname(out_mp4) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_mp4, fourcc, fps, (W0, H0), True)
    if not vw.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter: {out_mp4}")

    rows = []
    out_frame_cursor = 0

    def read_frame_at(cap: cv2.VideoCapture, idx: int):
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = int(np.clip(idx, 0, max(0, n - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            return None
        # ensure 3-channel BGR
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if resize_to_first and (frame.shape[1] != W0 or frame.shape[0] != H0):
            frame = cv2.resize(frame, (W0, H0), interpolation=cv2.INTER_NEAREST)
        return frame

    for film, path in zip(films, mp4_paths):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            vw.release()
            raise FileNotFoundError(f"Cannot open mp4: {path}")

        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release()
            print(f"[summary15][WARN] {film}: frame_count=0, skipping")
            continue

        picks = [
            ("begin", 0),
            ("middle", n // 2),
            ("end", max(0, n - 1)),
        ]

        for tag, idx in picks:
            frame = read_frame_at(cap, idx)
            if frame is None:
                print(f"[summary15][WARN] {film}:{tag} failed to read frame {idx}")
                continue

            start_out = out_frame_cursor
            for _ in range(hold):
                vw.write(frame)
                out_frame_cursor += 1

            rows.append({
                "film": film,
                "tag": tag,
                "src_frame": int(idx),
                "src_n_frames": int(n),
                "out_start_frame": int(start_out),
                "out_n_frames_held": int(hold),
                "seconds_per_frame": float(seconds_per_frame),
                "fps_out": float(fps),
            })

        cap.release()
        print(f"[summary15] {film}: sampled 3 frames (n={n})")

    vw.release()

    os.makedirs(os.path.dirname(out_index_csv) or ".", exist_ok=True)
    with open(out_index_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "film","tag","src_frame","src_n_frames",
                "out_start_frame","out_n_frames_held",
                "seconds_per_frame","fps_out"
            ],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"[summary15] wrote: {out_mp4}")
    print(f"[summary15] index: {out_index_csv}")
    return out_mp4, out_index_csv

