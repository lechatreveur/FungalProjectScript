# Report for Antigravity: stable per-cell color in the ground truth corrector

Written 2026-09-02, confirmed against the code on branch `feature/gtc-stable-color`
(cut from `feature/tracking-corrector-refactor` @ `a17145a`). This is a
"what was done + what is open" report, not a staged prompt — the user wants to
work through the open questions with you while building the tool. Follows the
shared-facts style of `docs/STAGED_HANDOFF.md`.

Rule being implemented: `docs/FLASK_APPS.md` "Shared UI conventions" — a cell's
color is a deterministic function of its **stable identity**, computed by one
helper shared between the server render and the client overlay, so the same cell
keeps one color across keyframes, across films in a linked sequence, and across
`ground_truth_corrector` / `tracking_corrector`.

---

## Status

- All changes are in `SingleCellQuantificationHPC/ground_truth_corrector/` and
  are **uncommitted** — that directory is entirely untracked in git (never
  committed, not gitignored). There is no clean baseline to diff against, so the
  ~8 touched files sit interleaved with the rest of the untracked app. The user
  is deciding how to land it (commit the app first, or a two-commit branch).
- Nothing in `tracking_corrector` was changed.
- New tests pass. One pre-existing GTC test fails on this branch **and on the
  base branch** — see Verification.

---

## The design

Three frame coordinate spaces matter here (full detail in
`docs/COORDINATE_SYSTEMS.md`):

- **local cell id** — the integer N in `TrackedCells_<film>/cell_N_masks.csv`.
  Stable across time *within one film*.
- **global_cell_id** — a linked cell's identity across a multi-film sequence
  (from `sequence_linkage.json`, `global_cells: {gid: [local_id_per_film, ...]}`,
  `-1` where absent). Stable across the whole sequence. This is what the client
  already holds in `state.selectedCell` (see `static/js/cells.js` `selectCell`).
- **raw `_seg.tif` label** — the Cellpose label integer in one frame's seg file.
  **NOT** a stable identity: the same physical cell gets a different label in
  each frame. Coloring by this was the bug.

Pipeline: `stable identity` → `stable_color_key()` (int passes through;
global_cell_id string → FNV-1a-32 hash) → `id_to_color()` (hash → hue). The same
two functions exist in Python and JS and must stay byte-identical.

---

## Files changed

### `services/gt_frames_service.py`

- `UNTRACKED_COLOR = (150, 150, 150)` (line 18) — BGR grey for a seg region with
  no tracked cell behind it.
- `fnv1a_32(s)` (line 29) — deterministic 32-bit FNV-1a over UTF-8 bytes.
  global_cell_ids are ASCII, so the JS `charCodeAt` port lines up.
- `stable_color_key(identity)` (line 40) — `int`/`np.integer` → itself;
  `str` → `fnv1a_32`.
- `id_to_color(cell_id)` (line 53) — **replaced** the old golden-ratio version
  with `tracking_corrector.services.frames_service.id_to_color`'s exact
  algorithm (Knuth multiplicative hash `* 2654435761`, s=0.8, v=0.95). That one
  returns RGB; this one returns **BGR** for cv2. The two copies are now
  algorithmically identical — deduplicating them is the tracked P9 task, not
  done here.
- `local_to_global_map(exp, film, sequence)` (line 170) — `{local_cell_id:
  global_cell_id}` for one film of a sequence, via `LinkageService` /
  `LinkageRepository` (same path `get_sequence_keyframe_map` already uses).
  Returns `{}` when there is no sequence (single-film view) or the film is not in
  the sequence — callers then key color on the local id.
- `seg_label_identity(exp, film, local_t, seg_lbl, H, W, local2global)`
  (line 202) — `{seg_label: (color_key_int, display_str)}` by **max mask
  overlap**: for each `cell_N_masks.csv`, decode the RLE at `local_t`, find which
  seg label its pixels fall on most, and claim that label for cell N's identity.
  Labels not claimed are absent → caller uses `UNTRACKED_COLOR`. First CSV to
  claim a label wins (deterministic per directory listing; ties are rare).
- `render_population_frame_jpeg(exp, film, t_val, sequence=None, quality=85)`
  (line 294):
  - cache key is now `(exp, film, t_val, sequence)` — color depends on the
    sequence.
  - `_seg.tif` fast path: builds the LUT from `seg_label_identity` instead of
    `id_to_color(r.label)`; unmatched labels → grey; centroid text shows the
    global id when known, else `?`.
  - CSV fallback path: `id_to_color(stable_color_key(local2global.get(cid, cid)))`;
    text shows the global id when known, else `cid`.
  - `clear_population_cache` still works — it filters on `k[0]/k[1]/k[2]`, which
    are unchanged; the 4th tuple element is just not filtered.

### `routes/frames_bp.py`

- `get_population_frame` (line 106) — passes `sequence=sequence` into
  `render_population_frame_jpeg`.
- `get_frame_boundaries` (line 135) — was one fixed cyan `(255,255,0,255)` for
  every contour. Now: `local_to_global_map` + `seg_label_identity`, then per
  unique seg label draw its contour in `id_to_color(key)` (BGRA), grey for
  unmatched (lines 170, 181-190).

### `static/js/color.js` (new)

- `fnv1a32(s)` (line 8), `idToColor(id)` (line 17) — ports of the Python
  functions; `idToColor` returns `[r, g, b]` (server returns BGR — order is the
  only difference, documented in the file header).
- `stableColorKey(identity)` (line 39), `selectedCellColorRGB()` (line 46) —
  uses `state.selectedCell` (already the global_cell_id) → `[r,g,b]` or `null`.

### `static/js/canvas.js`

- `drawMask()` (line 109) — the selected cell's mask overlay uses
  `selectedCellColorRGB()` (line 124), falling back to the old blue
  `[59,130,246]` if there is no selected cell. Alpha still 160.

### `templates/index.html`

- Loads `js/color.js` right after `js/state.js`, before `js/canvas.js`.

### `static/js/cells.js` + `static/css/style.css`

- Sidebar cell list: a `.cell-swatch` per row (line ~18), `background:
  rgb(idToColor(stableColorKey(String(c.global_id))))` — same key as everything
  else. `.cell-swatch` style added next to `.cell-item` in the CSS.

### `tests/test_stable_color.py` (new)

- 6 `unittest` tests: `id_to_color` determinism + bounds; same identity → same
  color; a global_cell_id's color is invariant across resolution paths; distinct
  identities mostly differ; FNV-1a reference vectors (`""` → `0x811C9DC5`, `"a"`
  → `0xE40C292C`, `"foobar"` → `0xBF9CF968`) for the JS port to match;
  `stable_color_key(int)` is identity.

---

## Verification done

- `python -m unittest ground_truth_corrector.tests.test_stable_color` → 6/6 pass.
- `create_app(Config())` imports and builds; `render_population_frame_jpeg`
  signature and the two new methods are present.
- `node --check` passes on `color.js` and `canvas.js`.
- **Pre-existing failure, not caused by this change:**
  `ground_truth_corrector.tests.test_ground_truth_tool.test_routes` fails at
  `mask_read[45,45] == 1` (a mask save/read roundtrip). Confirmed it fails
  identically on `feature/tracking-corrector-refactor` with these changes
  stashed.

---

## Open questions to work through with the user

1. **Single-film / unlinked cells.** The linked-sequence case (the actual bug —
   a cell across films/keyframes) is exact: both server and client hash the same
   `global_cell_id` string. For a single-film cell with no linkage, the server
   keys on the integer `cid` while the client keys on `String(state.selectedCell)`
   — if `state.selectedCell` for such a cell is not exactly `str(cid)`, the
   client mask color and the server population color can differ. Need to confirm
   what `list_cells` returns as `global_id` for a single-film experiment and
   align the two.
2. **`seg_label_identity` overlap heuristic.** It reads every `cell_*_masks.csv`
   in the film and decodes one RLE row each, per rendered frame. Fine for
   3-keyframe curation (cached), but if population view ever paginates or plays,
   this is N CSV reads per frame. Consider a per-`(film, local_t)` cache, or a
   persisted label↔cid map if the tracking pipeline writes one.
3. **`id_to_color` duplication.** GTC now carries an algorithmic copy of
   `tracking_corrector`'s function. `docs/FLASK_APPS.md` P9 wants one shared
   helper. Options: add the `sys.path` shim (`septum_alignment_board/app.py` has
   the pattern) and import it; or leave the copy with the comment that is there
   now. This is part of the larger GTC-un-forking task.
4. **Git baseline.** `ground_truth_corrector/` is untracked. Decide whether to
   commit the app on `feature/tracking-corrector-refactor` first (clean
   follow-up diff) or check it in on the feature branch.
5. **Boundary label text.** `get_frame_boundaries` colors outlines but draws no
   per-cell id text; the population view does. Decide if the boundary overlay
   should also label.

---

## Do NOT

- Do not color anything by a raw `_seg.tif` label, gallery/row order, render
  index, or `new_cell_id` / any flat table-row index (`docs/COORDINATE_SYSTEMS.md`,
  the M156 strip drift).
- Do not add a third `id_to_color` algorithm — converge on `tracking_corrector`'s.
- Do not change the FNV-1a or `id_to_color` math on one side (Python or JS)
  without changing the other; the `test_stable_color.py` reference vectors guard
  the Python side.
- Do not fork more `tracking_corrector` modules into GTC (`docs/FLASK_APPS.md` P9).
- If a stated fact here no longer matches the code, stop and flag it rather than
  building on it.
```
