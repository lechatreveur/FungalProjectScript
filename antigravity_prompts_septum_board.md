# Prompts for Antigravity: standalone septum alignment board (web)

Feed these to Antigravity one at a time, in order. Each stage should run and be checked
before moving to the next — this keeps the risky part (does it actually read/write the
same dataset as `tracking_corrector` without drift?) isolated to stage 1-2 before any
UI polish goes in.

Shared facts Antigravity should not have to re-derive (already confirmed against the
current code, dated 2026-08-06):

- `SingleCellQuantificationHPC/tracking_corrector/` is a Flask app with a clean
  `routes/ -> services/ -> repositories/` layering and a `create_app(cfg)` factory
  (`app.py`) that just registers blueprints. It is NOT a package today (no
  `SingleCellQuantificationHPC/__init__.py`); it's made importable by
  `manual_correction_tool.py` doing `sys.path.insert(0, <SingleCellQuantificationHPC dir>)`
  then `from tracking_corrector.__main__ import main`. The new tool must follow the
  same pattern so `tracking_corrector` is importable as a sibling package.
- A "global cell" = a `global_cell_id` inside a `sequence`, defined in each
  experiment's `sequence_linkage.json` (loaded via `LinkageRepository` /
  `LinkageService.get_sequences`). A sequence stitches multiple films together in
  order; `global_cells: {global_cell_id: [local_cell_id_per_film, ...]}` maps the
  global cell to its per-film local ids (-1 where absent in that film).
- The existing "dynamic gallery" (`tracking_corrector/static/js/gallery.js` +
  `GET /api/cell_strip_image?experiment=&sequence=&cell_id=<global_cell_id>&channel=`,
  implemented in `routes/frames_bp.py::get_cell_strip_image`) already renders exactly
  one global cell's full stitched-sequence strip as a single image, one 100x100 tile
  per sequence-wide frame index. This is the unit each board row should reuse.
- Septum labels are saved via `POST /api/save_septum_label`
  (`routes/septum_bp.py` -> `services/septum_service.py::save_septum_label`,
  schema `SaveSeptumRequest` in `schemas.py`). For a linked/multi-film global cell,
  the frontend sends `sequence` + `global_cell_id` + `start_aligned`/`end_aligned` as
  **plain sequence-wide frame numbers** (no per-film offset math client-side) — the
  backend fans the interval out across whichever films it actually touches. This
  exact logic and its reasoning is documented in the header comment of
  `tracking_corrector/static/js/septum.js` (lines 1-55) — read that comment before
  writing the save/load logic; it explains three coordinate systems (sequence frame /
  local frame / aligned frame) and which one the frontend is allowed to touch.
  `film`/`cell_id` in the request body just need to be *any* one (film, local_cell_id)
  pair belonging to that global cell's sequence — used only to look up the linkage
  file. Loading one cell's current label is `GET /api/get_septum_label?experiment=&film=&cell_id=`
  (local film + local cell id, not sequence/global_cell_id — see `loadSeptumLabels()`
  in `septum.js`).
- A cell can have up to two septum events (`has_septum`/`has_septum_2` and the
  matching `start_frame(_2)`/`end_frame(_2)`/`start_aligned(_2)`/`end_aligned(_2)`
  fields) — a row needs to be able to show/edit two intervals.
- Cached (offline batch) AI suggestions: `GET /api/get_septum_ai_cache?experiment=&sequence=&global_cell_id=`
  is a cheap file-only lookup (`services/septum_service.py::get_cached_ai_suggestion`),
  safe to call automatically per row. It returns `{"cached": false}` when nothing's
  been computed for that cell — treat that as "no suggestion," not an error. This is
  distinct from the live, review-only `POST /api/predict_septum`, which actually runs
  the model — do not call that automatically for a whole page of rows, only on
  explicit user action if at all.
- `GET /api/list_cells?experiment=&sequence=` (`routes/masks_bp.py` ->
  `TrackingService.list_sequence_cells`) returns `{cells: [{global_id, display_name}, ...], lineage: {...}}`
  — this is how to enumerate the rows for a sequence.
- The old desktop tool this is modeled on, `SingleCellDataAnalysis/alignment_board_gui.py`,
  never rendered every cell at once — it windowed to `n_rows` (default 12) visible
  rows with a vertical slider to page through the full cell list. Do the same here:
  `cell_strip_image` synthesizes each strip fresh on every request (decodes RLE masks,
  reads TIFFs, contrast-stretches per frame — no caching), so rendering an unbounded
  scroll of full-length strips will be slow. Fetch/render strips only for the
  currently visible page of rows.
- `tracking_corrector/config.yaml` already lists all known experiments with
  `display_name`s and the shared movie root (`/Volumes/X10 Pro/Movies` locally). Reuse
  `tracking_corrector.config.Config` rather than re-parsing this file.

---

## Stage 1 — scaffold + read-only board

```
Create a new standalone package at SingleCellQuantificationHPC/septum_alignment_board/,
sibling to SingleCellQuantificationHPC/tracking_corrector/. This is a lightweight web
tool for reviewing septum (cell division) alignment across an entire sequence at once,
modeled on the old desktop tool SingleCellDataAnalysis/alignment_board_gui.py, but
each row is a "global cell" (a sequence-spanning tracked cell, not a single-film
segment) and it must read/write the exact same dataset tracking_corrector already
manages — no new file formats, no duplicated file I/O logic.

Do this by importing tracking_corrector's existing modules directly rather than
reimplementing them: tracking_corrector.config.Config,
tracking_corrector.repositories.qc_repository.QCRepository,
tracking_corrector.repositories.linkage_repository.LinkageRepository,
tracking_corrector.services.audit_service.AuditService,
tracking_corrector.services.linkage_service.LinkageService,
tracking_corrector.services.septum_service.SeptumService,
tracking_corrector.services.frames_service.FramesService,
tracking_corrector.services.experiments_service.ExperimentsService. Follow the same
sys.path setup pattern as SingleCellQuantificationHPC/manual_correction_tool.py (which
inserts the SingleCellQuantificationHPC directory onto sys.path so tracking_corrector
is importable as a sibling package) so this new package can do the same for itself and
for importing tracking_corrector.

Build a Flask app (its own create_app() factory, its own __main__.py, its own
config.yaml or just reuse tracking_corrector's Config/config.yaml directly — your
call, but do not fork the experiment list into a second YAML file) that runs on port
5002 (tracking_corrector uses 5001, keep them distinct so both can run at once).
Register only the routes this tool needs: an experiment/sequence picker (reuse
ExperimentsService and LinkageService.get_sequences), a route to list the global cells
in a chosen sequence (reuse TrackingService.list_sequence_cells from
tracking_corrector.services.tracking_service, or call the equivalent logic directly),
and a route that reuses frames_bp.py's get_cell_strip_image logic (either by importing
and reusing FramesService directly, or by registering tracking_corrector's actual
frames_bp blueprint on this new app — prefer reusing the blueprint if it doesn't drag
in unwanted routes, otherwise extract the shared logic so there is one implementation,
not a copy). Do NOT register tracking_corrector's masks_bp or linkage_bp — no
brush/eraser mask editing, no lineage-tree editor in this tool.

Build a minimal frontend: experiment dropdown -> sequence dropdown -> a paginated grid
of rows, one row per global cell, showing that cell's full stitched strip image (reuse
GET /api/cell_strip_image?experiment=&sequence=&cell_id=<global_id>&channel= exactly
as tracking_corrector's gallery.js calls it). Show n_rows (default 12, configurable)
rows per page with prev/next page controls or a vertical slider, mirroring
alignment_board_gui.py's windowing (do not render the whole sequence's cell list at
once). No click interaction yet, no label editing yet, no AI overlay yet — this stage
is purely: pick a sequence, see a paged, scrollable grid of correctly-stitched global
cell strips. This validates the data-reuse plumbing before any write path is added.

Verify by running both tracking_corrector and this new tool against the same
experiment/sequence and confirming the strip images for a few global cells are
pixel-identical between the two tools' /api/cell_strip_image responses.
```

## Stage 2 — click-to-set interaction + save/load labels

```
Add labeling interaction to the septum alignment board built in stage 1. Read
tracking_corrector/static/js/septum.js lines 1-55 (the header comment) and
tracking_corrector/static/js/gallery.js in full before writing any of this — the
comment explains three frame-numbering systems in play (sequence frame / local frame
/ aligned frame) and exactly which one client-side code is allowed to send, and
gallery.js already implements the click-to-set-endpoint interaction this stage needs
to generalize from one selected cell to every visible row.

For each row, support up to two septum intervals (septum 1 / septum 2, matching
has_septum/has_septum_2 and the _2-suffixed fields in SaveSeptumRequest in
tracking_corrector/schemas.py). Add a mode selector (nav / set-start-1 / set-end-1 /
set-start-2 / set-end-2 — same modes as gallery.js's state.galleryClickMode) and
clicking a frame tile within a row's strip sets that row's corresponding endpoint,
exactly like gallery.js's onclick handler does for the single selected cell today —
just applied per-row instead of to one globally-selected cell.

Saving: POST to /api/save_septum_label with sequence + global_cell_id +
start_aligned/end_aligned/start_aligned_2/end_aligned_2 as plain sequence-wide frame
numbers (do not do local/offset math client-side for these — that's what
isLinkedSequenceCell() branches to in septum.js's saveSeptumLabels(), and every row
here is a linked/multi-film case by construction since it's a global cell). film/
cell_id in the request body just need to be any one (film, local_cell_id) pair
belonging to that row's global cell — pull it from the sequence's global_cells map
(same data list_cells / get_sequences already returns) rather than re-deriving it.

Loading: on page/row load, call GET /api/get_septum_label?experiment=&film=&cell_id=
using that same any-film/local-cell-id pair per row, and render the existing
has_septum/has_septum_2 state and interval highlighting using the same CSS-class
approach gallery.js already uses (septum-start-frame / septum-end-frame /
septum-during-frame and the _2 variants) so visual behavior matches the existing tool
exactly.

Autosave on each click (matching the existing tool's behavior of saving immediately
after a click sets an endpoint — see gallery.js's onclick handler calling
saveSeptumLabels() directly), with a lightweight per-row saved/unsaved indicator.

Verify by setting a septum interval on a multi-film global cell in the new tool, then
opening the same cell in tracking_corrector and confirming the interval displays
identically (and vice versa) — this is the real test that both tools are writing to
the same underlying per-film JSON files correctly.
```

## Stage 3 — cached AI suggestion overlay

```
Add a lightweight visual overlay to each row of the septum alignment board (stages 1-2
already built) showing the cached offline-batch AI suggestion for that global cell, if
one exists. Use GET /api/get_septum_ai_cache?experiment=&sequence=&global_cell_id=
(services/septum_service.py::get_cached_ai_suggestion in tracking_corrector) — this is
a cheap file-only lookup, safe to call automatically for every visible row on each page
load. It returns {"cached": false} when nothing has been computed for that cell yet;
treat that as "no suggestion available," not an error, and don't show anything for
that row in that case.

Do NOT call the live /api/predict_septum endpoint automatically for a page of rows —
that actually runs model inference per cell and tracking_corrector's own code
comments (septum_bp.py, septum.js) are explicit that it's meant to be an
explicit-click, review-only action for one cell at a time. If you want a "run live AI
for this row" action, make it an explicit per-row button, not automatic.

Render the cached suggestion distinctly from the human-set label (e.g. a thin marker
or different color under the frame tiles at the suggested start/end sequence frames —
the cached payload's shape is documented in septum.js's renderSeptumSparkline() and
the tracking_corrector_ai_cache memory: summary fields only (peak/start/end/
confidences), no per-frame probability array, so don't attempt to draw a full
sparkline from cached data — only the two endpoint markers plus whatever confidence
value is present). Make clear in the UI which frame markers are the AI's suggestion
and which are the saved human label, since they can disagree and that disagreement is
exactly what this overlay is meant to surface.
```
