# Fungal Septum Classifier: Development History & Lessons Learned

This document serves as a guide for future developers maintaining the Fungal Septum Classifier. It outlines the major architectural shifts, design decisions, and critical "traps" encountered during the development of version 2.0.

## 1. Multi-Film Architecture
### The Problem
Originally, the GUI used a single `global_interval` variable and a simplified indexing system. When the project scaled to multiple experiments (M92, M93, M96), state interference occurred—setting a red interval in one film would erroneously appear in others.

### The Solution (Multi-Film State)
- **Composite Keys**: Most internal state dictionaries (offsets, cell intervals) now use a `(film_name, cell_id)` tuple as the primary key.
- **Global Interval Map**: We implemented a `global_intervals_map` to store intervals per-film.
- **Forward-Filling Logic**: During dataset export, if a film hasn't been explicitly touched, it inherits the global interval from the previous film (chronological persistence).

## 2. The "White Septum" Trap (Critical)
### The Mistake
Early in development, "white septums" (highly contrastive, bright ridges) were labeled with the `i` key. However, the training script deterministically inverted these cells (`1.0 - x`). 
- **Result**: The AI *never* saw a bright septum during training. It became purely a "dark valley detector."
- **Failure**: When encountering white septums in the GUI during real-time inference, the model predicted 0% confidence because it was never trained on the original bright polarity.

### The Fix: Global Polarity Invariance
We removed the deterministic labeling-based flip and replaced it with **50% Random Polarity Inversion** during training.
- **Why**: This forces the CNN to detect the *structure* of the septum (the gradient and shape) regardless of whether it is pixel-bright or pixel-dark. 
- **Lesson**: Do not use hard-coded label-based inversions; let the data augmentation teach the model invariance.

## 3. Data Integrity & Export Pipeline
### The Baseline Bug
A critical bug existed where cells without explicit start/end alignments were being skipped during export, even if they had a valid global interval. This resulted in severely under-sampled training sets (e.g., M96 originally exported 18 samples instead of the expected 54).

### Key Logic
- The `septum_training_utils.py` script now correctly falls back: 
  1. Per-cell labels (most specific).
  2. Global film interval (if cell is not specifically labeled).
  3. Skips only if neither exists.

## 4. GUI Rendering & Visualization
- **Stable per-cell color** (correction tools): a cell's color must key off its
  stable identity (`global_cell_id`, or `(film, local_cell_id)`), never render
  order or a table-row index, so the same cell keeps its color across frames and
  across `tracking_corrector` / `ground_truth_corrector`. See
  `docs/FLASK_APPS.md` "Shared UI conventions".
- **Saliency (AI Vision)**: We integrated autograd-based saliency heatmaps. 
- **Transparency Fix**: When overlaying heatmaps, do not use simple `alpha` on a 2D array, or the "black/low" values will darken the biology. Instead, use a **4-channel RGBA map** where the intensity of the saliency is mapped directly to the **Alpha** channel. 
- **Axis Bounds**: When adding new `imshow` artists for overlays, always initialize them with the same dimensions as the main sheet (`np.zeros_like(sheet)`) to avoid Matplotlib's auto-scaling squishing the window.

## 5. Ground Truth Corrector & Multi-Film Sequence Linkage
### The Local ID vs Global Track Trap
In multi-film time-lapse experiments, Cellpose segments each film independently, assigning integer IDs starting from 1 ($1 \dots N$). Local `cell_248` in `FL1` has zero relation to local `cell_248` in `BF4` (they are 1000px apart).
- **Failure**: Naive broadcast linking (`[248, 248, 248, ...]`) or falling back to `int(req.cell_id.split('_cell_')[-1])` during mask saving blindly writes into unrelated local cell CSVs across films.
- **Rule**: Never parse integer IDs from global cell ID strings. In `/api/save_mask`, always use `track[f_idx]` or allocate a new local ID (`max_cid + 1`) within the target film.

### QC Track Preservation & Teleportation Prevention
- **Trap**: When relinking sequences with Hungarian bipartite matching, preserving user-reviewed tracks (`status in ('good', 'corrected')`) locked in legacy broken dummy broadcast tracks without displacement validation.
- **Rule**: Preserved tracks must undergo physical continuity validation ($\Delta d \le 30.0\text{ px}$). If a legacy QC track contains teleportation jumps ($> 30\text{ px}$), it must be repaired using forward/backward Hungarian propagation from the anchor film (`fix_jumping_curated_cells.py`).

### Single-Cell Mode Interaction Lock
- **Trap**: Allowing canvas double-click, Space+Click, or mousedown hit-testing to switch global cells while in Single-Cell view causes sudden, disruptive cell switching when drawing near adjacent boundaries.
- **Rule**: Canvas-based global cell switching is strictly restricted to **Population Mode**. In Single-Cell mode, active cell selection is 100% locked to explicit user clicks on the sidebar cell list.

### Canvas Coordinate Scaling
- **Trap**: Calling `canvas.getBoundingClientRect()` in viewers with CSS `transform: translate(panX, panY) scale(scale)` returns dimensions already transformed by scale. Dividing by `scale` again squares the scale factor.
- **Rule**: Always measure mouse coordinates relative to the untransformed viewport container (`viewport.getBoundingClientRect()`), then compute `(clientX - containerRect.left - panX) / scale`.

### RLE Mask Encoding (Fortran / Column-Major Order)
- **Trap**: Writing ad-hoc RLE decoders with row-major arithmetic (`cy = idx / width, cx = idx % width`) transposes/inverts $(x, y)$ coordinates. This produces scrambled centroids that misguide nearest-neighbor and tracking algorithms to jump across the canvas.
- **Rule**: Always import canonical `validate_and_decode_rle` from `SingleCellQuantificationHPC/ground_truth_corrector/schemas.py`. Pipeline RLEs are strictly 1-indexed Fortran-order (`order='F'`).

## 6. Sequence Linkage & Retracking Architecture
### Forward Tracking Redundancy vs Backward Shared-Mother Architecture
- **Trap**: Forward tracking from Film 0 causes a ~2x explosion of redundant global cells. When a cell divides or missegments, forward tracking spawns a new global cell with leading `[-1]`s and leaves a dead-end mother stub with trailing `[-1]`s.
- **Rule**: Always track backward from the terminal film ($Film_{N-1}$) to $Film_0$. Merge daughter cell tracks onto a shared mother track when $\text{Area}(\text{Mother}) \ge 1.4\times \text{Area}(\text{Daughter})$. Strictly enforce that a mother cell can be shared by at most 2 daughters.

### In-Place Mask Mutation & Auto-Duplication
- **Trap**: Modifying a local cell mask CSV in-place when that local cell is registered in another global cell (e.g. adjacent cell) silently corrupts the other global cell.
- **Rule**: In `/api/save_mask`, check if the local cell is shared across tracks. If shared, automatically fork to a new local ID (`max_cid + 1`), copy unchanged keyframes, write the edit to the new CSV, and update the track in `sequence_linkage.json`.

### Strict Curated Track Preservation
- **Trap**: Filtering preserved tracks with length heuristics (e.g., `valid_films >= 11`) drops valid curated tracks that legitimately span fewer films.
- **Rule**: Any track marked `corrected` or `good` in `qc_<seq>.json` is 100% immutable and protected in `protected_globals` regardless of track length.

## Summary Checklist for Updates
- [ ] If changing the neural network, update `FungalInferenceCore` in `inference_core.py`.
- [ ] If adding human labels, verify the keybinding doesn't conflict in `alignment_board_gui.py`.
- [ ] Always check `manifest.csv` after an export to ensure the `has_septum` count matches your GUI expectations.
- [ ] When saving masks in sequence mode, never parse local cell IDs from cell name strings.
- [ ] Never allow canvas-based cell switching while in single-cell editing mode.
- [ ] Always decode RLEs using Fortran column-major order (`order='F'`); never write ad-hoc C-order decoders.
- [ ] Track sequences backward from terminal film with shared-mother merging ($\ge 1.4\times$ area, $\le 2$ daughters).
- [ ] When saving modified masks, automatically fork to `max_cid + 1` if the local cell is shared.
- [ ] Always protect 100% of curated (`corrected`/`good`) tracks regardless of length.


