# Frame coordinate systems

Canonical reference for the frame-index coordinate systems used when a cell's
tracked timeline is stitched across multiple films. Every tool that reads or
writes septum intervals, alignment offsets, or sequence-linked cell data must
follow this. It consolidates rules that were previously spread across
`tracking_corrector/static/js/state.js`, `services/septum_service.py`, and
`routes/septum_bp.py`; those files remain the implementation and point here.

Referenced by [PROJECT_POLICY.md](PROJECT_POLICY.md) P6 and [AGENTS.md](../AGENTS.md).

## The four spaces

| Space | Definition | Where it lives |
|---|---|---|
| **Local frame** | Zero-based frame index *within one film*. Frame `t` of `Frames_<film>/` and row `t` of `TrackedCells_<film>/cell_<id>_masks.csv`. | Raw pipeline output; `one_cell_quantification_1CH.py`; mask/quant CSVs |
| **Aligned frame** | `local_frame + offset`, where `offset` is a per-`(film, local_cell_id)` integer stored in that film's septum QC JSON. This is how septum `start`/`end` are **stored on disk**. Different films can hold different offsets for the same global cell. | `cell_intervals[cid].start_aligned` / `end_aligned` (+ `_2`) and `offsets[cid]` in each film's septum JSON |
| **Sequence frame** | Position on the stitched multi-film timeline. Film 0 occupies `[0, L0)`, film 1 `[L0, L0+L1)`, … where `Lk` is film `k`'s frame count. The authoritative sequence-wide space for a linked cell. | Server: `SeptumService._sequence_film_bounds()`. Client: `getFilmSequenceBounds()` / `state.filmBoundaries` in `state.js` |
| **Global cell / sequence** | A `global_cell_id` inside a `sequence` from the experiment's `sequence_linkage.json`; `global_cells: {global_cell_id: [local_id_per_film, ...]}` maps it to one local id per film (`-1` where the cell is absent from that film). | `LinkageRepository` / `LinkageService.get_sequences`; `sequence_linkage.json` |

Conversions: `aligned_to_local_frame(a, offset) = a - offset`;
`local_to_aligned_frame(l, offset) = l + offset`;
`local → sequence = film_bounds[film].start + local`. `_sequence_film_bounds()`
(server) and `getFilmSequenceBounds()` (client) must produce the same numbering
so the two stay consistent.

## API contract

- **The frontend speaks *sequence frames* for linked cells.** When saving a
  septum interval on a linked/multi-film global cell, endpoints are sent as
  plain sequence-wide frame numbers with no per-film offset math on the client.
  The backend fans the interval out across whichever films it actually touches.
- **The backend returns fully resolved sequence frames.**
  `get_septum_alignment()` decodes each film's stored aligned value with **that
  film's own offset**, maps it onto the shared timeline via
  `_sequence_film_bounds()`, and returns the result with `offset: 0` as the
  signal "already resolved — do not subtract anything further."
- **Loading one cell's raw stored label** uses local film name + local cell id
  (not sequence / global id).
- **`film` / `cell_id` in a save request body** for a linked cell need only be
  *any* one `(film, local_cell_id)` pair in that global cell's sequence — used
  solely to locate the linkage file.
- A cell may carry **two septum events**: `has_septum` / `has_septum_2` with
  matching `start_frame(_2)` / `end_frame(_2)` / `start_aligned(_2)` /
  `end_aligned(_2)`.

## Rules

1. **A septum interval's two endpoints are not guaranteed to be in the same
   film.** A division can straddle a film boundary. Never assume "the film the
   playhead is on" is the film an endpoint belongs to.
2. **Decode each film's stored value with that film's own offset.** Offsets are
   saved independently per `(film, local_cell_id)`.
3. **Convert to sequence frames before combining** endpoints from different
   films (`min` of starts, `max` of ends across the touched films).
4. **`offset: 0` in a response means "resolved"** — the client must consume the
   value as-is and must not re-guess film membership.
5. Single-film / local-edit cells trivially have every frame in `filmIdx 0`;
   none of the cross-film handling applies, but the same helpers must still be
   used so behavior is uniform.

## Historical bugs this prevents

- **Save bug:** forcing both endpoints through the currently active film's local
  bounds rejected or mis-saved the endpoint that belonged to the other film —
  it appeared to vanish immediately after being set.
- **Load bug:** decoding every film's aligned value with a single offset
  (whichever film was the query entry point) made the same cell's endpoints
  appear to shift depending on which film was active at reload. Correct behavior
  only when all films' offsets coincided.

## Related invariants (elsewhere)

- Multi-film QC state (offsets, cell intervals, global interval map) is keyed by
  `(film_name, cell_id)` tuples — see `DEVELOPMENT_NOTES.md` §1.
- `skimage` `regionprops.orientation` is measured from the row (vertical) axis;
  crop rotation in the strip generators must use `-(angle_deg - 90)`, not
  `-angle_deg` — see `DEVELOPMENT_NOTES.md` and the M156 strip fix.
