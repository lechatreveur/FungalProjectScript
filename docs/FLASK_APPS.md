# Flask review apps: shared-core contract

`SingleCellQuantificationHPC/` has three Flask web tools that all read and write
the **same on-disk dataset** in the movie tree (`cell_<id>_masks.csv`,
`sequence_linkage.json`, `qc_<film>.json`, septum labels). This document defines
which code owns that dataset layer and what any app must honor when it writes.

Referenced by [PROJECT_POLICY.md](PROJECT_POLICY.md) P9. Frame coordinates are
in [COORDINATE_SYSTEMS.md](COORDINATE_SYSTEMS.md).

## The three apps

| App | Port | Role | Integration |
|---|---|---|---|
| `tracking_corrector` | 5001 | Full corrector: mask editing, tracking QC, linkage, septum, jobs. **Owns the shared dataset-access layer.** | self-contained |
| `septum_alignment_board` | 5002 | Read/label septum intervals across a whole sequence at once. | **imports `tracking_corrector`** — config, repositories, services, and re-registers its blueprints. No forked logic. |
| `ground_truth_corrector` | 5002* | 3-keyframe ground-truth curation + Cellpose training export. | **currently forks** `tracking_corrector` modules — see "Non-conforming" below. |

\* `ground_truth_corrector` also defaults to 5002; set `GT_TRACKING_PORT=5003`
so all three can run at once.

## Canonical owner

`tracking_corrector` owns, and is the single source of truth for:

- **`config.py`** — the `env var → config.yaml → default` path resolver (P4).
- **`repositories/`** — `qc_repository.py`, `linkage_repository.py`,
  `mask_repository.py`: all file I/O for the shared dataset, including atomic
  writes and revision checks.
- **Dataset-writing services** — `qc_service.py`, `septum_service.py`,
  `masks_service.py`, `linkage_service.py`.
- **`qc_schema.py`** — the QC entry schema.

`septum_alignment_board` is the reference pattern for a new app: put the parent
`SingleCellQuantificationHPC` dir on `sys.path`, then
`from tracking_corrector.<...> import <...>`, and reuse — repositories, services,
and whole blueprints (`experiments_bp`, `frames_bp`, `septum_bp`) — rather than
reimplementing.

## On-disk dataset contract

Any code that writes a shared dataset file must:

1. **Write atomically.** Temp file in the same directory, then `os.replace()`.
   Never a partial write to `cell_<id>_masks.csv`, `sequence_linkage.json`,
   `qc_<film>.json`, or a septum label file. `tracking_corrector` has
   `atomic_write_text()` / `atomic_write_dataframe()` and
   `tests/test_atomic_writes.py` — use them.
2. **Check revision before overwrite.** Pass the `expected_revision` /
   file-hash the read returned; refuse the write if the file changed underneath
   (optimistic concurrency — two tools can be open at once).
3. **Honor one canonical writer per file class.** `sequence_linkage.json` and
   `cell_<id>_masks.csv` are written through `tracking_corrector`'s
   repositories. A second app that needs to write them imports those, it does
   not carry its own copy.
4. **Use the frame coordinate rules** in `COORDINATE_SYSTEMS.md` and the QC
   schema in `tracking_corrector/qc_schema.py`.

**Read-only reuse is unrestricted** — registering another app's blueprint,
calling a read-only repository method, importing a pure helper. The restriction
is on forking a *writer*.

## Non-conforming: `ground_truth_corrector`

It carries its own diverged copies of `qc_repository.py`,
`linkage_repository.py`, `mask_repository.py`, `linkage_service.py`,
`tracking_service.py`, `experiments_service.py`, and `security.py`. Its mask
writer (`atomic_write_dataframe` + `compute_file_hash`) and its inline
`save_linkage` tmp-file logic are a **second implementation** of the contract
above and have already drifted from `tracking_corrector`'s versions. Because
both apps write `sequence_linkage.json` and mask CSVs in the same movie tree,
their revision-check semantics must not disagree.

**Cleanup task (not yet done):** reconcile `ground_truth_corrector` to import
the shared layer from `tracking_corrector`, or document per module why a fork is
required and keep it explicitly in sync. Do this as a staged handoff (P7), not a
silent refactor — it touches the write path for the shared dataset.

## Shared UI conventions

Presentation invariants that every app's frontend must honor so the same cell
looks the same in `tracking_corrector` (which `manual_correction_tool.py` wraps),
`ground_truth_corrector`, and `septum_alignment_board`.

- **Stable per-cell color.** A cell's display color is a deterministic function
  of its **stable identity** — `global_cell_id` for a linked cell,
  `(film, local_cell_id)` for a single-film cell — hashed to a hue. It must
  **not** depend on render order, gallery or page position, row index, or
  `new_cell_id` / table-row-id (a flat row index into `id_map_unaligned.csv`,
  not an identity — see [COORDINATE_SYSTEMS.md](COORDINATE_SYSTEMS.md) and the
  M156 strip drift). Consequence: the same cell keeps its color across frames,
  re-renders, pagination, and across all three apps.
- **One shared color helper**, not a copy per app — the "don't fork" rule
  applies to shared frontend logic too. There is no such helper today (colors
  are currently all status-driven: good / bad / mistracked / corrected); adding
  one introduces the shared function.
- Status colors (QC good/bad/mistracked, save state) are a separate axis and
  stay as they are; the per-cell identity color is an addition, not a
  replacement.

## Rules for a new or changed app

- New app → `septum_alignment_board` pattern. No forked dataset module.
- Shared frontend conventions above are honored, not reimplemented per app.
- Changing a `tracking_corrector` repository or dataset-writing service: check
  every consumer (`septum_alignment_board`, and `ground_truth_corrector` until
  it is reconciled), and keep `test_atomic_writes.py` / `test_septum_*` green.
- Adding a write path for a new file class: decide and document its canonical
  writer here before shipping it.
