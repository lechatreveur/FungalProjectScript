# Project policy

Version 1. Effective 2026-09-02. Version 1.1 (2026-09-02) adds P4–P6.
Version 1.2 (2026-09-02) adds P7. Version 1.3 (2026-09-02) adds P10.
Version 1.4 (2026-09-02) adds P9. Version 1.5 (2026-09-02) adds P11.
Version 1.6 (2026-09-02) adds P8.

The working policy for changes to this repository, whether made by a person or an
AI agent. [AGENTS.md](../AGENTS.md) is the entry router; this file is the
rulebook. It currently covers:

- **P1 — Charter and working principles** (below).
- **P2 — Claim-to-artifact validation ladder.**
- **P3 — Provenance: no orphan artifacts.**
- **P4 — Path resolution.**
- **P5 — Irreversible-action gate.**
- **P6 — Frame coordinate systems** (reference doc:
  [COORDINATE_SYSTEMS.md](COORDINATE_SYSTEMS.md)).
- **P7 — Staged handoff prompts** (reference doc:
  [STAGED_HANDOFF.md](STAGED_HANDOFF.md)).
- **P9 — Flask apps shared-core contract** (reference doc:
  [FLASK_APPS.md](FLASK_APPS.md)).
- **P10 — Training reproducibility** (reference doc:
  [TRAINING_CHECKLIST.md](TRAINING_CHECKLIST.md)).
- **P8 — Experiments ledger** (reference doc:
  [EXPERIMENTS.md](EXPERIMENTS.md)).
- **P11 — Sharing hygiene** (reference doc:
  [SHARING_HYGIENE.md](SHARING_HYGIENE.md)).

Cross-model verification (using an external CLI agent as an independent
reviewer) is planned for version 2 and is not policy yet.

---

## P1 — Charter and working principles

### Purpose

This repository converts time-lapse fungal microscopy movies into aligned,
quantitative single-cell data and models cell-cycle and septum behavior from it.
The pipeline stages and their entry-point scripts are listed in
[AGENTS.md](../AGENTS.md) and detailed in
[SingleCellQuantificationHPC/PIPELINE_PROTOCOL.md](../SingleCellQuantificationHPC/PIPELINE_PROTOCOL.md).

### Document hierarchy

Apply authority in this order. Higher layers control lower ones.

1. Current explicit instructions from the repository owner.
2. This policy (`docs/PROJECT_POLICY.md`).
3. The pipeline protocol, project map, and development notes
   (`PIPELINE_PROTOCOL.md`, `PROJECT_MAP.md`, `DEVELOPMENT_NOTES.md`) — canonical
   for how the pipeline behaves and what has already gone wrong.
4. Component READMEs and config schemas (e.g.
   `manifold_explorer/README.md`, each Flask app's layout).
5. Historical prompt logs (`antigravity_prompts_*.md`), training logs, experiment
   ledgers, and superseded artifacts — evidence, not policy.
6. Chat history, attachments, and model memory — navigation aids only.

If two documents at the same layer conflict, stop and name the conflict rather
than picking the convenient rule.

### Working principles

- **Smallest distinguishing test first.** Before a large run or a production
  change, state the competing explanations (including "bug" and "numerical
  artifact") and run the cheapest experiment that separates them.
- **Read before write.** Prefer read-only inspection, minimal tests, and
  narrowly scoped prototypes before touching production code or canonical data.
- **Separate exploratory from canonical.** Never overwrite a canonical output to
  run a diagnostic. Redirect output paths for tiny runs; back up and restore if
  a script only writes to the canonical location.
- **Experiment-dated scripts are frozen records.** `IAonNAS_<date>_*.py`,
  `quantify_M*.py`, `submit_array_M*.sh`, `generate_M*_strips.py`, and similar
  encode acquisition-specific paths and parameters. Do not "modernize" them in
  place. Archive one only when a named general replacement exists and a
  repository-wide search finds no active references (`PROJECT_MAP.md` rule).
- **Irreversible actions are gated.** File moves and deletes, in-place
  regeneration that overwrites canonical output, `rsync --delete`, and
  checkpoint relocation run only after the fix they depend on has passed its
  validation level, and only with a backup and an explicit go-ahead. "Do not run
  cleanup against an unverified fix."
- **Respect the known traps.** See `DEVELOPMENT_NOTES.md` and the
  non-negotiables list in `AGENTS.md`. The recurring ones: septum polarity
  invariance via 50% random inversion (never a label-based flip); export label
  fallback (per-cell → global film interval → skip); multi-film state keyed by
  `(film_name, cell_id)`; `skimage` `regionprops.orientation` measured from the
  row axis, not horizontal; manifold reference scaling and UMAP fit computed on
  the reference experiment only, mutants projected with `.transform()`.
- **Hygiene.** No credentials, host addresses, cookies, or payload dumps in
  commits. Environment-specific absolute paths belong in config or a resolver,
  not hard-coded into new scripts.

### Reproducibility baseline

- `requirements.txt` is the frozen dependency state. Do not bump a pinned
  version as a side effect of another change.
- Any run whose output is kept records: parameters, seeds, input paths, output
  path, git commit, and the validation result (see P3).

---

## P2 — Claim-to-artifact validation ladder

**Core principle:** every correctness claim is tied to an artifact — a
read-back, a schema/shape check, a minimal run, a metric, a comparison, or an
explicit uncertainty note. Confident prose is not evidence.

Pick the **lowest level that supports the claim you are making**. A higher-level
claim requires every lower level to have passed as well.

| Level | Name | Establishes | Typical cost |
|---|---|---|---|
| L0 | Documentation / read-back | The file or config says what you think, and every path it cites exists | seconds |
| L1 | Schema & shape | Output has the right columns / keys / dtypes; arrays have expected shape and value range; RLE masks decode; frame counts match the manifest | seconds |
| L2 | Minimal smoke run | A pipeline stage runs on one cell or a tiny movie and returns sane values | seconds–minutes |
| L3 | Regression vs trusted output | New or changed code reproduces an existing known-good artifact at the same inputs | minutes |
| L4 | Scientific validation | A claim about a measurement or biology is checked against curated ground truth or an independent recompute | minutes–hours |
| L5 | Method-defining review | A choice that changes what a number *means* — needs the owner's sign-off | owner + review |

### Claim → level mapping (examples)

- "The protocol doc now says X" / "the config points at the right movie" → **L0**.
- "`one_cell_quantification_1CH.py` still emits the standard `cell_<id>_data.csv`
  columns" / "the adapter produces the manifold-explorer input schema" → **L1**.
- "The tracker runs end to end on cell N and the per-frame IoU is in [0, 1] with
  no NaN areas" → **L2**.
- "The refactored `tracking_corrector` service returns the same septum labels as
  before for experiment M133" / "the new UMAP builder reproduces the current
  `umap_m156_*` plane at seed 42" → **L3**.
- "The retracked masks improve mean IoU vs the curated ground-truth set" /
  "`has_septum` count after export matches the GUI" / "the division-time fit
  residuals are within tolerance" → **L4**.
- "This is the correct augmentation polarity rule / label semantics / coordinate
  transform / QC threshold / manifold reference-scaling rule / segmentation
  model" → **L5**.

### Level detail

**L0 — Read-back.** After creating or editing any doc, config, or handoff file,
read the whole file back (targeted read-back only if context is tight and the
wrap-up says so). Confirm it exists, the intended sections are present, and the
edited passage reads correctly in context — not merely that a string was
inserted. For any path or filename mentioned in prose, confirm it exists.

**L1 — Schema & shape.** For a CSV: load it, assert the required columns are
present with the right dtype, and that value ranges are physical (areas > 0,
probabilities in [0, 1], frame indices contiguous). For a mask store: decode a
sample RLE and check the mask is non-empty and within frame bounds. For a JSON
(`sequence_linkage.json`, `qc_*.json`): validate against its schema module where
one exists (`schemas.py`), else check the key structure. Schema changes are
additive only; a renamed or removed field is a breaking change and needs every
reader updated in the same change.

**L2 — Smoke run.** Run at the smallest scale that exercises the code path (one
cell, a few frames, `res` small). A smoke test asserts, it does not just print.
Baseline assertions: no NaN in quantification columns; IoU and probability in
[0, 1]; segmented/tracked masks non-empty; failed cells are represented as
empty/masked, never silently substituted. Run from the directory the script
expects so relative paths resolve; write scratch output to the scratchpad or a
redirected path, never the canonical location.

**L3 — Regression.** Any code path that replaces or generalizes an existing one
must reproduce the existing trusted result before it is trusted for new cases.
Pin the comparison: same inputs, same seed, same resolution. Compare the
artifact, not a screenshot — column values, metric numbers, or a hash of the
serialized output. State the tolerance used.

**L4 — Scientific validation.** For a claim about a measurement or biological
quantity, name the ground truth and the metric. Tracking: IoU / survival rate /
final-frame IoU against the curated set (cf.
`SingleCellQuantificationHPC/tracker_comparison_summary.md`). Septum
classification: label-count parity with the GUI and per-interval agreement on a
held-out film. Division timing: fit residuals against manually marked events.
Store the full per-item result (per cell, per frame), not just the aggregate.

**L5 — Method-defining review.** Not settled by a passing smoke test. These need
the owner's reasoning and sign-off:
- Augmentation or label semantics (e.g. any change to septum polarity handling).
- Coordinate-system definitions (sequence frame / local frame / aligned frame).
- QC rule thresholds that turn a continuous quantity into an accept/reject.
- The manifold reference-scaling and UMAP-fit rule.
- Swapping the segmentation or tracking model, or its checkpoint, for production.
- Any threshold that converts a continuous metric into an on/off classification.

### Reporting unresolved uncertainty

When evidence is missing or ambiguous, do not round up to a confident claim.
Report in this shape:

```
UNCERTAIN: <one line: what is not established>
Evidence I have: <the L0–L4 check that did run, with its result>
Evidence I lack: <the check that would settle it, and why it did not run>
Options: (a) run <specific check>  (b) escalate to the owner for L5
         (c) mark provisional and proceed with <named risk>
Recommended: <one option>
```

Escalate rather than guess when the open question is a modeling or taste
decision (L5). Decomposition and more tests do not resolve those.

---

## P3 — Provenance: no orphan artifacts

Almost all data in this project is gitignored (`*.npz`, `*.pt`, `*.tif`,
`*.csv`, `*.mp4`, `*.png`, `data/`, `logs/`, `checkpoints/`,
`tracker_checkpoints*/`). Reproducibility therefore depends on every generated
artifact carrying its own provenance.

### What must carry provenance

Anything generated outside git that a later step, figure, or claim depends on:

- model checkpoints (`.pt`, `.pth`);
- curated / exported datasets (`.npz`, extracted crop sets);
- strip directories (`STRIPS_DIR` and similar);
- quantification tables (`cell_<id>_data.csv`, merged per-experiment tables);
- manifold dashboards (`umap_*.html`, `web_data.json`);
- population movies and overlay movies;
- any `qc_*.json` or linkage file produced by a script rather than hand-edited.

### Provenance record

Write a sidecar `<artifact>.provenance.json` next to the file, or a single
`_provenance.json` in the output directory for a multi-file artifact. Minimum
fields:

```json
{
  "artifact": "umap_m156_retrack_200.html",
  "created": "2026-09-02T11:47:00Z",
  "created_by": "build_umap_html_m156_retrack_200.py",
  "git_commit": "127a666",
  "git_dirty": true,
  "inputs": [
    "umap_features_m156_v5_200.csv",
    "/Volumes/X10 Pro/Movies/<experiment>/TrackedCells_<movie>/"
  ],
  "params": { "seed": 42, "n_cells": 200, "L_max": 81 },
  "host": "workstation-ssd",
  "notes": "retrack v5 features; reference scaling from Sept17"
}
```

For CSV or HTML outputs where a sidecar is inconvenient, an equivalent comment
header or embedded `<!-- provenance: ... -->` block is acceptable as long as it
carries the same fields.

The non-negotiable subset: **git commit, source inputs, generating script,
seed/params, date.**

### Rules

- A figure or claim built from an artifact must be able to name that artifact's
  provenance. An artifact with no provenance record is treated as exploratory
  and untrusted until it is regenerated with one.
- When an artifact's identifier scheme changes (for example the M156 vertical
  strip old vs new naming), the new run writes provenance and the obsolete files
  are moved to a `_superseded/` subdirectory with a short note — never left
  interleaved with current files under a different convention.
- Keep failed, refused, and superseded artifacts with their original status;
  do not delete them to tidy up.
- One canonical location per current artifact. Use routers and cross-references
  rather than copies.

### Amendment

A validation level, provenance field, or method-defining rule changes only by a
dated edit to this file that states: the old rule, the evidence that exposed its
limitation, the replacement, and which earlier results remain unaffected.
Ordinary prose fixes use normal version history.

---

## P4 — Path resolution

Environment-specific absolute paths (`/Volumes/X10 Pro/...`, `/Volumes/Movies/...`,
the HPC `/RAID1/.../FungalProject/Movies/...`) are the single biggest source of
"works on my machine" breakage in this repo. Recent history is full of
checkpoint-relocation and path-resolution fixes.

### The standard precedence

New code resolves every external location as:

```
environment variable  ->  config file value  ->  documented default
```

The reference implementation is
[`SingleCellQuantificationHPC/tracking_corrector/config.py`](../SingleCellQuantificationHPC/tracking_corrector/config.py)
(`local_movie_root`, `nas_movie_root`, `*_cache_root`). The ad-hoc checkpoint
search in `one_cell_quantification_1CH.py` (try SSD, then NAS, then repo-local
fallback) is the same idea for a read-only asset and is acceptable, but new
call sites should prefer the config precedence over inlining a candidate list.

### Standard names

| Location | Env var | Typical default |
|---|---|---|
| Workstation SSD movie root | `LOCAL_MOVIE_ROOT` | `/Volumes/X10 Pro/Movies` |
| NAS movie root | `NAS_MOVIE_ROOT` | `/Volumes/Movies` |
| AI checkpoints root | `FUNGAL_AI_ROOT` | `<movie root>/AI` then repo-local |
| Pipeline outputs root | `FUNGAL_OUTPUTS_ROOT` | `/Volumes/X10 Pro/FungalProject_Outputs` |
| HPC project root | `FUNGAL_HPC_ROOT` | `/RAID1/working/R402/hsushen/FungalProject` |
| Render/scratch cache | `TRACKING_CORRECTOR_CACHE_ROOT` | OS temp dir |

`LOCAL_MOVIE_ROOT`, `NAS_MOVIE_ROOT`, and `TRACKING_CORRECTOR_CACHE_ROOT` are
already read by `tracking_corrector/config.py`. The `FUNGAL_*` names are the
convention for new code; wire them into a component's config as that component
next needs them rather than in one sweep.

The current physical mount points per machine (workstation, NAS via SMB, HPC via
SSH) are documented in
[SingleCellQuantificationHPC/COWORKER_GUIDE.md](../SingleCellQuantificationHPC/COWORKER_GUIDE.md).
Keep that table current; it is the one place a new machine's paths are recorded.

### Rules

- **New scripts and new modules must not hard-code an absolute external path.**
  Use the precedence above, reading from a `config.yaml` where the component has
  one.
- **Experiment-dated scripts are exempt** as frozen records (P1). Do not rewrite
  their paths in place; if one must run on a new machine, copy it to a new dated
  file or drive it with an env override, and note which was done.
- A path default may point at a mount that does not exist on the current
  machine; resolution code must fail with a clear message naming the env var to
  set, not a bare `FileNotFoundError` deep in a loop.
- Do not commit a machine-specific `config.yaml`. Commit a
  `config.example.yaml` or keep machine values in env vars.

---

## P5 — Irreversible-action gate

Formalizes the discipline the `antigravity_prompts_*.md` staging files already
apply by hand: the risky, hard-to-undo step runs last and only against a
verified fix.

### Actions in scope

- Deleting or moving files, especially generated data, checkpoints, or strip
  directories.
- In-place regeneration that overwrites a canonical output (e.g. rerunning a
  `generate_M*_strips.py` over `STRIPS_DIR`, or a sweeper over its canonical
  results file).
- `rsync --delete`, and any sync whose target is authoritative storage
  (NAS, HPC RAID1).
- Relocating model checkpoints between SSD / NAS / repo.
- Bulk edits across many experiment files or QC JSONs.
- `git` history rewrites, force-pushes, branch deletion.

### Required before running one

1. **Verified precondition.** The fix or change the action depends on has passed
   its P2 validation level. "Do not run cleanup against an unverified fix."
2. **Backup.** The affected files are copied somewhere out of the blast radius,
   or the action is proven reversible (e.g. `git`-tracked and clean). State
   where the backup is.
3. **Explicit go-ahead** from the owner for that specific action, in this
   session. Prior approval of a related action does not carry over.
4. **Dry run first** where the tool supports it (`rsync -n`, list-what-would-move
   before moving).

### After running one

Record what was done, the file counts before/after, and the backup location —
in the commit message, a provenance record (P3), or the staging file. A move
that changes an artifact's identifier scheme also follows the `_superseded/`
rule in P3.

---

## P6 — Frame coordinate systems

Multi-film stitched cell timelines use four distinct frame-index spaces (local /
aligned / sequence / global-cell). Getting the conversions wrong has already
caused two separate septum save/load bugs. The full contract — the four spaces,
the API rules, and the historical bugs — is in
[COORDINATE_SYSTEMS.md](COORDINATE_SYSTEMS.md).

### Rules

- Any change to septum interval storage, alignment offsets, sequence linkage
  handling, or the frame math in `tracking_corrector` /
  `septum_alignment_board` / `ground_truth_corrector` must conform to
  `COORDINATE_SYSTEMS.md`, and must update it in the same change if the contract
  itself changes.
- A change to what any of these spaces *means* — the offset definition, the
  sequence-bounds numbering, the `offset: 0` "resolved" signal — is a **Level 5**
  decision (P2): owner sign-off required.
- `_sequence_film_bounds()` (server) and `getFilmSequenceBounds()` /
  `state.filmBoundaries` (client) must stay in the same numbering. A change to
  one requires the matching change to the other, verified at Level 3 against an
  existing linked cell.
- New code that combines endpoints from different films converts each to a
  sequence frame first (decode with that film's own offset), then combines.

---

## P7 — Staged handoff prompts

When a multi-step change is delegated to another agent, or is large enough that
a cold start would re-derive most of its context, write it as a sequence of
individually verifiable stages. The method, the shared-facts preamble rules, and
a fill-in template are in [STAGED_HANDOFF.md](STAGED_HANDOFF.md); the worked
examples are the `antigravity_prompts_*.md` files at repository root.

### Rules

- **One stage, one verifiable outcome.** Every stage ends with a concrete
  `Verify by ...` check named in the prompt itself — the P2 ladder applied per
  stage.
- **Stage 1 de-risks the load-bearing assumption** (data plumbing, a correctness
  fix, a schema) before any UI, cleanup, or optimization is built on it.
- **The irreversible stage is last and explicitly gated** on the earlier stages
  being verified. It follows P5: move don't delete, back up, print before/after
  counts.
- **The shared-facts preamble is dated and sourced** ("confirmed against the
  current code, dated YYYY-MM-DD") and uses exact identifiers — paths, function
  names, line ranges, concrete ids — not descriptions.
- **The preamble states what to do if a premise turns out false:** stop and
  report, do not proceed or silently repair.
- A handoff file is a working artifact. Keep it (it records what was verified and
  when); it is evidence, not policy (P1 hierarchy).

---

## P8 — Experiments ledger

Each imaging experiment (M92, M93, … M160, plus the pre-pipeline and
representation-learning datasets) has a row in
[EXPERIMENTS.md](EXPERIMENTS.md): date, movie folder, strain/condition, channels,
film-sequence structure, driving scripts, and deliverables. The canonical
machine-readable registry is
`SingleCellQuantificationHPC/tracking_corrector/config.yaml`; EXPERIMENTS.md is
the human-readable companion that also records what the config cannot (strain,
aim, which analyses were run).

### Rules

- **Adding an experiment folder to `tracking_corrector/config.yaml` means adding
  its row to `EXPERIMENTS.md`** in the same change.
- Repo-derived columns (date, channels, films, scripts) are facts; the
  strain/condition column is inferred from film naming until someone confirms it
  against the lab notebook; the **Aim** column is filled in by the owner.
- An experiment-dated script is a frozen record for its row (P1) — do not
  repurpose it; write a new dated script for a new experiment.

---

## P9 — Flask apps shared-core contract

The three review apps in `SingleCellQuantificationHPC/` read and write the same
on-disk dataset. `tracking_corrector` owns the shared dataset-access layer
(`config.py`, `repositories/`, the dataset-writing services, `qc_schema.py`).
The full contract — the owner, the on-disk write rules, the reference
integration pattern, and the `ground_truth_corrector` cleanup task — is in
[FLASK_APPS.md](FLASK_APPS.md).

### Rules

- **A new or refactored app reuses `tracking_corrector`'s layer** via the
  `septum_alignment_board` pattern (`sys.path` insert + `from tracking_corrector
  ... import`). It does not fork a repository or a dataset-writing service.
- **Every writer of a shared dataset file writes atomically** (temp file +
  `os.replace`) and **checks the revision it read** before overwriting.
- **One canonical writer per file class.** `sequence_linkage.json` and
  `cell_<id>_masks.csv` are written through `tracking_corrector`'s repositories.
- **Read-only reuse is unrestricted;** forking a writer is not.
- **Shared UI conventions are honored, not reimplemented per app.** A cell's
  display color is a deterministic function of its stable identity
  (`global_cell_id`, or `(film, local_cell_id)`), never of render order or a
  table-row index — so the same cell keeps its color across frames and across
  apps. Details in [FLASK_APPS.md](FLASK_APPS.md).
- Changing a shared repository or service: check every consumer and keep
  `tracking_corrector/tests/test_atomic_writes.py` and the septum tests green.
- `ground_truth_corrector` currently forks and has drifted — reconciling it is a
  P7 staged-handoff task, not a silent refactor.

---

## P10 — Training reproducibility

Every trained model kept in the project carries enough record to say what data,
code, and settings produced it, and is benchmarked before it becomes the one an
inference path loads. The full pre-flight / provenance / promotion checklist is
in [TRAINING_CHECKLIST.md](TRAINING_CHECKLIST.md).

### Rules

- **Every kept checkpoint has a `<checkpoint>.provenance.json` sidecar** (P3
  specialized): git commit, training-data identity (working dirs + manifest row
  counts + manifest hashes), generating script and verbatim command line, seed
  and `PYTHONHASHSEED`, hyperparameters, `pos_weight` actually used,
  augmentation summary, date, host, epoch saved / best epoch, val metrics.
- **Pin `PYTHONHASHSEED`** for every run of a given model and record it — the
  train/val split depends on it (see the checklist §5).
- **The per-run training log is kept with the checkpoint,** not left only in the
  shared append-mode `training.log`.
- **Promoting a checkpoint to production is Level 4–5** (P2): benchmark against
  curated ground truth, compare against the current production checkpoint on the
  same benchmark, and get owner sign-off for any change to augmentation, label
  semantics, decision threshold, or input normalization.
- **Checkpoint relocation between SSD / NAS / repo follows P5:** back up, keep
  the previous one until the new one is benchmarked, record the move.
- Superseded checkpoints stay with their provenance and their benchmark numbers.

---

## P11 — Sharing hygiene

The repo is shared with coworkers and pushed to GitHub, and the genomics
workflow authenticates to an external service. The rules for what leaves the
machine, the pre-push scrub, and the remediation list for what is already
committed (a JGI session cookie, a password-on-command-line) are in
[SHARING_HYGIENE.md](SHARING_HYGIENE.md).

### Rules

- **Never commit** session cookies, tokens, API keys, or passwords — not in a
  file, a script literal, or a command line. Read secrets from the environment
  or an interactive prompt.
- **A committed secret is compromised:** remove the file *and* rotate/revoke it
  at the provider. Deleting the file is not sufficient.
- **No machine-specific `config.yaml`** and **no personal email as a code
  literal** — use `config.example.yaml` and env vars (P4).
- **Run the pre-push scrub** (SHARING_HYGIENE.md) before sharing any branch that
  touches the genomics workflow, shell scripts, or config.
- Lab-internal names (HPC private IP, NAS host, `hsushen`) are low risk but
  should be centralized in one gitignored place rather than scattered.
- Purging a committed secret from git history rewrites shared branches — that is
  the owner's decision, not an automatic step.
