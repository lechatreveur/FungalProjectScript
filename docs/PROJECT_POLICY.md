# Project policy

Version 1. Effective 2026-09-02. Version 1.1 (2026-09-02) adds P4–P6.
Version 1.2 (2026-09-02) adds P7. Version 1.3 (2026-09-02) adds P10.
Version 1.4 (2026-09-02) adds P9. Version 1.5 (2026-09-02) adds P11.
Version 1.6 (2026-09-02) adds P8. Version 1.7 (2026-09-08) adds P12.
Version 1.8 (2026-09-09) adds P13. Version 1.9 (2026-09-10) adds P14.
Version 1.10 (2026-09-11) adds P15 and revises P14 stage 3 for model-based
dense tracking.

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
- **P12 — Cell color coding and segmentation overlay policy** (below).
- **P13 — Pole completeness for polarity-site quantification** (below).
- **P14 — Pipeline stage order and per-stage objective priority** (below).
- **P15 — Canonical modules: which script to use, and the copy-to-modify rule**
  (below).

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
  the reference experiment only, mutants projected with `.transform()`;
  pole-clipped masks (a centroid-correct mask that truncates a pole silently
  drops the pixels polarity-site quantification depends on — see P13).
- **Hygiene.** No credentials, host addresses, cookies, or payload dumps in
  commits. Environment-specific absolute paths belong in config or a resolver,
  not hard-coded into new scripts.
- **Chat Math & Formula Notation (Unicode).** The IDE chat panel does not parse
  raw LaTeX delimiters. All formulas, variables, and Greek symbols in conversational
  responses must be rendered using clean Unicode characters (e.g., ΔI, θ, μ, σ,
  Ī_S, Ī_C, u_long, u_short, Δp_long, √(Δx² + Δy²), ≤, ≥, ≈, ×, →) rather than
  raw LaTeX `$ ... $` or `$$ ... $$` delimiters.

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
`SingleCellQuantificationHPC/tracker_comparison_summary.md`), **and pole-span
agreement vs. the phase-local expected span for any track feeding polarity /
pole-intensity analysis (P13) — centroid-continuous is necessary, not
sufficient.** Septum classification: label-count parity with the GUI and
per-interval agreement on a held-out film. Division timing: fit residuals
against manually marked events. Store the full per-item result (per cell, per
frame), not just the aggregate.

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
- **Hardware resource limits & MPS memory caps (Local Workstations):**
  - PyTorch MPS unified memory on Apple Silicon does not enforce a safe ceiling by
    default and can balloon into tens of gigabytes of virtual swap (>28 GB), causing
    system freeze or Out-Of-Memory (OOM) failure.
  - Any local MPS training pipeline (`train_cellposesam.py`, representation-learning
    scripts) must enforce an explicit memory cap via
    `torch.mps.set_per_process_memory_fraction(fraction)` calibrated against Metal's
    `torch.mps.recommended_max_memory()`.
  - For an 18 GB workstation, cap MPS at ~10.0 GiB (~75% of device working limit),
    leaving at least 6–8 GB of physical RAM free for the OS and IDE, while
    comfortably fitting ViT multi-head attention peaks (~6.8 GiB).
  - Do not set uncalibrated `PYTORCH_MPS_HIGH_WATERMARK_RATIO` below peak attention
    requirements, which causes artificial `MPS backend out of memory` errors.
  - Enforce bounded working set churn: tile size `bsize=256`, `batch_size=1`, and
    `--nimg_per_epoch` sub-sampling (e.g. 25 images/epoch) rather than loading
    unbounded image lists into memory simultaneously.
- **Resume lineage & in-place immutability:**
  - Resumed runs must record `resumed_from` in the `<checkpoint>.provenance.json`
    sidecar pointing to the parent checkpoint.
  - Never overwrite the parent checkpoint in-place. Resumed artifacts must be saved
    with a distinct target name (e.g., `<name>_resumed`).
  - Output path discipline: avoid framework-level directory nesting (e.g., Cellpose
    automatically appends `models/` to `save_path`). Ensure checkpoints mirror
    cleanly to `models/` and `~/.cellpose/models/`.
- **Superseded checkpoints stay with their provenance and their benchmark numbers.**

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

---

## P12 — Cell color coding and segmentation overlay policy

Every visualization across all review tools (`tracking_corrector`, `ground_truth_corrector`, `septum_alignment_board`, manifold explorer dashboards, and population movies) must adhere to strict deterministic color identity, overlay provenance, and unassigned segment rendering rules.

### Rules

1. **Deterministic Color Identity Mapping**:
   - A cell's color is a strictly deterministic function of its **stable identity**:
     - For multi-film linked sequences: `global_cell_id` (e.g. `"5_1_N1_F0_cell_50"`).
     - For single-film local views: `local_cell_id` (integer).
   - In sequence mode, **only cells linked to an active `global_cell_id` in `sequence_linkage.json` receive a color hue and label badge**. Any local cell in `TrackedCells_<film>` not mapped to a global cell must **never fall back to its local cell ID**; it must be treated as unassigned and rendered as white (see Rule 4).
   - Never key color on transient UI indices (render order, pagination index, table row ID, or `new_cell_id` row numbers).
   - String global IDs are hashed to 32-bit unsigned integers using **FNV-1a 32-bit**:
     `h = 0x811C9DC5`, `h = ((h ^ byte) * 0x01000193) % 2^32`.
   - The integer key maps to hue using **Knuth's multiplicative hash**:
     `val = (key * 2654435761) % 2^32`, `h = (val % 360) / 360.0`, with fixed saturation `s = 0.8` and value `v = 0.95`.

2. **Frontend-Backend Exact RGB Parity (The BigInt Rule)**:
   - In JavaScript, multiplying 32-bit unsigned integers (such as `key * 2654435761`) exceeds standard `Number` safe integer precision without BigInt, causing floating-point rounding divergence from Python's 64-bit integer arithmetic.
   - All client-side JavaScript color functions (`static/js/color.js`) must calculate the hash using explicit `BigInt` operations:
     `Number((BigInt(id >>> 0) * 2654435761n) % 4294967296n)`.
   - Client-side outlines, timeline swatches, and cell selection buttons must match server-rendered mask overlays byte-for-byte in RGB.

3. **Individual Masks as Authoritative Ground Truth Overlays**:
   - Population frames and boundary overlays must always be rendered from the updated per-cell mask CSV files (`TrackedCells_<film>/cell_<cid>_masks.csv`), never directly from raw Cellpose masks (`*_seg.tif`).
   - Any manual or algorithmic corrections saved to `cell_<cid>_masks.csv` must immediately reflect in the population overlays.

4. **White Rendering for Untracked / Unassigned Segments**:
   - Any segment that is not part of an active tracked global cell is unassigned and must be rendered in pure **WHITE** (`(255, 255, 255)` fill and 1px white boundary outline) with **no text label**:
     - Local cells in `TrackedCells_<film>` that are unlinked / unmapped to a `global_cell_id` in sequence mode.
     - Residual Cellpose segmentation regions in `_seg.tif` that have no active tracked cell behind them.
   - White rendering prevents false color attribution and immediately alerts the curator to unlinked or extraneous segments.

---

## P13 — Pole completeness for polarity-site quantification

Polarity-site dynamics (pole cap intensity, pole-to-pole oscillation, new-end
take-off) are measured **at the two poles of the _S. pombe_ rod**. A per-frame
mask that is centroid-correct but **pole-clipped** silently drops the exact
pixels the measurement depends on: a truncated pole reads as "no polarity
signal" rather than "signal not captured." Mask quality for any pole / polarity
analysis is judged by **pole inclusion**, not by IoU or centroid continuity
alone.

Context: this is the recurring failure mode behind the M156 vertical-strip
rotation bug, the `recover_missegmented_poles.py` tooling, and the dense
retracking work — Cellpose under-segments a pole for 1–3 frames, or the tracker
picks a `_seg.tif` label that is itself clipped.

### Rules

- **Both physical poles must be present in every per-frame mask** used for
  polarity, pole-intensity, cell-length, or division-geometry quantification.
  Centroid-continuous and IoU-passing are necessary, not sufficient.
- **Pole completeness is an L4 check (P2).** Validate a retracked / re-segmented
  / pole-recovered mask series against the **phase-local expected major-axis
  span**, interpolated from the curated keyframes and good neighbour frames —
  **phase-aware**: never interpolate pre-division mother geometry into
  post-division daughter frames or vice versa (P6, and
  `interpolate_expected_geometry` in `recover_missegmented_poles.py`). A frame
  whose span is `< ~0.85 ×` the expected phase-local span is flagged
  **pole-short** and is excluded from polarity quantification until repaired.
- **Repair order** (`recover_missegmented_poles.py`): (1) adopt a fully
  overlapping `_seg.tif` label of the right length; (2) fuse an adjacent split
  `_seg.tif` fragment lying beyond the truncated tip along `u_long`; (3) extrude
  a minor-axis-width brush along `u_long` to the expected tip. A frame that none
  of these fixes is **marked, not silently shipped**.
- **Dense in-film retracking** (the all-frame plan, and any gap-fill against
  `_seg.tif`) runs the pole-completeness stage on every re-linked frame.
  Recovered masks carry a `composition` / provenance tag recording how the poles
  were obtained: `single`, `fuse`, `extrude`, `pole_short`, `interp`.
- **Downstream artifacts inherit pole clipping.** Vertical strips
  (`generate_M*_strips.py`), Video-AE crops, and manifold features are computed
  from the mask; a pole-short mask propagates a truncated crop into every
  representation built on it. Regenerate these after a pole-recovery pass.
- **The segmentation-model target is pole inclusion.** A Cellpose checkpoint
  trained or promoted under P10 is benchmarked on **pole-span agreement vs. the
  curated keyframes**, not mask IoU alone. A model that clips poles is not an
  improvement even at higher IoU.
- Superseded and pole-short masks keep their status (P3) — excluded from
  polarity analysis, not deleted.

Reference: `recover_missegmented_poles.py` (expected-geometry interpolation,
truncation detection, three-strategy recovery), `DEVELOPMENT_NOTES.md`
(pole-recovery and strip-rotation history), `COORDINATE_SYSTEMS.md` (phase and
frame spaces).

---

## P14 — Pipeline stage order and per-stage objective priority

The single-cell pipeline has a **fixed stage order**, and each stage has **one
primary objective**. Work that optimises a later stage's metric before its
prerequisite stage is settled is wasted, and work that optimises a stage's
*secondary* objective at the cost of its primary one is a regression even when
the secondary number improves.

### Canonical stage order

1. **GTC + ABBT — keyframe curation and quick QC.**
   Curate the keyframe backbone in the ground-truth corrector; run the Advanced
   Backward Bayesian Tracker over it to establish per-cell identity, inter-film
   linkage, sister assignment, and **division timing at keyframe resolution**.
   Output of this stage is authoritative for everything downstream.
2. **Segmentation — train Cellpose on the curated keyframes, then segment every
   timeframe.** Model training follows P10; promotion follows P10 §3 and is
   benchmarked on pole-span agreement (P13), not IoU alone.
3. **Model-based dense tracking — all-frame decision against an explicit shape
   model** built from the stage-1 keyframes, applied to the stage-2 `_seg.tif`
   series. Canonical module in P15.
4. **Quantification — polarity-site dynamics** on the dense mask series.

### Rules

- **Do not reorder the stages.** In particular, dense tracking runs *after*
  resegmentation, not before. A tracker can only choose among the labels
  segmentation hands it; it cannot recover a cell that was never segmented.
  Attempting to compensate for poor segmentation inside the tracker produces
  synthetic masks that are plausible-looking and wrong.
- **Keyframes are read-only from stage 2 onward.** They are the product of
  stage 1 and represent days of curation. No later stage rewrites a keyframe
  mask; a keyframe that appears wrong is escalated to GTC, not patched in place.
- **Stage 3 objective priority is fixed:**
  1. **Primary — pole inclusion, and therefore major-axis length stability.**
     Every per-frame mask must contain both physical poles (P13). The expected
     length is **inferred from the curated keyframes and the interval anchors**,
     never from the interior frames being repaired. This is the metric dense
     tracking is tuned, accepted, and reported on.
  2. **Secondary — dense division timeframe.** Division timing is *already
     established* at keyframe resolution by stage 1 (ABBT: 98.74% exact,
     99.37% within ±1 keyframe over 316 curated cells). Refining `t*` to the
     individual frame is a convenience for phase-splitting, not a deliverable.
- **A low-confidence dense `t*` defers to the stage-1 keyframe call.** When the
  dense localiser's confidence is low or its QC flags fire, use the keyframe
  division interval rather than emitting a shaky per-frame `t*`. Never trade
  pole completeness for a sharper `t*`.
- **Never derive expected geometry from unvalidated interior frames.** Frames
  whose length is a merge-scale outlier against the robust interval median —
  including *canonical* frames that were never flagged, because their RLE is
  neither empty nor duplicated — are excluded from the expected-geometry basis
  and from any write-back into it. Without this, one sister-fusion frame sets
  the expected length for its whole neighbourhood and the pole-recovery
  machinery faithfully extends every real cell to the fused scale.
- **The expected shape is a model, not an accumulation.** Stage 3 carries an
  explicit per-cell shape — two pole tips, a centre, a width and a bend angle —
  built from the two bracketing curated keyframes. Every per-frame mask is a
  decision against that shape, not the product of a chain of repairs.
- **Length is anchored; only the angle may be fitted.** Arm lengths and width
  derive solely from curated keyframes and from frames the tracker accepted as
  GOOD. No other frame writes back into the reference. The bend angle may be
  fitted per frame within a bounded window, because cells bend and twist through
  division and the keyframes cannot predict it.
- **Tip presence is not sufficient evidence.** A segment containing both expected
  tips is accepted only if its span is also within `1.25 ×` the expected span. A
  fused blob contains both of our tips *and* a whole neighbour, so the tip test
  alone silently passes fusions.
- **Every branch output is length-checked, in both directions.** Any branch that
  adds pixels is rejected if the result exceeds `1.25 ×` expected; any branch that
  removes pixels is rejected if the result falls below `0.75 ×` expected. The two
  failure modes of this family are the unbounded graft and the over-eager cut, and
  both have occurred.
- **Every interior frame is decided.** There is no pass-through category, so no
  frame can be emitted without having been inspected.
- **Report stage 3 as spans-in-band first**, over *all* emitted frames with
  nothing excluded, with the GOOD-branch share and the branch histogram alongside.
  Frames carrying no image evidence (`NO_SEG`, `BOTH_MISSED`) are counted and
  reported separately, because they are model output and must be excluded or
  flagged in stage 4 (P13). Division timing is reported as agreement with stage 1,
  not as an independent accuracy claim.
- **Stage boundaries are P2 checkpoints.** Promoting a stage's output to the
  input of the next is at least L4: benchmark against the curated set, compare
  to the current production artifact, and record the result (P3, P8).

Reference: `docs/development_report_2026_09_08.md` §6 (dense all-frame
strategy), `advanced_backward_bayesian_tracker.py` (stage 1),
`train_cellposesam.py` + [TRAINING_CHECKLIST.md](TRAINING_CHECKLIST.md)
(stage 2), P13 (stage 3 acceptance criterion).

---

## P15 — Canonical modules: which script to use, and the copy-to-modify rule

### Why this exists

Several scripts in this repository do superficially the same job, and the
superseded ones still run without error. Choosing the wrong one produces numbers
that look plausible and are **not comparable** with the rest of the pipeline.

This is not hypothetical. On 2026-09-11 the model-based dense masks were first
quantified with `ImageQuantification` from `Image_quantification_functions.py` —
the pre-1-channel routine — instead of `quantify_one_object`, which is what
`one_cell_quantification_1CH.py` actually calls. Every row came out populated and
sane-looking, while silently omitting the septum pattern recognition and the
two-half split that the production path emits for every frame.

**Before running or extending a stage, look up its canonical module here. Do not
infer it from a filename.**

### The registry

#### Stage 1 — keyframe curation and quick QC

| Module | Method |
| :--- | :--- |
| `SingleCellQuantificationHPC/ground_truth_corrector/` (**GTC**, package) | Flask app for human curation of the keyframe backbone. Serves keyframe images with per-cell mask overlays, takes mask and identity corrections, and writes the canonical keyframes plus `sequence_linkage.json` (per-sequence film list and the global-cell → per-film local-id table). Shared-core contract in P9; colour and overlay rules in P12. Authoritative for everything downstream. |
| `SingleCellQuantificationHPC/advanced_backward_bayesian_tracker.py` (**ABBT**) | Refined hard-EM backward Bayesian tracker over the curated keyframe backbone. Runs backward from the last keyframe with hypothesis-conditioned cleavage bisection, phase-aware geometry interpolation between keyframes, and sister-swap rectification. Establishes per-cell identity, inter-film linkage, sister assignment, and **division timing at keyframe resolution** (98.74% exact, 99.37% within ±1 keyframe over 316 curated cells). |

#### Stage 2 — segmentation

| Module | Method |
| :--- | :--- |
| `SingleCellQuantificationHPC/train_cellposesam.py` | Fine-tunes Cellpose-SAM on the curated keyframe image/mask pairs from stage 1. Reproducibility requirements and promotion gate in P10 and `TRAINING_CHECKLIST.md`; promotion is benchmarked on pole-span agreement (P13), not IoU alone. |
| `SingleCellQuantificationHPC/batch_segment_ims_1CH.py` | Applies the promoted model to every timeframe of a film, writing `Masks_<film>/<film>_t_###_c_0_seg.tif` as labelled images. One label per object; no identity across frames. |

#### Stage 3 — model-based dense tracking

| Module | Method |
| :--- | :--- |
| `SingleCellQuantificationHPC/model_based_dense_tracking.py` | Carries an explicit per-cell shape — pole tips `E1` and `E2`, centre `C`, stroke radius `r`, bend angle `theta` — built from the two bracketing curated keyframes. The expected mask (`E1CE2`) is the union of two round-capped strokes `E1->C` and `E2->C`, each started one radius in from its tip so the cap lands on the tip. Each interior frame is a four-way decision against that shape: keep the segment, cut it against `E1CE2`, union it with a stroke to the missing tip, or fall back to `E1CE2`. Arm lengths and radius come only from curated keyframes and from frames accepted as GOOD; only the angle is fitted per frame. Non-dividing intervals are solved by bisection with good-gated anchor promotion; dividing intervals by a two-pass good-segment changepoint with a confidence gate. |
| `SingleCellQuantificationHPC/run_model_based_dense_tracking.py` | Selects cells from the QC work queue by status, expands them through `sequence_linkage.json` to per-film keyframe intervals, runs the tracker, and writes one dense-mask CSV per (film, cell) plus a per-interval summary. Resumable; scratch-only output. |

#### Stage 4 — quantification

| Module | Method |
| :--- | :--- |
| `SingleCellQuantificationHPC/one_cell_quantification_1CH.py` | The per-cell entry point for 1-channel data. Resolves the channel (`--track_channel gfp` or `bf`), tracks or re-uses the mask series, and dispatches per frame to the routine below. Frame naming convention is `<film>_t_###_c_<channel>.tif` with no z index. |
| `quant_helpers.py::quantify_one_object` | **The GFP main entry.** Per frame it (1) quantifies the segment as one object and emits a row; (2) runs touching-circles septum pattern recognition, saving the pattern scores and centre; (3) splits the mask by the minor-axis line through the pattern centre, pastes both halves back to the full frame, and quantifies each, emitting one row per half. So a parent call emits **three rows per frame**: `<cell_id>`, `<cell_id>_1`, `<cell_id>_2`. A single `ep_refs` dict, with independent `single` / `1` / `2` entries, is carried across the whole series so the EM endpoint references persist and pole 1 and pole 2 cannot swap identity mid-track. |
| `bf_pattern.py::bf_pattern_only` | The brightfield path: septum pattern recognition without the GFP mixture fit. |
| `Cell_tracking_functions.py::rle_encode` / `rle_decode` | The mask serialisation used by every mask CSV in the pipeline. Use these, not an ad-hoc encoder. |
| `SingleCellQuantificationHPC/generate_cell_ids_1CH.py` + `generate_cell_jobs.py` | Build `cell_ids.txt` and the SLURM array scripts under `sb_scripts/`. Channel logic is in `PIPELINE_PROTOCOL.md` §4. |
| `SingleCellQuantificationHPC/merge_cell_data.py` | Merges per-cell `cell_<id>_data.csv` into the per-film table. |
| `SingleCellQuantificationHPC/pull_cells.sh` | Syncs quantification results back from the HPC. |

Film intensity scale for the GFP path is `FindMovieMaxMin`: pool every 10th pixel
of **every** frame of the film, then take the 99.5th and 1st percentiles. Using
the first frame alone ignores photobleaching and shifts the scale.

#### Superseded — do not use for 1-channel experiments

| Module | Superseded by | Why |
| :--- | :--- | :--- |
| `quantify_cell.py` | `one_cell_quantification_1CH.py` | Two-digit frame indices, `Masks_*/GFP_seg` and `brightfield_seg` subfolders, a hard-coded HPC `sys.path`, and its own overlap-based re-tracking. Wrong layout for 1-channel data. |
| `Image_quantification_functions.py::ImageQuantification` | `quant_helpers.py::quantify_one_object` | Emits one row per frame with no septum pattern and no two-half split. |
| `scratch/dense_abbt_interval.py` | `model_based_dense_tracking.py` | Heuristic stage 3; retained as an evidence artifact for the 2026-09-11 dense-ABBT report, not as a runnable path. |

A module listed as superseded stays in the tree as evidence (P3) and is not
deleted. It is also not run.

### The copy-to-modify rule

Every module in the registry is a **reliable module**: downstream results and
published numbers depend on it behaving exactly as it does today.

- **Do not modify a registry module in place to explore, debug, or try an idea.**
  This includes GTC, ABBT, the model-based dense tracker, and the stage-4
  quantification modules.
- **Work on a copy.** Copy the module into `scratch/` under a name that says what
  it is and when it was taken, for example
  `scratch/quantify_one_object_20260911_septum_test.py`. Modify and run the copy.
- **Record the provenance** of the copy: the source path, the commit it was taken
  at, and what is being tried (P3).
- **Promotion back into the registry is a P2 L4 checkpoint** — benchmark against
  the curated set, compare against the current production artifact, record the
  result — and is an irreversible action under P5, so it needs a backup and an
  explicit go-ahead.
- **New capability goes in a new module**, not as a flag bolted onto a registry
  module. The model-based dense tracker was added this way rather than by editing
  `dense_abbt_interval.py`.
- Experiment-dated scripts (`IAonNAS_<date>_*.py`, `quantify_M*.py`,
  `generate_M*_strips.py`, `submit_array_M*.sh`) remain frozen records under P1
  and are never modernised in place.

### Changing this registry

Adding, renaming, or superseding an entry is a policy change: update this table
in the same commit that introduces the module, and say in the commit message
which entry moved and why.


