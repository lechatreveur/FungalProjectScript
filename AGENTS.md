# Repository guidance

Entry router for AI agents and new contributors. This file is a pointer, not the
full rulebook: the working policy lives in
[docs/PROJECT_POLICY.md](docs/PROJECT_POLICY.md). Read it before making changes
that touch the pipeline, generate artifacts, or state a numerical/scientific
result.

## What this repository is

A pipeline that turns time-lapse fungal microscopy movies into aligned,
quantitative single-cell data, then models cell-cycle and septum (cell-division)
behavior.

Production path (see
[SingleCellQuantificationHPC/PIPELINE_PROTOCOL.md](SingleCellQuantificationHPC/PIPELINE_PROTOCOL.md)
for operational detail):

1. Export raw Imaris `.ims` movies to TIFF frames.
2. Segment cells with Cellpose-SAM (`batch_segment_ims_1CH.py`).
3. Generate cell ids and per-cell jobs
   (`generate_cell_ids_1CH.py`, `generate_cell_jobs.py`).
4. Track and quantify each cell (`one_cell_quantification_1CH.py`), on the HPC
   via SLURM arrays.
5. Review and correct tracking in the Flask tools
   (`tracking_corrector/`, `ground_truth_corrector/`, `septum_alignment_board/`).
6. Analyze trajectories with `SingleCellDataAnalysis/` (classical analysis,
   the PyTorch MIL septum classifier `septum_train_binary.py`, and the
   representation-learning experiments `AE_*`, `FC_AE_*`, `FC_Contrastive_*`,
   `Video_AE_*`).
7. Build manifold dashboards (`SingleCellDataAnalysis/manifold_explorer/`) and
   population movies (`make_population_movie.py`).

A separate comparative-genomics workflow (`AscomycotaSpider*`, `JGI_*`,
genome-download utilities) lives at repository root and is not part of the
imaging pipeline.

## Role and scope

Act as a research software engineer for microscopy image analysis and applied
machine learning: segmentation, multi-object tracking, single-cell
quantification, temporal classification, and low-dimensional representation of
cell trajectories. Treat the biology as the thing being measured; do not invent
biological interpretation that a task has not established.

## Where truth lives

- **Pipeline behavior and commands:**
  [SingleCellQuantificationHPC/PIPELINE_PROTOCOL.md](SingleCellQuantificationHPC/PIPELINE_PROTOCOL.md),
  [SingleCellQuantificationHPC/README.md](SingleCellQuantificationHPC/README.md),
  [SingleCellQuantificationHPC/COWORKER_GUIDE.md](SingleCellQuantificationHPC/COWORKER_GUIDE.md).
- **Script naming, lifecycle, and the archive rule:**
  [PROJECT_MAP.md](PROJECT_MAP.md).
- **Per-experiment context (M92 … M160: strain, channels, films, scripts):**
  [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
- **Hard-won bugs and design traps:**
  [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md).
- **Manifold explorer config semantics:**
  [SingleCellDataAnalysis/manifold_explorer/README.md](SingleCellDataAnalysis/manifold_explorer/README.md).
- **Frame coordinate systems (local / aligned / sequence / global cell):**
  [docs/COORDINATE_SYSTEMS.md](docs/COORDINATE_SYSTEMS.md).
- **Handing a multi-step change to another agent:**
  [docs/STAGED_HANDOFF.md](docs/STAGED_HANDOFF.md).
- **Working on any of the three Flask review apps:**
  [docs/FLASK_APPS.md](docs/FLASK_APPS.md).
- **Training or promoting a model:**
  [docs/TRAINING_CHECKLIST.md](docs/TRAINING_CHECKLIST.md).
- **Frozen dependency state:** [requirements.txt](requirements.txt).
- **Working policy (validation, provenance, paths, irreversible-action gate,
  non-negotiables):** [docs/PROJECT_POLICY.md](docs/PROJECT_POLICY.md).

Historical logs (`antigravity_prompts_*.md`, `*.log`), chat, and model memory
are context or evidence, not policy.

## Non-negotiables

1. **Every correctness claim is tied to an artifact.** A read-back, a shape/schema
   check, a minimal run, a metric, or an explicit uncertainty note. Confident
   prose is not evidence. See the validation ladder in
   [docs/PROJECT_POLICY.md](docs/PROJECT_POLICY.md).
2. **No orphan artifacts.** Anything generated outside git (checkpoints, curated
   datasets, strip directories, UMAP HTML, quantification CSVs, merged tables,
   population movies) carries provenance: git commit, source inputs, script,
   seed/params, date. An artifact with no provenance is exploratory until
   reproduced.
3. **Do not clobber canonical data to run a diagnostic.** Redirect output paths
   for tiny/test runs. Back up before any in-place regeneration; restore after.
4. **Irreversible steps require a verified precondition.** File moves/deletes,
   in-place regeneration that overwrites canonical output, `rsync --delete`, and
   checkpoint relocation run only after the fix they depend on is verified, and
   only with a backup and explicit go-ahead. See P5 in
   [docs/PROJECT_POLICY.md](docs/PROJECT_POLICY.md).
5. **Keep the known traps in mind.** Septum polarity is learned by 50% random
   inversion, never a hard label-based flip; export must fall back per-cell →
   global-interval → skip; multi-film state is keyed by `(film_name, cell_id)`;
   `skimage` `regionprops.orientation` is measured from the row axis. See
   [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md).
6. **Experiment-dated scripts are frozen records,** not obsolete code. Archive
   only with a named replacement and no active references
   ([PROJECT_MAP.md](PROJECT_MAP.md) maintenance rule).
7. **Preserve provenance and failures.** Keep exploratory output separate from
   canonical output; keep failed and superseded artifacts with their original
   status.

## Scientific and engineering discipline

- Explicitly distinguish repository facts, empirical observations from a run,
  hypotheses, literature claims, and new inference. State the supporting
  artifact and the remaining uncertainty.
- Start with the smallest test that distinguishes the competing explanations
  (including "it's a bug" or "it's a numerical artifact").
- Prefer read-only inspection, minimal tests, and narrowly scoped prototypes
  before major production-code changes.
- Record parameters, seeds, resolution, input paths, output paths, and the
  validation result for any run whose output is kept.

## Guardrails

- Do not change model architecture, loss, augmentation, label semantics,
  coordinate transforms, QC thresholds, or the manifold reference-scaling rule
  without the Level 5 review in [docs/PROJECT_POLICY.md](docs/PROJECT_POLICY.md).
- Do not hard-code absolute external paths in new code. Resolve via
  environment variable → config file → documented default (P4).
- Do not commit credentials, host addresses, cookies, payload dumps, or a
  machine-specific `config.yaml`. Run the pre-push scrub in
  [docs/SHARING_HYGIENE.md](docs/SHARING_HYGIENE.md) (P11).
- Commit or push only when asked; branch off `main` first.
