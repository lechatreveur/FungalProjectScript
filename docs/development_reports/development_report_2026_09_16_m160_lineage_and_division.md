# Development Report: M160 Full Cohort, Lineage, and Division Anchoring

**Date**: September 16, 2026
**Experiment**: `2026_08_28_M160`
**Follows**: `development_report_2026_09_15_m160_hpc_cohort_run.md`
**Branch**: `mbdt-stage3`
**Status**: full cohort complete through stage 6. Lineage resolver built and
validated. Division anchoring analysed; two findings change the plan.

---

## 1. Executive Summary

The HPC array finished and the full cohort is through the pipeline: **6,244
datapoints over 933 global cells**, with a manifold built on autoencoder latents
and link lines showing 918 cells moving across it.

Three things were found along the way that matter more than the throughput.

1. **Division is encoded implicitly in the linkage**, by a mother and her
   daughters sharing a local cell id before the split. The `lineage` key exists
   but is empty. Treating the sharing as duplication collapsed lineage and cost
   18 cells all of their films.
2. **A fork is not a division film.** Only 5% of fork films contain the area
   drop, against 73% of `t_div` films. A fork tells you a division happened, not
   when.
3. **Missegmentation separates from true division by the bounce**, not by the
   depth of the drop, and not by `nu_dis` or septum, both of which collapse in
   either case.

---

## 2. Full Cohort Completion

| Stage | Result |
| :--- | ---: |
| Requantification with keyframes (array 2460385) | 603 cells, 90 min, no errors |
| Retrieved from cluster | 1.7 GB |
| Gaps filled locally | 31 of 34 recovered |
| Dense mask tables | 6,244 |
| Quantification tables | 6,244 |
| Strips (16×56 tiles) | 6,244 |
| Feature datapoints | 6,244, eleven features, no nulls |
| Autoencoder | 6,243 datapoints, 300 epochs, loss 0.894 → 0.237 |
| Explorer | 18 MB, strips linked not embedded |

**The cohort fix mattered.** Against the earlier `sus_pos` selection:

| | sus_pos run | Full cohort |
| :--- | ---: | ---: |
| Datapoints | 605 | 6,244 |
| Global cells | 328 | 933 |
| Cells in all 7 films | 1 | 857 |
| Mean stage-3 GOOD | 73.7% | **94.5%** |
| Median model-only | 1.02% | **0.00%** |

The old cohort was the difficult tail of the data, not a sample of it.

### 2.1 The 34 missing pairs

The cluster's `TrackedCells` folders lacked those cells' mask CSVs, though the
SSD had them. 31 recovered locally in five minutes. Three remain: `FL7_F0`
cells 420 and 421 and `FL7_F1` cell 519, all `KF_NO_MASK` — a curated keyframe
with no mask. Stage-1 escalations, not tracker faults.

### 2.2 Two bugs fixed

**Array tasks clobbered the stage-3 summary.** All 21 wrote to one path, so 20
were lost. No measurement was affected — per-frame provenance lives in the mask
tables and carries into the quant tables — but the interval diagnostics went.
Film-scoped runs now write `model_based_dense_summary_<film>.csv`.

**The tracker could not deploy.** It imported its RLE codec from
`ground_truth_corrector.schemas`, which pulls pydantic and, through that
package's `__init__`, Flask. Neither is on the cluster. It now uses
`Cell_tracking_functions`, which P15 already names as canonical; verified
byte-identical on all 101 frames of a real cell in both directions.

---

## 3. Lineage: mothers that fork into daughters

### 3.1 What was wrong

`build_tasks` deduplicated on `(film, local_cell_id)` so a physical cell is not
tracked twice. Correct. But the datapoint was then attributed to whichever
`global_cell_id` was seen first, and the other silently lost that film.

| FL films per selected cell | Before | After |
| :--- | ---: | ---: |
| 7 films | 860 | **950** |
| 1–6 films | 72 | 0 |
| **0 films** | **18** | 0 |

Eighteen cells got nothing at all.

### 3.2 Why it happens

`sequence_linkage.json` carries a `lineage` key that is **empty**. Division is
encoded implicitly instead: a mother and her daughters share the same local id
in every film before the split. Classifying every contested pair:

| Kind | Pairs | Meaning |
| :--- | ---: | :--- |
| Shared prefix then diverge | 73 | a division |
| Identical throughout | 22 | a true duplicate entry |
| Diverge then re-converge | 14 | a linkage fault |

### 3.3 The resolver

`SingleCellQuantificationHPC/lineage_m160.py` builds a trie over each cell's
local ids across the fluorescence films. A branch is a division: the path to the
first branch is the mother, each branch below it a daughter, and a daughter that
branches again is a mother further down.

| | |
| :--- | ---: |
| Segments | 992 |
| Founders (depth 0) | 856 |
| Daughters (depth 1) | 136 |
| Forks | 68, **all binary** |
| Segments spanning all 7 films | 788 |
| Points, each in exactly one segment | 6,204 |

Every fork being binary is consistent with cell division and nothing in the data
contradicts it. Re-converging tracks are excluded (42 points) rather than
modelled, because a cell does not un-divide and leaving them in puts the shared
tail in two segments at once.

---

## 4. Division anchoring

### 4.1 The two signals barely overlap

| | Cells |
| :--- | ---: |
| `t_div` from stage 3's scan | 82 |
| Linkage fork | 136 |
| **Both** | **18** |
| Neither | 733 |

`t_div` concentrates in the early films (FL1, FL2, FL4, FL5); forks concentrate
late (FL6, FL7 carry half). Frame position of `t_div` within its film: min 7,
quartiles 25 / 48 / 75, max 100.

`t_div` coverage is low because stage 3 only ran its division scan where the
seed had flagged the cell, and the seed only ever saw `good` and `corrected`
cells.

### 4.2 A fork is not the division film

Testing whether a film contains a detectable sustained area drop:

| Positive source | Candidate drop found |
| :--- | ---: |
| `t_div` film | **60 / 82 (73%)** |
| Fork film | **6 / 126 (5%)** |

The fork marks where the linker first gave the daughters different ids, which
can be well after the physical division. **Forks confirm that a division
happened; they do not date it.** Using a fork film as a cycle anchor would place
the anchor in the wrong film most of the time.

### 4.3 nu_dis and septum do not predict; they confirm

Median value as a fraction of each cell's own baseline 40 to 20 frames earlier,
over 61 curated cells with a frame-level `t_div`:

| Δ frames | nu_dis | septum | area |
| ---: | ---: | ---: | ---: |
| −20 | 1.01 | 1.03 | 1.01 |
| −10 | 1.04 | 1.06 | 1.01 |
| −2 | 1.06 | 1.10 | 1.01 |
| −1 | 1.04 | 0.97 | 1.01 |
| **0** | **0.00** | **0.24** | **0.61** |
| +10 | 0.00 | 0.19 | 0.57 |

Flat within a few percent for twenty frames, then a cliff. Raw: nu_dis 25–28 px
through the mother phase collapsing to near zero; septum-minus-cytoplasm 3.6 to
0.4; area 4,310 to 2,400.

`nu_dis` is effectively binary — two separating nuclei against one central one —
and is the sharpest single indicator of mother-state versus daughter-state. But
nothing here gives warning of an upcoming division within 60 frames.

**Caveat**: `t_div` is defined as the frame where switching from mother to
daughter parameters maximises GOOD frames, so the object being measured changes
at exactly that frame. The step is partly guaranteed by construction. Breaking
the circularity needs an independent anchor.

### 4.4 Separating true division from missegmentation

Framing: a cell divides in at most one film, so for a cell with a known dividing
film the other six are negatives. 208 positives against 919 negatives.

A candidate drop event is enriched about tenfold in dividing films (32% against
3%), so its presence is already informative. Among films that have one:

| Statistic | Dividing | Other |
| :--- | ---: | ---: |
| Area ratio | 0.53 | 0.59 |
| nu_dis ratio | 0.00 | 0.00 |
| Septum ratio | 0.02 | 0.11 |
| **Bounce** | **0.62** | **0.93** |

**The depth of the drop does not discriminate. The bounce does.** A
missegmentation recovers; a division does not.

| Rule | Precision | Recall |
| :--- | ---: | ---: |
| bounce ≤ 0.70 | **0.89** | 0.62 |
| bounce ≤ 0.85 | 0.83 | 0.73 |
| area ≤ 0.70 (the old criterion) | 0.77 | 0.97 |
| area ≤ 0.75 and nu ≤ 0.35 | 0.74 | 0.98 |

Precision here is a lower bound: some "false positives" are films that contain a
real division we had no label for.

Adding `nu_dis` and septum to the rule does **not** help, because both collapse
in either class — a missegmentation also loses part of the object.

---

## 5. Cell cycle: the old method, reviewed and set aside

`Video_AE_extract_cycle_stage.py` averages five quantities over a cell's 101
frames and assigns a stage backwards with a length gate. It has four problems:

1. **It uses no division information at all.** Cells with and without a division
   are estimated identically.
2. **It is a scalar per cell**, averaging pre- and post-division frames together.
3. **Its thresholds are half absolute, half cohort-relative** (pattern ≥ 0.90 and
   nu_dis ≥ 18.0 against septum ≥ 80th percentile and length ≥ median), so
   changing the cohort changes cells' stages. The cohort just grew tenfold.
4. **Anything below median length is forced to Stage 1** by construction.

It also never covered M160; its scores are for the 419-cell video cache.

Superseded. The project owner's direction is to anchor on the division film
instead, at film resolution, where a 20-minute error is acceptable.

### 5.1 The size-regression precedent

`SingleCellDataAnalysis/plot_area_vs_stage_global.py` already implements the
right skeleton: anchor each cell at its division time, build `stage_min = frame
time − division time`, regress area on stage over pre-division points per
channel, then place unanchored cells by inverting the fit on their mean area,
`tau = mean_T − (mean_A − c) / m`. Area comes from the decoded mask, and BF and
GFP are fitted separately because BF segmentations run ~1.25× larger.

For M160 three things change: the anchor becomes `t_div` (not the septum
endpoint, and not the fork), M160's seven FL films need their start times and
resolutions, and with few anchored cells the fit's r value carries more weight.

---

## 6. Artifacts

Under `/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking/2026_08_28_M160/`:

| Path | Contents |
| :--- | :--- |
| `dense_masks/` | 6,244 stage-3 tables |
| `quant/` | 6,244 stage-4 tables, 101 frames each |
| `strips/` | 6,244 PNGs at 16×56 tiles |
| `features/umap_features_m160.csv` | 6,244 × eleven features |
| `fc_ae_3d_m160.pth` | the M160 autoencoder |
| `umap_m160_standalone.html` | the explorer, 18 MB |
| `hpc_jobs/` | array, post job, README |
| `_superseded_sus_pos_run_20260914/` | the biased run, kept with a note (P3) |

New modules: `lineage_m160.py`, `generate_model_based_jobs_m160.py`,
`train_fc_ae_m160.py`, `build_strips_m160.py`, `build_features_m160.py`,
`build_umap_html_m160.py`.

---

## 7. Open items

1. **Wire the lineage resolver into stages 5 and 6** so features are keyed by
   segment and the explorer draws the fork. No re-tracking needed; the mask and
   quant tables are keyed by `(film, local_cell_id)` and stay valid.
2. **Extend the division detector to unreviewed cells.** The bounce criterion
   needs no curation, so it applies to all 933. At precision 0.89 this would
   give a film-level anchor for many of the 733 cells that currently have none.
3. **Exclude the division film from autoencoder training** — the project
   owner's instruction, not yet implemented.
4. **Three cells blocked on `KF_NO_MASK`** need a curated keyframe mask.
5. **Seventeen duplicate `global_cell_id` groups and four re-converging pairs**
   need resolution in the linkage, not downstream. Duplicates also carry
   conflicting QC statuses — one track is `good` under one entry and `bad` under
   another — so a cell's status currently depends on which entry is read.
6. **The ABBT tracker never runs on unreviewed cells.** Nothing in the model
   requires curation; only the benchmark loop filters. Its training labels come
   from the area-drop heuristic now known to be weak, so retraining on the bounce
   criterion would matter more than widening inference.
