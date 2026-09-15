# Development Report: M160 Feature Extraction on Model-Based Dense Tracking

**Date**: September 14, 2026
**Module**: `SingleCellQuantificationHPC/build_features_m160.py`
**Dataset**: `2026_08_28_M160`, sequences `5_1_N1_F0` / `F1` / `F2`
**Stage**: 5 (feature extraction), consuming stage-4 output from
`quantify_model_based_dense.py`
**Status**: complete. 328 global cells, 59,241 frames, eleven features per cell.
One author error found after the first build and corrected in §5.
**Policy**: identity per P12; output location per P4; canonical modules per P15;
this report filed per P3.

---

## 1. Executive Summary

M160 is the first experiment tracked with the model-based dense tracker, so this
is the first feature table built on masks that are decided against a shape model
rather than taken from segmentation alone. 605 stage-4 quantification tables
reduce to **328 global cells**, because a global cell is one biological cell
followed across consecutive films and several tables merge into one series.

Two defects were found and fixed before the full run. One of them, a time-base
collision, would have silently corrupted a third of the input rows and is the
more important result in this report.

Intensities from model-inferred segments are **included** in the fits, on the
project owner's instruction, with the per-cell share recorded so any downstream
result can be filtered or weighted by it.

---

## 2. The Chain

Canonical modules only (P15). Nothing in `SingleCellDataAnalysis/` was modified.

| Step | Module | What it does |
| :--- | :--- | :--- |
| 1 | this script | Keep primary-object rows; the two split rows per frame are dropped |
| 2 | this script | Cytoplasm-correct: `pol{1,2}_int_corr = pol{1,2}_int − cyt_int` |
| 3 | this script | Place each film on a sequence-continuous time axis (§4) |
| 4 | `signal_analysis.quantify_all_cells` | Per cell and pole, select trend vs trend+oscillation by AIC; emit parameters and phase offsets |
| 5 | `signal_cor.quantify_all_cells_acor` | Detrended autocorrelation per cell |
| 6 | `PCA_utils.load_experiment_features` | Assemble the feature row, re-ordering the poles so `pol1` is the brighter one |

Feature row, eleven columns: `pol1_a`, `pol1_mid`, `pol1_v`, `pol2_a`,
`pol2_mid`, `pol2_v`, `NC_score`, `Periodicity`, `a1a2` (= a1·a2),
`d` (= |mid1 − mid2|), `dd` (= a1 − a2).

- `a` is the trend slope, `mid = a·50 + b` the fitted value at the midpoint.
- `v` is the **residual variance of a straight-line fit** to the corrected
  trace: how far the signal departs from a linear trend, carrying the
  oscillation and noise energy that `a` and `mid` do not.
- `Periodicity` = precision_sum − freq_distance_sum.

`load_experiment_features` reads its three inputs from
`<dir>/unaligned_pairs_quant/`, and for the stacked file it looks **only**
there, with no fallback. This build writes that layout rather than modifying the
module (P15).

---

## 3. Identity — global_cell_id, not row numbers

P12 is explicit: *"Never key colour on transient UI indices (render order,
pagination index, table row ID, or `new_cell_id` row numbers)."*

The M156 build (`build_umap_from_retrack_data.py`) keyed its feature rows on
`new_cell_id`, an integer assigned by row position in an id-map CSV. That is the
construct P12 names. This build keys on **`global_cell_id`** from
`sequence_linkage.json` — for example `5_1_N1_F0_cell_238` — and carries `films`
and `local_cids` as attributes rather than as identity.

Nothing is written into the M156 tables. Output is experiment-scoped under M160.

---

## 4. Defect 1 — the time-base collision (found and fixed)

### Observation

The first run produced 19 feature rows for 12 cells, and the stacked input had
**392 of 1,176 (cell, time_point) pairs carrying more than one row**.

### Mechanism

A sequence's films are **consecutive acquisition blocks**, each restarting at
`t = 0`. `5_1_N1_F0` has 13 films, seven of them fluorescence. A global cell
seen in `FL1_F0` and `FL4_F0` therefore has two separate stretches of its life,
both labelled t = 1..99.

Keying the stacked table on `global_cell_id` without a per-film offset lays
those stretches on top of each other. `quantify_all_cells` sorts by time point
and fits, so it would have fitted a curve through two interleaved series,
treating a gap of three films as if it were the same instant.

The M156 build never hit this because its `new_cell_id` was assigned per (film,
cell), so films could not merge — the identity that violates P12 happened to be
protective here. Using the correct identity exposes the problem.

### Fix

Each fluorescence film takes its ordinal within its sequence, and
`time_point = ordinal × 101 + local_time`, with the original kept as
`local_time`. Duplicate (cell, time) pairs after the fix: **0**.

This is asserted in the script on every run and printed, rather than left to be
rediscovered.

---

## 5. Defect 2 — the wrong assembly module (author error, corrected)

The first version of this build produced **nine** features and recorded the two
missing ones as a pre-existing gap in the codebase. That was wrong, and the
claim is retracted here rather than quietly dropped (P1, P3).

**What happened.** The chain was traced from the wrong end. Searching for a
function that builds a row resembling the feature row finds
`clustering.cluster_cells_by_amplitude_and_delay`, and that is what the first
version called. It is a clustering routine: it emits only `a` and `mid` per
pole, its velocity terms are commented out, and it returns its *weight-
normalised* frame, with standard deviations pinned to the clustering weights of
3 and 1 rather than to the data.

The correct module is the one `FC_AE_data_loader` actually imports,
`PCA_utils.load_experiment_features`. It emits all eleven columns, including
both `v` terms, as raw values, leaving scaling to the loader.

**This is exactly the failure P15 exists to prevent** — choosing a module by
what it looks like rather than by what the consumer calls — committed in the
same session P15 was written. The stage-5 entry in the P15 registry has been
corrected to name `load_experiment_features`.

Nothing upstream of the assembly step changed: the fits and autocorrelation
inputs are identical, so only the final table was rebuilt.

---

## 6. Model-Only Frames

M160's tracker emits a mask for every interior frame. Where the segmentation
gave nothing usable — branch `NO_SEG` or `BOTH_MISSED` — the mask is the shape
model itself, and the intensity is measured from the best inferred segment.

Per the project owner's instruction these frames are **included in the fits**,
and each cell carries `n_model_only`, `model_only_pct` and `stage3_good_pct`.

Overall: **2,093 of 59,241 frames (3.53%)**. Per cell:

| Share of frames model-only | Cells |
| :--- | ---: |
| 0% | 121 |
| 0 to 1% | 27 |
| 1 to 5% | 126 |
| 5 to 10% | 36 |
| 10 to 25% | 12 |
| 25 to 50% | 5 |
| Above 50% | 1 |

Mean 2.81%, median 1.02%. The distribution is strongly skewed: three quarters of
cells are below 5%, and the tail is short.

The six worst cells, which are the ones to treat with suspicion in any result:

| Cell | Films | Frames | Model-only | Stage-3 GOOD |
| :--- | ---: | ---: | ---: | ---: |
| `5_1_N1_F0_cell_238` | 1 | 98 | 69.4% | 15.3% |
| `5_1_N1_F0_5_1_N1_FL7_F0_cell_202` | 2 | 196 | 43.9% | 46.4% |
| `5_1_N1_F0_5_1_N1_FL7_F0_cell_278` | 4 | 392 | 32.9% | 37.8% |
| `5_1_N1_F0_5_1_N1_BF2_F0_cell_138` | 3 | 294 | 32.0% | 44.6% |
| `5_1_N1_F1_cell_340` | 3 | 294 | 32.0% | 52.4% |
| `5_1_N1_F1_5_1_N1_FL7_F1_cell_253` | 5 | 490 | 26.5% | 32.4% |

Model-only share and stage-3 GOOD share move together, as they should: a cell
whose segmentation was poor produced both more inferred masks and fewer frames
where the segmentation was taken unchanged.

---

## 7. Cohort

| | |
| :--- | ---: |
| Stage-4 tables in | 605 |
| Global cells out | 328 |
| Frames | 59,241 |
| Unmapped cells skipped | 0 |
| Feature rows with a null | 0 |
| Mean stage-3 GOOD share | 73.7% |

Films per cell: 176 cells in one film, 88 in two, 27 in three, 22 in four, 7 in
five, 7 in six, 1 in seven. Frames per cell: median 98, range 49 to 686.

**The written feature values are raw**, not normalised. `load_experiment_features`
returns the fitted quantities directly and the downstream loader applies its own
scaling. Eleven columns, 328 rows, no nulls.

---

## 8. Artifacts

All under
`/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking/2026_08_28_M160/features/`
(P4 — generated data never on the system disk):

| File | Contents |
| :--- | :--- |
| `umap_features_m160.csv` | 328 cells × eleven features, plus provenance columns |
| `cell_provenance_m160.csv` | Per global cell: films, local ids, frame and model-only counts |
| `cell_provenance_per_film_m160.csv` | The same before merging films |
| `unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv` | Cytoplasm-corrected series on the continuous time axis |
| `unaligned_pairs_quant/model_fits_by_cell.csv` | Trend and oscillation parameters |
| `unaligned_pairs_quant/acor_detrended_results.csv` | Autocorrelation results |
| `_provenance.json` | Run record (P3) |
| `../features_build.log` | Run log |

---

## 8.1 Three corrections after review (2026-09-15)

The build described above was wrong in three ways, all found by comparing
against the Sept17 reference rather than by reasoning from the code as written.
Recorded here rather than dropped (P1, P3).

**What a datapoint is.** One cell in **one film** over exactly 101 frames, keyed
`<global_cell_id>__<film>`, mirroring the reference's experiment + global cell +
source. The earlier build merged each global cell's consecutive films into one
49-to-686 frame series. The reference trajectory loader requires exactly 101 and
hard-skips anything else, so every M160 cell would have been dropped.

**Keyframes were never quantified.** Stage 3 emits interior frames only, so each
film trace was 98 frames running t = 1..99 with 0, 50 and 100 absent. Stage 4
now also quantifies the three curated keyframes from the canonical masks, tagged
`KEYFRAME`. After requantification, 604 of 605 traces are exactly 101 frames;
the exception is `FL7_F0` cell 421, whose stage-3 run failed on a missing
keyframe mask and which is correctly dropped.

**Strip contrast.** `ImageQuantification` rescales each crop to the **film's**
intensity range before handing it to `build_strip_tile`, using the 1st and
99.5th percentiles. The first strip build passed raw camera counts, leaving the
background offset in for the per-strip normalisation to absorb and flattening
the signal. Strips are now one per datapoint, so a strip and the trajectory
beside it cover the same frames.

Cohort after these fixes: **604 datapoints over 327 global cells**, 152 of them
spanning two or more films.

---

## 9. Stage 6 — the standalone M160 map

`SingleCellQuantificationHPC/build_umap_html_m160.py` →
`umap_m160_standalone.html` (183 MB), 328 cells.

**This is a standalone map and is stated as such, in the report and in a banner
on the page itself.** The scaler and the UMAP are fit on M160's own cells, so
the coordinates are M160's own and are not comparable with the reference
manifold or with the M156 maps, which are each standalone in the same way. P1
requires the reference fit plus `.transform()` for cross-experiment work; that
was not done here because a standalone map is what was asked for.

**UMAP is fit on autoencoder latents, not on the features.** This was the third
error in the first build and the most consequential: the autoencoder folds the
101-frame Pol1/Pol2 trajectory together with the eleven features into one
vector, and fitting UMAP on the features alone discards the trajectory shape
entirely, which is the thing the model exists to encode.

A model was trained on M160 alone (`train_fc_ae_m160.py`, 300 epochs, CPU, seed
42), reusing `MultimodalAutoencoder3D` and `load_feature_constrained_data`
unmodified. Loss fell from 1.99 to 0.276, split 0.181 trajectory and 0.094
feature. Training on M160 alone is what makes the manifold standalone;
projecting through the Sept17 model would have put M160 on the reference
manifold instead.

UMAP then runs on those latents with `random_state=42, n_jobs=1`, at three and
two components.

### 9.3 Link lines — following a cell across the manifold

Because a datapoint is one cell in one film, a global cell followed across
consecutive films appears as several points, and the point of the map is to see
where it moves between them. The link function follows the M156 explorer:

- Datapoints are grouped by `global_cell_id` and joined in **film order**.
- Each link is split into six sub-segments with a **graded opacity ramp**, so
  direction of travel is readable rather than just connectivity.
- Two styles: dull grey `rgba(160,174,192, 0.10→0.25)` at width 1.2 for every
  cell, and bold sky blue `rgba(2,132,199, 0.50→1.00)` at width 4.8 in 3D and
  3.8 in 2D for the selected cell.
- Lines carry `hoverinfo:'none'` and stay out of the legend, so they never
  interfere with picking points, and sit behind the markers.
- A checkbox toggles them, and the sidebar lists the selected cell's path film
  by film.

152 of the 327 global cells span two or more films and therefore draw a link.

Manual colour range inputs were also added, which the M156 page has and the
first build omitted.

**Format follows the Sept17 reference**, `SingleCellDataAnalysis/FC_AE_3d_umap.py`:
light theme (`#f4f6f8` page, white panels, `#1e293b` toolbar), 3D/2D dimension
toggle, "Color by" dropdown, Viridis, and a sidebar of cards holding the cell
statistics, the intensity profile with Pol1 in `#ef4444` and Pol2 in `#3b82f6`,
and the vertical strip.

### 9.2 Colour range — why the first build was unreadable

The reference sets `cmin`/`cmax` from the raw minimum and maximum of the colour
array. On a skewed axis a single outlier compresses every other cell into one
end of the scale. The reference already concedes this for one axis, with a
hand-tuned piecewise `remap_for_display` for the cycle score.

This build takes the general form: robust limits from the 2nd and 98th
percentile per axis, with values beyond simply clamping. The difference is not
marginal:

| Colour axis | Raw range | 2nd–98th percentile |
| :--- | :--- | :--- |
| Pol1 mid intensity | −6.57 to 49.9 | −2.96 to 9.88 |
| Pole distance | 0.0006 to 50.1 | 0.021 to 21.2 |
| Pol1 variability | 0.074 to 29.8 | 0.094 to 11.0 |
| Model-only % | 0 to 69.4 | 0 to 20.1 |

On Pol1 mid intensity the usable gradient was spanning about a fifth of the bar,
so most cells rendered as the same dark purple.

The polarity panel was also widened from 240 to 300 pixels tall and given a
dotted zero line at the cytoplasm level plus a dashed marker at the division
frame, so the Pol1 against Pol2 relationship is legible rather than two thin
traces in a small box.

**Why a new module.** `build_umap_html_m156_*.py` does not build an explorer. It
fits a UMAP and then re-embeds into an existing explorer HTML, lifting each
cell's trajectories, autocorrelation arrays, fit parameters and strips out of it.
M160 had no such template, so the cell objects are constructed here from the
stage-4 and stage-5 artifacts directly.

The page carries a scatter with eight selectable colour axes, including
`model_only_pct` and `stage3_good_pct` so tracking quality can be inspected
against position on the map, plus per-cell metadata, the cytoplasm-corrected
Pol1 and Pol2 traces on the sequence-continuous axis, and the vertical strip.

### 9.1 Strips were missing, and that was an omission

Vertical strips are a **stage-4** artifact: `quantify_one_object` appends one
tile per frame to the `strip_tiles` list it is handed, and `--make_strips`
writes the PNG. `quantify_model_based_dense.py` did not pass that list, so M160
had none of the 3,816 strips on disk.

They did not need requantifying. A strip is fully determined by the frame image
and the mask — `build_strip_tile` derives its own rotation from
`regionprops.orientation` and touches none of the EM or pole fitting — which is
the reasoning `build_strips_only.py` was already written on.
`build_strips_m160.py` does the same for dense masks: 328 strips in 48 minutes,
no errors, one per global cell with tiles concatenated across films in sequence
order. The rule is now in P15.

---

## 10. Limitations and Next Steps

1. **The autoencoder also needs trajectories, not just features.**
   `load_feature_constrained_data` pairs this table with 101-frame trajectories
   from `AE_data_loader.load_and_preprocess_trajectories` and keeps only cells
   present in both. Our cells run 49 to 686 frames on a sequence-continuous
   axis, not 101, so that pairing has not been attempted and is the next thing
   to resolve before any autoencoder run.
2. **Model-only frames are in the fits.** That is the instruction and it is
   recorded, but an oscillation amplitude fitted partly over inferred masks is
   not the same measurement as one fitted over segmented cells. The sensible
   check is to rebuild with those frames dropped and compare the feature table,
   which the script supports by filtering the stacked file.
3. **The continuous time axis assumes films are contiguous** with no gap between
   blocks. If the acquisition has real dead time between films, the offset needs
   the true interval rather than 101 frames.
4. **One cell is 69% model-only.** It should probably be excluded outright
   rather than carried with a flag.
5. **Comparability with M156 is not established.** The two feature tables were
   built from different trackers, different identities and different time bases.
   They should not be pooled without a deliberate check.
