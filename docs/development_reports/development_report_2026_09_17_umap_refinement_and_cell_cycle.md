# Development Report: Explorer Refinement, Latent Sweep, and Cell-Cycle Stage

**Date**: September 17, 2026, with corrections made on September 18
**Experiment**: `2026_08_28_M160`
**Follows**: `development_report_2026_09_16_m160_lineage_and_division.md`
**Branch**: `mbdt-stage3`
**Status**: explorer refined and rebuilt; latent space moved from 3 to 6
dimensions; cell-cycle stage implemented, and then substantially rebuilt after
the project owner found two faults in it.

---

## 1. Executive Summary

Four explorer functions were added, the latent dimension was chosen by sweep
rather than inheritance, and a cell-cycle stage axis was built.

Three findings are worth more than the features themselves.

1. **The M156 oscillation thresholds were unreachable in M160.** Both
   oscillatory categories would have shown zero cells no matter what the user
   clicked.
2. **Fluorescence area is not cell size.** It measures the GFP-positive region,
   which *shrinks* 25% before division while the cell body grows 27%. The first
   cell-cycle attempt failed entirely because of this.
3. **The brightfield films are half the time course and were being ignored.**
   41% of divisions occur in them, and skipping them also left every stage
   estimate temporally incoherent.

---

## 2. Explorer: four functions

All four follow the M156 explorer's precedent, extracted from its HTML.

**Dynamic mode.** A categorical axis over five modes — non-polarized, monopolar,
monopolar oscillatory, bipolar, bipolar oscillatory — using the M156 classifier
and its discrete five-colour scale, with live thresholds shown only when that
axis is selected.

**Arrow keys** step through the datapoints of one global cell in film order,
ignored while a form control has focus. This required lifting the sidebar
builder out of the click handler into a reusable function.

**Autocorrelation card**, collapsed by default, showing the measured ACF against
the fitted double-exponential-times-cosine model and spelling out
`Periodicity = precision_sum − freq_distance_sum` with its arithmetic. Drawn
lazily on first open.

**Sticky trajectories card**, so the traces stay in view while the sidebar
scrolls.

### 2.1 NC score, and thresholds that did not transfer

The project owner corrected a sign error: `signal_cor` computes
`NC_score = -(A + C)` over the cross-correlation fit, so the score is **positive**
when the poles are negatively correlated. The oscillation gate needs `NC >= thr`,
not `<=`.

Checking that exposed a larger problem. The M156 thresholds were carried over
whole, and only two of four transfer:

| Threshold | M156 value | M160 range | Cells clearing it |
| :--- | ---: | :--- | ---: |
| pol1_mid | 4.04 | −1.77 to 17.4 | 16.2% |
| pol2_mid | 2.0 | −2.40 to 11.3 | 34.1% |
| Periodicity, mono | 5.0 | −0.35 to **4.24** | **0.00%** |
| Periodicity, bi | 6.5 | −0.35 to **4.24** | **0.00%** |

M160's Periodicity never reaches 5, so **both oscillatory categories were
unreachable**. Recalibrated to this dataset's own distribution — 1.14 is its
90th percentile, 1.60 roughly its 96th — the counts go from 0 and 0 to 9 and 72.
Turning the NC gate off gives 15 and 99, so NC rejects about a quarter of
otherwise-periodic cells as in-phase.

---

## 3. Latent dimension: chosen, not inherited

`FC_AE_dimension_sweep.py` sweeps a different architecture — one that takes the
trajectory alone and predicts the features — so a new sweep was written for
`MultimodalAutoencoder3D`, the model actually in use.

Nine dimensions, three repeats, early stopping, 61 minutes over 6,184
datapoints:

| dim | val total | traj | feat |
| ---: | ---: | ---: | ---: |
| 2 | 0.583 | 0.231 | 0.352 |
| 3 | 0.435 | 0.183 | 0.252 |
| 5 | 0.341 | 0.170 | 0.172 |
| **6** | **0.308** | 0.165 | 0.143 |
| 10 | 0.257 | 0.154 | 0.102 |
| 16 | 0.231 | 0.150 | 0.081 |

**Reading it needed care.** The raw standard deviation reaches 74% of the mean,
which would normally make the curve unusable. It is not noise: the three repeats
differ by validation split, and repeat 2 is about three times worse at *every*
dimension. Computing the elbow *within* each repeat removes that offset and is
consistent — 6, 5, 6.

The decomposition matters more than the headline. Marginal gain in the
trajectory term runs +20.8%, +4.0%, +3.5%, +3.0%, then goes negative. **The
trajectory term is flat past dimension 5**, while the feature term keeps falling
from 60% of the loss at dim 2 to 33% at dim 12. Extra dimensions buy feature
reconstruction, not dynamics.

**Moved to 6.** Loss fell from 0.2293 to 0.1231, trajectory 0.1479 to 0.1140,
features 0.0814 to 0.0091. It also gives UMAP a genuine 6→3 and 6→2 reduction
rather than the 3→3 near-identity it had been doing, so the map is now UMAP's
work rather than a re-rendering of the bottleneck.

The explorer reads the latent dimension out of the checkpoint, so it cannot
drift from the trainer.

---

## 4. Cell-cycle stage

### 4.1 First attempt: a negative result

The method of `plot_area_vs_stage_global.py` — anchor at division, regress area
on stage, slide unanchored cells onto the fit — was implemented for M160 and
**failed**. Pooled r was ~0 at every pre-division window from 20 to 300 minutes;
per cell, median within-cell r was 0.060 with only 16% of cells reaching 0.5.

The reason was scale: growth over 60 minutes is about 55 px against a
between-cell standard deviation of 672 px.

Checking the metadata was worthwhile on its own. M160's fluorescence films run
101 frames at 12 s, 20.0 min each, but start every **40.6 min** because a
brightfield block sits between them. Treating them as contiguous would have
compressed the stage axis twofold.

### 4.2 The brightfield films changed the result

On the project owner's prompt, the brightfield films were added — and the first
attempt had been measuring the wrong quantity entirely:

| | BF (cell body) | FL (GFP region) |
| :--- | ---: | ---: |
| Median area, −220 → −20 min | 4,184 → 5,330 px (**+27%**) | 4,162 → 3,140 px (**−25%**) |
| Median-trend r | **+0.775** | −0.119 |
| Cells with within-cell r > 0.5 | **57%** | 7% |

Fluorescence area tracks the GFP-positive region, which concentrates and shrinks
approaching division. Brightfield area is the cell outline, which grows. The
empirical BF:FL ratio is 1.200, matching the 1.25 the codebase already assumes.

### 4.3 Several quantities beat size alone

Grouped cross-validation over 648 curated datapoints from 97 cells:

| Features | r² | MAE |
| :--- | ---: | ---: |
| predict the mean | 0.000 | 90.2 min |
| brightfield area alone | +0.282 | 78.2 min |
| cell area alone | +0.420 | 66.9 min |
| cell length alone | +0.443 | 65.4 min |
| septum only | +0.175 | 82.2 min |
| nuclear only | +0.148 | 84.0 min |
| **all of them** | **+0.682** | **48.9 min** |

Septum and nuclear signals are weak alone but pull real weight in the ensemble.
**Cell length is the strongest single predictor**, as expected for a rod growing
by tip extension.

---

## 5. Two faults in that design, found by the project owner (Sept 18)

Both were exposed by one cell, `5_1_N1_F1_5_1_N1_BF4_F1_cell_418`.

### 5.1 Brightfield divisions were invisible

The detector read only fluorescence films, which cover 20 of every 40.6 minutes,
and worked one film at a time so a drop spanning a boundary was unseeable from
either side. That cell's area halves from 6868 to 3350 at **t = 162.0 min, the
final frame of BF4**, and it was reported as having no division at all.

`division_detect_m160.py` now assembles ONE area series per cell across every
film, both channels, on the real clock, and scans it with windows in minutes
rather than frames.

| | Before | After |
| :--- | ---: | ---: |
| Cells with a division | 97 | **237** |
| Divisions in brightfield films | 0, invisible | **98 (41%)** |

It locates the flagged cell at 161.5 min against a true 162.0.

### 5.2 Stages ignored the acquisition clock

Consecutive fluorescence films are 40.7 minutes apart, always. The per-datapoint
regression did not know that, and produced for that cell:

| Film | Old stage | Gap | New stage | Gap |
| :--- | ---: | ---: | ---: | ---: |
| FL1 | −335.9 | | −151.5 | |
| FL2 | −250.1 | 85.7 | −110.9 | 40.6 |
| FL3 | −238.5 | 11.6 | −70.3 | 40.6 |
| FL4 | −119.0 | 119.5 | −29.8 | 40.6 |
| FL5 | −18.1 | 100.9 | **+10.8** | 40.6 |
| FL6 | −19.4 | **−1.2** | +51.4 | 40.6 |
| FL7 | −18.9 | 0.4 | +92.0 | 40.6 |

Three films 40 minutes apart all placed at −19 min, and every film called
pre-division when the cell had divided before FL5.

**The redesign**: a cell has ONE division time, and
`stage = film midpoint − t_division`. Consecutive films then differ by exactly
40.6 minutes by construction. Where the division is measured the cell is
*curated*; otherwise each film votes on the cell's division time through a ridge
and the median vote is taken.

Held-out cells: division-time MAE 27 min median, r² +0.519, **39% within one
film and 68% within two**. Curated datapoints rose from 648 to 1,547.

A third, smaller bug was fixed alongside: division film attribution searched all
39 films by start time and so picked an arbitrary field, since the three fields
are imaged seconds apart.

---

## 6. Artifacts

Under `FungalProject_Outputs/model_based_dense_tracking/2026_08_28_M160/`:

| Path | Contents |
| :--- | :--- |
| `umap_m160_standalone.html` | explorer, 26 MB, 13 colour axes |
| `fc_ae_3d_m160.pth` | autoencoder, latent_dim 6 |
| `latent_sweep/` | sweep results, summary, plot |
| `cell_cycle/division_times_m160.csv` | 237 measured divisions |
| `cell_cycle/cell_cycle_stage_by_datapoint.csv` | 6,244 stages, 1,547 curated |
| `cell_cycle/*.png` | size regression, prediction quality |

New modules: `sweep_fc_ae_m160.py`, `cell_cycle_stage_m160.py`,
`cell_cycle_regress_m160.py`, `division_detect_m160.py`,
`cell_cycle_stage_build.py`.

---

## 7. Limitations

1. **696 cells show no division** in their 4-hour window, so their stage rests
   entirely on the regression. For a cycle of this length that is expected, but
   it is most of the cohort.
2. **The estimated stage is coarse.** 27 minutes median error, and only 39% of
   cells within one film.
3. **Detector thresholds are a judgement call.** A drop to ≤75% that does not
   recover above 80%; loosening them would find more divisions at the cost of
   admitting missegmentations.
4. **The mode thresholds are defaults, not findings.** They were set from M160's
   own percentiles because the M156 values were unreachable, not from a
   biological criterion.
5. **The ACF card's measured curve uses a linear detrend**, while the fitted
   curve uses the canonical AIC-selected trend. A visible gap between them means
   the two disagreed for that cell.
6. **Laser exposure is uncontrolled.** Later fluorescence films have received
   more cumulative exposure than earlier ones, and nothing in the current
   analysis separates that from biology. A control restricted to FL1 is the
   next task.
