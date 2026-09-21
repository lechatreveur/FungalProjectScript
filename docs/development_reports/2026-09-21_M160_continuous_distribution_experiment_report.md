# M160: the polarity manifold is continuous, and it is not the laser

**Date:** 2026-09-21
**Experiment:** 2026_08_28_M160 (`5_1_N1`, FL1-FL7 + BF1-BF6, fields F0-F2)
**Controls used:** Sept17 (2025_09_17), M160 FL1-only
**Artifacts:** `/Volumes/X10 Pro/FungalProject_Outputs/umap_control/`
**Related:** P16 in `docs/PROJECT_POLICY.md`;
`2026-09-21_umap_clustering_is_not_measurable_with_silhouette_hopkins.md`

---

## 1. Question

M156/Sept17 work had left the expectation that single-cell polarity dynamics
fall into discrete modes - non-polar, monopolar, bipolar, and their oscillatory
variants - and that a UMAP of the learned latent space would show them as
separated clusters. M160's map did not. The question was whether that is a
biological difference, a pipeline defect, or an artifact of how the map is
made and read.

## 2. What was built

Stages 1-4 are covered by earlier reports. This report covers stage 5 onward.

| Stage | Module | Output |
| --- | --- | --- |
| 5 features | `build_features_m160.py` | 11 engineered features + 101x2 trajectory per datapoint |
| 5 lineage | `lineage_m160.py` | 992 segments, 68 binary forks |
| 5 division | `division_detect_m160.py` | 237/933 cells with a division; 41% detected in BF |
| 5 cell cycle | `cell_cycle_stage_build.py` | stage estimate, held-out MAE 27 min, r2 +0.519 |
| 6 representation | `train_fc_ae_m160.py` | `fc_ae_3d_m160.pth`, 6D latent (elbow sweep) |
| 6 representation | `train_fc_ae_m160.py --film-contains FL1_` | `fc_ae_3d_m160_FL1.pth`, 3D latent, unseeded |
| 6 manifold | `build_umap_html_m160.py` | interactive explorer |
| 6 validation | `SingleCellDataAnalysis/umap_validation/` | the four scripts below |

**Datapoint definition (P12-compliant):** one cell in ONE film over exactly 101
frames, keyed `<global_cell_id>__<film>`. Full cohort n=6243; FL1 subset n=861.
One datapoint is dropped by the loader
(`M160_5_1_N1_F0_cell_165_5_1_N1_FL7_F0`, 51 frames instead of 101).

## 3. Results

### 3.1 The map is continuous, and n_neighbors cannot change that

Full cohort, n=6243, 6D latents, everything fixed but the neighbour count:

| n_neighbors | % of n | best k | silhouette |
| ---: | ---: | ---: | ---: |
| 5 | 0.08% | 2 | +0.447 |
| 15 (library default) | 0.24% | 2 | +0.458 |
| 248 (fraction-matched) | 3.97% | 2 | +0.481 |
| 1000 | 16.02% | 2 | +0.482 |

Eight settings spanning 0.08% to 16% of the population give the same crescent
with the same hook and tail. The parameter changes texture, not topology.

### 3.2 The metrics do not distinguish M160 from noise

Identical pipeline on structureless nulls:

| dataset | M160 @248 (6D) | FL1 @34 (3D) |
| --- | ---: | ---: |
| isotropic Gaussian, no structure | +0.379 | +0.433 |
| each latent dim independently permuted | +0.435 | **+0.480** |
| real data | +0.481 | **+0.452** |
| 3 genuinely separated Gaussian blobs | +0.926 | +0.869 |

The silhouette floor on a UMAP embedding is **+0.38-0.43, not 0** - k-means
bisects any elongated cloud and silhouette rewards it. Genuine clusters score
~+0.9. Hopkins is unusable here entirely (noise scores 0.78-0.80 rather than
the textbook 0.5, and the shuffled null outscored the real data).

**Sept17's +0.514, M160's +0.481 and M160-subsampled-to-n=378's +0.502 all sit
in the noise band.** The remembered Sept17 clustering is not supported by
measurement. There was never a Sept17 result for M160 to fail to reproduce.

### 3.3 FL1, the laser control, is the weakest of all

FL1 is the first fluorescent film at each position, so the least-exposed cells,
trained as an independent 3D autoencoder. It **ties its shuffled null at
n_neighbors=15 (+0.470 vs +0.469) and loses to it at the fraction-matched 34
(+0.452 vs +0.480)**. Destroying every joint relationship between features
while preserving their distributions produces a map that scores as well or
better than the real data.

At n=861 the map *does* visually fragment into islands at n_neighbors 5-15
(best k jumps to 8) while silhouette stays flat at ~+0.47. The picture and the
metric are independent failure modes; neither validates the other. This is now
P16 clause 3.

### 3.4 Manual trajectory review (user, 2026-09-21)

Trajectories were inspected directly against their positions on the M160 UMAP.
**No discrete dynamic modes were found.** The trajectory morphology varies
smoothly across the map and agrees with the continuous distribution the
measurements indicate. This is the decisive evidence: it is an independent,
non-statistical confirmation that does not depend on any of the caveats above.

### 3.5 The within-experiment laser trend is real but is not the explanation

Across the 858 cells present in all seven FL films, FL1 -> FL7:

| quantity | FL1 | FL7 | |
| --- | ---: | ---: | --- |
| pol1_mid | 5.40 | 1.71 | p < 1e-19 |
| pole distance | 1.63 | 0.31 | p < 1e-19 |
| Periodicity | 0.66 | 0.28 | p < 1e-19 |
| pole/cytoplasm ratio | 1.046 | 1.013 | genuine contrast loss, not pure bleaching |

Tracking quality is flat across the series, so this is not a segmentation
artifact. Polarity signal genuinely degrades over the acquisition.

**But this cannot be what removed the discrete modes, because the modes are
already absent in FL1**, which has the least exposure of any film and still
fails to beat its own shuffled null. Illumination damages the signal
progressively; it did not create the continuous distribution.

## 4. Interpretation

The M160 population is polarity-continuous rather than mode-discrete. Given
3.5, the most probable reading is that the cells were already stressed at the
start of imaging - by some factor of the microscopy session or the sample
preparation preceding it - and that this stress collapsed the discrete
dynamical repertoire into a continuum before the first fluorescent frame.

Candidate factors, none yet excluded: temperature at the stage, medium
composition or exhaustion, agar pad preparation and drying, culture age or
density at mounting, time between mounting and first acquisition, focus drift
and the BF interleave, strain background.

## 5. Next step: compare M160 FL1 against M161 and M162 FL1

### 5.1 M161 does not exist

Searched the SSD, the home tree, and `docs/EXPERIMENTS.md`. **There is no M161
anywhere accessible.** The ledger goes M160 (2026-08-28) -> M162 (2026-09-09).
If M161 was acquired it is on the NAS or a machine not mounted here; if it was
never acquired, the comparison is M160 vs M162 only. **This needs your
confirmation before planning around it.**

### 5.2 M162 is an excellent illumination-matched control

From the acquisition metadata:

| | M160 FL | M162 FL1 |
| --- | --- | --- |
| date | 2026-08-28 14:44 | 2026-09-09 10:21 |
| frames x interval | 101 x 12 s | 101 x 12 s |
| duration | 20.0 min | 20.2 min |
| exposure ch1 / ch2 | 400 / 100 ms | 350 / 120 ms |
| laser line | Laser 2 on | Laser 2 on |
| **laser intensity** | **5** | **5** |
| binning | 1x1 | 1x1 |

Same line, same intensity, same cadence, exposure within 15%. **If M162 FL1
shows discrete modes under this pipeline, illumination dose is definitively
excluded and the cause lies in strain, medium, or handling.** If M162 FL1 is
also continuous, then either the condition is shared or the discrete-mode model
itself needs revisiting.

**Important limitation:** M162 differs from M160 in strain *and* medium
(`NeonG_YES` vs `5_1_N1`) and in date. It is therefore a "does any other
dataset show discrete modes" test, **not** a controlled single-factor
experiment. A positive result narrows the field; it does not identify the
factor. Isolating one factor requires a purpose-designed acquisition (6.3).

### 5.3 M162 pipeline state

Stage 3 (model-based dense tracking) is **complete**: 4 FL films x 4 fields +
3 BF films x 4 fields, with per-film provenance JSONs and `dense_masks`.

FL1 tracked rows: F0 198, F1 514, F2 238, F3 296 = **1,246**, comfortably above
M160 FL1's n=861, so the comparison will not be sample-size limited.

**Missing:** stage 4 quantification, stage 4b strips, stage 5 features, stage 6
autoencoder. No `features/` directory, no `.pth`.

## 6. What to prepare

### 6.1 Blocking questions (need your answer)

1. **Does M161 exist?** If yes, where - NAS path or HPC path? If no, confirm
   the comparison is M160 vs M162.
2. **M162 keyframe curation status.** `m162_abbt_results.csv` exists, but P14
   stage 1 requires curated keyframes before quantification is trustworthy.
   Has M162 been through curation, or does it need a review pass?
3. **Which factor do you most suspect?** It determines what metadata is worth
   extracting alongside (stage temperature, mount-to-acquisition delay, culture
   OD) and whether a purpose-designed M163 is warranted.

### 6.2 Work that can start immediately on M162

In P14 stage order, reusing the M160 modules with an M162 id map:

1. **Stage 4 quantification** - `quantify_model_based_dense.py --films FL1...`
   FL1 alone is enough for the first comparison, which keeps it cheap. Expect
   HPC time comparable to M160's 2.5-3.5 h/task.
2. **Stage 4b strips** - `build_strips_m160.py` generalized to M162. Remember
   the contrast rule: rescale to each film's [C1min, C1max] before writing,
   never pass raw camera counts.
3. **Stage 5 features** - `build_features_m160.py` against M162 paths. Must use
   `PCA_utils.load_experiment_features` for the full 11 features, not
   `clustering.cluster_cells_by_amplitude_and_delay` (9, weight-normalised).
4. **Stage 6 autoencoder** - `train_fc_ae_m160.py --film-contains FL1_`, 3D
   latent unseeded, to match the M160 FL1 model exactly.
5. **Stage 6 validation** - `umap_validation/nn_sweep_fl1.py` parameterised for
   M162. **P16 requires the shuffled-dimension null at M162's own n and latent
   dimension**; the M160 floors do not transfer.

### 6.3 Code changes required

- Generalize the four `*_m160.py` modules to take an experiment root rather
  than the hard-coded M160 path. They are currently M160-specific by
  construction (P15 copy-to-modify), so this is a real refactor, not a flag.
- `umap_validation/nn_sweep_fl1.py` hard-codes the M160 base and the `FL1_`
  filter; parameterise both.
- M162 uses 4 fields (F0-F3) against M160's 3 (F0-F2), and film names
  `NeonG_YES_*` against `5_1_N1_*`. Any regex or field enumeration assuming
  M160's layout will need widening.
- An M162 id map is needed. P12 forbids `new_cell_id` row numbers; follow the
  M160 id map convention, not M156's.

### 6.4 The analysis that actually answers the question

Comparing two continuous blobs by silhouette will not work - section 3.2 shows
the metric cannot resolve differences at this effect size. Two better designs:

1. **Conditioned test, within each experiment.** Do the Mode categories (or
   cell cycle stage) occupy distinct regions of latent space? The null is label
   permutation with the geometry held fixed. This returns a signal even when
   the unconditioned "are there clusters?" cannot, and it is the right
   instrument for "does M162 have discrete modes?".
2. **Direct distributional comparison, between experiments.** Compare M160 FL1
   and M162 FL1 on the 11 engineered features directly - no autoencoder, no
   UMAP, no clustering. Per-feature distributions with effect sizes. If M162
   cells are less stressed, `pol1_mid`, pole distance and Periodicity should
   all sit higher, exactly as they do at M160's FL1 relative to its FL7. This
   is cheap, interpretable, and does not depend on any manifold assumption.

Design 2 should be run **first**: it needs only stages 4-5 on M162 FL1, skips
the autoencoder entirely, and would likely settle the stress question on its
own.

## 7. Reproduction

```
python3 SingleCellDataAnalysis/umap_validation/cluster_compare.py
python3 SingleCellDataAnalysis/umap_validation/subsample_matched_n.py
python3 SingleCellDataAnalysis/umap_validation/nn_sweep.py
python3 SingleCellDataAnalysis/umap_validation/null_calibration.py
python3 SingleCellDataAnalysis/umap_validation/nn_sweep_fl1.py
```

All write to `/Volumes/X10 Pro/FungalProject_Outputs/umap_control/`. Run from
the repo root with the SSD mounted.
