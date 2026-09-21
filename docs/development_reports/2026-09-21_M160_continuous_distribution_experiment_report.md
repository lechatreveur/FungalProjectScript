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

## 5. Next step: compare M160, M161 and M162 FL1

### 5.1 M161 is on the NAS

`/Volumes/Movies/2026_09_03_M161`, acquired 2026-09-03, condition
`NeonG_YES_1`. 7.8 GB, 60 `.ims` files, FL1-FL6 + BF1-BF5 across 4 fields
(F0-F3). **No processing exists anywhere** - it has not been imported to the
SSD and has no segmentation, tracking or quantification.

(An earlier draft of this report stated M161 did not exist. That search covered
the SSD and the home tree but not the NAS mount at `/Volumes/Movies`. The NAS
also holds M157, M158, M159, M163, M164, M165, M166 and M167, none of which are
in `docs/EXPERIMENTS.md`.)

### 5.2 M161 and M162 are the same condition, which makes this a real design

| | M160 | M161 | M162 |
| --- | --- | --- | --- |
| date | 2026-08-28 | 2026-09-03 | 2026-09-09 |
| condition | `5_1_N1` | `NeonG_YES_1` | `NeonG_YES` |
| FL films | FL1-FL7 | FL1-FL6 | FL1-FL4 |
| BF films | BF1-BF6 | BF1-BF5 | BF1-BF3 |
| fields | F0-F2 (3) | F0-F3 (4) | F0-F3 (4) |
| FL cadence | 101 x 12 s | 101 x 12 s | 101 x 12 s |
| exposure ch1/ch2 | 400 / 100 ms | **350 / 120 ms** | **350 / 120 ms** |
| laser line | Laser 2 on | Laser 2 on | Laser 2 on |
| laser intensity | **5** | **5** | **5** |
| binning | 1x1 | 1x1 | 1x1 |

M161 and M162 share strain, medium **and** acquisition settings exactly, and
differ only in session and day. M160 differs in strain/medium and marginally in
exposure, at the same laser intensity.

So M161 is not a redundant third dataset - it is the **replicate control** that
the M160-vs-M162 comparison was missing. The three outcomes are now separable:

| outcome | reading |
| --- | --- |
| M161 ~ M162, both differ from M160 | strain/medium (or the minor exposure difference) |
| **M161 differs from M162** | **session-level: handling, temperature, mounting, culture age** |
| all three continuous | the stressor is common to the whole setup, or the discrete-mode model itself needs revisiting |

The middle row is the one that directly tests section 4's hypothesis, and it is
only testable because two sessions share a condition. Without M161 a difference
between M160 and M162 could never have been attributed to anything.

Note that a M161-vs-M162 difference is the *informative* result here even
though it is the noisier-looking one: it would localise the stressor to
something that varies between sessions, which is exactly the class of factor
listed in section 4.

### 5.3 Pipeline state

| | M160 | M161 | M162 |
| --- | --- | --- | --- |
| stage 1 keyframes | done | **none** | `m162_abbt_results.csv` exists, curation status unconfirmed |
| stage 2 segmentation | done | **none** | done |
| stage 3 dense tracking | done | **none** | **done** (4 FL x 4 fields + 3 BF x 4 fields) |
| stage 4 quantification | done | none | none |
| stage 5 features | done | none | none |
| stage 6 autoencoder | done | none | none |
| on SSD? | yes | **no, NAS only** | yes |

FL1 cohort sizes: M160 861 datapoints; M162 1,246 tracked rows (F0 198, F1 514,
F2 238, F3 296); M161 unknown until tracked, but 4 fields at M162's density
suggests a comparable figure. None of the three will be sample-size limited.

**M161 is the long pole.** It needs stages 1-5 from raw, where M162 needs only
4-5. Budget accordingly.

## 6. What to prepare

### 6.1 Questions (no longer blocking, but they shape the plan)

1. **M162 keyframe curation status.** `m162_abbt_results.csv` exists, but P14
   stage 1 wants curated keyframes before quantification is trustworthy. Has
   M162 had a review pass, or does it need one?
2. **Which factor do you most suspect?** Determines what session metadata is
   worth capturing alongside - stage temperature, mount-to-acquisition delay,
   culture OD at mounting - and whether a purpose-designed M168 is warranted.
3. **Should the other NAS experiments be catalogued?** M157-M159 and M163-M167
   are unrecorded. Some may be better-matched controls than M162.

### 6.2 Recommended order of work

**Run the cheap decisive analysis first.** Section 3.2 shows silhouette cannot
resolve differences at this effect size, so routing this through autoencoders
and UMAPs would answer nothing. The 11 engineered features compared directly -
per-feature distributions with effect sizes - needs no manifold and no
representation learning.

1. **M162 FL1 through stages 4-5.** Stage 3 is already done, so this is the
   fastest route to a first comparison point.
2. **M160 FL1 vs M162 FL1 on the 11 features.** M160's side already exists.
   If M162 cells are less stressed, `pol1_mid`, pole distance and Periodicity
   should sit higher - the same signature M160 shows at FL1 relative to FL7.
3. **Copy M161 to the SSD** (7.8 GB of 220 GB free; FL1 alone is ~1.1 GB).
   Per P4, working data goes on the SSD, not the internal disk. FL1-only is
   enough to start.
4. **M161 FL1 through stages 1-5.** The long pole - segmentation and keyframe
   curation from scratch. HPC, by analogy with M160's 2.5-3.5 h/task.
5. **Three-way comparison**, reading off the table in 5.2.
6. **Only then**, if the distributional result warrants it, stage 6 and the
   manifold - with P16's shuffled-dimension nulls at each experiment's own n
   and latent dimension. The M160 floors do not transfer.

### 6.3 Code changes required

- Generalize the four `*_m160.py` modules to take an experiment root rather
  than the hard-coded M160 path. They are M160-specific by construction under
  P15's copy-to-modify rule, so this is a real refactor, not a flag.
- `umap_validation/nn_sweep_fl1.py` hard-codes the M160 base and the `FL1_`
  filter; parameterise both.
- Field count differs: M160 has 3 (F0-F2), M161 and M162 have 4 (F0-F3). Film
  prefixes differ three ways: `5_1_N1_*`, `NeonG_YES_1_*`, `NeonG_YES_*`. Note
  that `NeonG_YES_1_FL1` and `NeonG_YES_FL1` differ by one underscore-delimited
  token, so any prefix match must be anchored or it will cross-match M161 and
  M162.
- An id map is needed for each of M161 and M162. P12 forbids `new_cell_id` row
  numbers; follow the M160 convention, not M156's.
- Stage 4b strips: rescale to each film's `[C1min, C1max]` before writing,
  never pass raw camera counts.

### 6.4 Ledger

`docs/EXPERIMENTS.md` now carries M161 and the corrected M162 state. The
unrecorded NAS experiments (M157-M159, M163-M167) are noted but not catalogued;
see 6.1 question 3.

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
