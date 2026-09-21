# UMAP clustering claims: what the metrics actually support

**Date:** 2026-09-21
**Experiment:** 2026_08_28_M160, with Sept17 (2025_09_17) as pipeline control
**Outputs:** `/Volumes/X10 Pro/FungalProject_Outputs/umap_control/`

## Question

Sept17's UMAP was remembered as showing obvious clustering; M160's does not.
Is that a real biological difference, a pipeline difference, or an artifact?

## What was tested

Three experiments, each holding everything fixed except one variable.

### 1. Sept17 through the M160 pipeline (`cluster_comparison.png`)

Sept17 re-trained with the M160 trainer at latent 6 and embedded with
`n_neighbors` at the Sept17 reference fraction (3.97%). Sept17 scored
silhouette +0.514, M160 full cohort +0.481.

### 2. Matched sample size (`subsample_matched_n.png`)

M160 subsampled to Sept17's n=378, ten independent draws, `n_neighbors=15`.
Result: +0.502 +/- 0.022. Sept17's +0.514 sits 0.5 sd inside that distribution.
So the two datasets are not distinguishable by this measure.

### 3. `n_neighbors` sweep on M160 (`m160_n_neighbors_sweep.png`)

n=6243, 6D latents, `n_components=2, random_state=42, n_jobs=1` fixed.

| n_neighbors | % of n | best k | silhouette | Hopkins |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 0.08% | 2 | +0.447 | 0.936 |
| 15 (library default) | 0.24% | 2 | +0.458 | 0.894 |
| 30 | 0.48% | 2 | +0.435 | 0.906 |
| 60 | 0.96% | 2 | +0.456 | 0.867 |
| 125 | 2.00% | 2 | +0.480 | 0.867 |
| 248 (current) | 3.97% | 2 | +0.481 | 0.867 |
| 500 | 8.01% | 2 | +0.484 | 0.899 |
| 1000 | 16.02% | 2 | +0.482 | 0.920 |

`n_neighbors` changes texture, not topology. All eight panels show one
connected crescent with the same hook and tail. The setting is not
load-bearing, and the hypothesis that small `n_neighbors` fragments the cohort
into apparent islands is **false** for this dataset.

### 4. Null calibration (`metric_null_calibration.png`) — the decisive test

Four datasets, n=6243 in 6D, identical pipeline:

| dataset | sil @15 | sil @248 | Hopkins |
| --- | ---: | ---: | ---: |
| isotropic Gaussian (no structure) | +0.394 | +0.379 | 0.78-0.80 |
| M160 with each dim independently permuted | +0.417 | +0.435 | 0.89-0.92 |
| **M160 real** | **+0.458** | **+0.481** | 0.87-0.89 |
| 3 genuinely separated Gaussian blobs | +0.886 | +0.926 | 0.99 |

## Findings

**The silhouette floor on a UMAP embedding is ~+0.39, not 0.** K-means bisects
any elongated cloud and silhouette rewards it. Genuine cluster structure scores
~+0.89. M160's +0.48 is a third of the way up from noise and barely above its
own shuffled null.

**Hopkins is unusable on a UMAP embedding.** Its 0.5-means-uniform baseline
assumes a uniform-box null; a UMAP output is never uniform, so structureless
noise scores 0.78-0.80. The shuffled null scored *higher* (0.917) than the real
data (0.867). Discard all Hopkins figures previously quoted in this thread.

**Neither dataset clusters.** Sept17's +0.514, M160's +0.481 and the matched-n
subsamples' +0.502 all sit in the noise band. There was no Sept17 clustering
result for M160 to fail to reproduce. The remembered "obvious clustering" in
the Sept17 explorer is not supported by these measurements.

**UMAP manufactures convincing structure from nothing.** The shuffled-dimension
null - all joint structure destroyed, every marginal preserved - renders as a
vivid multi-armed star that looks *more* structured than the real cohort.

## Consequences for the pipeline

1. Do not use silhouette or Hopkins on a UMAP embedding as evidence of
   clustering without running the matched nulls alongside. If they are
   reported, report the shuffled-dimension null and the noise floor in the
   same table.
2. Do not read cluster structure off a UMAP scatter by eye. Panel 2 of
   `metric_null_calibration.png` is the standing counterexample.
3. `n_neighbors` on M160 is not a tuning knob for structure. Leave it at the
   fraction-matched 248; nothing between 125 and 1000 changes the map.
4. Claims about M160 vs Sept17 structure need a different instrument -
   one that tests a specific hypothesis (e.g. do the Mode categories, or cell
   cycle stage, separate in latent space?) rather than asking the
   unconditioned question "are there clusters?".

## Reproduction

All four scripts live in `SingleCellDataAnalysis/umap_validation/` and are run
from the repo root with the SSD mounted:

- `cluster_compare.py` - Sept17 vs M160 vs M160-FL1 at matched fraction
- `subsample_matched_n.py` - M160 down to n=378, 10 draws
- `nn_sweep.py` - the eight neighbour counts
- `null_calibration.py` - the null calibration

Each writes its plot to `/Volumes/X10 Pro/FungalProject_Outputs/umap_control/`.
`null_calibration.py` is the one to re-run before making any future claim about
cluster structure in a UMAP embedding.

The Sept17 control run itself is config-driven and reproducible:

```
python3 -m SingleCellDataAnalysis.manifold_explorer.cli \
    build SingleCellDataAnalysis/config_sept17_control.yaml
```
