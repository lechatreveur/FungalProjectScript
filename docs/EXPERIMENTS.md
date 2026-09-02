# Experiments ledger

One row per imaging experiment. **Repo-derived** columns (date, channels, film
structure, driving scripts, linkage rule, deliverables) are facts pulled from
`tracking_corrector/config.yaml`, `generate_sequence_linkages.py`, git history,
and script names. **Strain / condition** is inferred from film-folder naming and
is *not* a verified genotype — confirm against the lab notebook and fill in
**Aim**, which the repo does not record.

Referenced by [PROJECT_POLICY.md](PROJECT_POLICY.md) P8. Canonical experiment
registry: `SingleCellQuantificationHPC/tracking_corrector/config.yaml`.

## Naming shorthand seen in film folders (unverified)

- `A14` — recurring strain background.
- `Scd1_D`, `Scd1S573A`, `Scd1S573D` — Scd1 polarity-factor mutants; `S573A` /
  `S573D` read as phospho-site (Ser573) alanine / aspartate substitutions.
- `tea1`, `dcd10` — additional markers / mutants.
- `YES` — rich media. `FBFBF` — fluorescence/brightfield alternating
  acquisition. `1t` / `t` — single timepoint / temperature block. `20min`,
  `1TP1`/`1TP2`, `FL1..FLn` — acquisition interval / multi-block scheme.

## Ledger

| Exp | Date | Movie folder | Strain / condition (from film names — confirm) | Ch | Film sequence (per `generate_sequence_linkages.py`) | Driving scripts / deliverables | Aim (fill in) |
|---|---|---|---|---|---|---|---|
| wt_tea1 | 2025-03-14 | — | wild-type + `tea1` | ? | pre-pipeline | `IAonNAS_20250314_wt_tea1*.py` (per-field one-offs) | |
| M57 | 2025-04-30 | — | `WT_dcd10_G1` | ? | pre-pipeline | `IAonNAS_2025_04_30_M57_WT_dcd10_G1_1_F0.py` | |
| M63 | 2025-05-15 | — | `A14` | ? | pre-pipeline | `IAonNAS_2025_05_15_M63_A14_5_F1.py` | |
| M68 | 2025-06-04 | — | `A14` | ? | pre-pipeline | `IAonNAS_2025_06_04_M68_A14_1.py` | |
| June25_20m | 2025-06-25 | `2025_06_25/A14_10_20min` | `A14`, 20-min interval | bf+gfp | — | AE / contrastive training data (`FC_AE_*`, `FC_Contrastive_*`, `Video_AE_*`) | |
| Sept17 | 2025-09-17 | `2025_09_17` | `A14` `1TP1` / `1TP2` (BF+FL each) | bf+gfp | `A14_1TP1_{F} , A14_1TP1_BF_{F} , A14_1TP2_{F} , A14_1TP2_BF_{F}` | original manifold-explorer **reference** experiment; tracker benchmark set | |
| M92 | 2025-12-31 | `2025_12_31_M92` | `A14-YES-1t-FBFBF` | bf | 5 films `FBFBF_{F} … -5_{F}` | first multi-film-GUI integration; `main_process_2025_12_31_M92.py`, `quantify_M92_M93.py`, PCA + trajectory AE; septum classifier training dir | |
| M93 | 2026-01-08 | `2026_01_08_M93` | `A14` (`FL_1,BF_1,FL_2,BF_2,FL_3,BF_2b`) | bf | 6-film `A14_{F}` group | `main_process_2026_01_08_M93.py`, `quantify_M92_M93.py`; tracker benchmark set (`tracker_comparison_summary.md`); septum training dir | |
| M96 | 2026-01-16 | `2026_01_16_M96` | `A14` (`FL_1,BF_1,FL_2,BF_2,FL_3`) | bf | 5-film `A14_{F}` group | septum export example in `DEVELOPMENT_NOTES.md` §3 | |
| M97 | 2026-01-18 | `2026_01_18_M97` | `A14-YES-t` (5 films) | ? | 5-film `A14-YES-t_{F}`; "remove item 0" rule | `submit_array_M97.sh`; "M97/M125 analysis on SSD" commit | |
| M125 | ? | — | ? | ? | ? | referenced once (M97 commit); no other trace | |
| M130 | 2026-04-23 | `2026_04_23_M130` | ? | bf+gfp | generic `FL/BF` grouping | registry only; no scripts or commits — minimal footprint | |
| M133 | 2026-04-29 | `2026_04_29_M133` | `YES_Scd1_D` (Scd1 mutant) | bf+gfp | 6-film `YES_Scd1_D_{F} … _5_{F}` | `quantify_M133.py`, `generate_M133_strips.py`, `FC_AE_3d_umap_M133.py`, `FC_Contrastive_train_M133_mix.py` | |
| M135 | 2026-04-30 | `2026_04_30_M135` | `A14` (`FL1,BF1,FL2,BF2,FL3`) | bf+gfp | 5-film `A14_{F}` group | `quantify_M135.py`, `generate_M135_strips.py`, `plot_area_heatmap_M135.py`, `plot_area_vs_stage_M135.py` | |
| M143 | 2026-06-03 | `2026_06_03_M143` | `Scd1S573A` / `Scd1S573D` (phospho-site mutants) | gfp (per `run_local_M143_gfp.sh`) | 6-film per strain, `{strain}_{F} … _5_{F}` | `submit_array_M143.sh`, `submit_single_M143.sh`, `run_local_M143_gfp.sh` | |
| M156 | 2026-07-16 | `2026_07_16_M156` | field `3_`, `FL1`–`FL6` (6 GFP films) | bf+gfp | 6-film `3_FL1..FL6_{F}` | **major current subject**: retrack (`retrack_F2_improved.py`, `submit_array_M156_fl_retrack.sh`), curated dataset, vertical strips (`generate_M156_strips.py`), `merge_cell_data_M156.py`, `FC_AE_3d_train_M156.py`, many `umap_m156_*.html` / `build_umap_*` deliverables; current manifold-explorer reference in `SingleCellDataAnalysis/config.yaml` | |
| M160 | 2026-08-28 | `2026_08_28_M160` | `5_1_N1`, `FL1`–`FL7` + `BF1`–`BF6` | bf+gfp | 13-film `5_1_N1_{F}` (excl. `N1_2`, `snap`) | newest; `retrack_m160_sequences.py`, `retrack_3frames_m160.py`, `submit_array_M160.sh` | |

## Status legend for the "Aim" work

- **pre-pipeline** rows (`IAonNAS_*`, wt_tea1, M57/M63/M68) predate the current
  segment→track→quantify pipeline; their scripts are frozen records (P1).
- Rows with a `quantify_M*` / `main_process_*` script have been through classical
  analysis; rows with `FC_AE_*` / `FC_Contrastive_*` / `generate_M*_strips` have
  been through representation-learning.
- Keep this table updated when a new experiment folder is added to
  `tracking_corrector/config.yaml`.
