# Development Report: `manifold_explorer` Is the Reference Builder

**Date**: September 18, 2026
**Experiment**: `2026_08_28_M160` (and the Sept17 / M156 reference family)
**Branch**: `mbdt-stage3`
**Status**: review complete; P15 corrected. No migration performed.

---

## 1. Executive Summary

The M160 manifold pipeline built over the last several days is a **parallel
implementation of an existing, more careful one**:
`SingleCellDataAnalysis/manifold_explorer/`.

That package is the generator for
`fc_ae_3d_manifold_explorer_curated_Sept17.html`,
`..._curated_M156_qcfiltered.html` and the rest of the curated family — the very
templates every `build_umap_html_m156_*.py` re-embeds into, and the source of
the explorer features reimplemented for M160 from the M156 HTML.

I had concluded, and recorded in P15, that no such generator existed. That was
wrong. The error was in the search, not the repository.

---

## 2. How the generator was missed

The builder is **config-driven**. `exporter.py` takes `output_path` as a
parameter, so the output filename never appears in any `.py` file — it lives in
`config.yaml` and `README.md`.

My search was `grep -rn --include="*.py" "curated_Sept17"`. The `--include`
filter excluded the two file types that actually held the answer. I compounded
it by also searching for the literal string `fc_ae_3d/`, which likewise appears
only in configuration.

**The lesson worth keeping**: when looking for what produced an artifact, search
the whole tree before concluding absence, and treat a config-driven tool as
invisible to source-only greps. A negative result from a filtered search is not
evidence of absence.

The fingerprints that should have led there are unambiguous in hindsight —
`templates/viewer.html` carries the exact title of the curated explorers, and
`app.js` holds `getCategory`, `thresholds-container`, `sticky-card` and the ACF
handling I reconstructed from the minified M156 page.

---

## 3. What `manifold_explorer` already provides

Four of these were rebuilt from scratch for M160 for want of finding them.

**A declared film clock.** Each source in `config.yaml` carries
`start_time_min`, `midpoint_min` and `time_res_min`; the experiment carries
`cycle_length_min`. M156's fluorescence films are declared at 0.0, 40.4, 80.9,
121.3, 161.8, 202.2 minutes. I derived M160's equivalent (40.6 minute spacing)
from Imaris timestamps instead — the same quantity, obtained the hard way.

**Reference fitting, as P1 requires.** The scaler is fitted on the reference
experiment's curated cells only; UMAP is then `fit` on reference latents and
every other experiment placed with `.transform()`:

```python
reducer_3d.fit(latents_ref)
coords_3d = reducer_3d.transform(latents)
```

This is the part the M160 path gets wrong by design. Building standalone maps
per experiment — which is what was asked for and what I built — yields
coordinates that cannot be compared across experiments. It also dissolves the
"each model has its own scaler, so losses are not comparable" problem I reported
as unavoidable: here one scaler is shared by construction.

**Division time with brightfield alignment.**
`division_time.py::estimate_missing_division_times` fits area against time on
cells with known division times and places the rest by
`tau = mean_T − (mean_A − c) / m`, using **brightfield** areas decoded from
masks. That is the same estimator and the same channel choice that M160 arrived
at only after the project owner pointed out the brightfield films were being
ignored.

**Composable QC.** `qc.py` excludes cells with missing trajectories or a `bad`
curation status and reports counts by reason — the origin of "curated" and
"qcfiltered" in those filenames.

Also present: `adapters.py` (a pluggable metadata layer, `GenericAdapter` plus a
`Sept17Adapter`), `schemas.py` (input validation and trajectory-length audits),
and `exporter.py` (single-file HTML or a static site).

---

## 4. What the M160 path has that it does not

The M160 work is not redundant. It handles inputs `manifold_explorer` was never
built for:

| Capability | Where |
| :--- | :--- |
| Model-based dense masks as the mask source | stage 3 / stage 4 |
| 101-frame completion by quantifying curated keyframes | `quantify_model_based_dense.py` |
| Lineage as a mother/daughter trie, drawn as forks | `lineage_m160.py` |
| Division detection on a continuous two-channel series | `division_detect_m160.py` |
| Division vs missegmentation by the bounce criterion | same |
| Strips rebuilt from dense masks without requantifying | `build_strips_m160.py` |
| Latent-dimension sweep | `sweep_fc_ae_m160.py` |
| Stage per datapoint, flagged curated or estimated | `cell_cycle_stage_build.py` |

---

## 5. What migration would involve

**The blocker is the adapter.** `GenericAdapter.load_metadata` requires an
`id_map_csv` with columns `new_cell_id`, `orig_str_id`, `field`, `source`, and
raises `FileNotFoundError` without it. **M160 has no
`unaligned_pairs_quant/id_map_unaligned.csv`.**

That is not an accident of housekeeping. `new_cell_id` is the row-number
identity P12 forbids, and the M160 pipeline was deliberately keyed on
`global_cell_id` instead. So M160 cannot use the generic adapter without
manufacturing exactly the identity the policy rules out.

M160 does have `sequence_linkage.json` and the per-sequence `qc_*.json` files,
so the pieces for an **`M160Adapter`** are present: read identity from the
linkage rather than an id map, as `Sept17Adapter` already does for its own
format. That is the honest shape of the work — one new adapter class, a
`sources` block with the film clock (already measured), and a config entry.

**What would have to be decided, not just coded**: the M160 maps are standalone
by request, and `manifold_explorer` is built around a reference experiment with
everything else projected onto it. Migrating means choosing whether M160 joins
an existing reference manifold or becomes its own reference — a scientific
choice, not a configuration one.

---

## 6. Recommendation

Do not migrate yet. The M160 pipeline works and its outputs are current. But new
manifold capability should go into `manifold_explorer` rather than alongside it,
and the M160 features worth keeping — lineage forks, the bounce division
detector, the dense-mask strip builder — are the candidates to port.

The immediate value of this review is the correction: P15 now points at the real
builder, so the next person does not rebuild a fifth copy of the film clock.

---

## 7. Corrections issued

- **P15** previously stated that the curated-family explorers had no generator
  and should be treated as opaque artifacts never to be rebuilt. Both claims
  were false and have been replaced.
- The surrounding facts in that entry survive and are unchanged:
  `FC_AE_3d_umap.py` is a different, plainer builder; its output
  (2026-07-06) predates `fc_ae_3d_final.pth` (2026-07-20); it now crashes on
  Sept17 data because the stacked file gained a `global_cell_id` column that
  changes the id format `load_cell_areas` assumes; and `video_gids.txt` and
  `cycle_stage_scores.npy` are absent from the disk.
