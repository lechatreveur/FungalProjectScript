# Fungal Manifold Explorer Builder

A modular, config-driven pipeline for building interactive 2D/3D UMAP manifold visualization dashboards from fungal time-course single-cell trajectories.

## 1. Directory Structure

The project has been refactored into a structured Python package:
* `config.py`: Dataclasses for dataset settings and timing configurations.
* `schemas.py`: Input schema validation and trajectory length audits.
* `adapters.py`: Normalization adapter pattern to load metadata formats dynamically.
* `qc.py`: Composable quality control rules.
* `division_time.py`: Division time calculation and area-growth alignment fitting.
* `pipeline.py`: Pipeline coordinator for scaling, autoencoder latent projection, and UMAP space alignment.
* `exporter.py`: Outputs single-file HTML or folder-based static site directory structures.
* `cli.py`: Command Line Interface runner.
* `templates/`: HTML, CSS, and Plotly interactive application templates.

---

## 2. Configuration (`config.yaml`)

All experiments are registered and configured inside a unified `config.yaml` file. Here is an example configuration adding both a reference experiment (`Sept17`) and a mutant experiment (`M133`):

```yaml
output:
  directory: "/Volumes/X10 Pro/FungalProject_Outputs/fc_ae_3d"
  mode: "single-html" # Options: "single-html" or "static-site"
  filename: "fc_ae_3d_manifold_explorer_curated_Sept17.html"

model:
  path: "/Volumes/X10 Pro/FungalProject_Outputs/fc_ae_3d/fc_ae_3d_final.pth"
  reference_experiment: "Sept17"
  umap_seed: 42

experiments:
  Sept17:
    root: "/Volumes/X10 Pro/Movies/2025_09_17"
    stacked_csv: "/Volumes/X10 Pro/Movies/2025_09_17/unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv"
    id_map_csv: "/Volumes/X10 Pro/Movies/2025_09_17/unaligned_pairs_quant/id_map_unaligned.csv"
    linkage_json: "/Volumes/X10 Pro/Movies/2025_09_17/sequence_linkage.json"
    qc_jsons:
      - "/Volumes/X10 Pro/Movies/2025_09_17/F0/qc_F0.json"
      - "/Volumes/X10 Pro/Movies/2025_09_17/F1/qc_F1.json"
    cycle_length_min: 300.0
    reference: true
    adapter_type: "sept17"
    sources:
      GFP1:
        film_name: "A14_1TP1_{field}"
        midpoint_min: 10.0
        time_res_min: 0.2
        start_time_min: 0.0
      GFP2:
        film_name: "A14_1TP2_{field}"
        midpoint_min: 150.0
        time_res_min: 0.2
        start_time_min: 140.0

  M133:
    root: "/Volumes/X10 Pro/Movies/2026_04_29_M133"
    stacked_csv: "/Volumes/X10 Pro/Movies/2026_04_29_M133/unaligned_pairs_quant/stacked_gfp1_gfp2_for_unaligned_pairs.csv"
    id_map_csv: "/Volumes/X10 Pro/Movies/2026_04_29_M133/unaligned_pairs_quant/id_map_unaligned.csv"
    linkage_json: "/Volumes/X10 Pro/Movies/2026_04_29_M133/sequence_linkage.json"
    qc_jsons:
      - "/Volumes/X10 Pro/Movies/2026_04_29_M133/F0/qc_F0.json"
      - "/Volumes/X10 Pro/Movies/2026_04_29_M133/F1/qc_F1.json"
    cycle_length_min: 300.0
    reference: false
    adapter_type: "generic"
    sources:
      GFP1:
        film_name: "A14_TP1_{field}"
        midpoint_min: 10.0
        time_res_min: 0.2
        start_time_min: 0.0
      GFP2:
        film_name: "A14_TP2_{field}"
        midpoint_min: 70.0
        time_res_min: 0.5
        start_time_min: 20.0
      GFP3:
        film_name: "A14_TP3_{field}"
        midpoint_min: 130.0
        time_res_min: 0.2
        start_time_min: 60.0
```

---

## 3. How to Build the Dashboard

Run the pipeline from the project root using the CLI module:

```bash
# Build the dashboard with the configured experiments
python3 -m SingleCellDataAnalysis.manifold_explorer.cli build SingleCellDataAnalysis/config.yaml

# Fail execution immediately if warning validation issues occur (Strict mode)
python3 -m SingleCellDataAnalysis.manifold_explorer.cli build SingleCellDataAnalysis/config.yaml --strict
```

---

## 4. Scientific Scaling & UMAP fit Semantics

To ensure reproducibility and correct scientific comparison:
1. **Reference Scaling**: The feature scaling parameters (means, standard deviations) are computed **exclusively** on the reference dataset (`Sept17`). All other experiments (e.g., `M133` mutant) are scaled using these reference statistics.
2. **Reference UMAP Space**: The 2D and 3D UMAP coordinate reductions are fitted **exclusively** on the reference latents. Mutant cell latents are projected onto this pre-fitted space using UMAP's `.transform()` method.

---

## 5. Sharing and Static-Site Serving

If `output.mode` is set to `static-site`, the output folder will contain the dashboard structure with separately optimized image strips in `.webp` format and decoupled `.json` data files.

Because modern browsers block local asynchronous HTTP fetch requests (`fetch()`) when opening HTML directly from the local file system (`file://`), you should serve the static directory using a simple web server:

```bash
# Start server in the output directory
python3 -m http.server 8000 --directory "/Volumes/X10 Pro/FungalProject_Outputs/fc_ae_3d/"
```
Then visit `http://localhost:8000` in your browser.
