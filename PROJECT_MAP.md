# Project map

The repository turns time-lapse fungal microscopy movies into aligned, quantitative
single-cell data and then models cell-cycle and septum behavior.

## Production path

1. **Export raw Imaris movies**
   - `run_local_ims_export.py`
   - `batch_ims_export.py`
2. **Segment cells**
   - `SingleCellQuantificationHPC/batch_segment_ims_1CH.py`
3. **Create cell jobs and track/quantify cells**
   - `SingleCellQuantificationHPC/generate_cell_ids_1CH.py`
   - `SingleCellQuantificationHPC/generate_cell_jobs.py`
   - `SingleCellQuantificationHPC/one_cell_quantification_1CH.py`
4. **Review and correct tracking**
   - `SingleCellQuantificationHPC/manual_correction_tool.py`
5. **Analyze cell trajectories**
   - reusable analysis code is in `SingleCellDataAnalysis/`
   - `SingleCellDataAnalysis/main.py` is the generic classical-analysis entry point
   - `SingleCellDataAnalysis/septum_train_binary.py` trains the production septum classifier
   - `SingleCellDataAnalysis/AE_*`, `FC_AE_*`, `FC_Contrastive_*`, and `Video_AE_*`
     are related representation-learning experiments
6. **Render population movies**
   - `make_all_population_movies.sh`
   - `make_population_movie.py`

The operational details and storage locations are documented in
`SingleCellQuantificationHPC/PIPELINE_PROTOCOL.md`.

## Naming guide

- `main_process_<date>_<experiment>.py`, `quantify_M*.py`, `submit_array_M*.sh`, and
  `IAonNAS_<date>_*.py` preserve experiment-specific parameters. They are historical,
  but not automatically obsolete.
- `*_test.py`, `test_*.py`, `debug_*.py`, `scratch_*.py`, plotting scripts, and saliency
  scripts are diagnostics or one-off investigations rather than production entry points.
- `AscomycotaSpider*`, `JGI_*`, and genome-download utilities form a separate comparative
  genomics workflow; their latest versions remain at repository root.
- Superseded files are kept under `archive/`, with an index explaining each move.

## Maintenance rule

Archive a script only when it has a clearly identified replacement and no active
references. Keep experiment-specific scripts until their results and parameters have a
separate reproducibility record.
