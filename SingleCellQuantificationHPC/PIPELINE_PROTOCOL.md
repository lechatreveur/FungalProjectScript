# Fungal Pipeline: Single-Cell Quantification Protocol

This protocol outlines the steps to process experimental films from raw `.ims` files to single-cell quantification tables on the HPC.

## 1. IMS Export (Local SSD)
Raw `.ims` files are exported to TIFF frames using the local workstation for speed.
- **Script**: `run_local_ims_export.py` (wraps `batch_ims_export.py`)
- **Action**: Updates `working_dir` and runs the export environment.
- **Output**: `Frames_<movie_name>/` and `<movie_name>_c_0.mp4` in the movie directory on `/Volumes/X10 Pro/`.

## 2. Segmentation (Local SSD)
Segmentation is performed locally using Cellpose-SAM.
- **Script**: `batch_segment_ims_1CH.py`
- **Action**: Iterates through `.ims` files in a directory and generates masks.
- **Output**: `Masks_<movie_name>/` and `DONE_segmentation.txt`.

## 3. Data Sync to HPC (RAID1)
Processed frames and masks must be uploaded to the HPC storage for large-scale quantification.
- **Script**: `upload_M13x_data.sh` (or manual `rsync`)
- **Source**: `/Volumes/X10 Pro/Movies/<Experiment>/`
- **Destination**: `hsushen@172.20.97.21:/RAID1/working/R402/hsushen/FungalProject/Movies/<Experiment>/`
- **Note**: Ensure `Frames_` and `Masks_` are synced. The upload script automatically excludes macOS metadata files (`._*`).
- **Cleanup**: Periodically remove `._*` files if they appear using: `find . -name "._*" -exec rm -rf {} +`.

## 4. Quantification Setup (HPC/Local)
Identify cells and generate SLURM job scripts.
- **Script**: `submit_array_M13x.sh` (wraps `generate_cell_ids_1CH.py` and `generate_cell_jobs.py`)
- **Channel Logic**:
    - `_BF_` or `YES_Scd1_D` (1, 3, 5) -> **bf** (uses `bf_pattern_only`)
    - `_FL_` or `YES_Scd1_D` (2, 4) -> **gfp** (uses `quantify_one_object`)
- **Output**: `cell_ids.txt` and `sb_scripts/*.sh` in the quantification directory.

## 5. Execution (HPC)
Submit the generated array jobs to the SLURM scheduler.
- **Action**: `cd /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/<Experiment>/<Movie>/sb_scripts/`
- **Action**: `sbatch cell_job_1.sh` (or equivalent master script).

## 6. Data Retrieval
Quantification results (`cell_<id>_data.csv`) are generated in `TrackedCells_<movie_name>/`. Sync these back to the local machine or NAS for analysis.
- **Script**: `pull_cells.sh` (pass movie folder names as arguments).

## 7. Population Movies
Generate visual population movies using the synced frames and tracking data.
- **Script**: `make_all_population_movies.sh` (wraps `make_population_movie.py`).
- **Action**: Run the script passing the movie folder names as arguments (e.g., `./make_all_population_movies.sh 2026_04_23_M130`).
- **Output**: `<movie_name>_population.mp4` saved in the `population_movies` subfolder within each movie directory.

