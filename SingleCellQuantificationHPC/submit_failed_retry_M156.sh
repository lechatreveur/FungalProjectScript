#!/usr/bin/env bash
set -euo pipefail

BASE_EXP_ROOT="/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC"
MOVIE_ROOT="/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156"
EXP_ROOT="${BASE_EXP_ROOT}/2026_07_16_M156"
JOB_ID_FILE="${EXP_ROOT}/slurm_job_ids.tsv"
DONE_MARKER="${EXP_ROOT}/FAILED_RETRY_SUBMISSION_COMPLETE"

[[ ! -f "$DONE_MARKER" ]] || exit 0
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cellpose_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$BASE_EXP_ROOT"

submitted_movies=0
for retry_ids in "${EXP_ROOT}"/3_*/retry_failed_cell_ids.txt; do
    [[ -s "$retry_ids" ]] || continue
    movie="$(basename "$(dirname "$retry_ids")")"
    if [[ "$movie" == 3_BF* ]]; then
        track_channel="bf"
    else
        track_channel="gfp"
    fi
    echo "Retrying $movie with $(wc -l < "$retry_ids") failed cells."
    python generate_cell_jobs.py \
        -w "$EXP_ROOT/$movie/failed_retry_sb_scripts/" \
        -s "$BASE_EXP_ROOT/one_cell_quantification_1CH.py" \
        -i "$retry_ids" \
        -e "$MOVIE_ROOT" \
        -f "$movie" \
        -c "$track_channel" \
        -n 9 \
        -d 10 \
        -a 2000 \
        --direction forward \
        --job-name-prefix "M156_FINALRETRY_" \
        --job-id-file "$JOB_ID_FILE" \
        --submit slurm
    submitted_movies=$((submitted_movies + 1))
done

date -Is > "$DONE_MARKER"
echo "M156 failed-cell retry submission complete for ${submitted_movies} movies."
