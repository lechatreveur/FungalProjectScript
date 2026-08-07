#!/usr/bin/env bash
set -euo pipefail

BASE_EXP_ROOT="/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC"
MOVIE_ROOT="/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156"
EXP_ROOT="${BASE_EXP_ROOT}/2026_07_16_M156"
JOB_ID_FILE="${EXP_ROOT}/slurm_job_ids.tsv"
DONE_MARKER="${EXP_ROOT}/REPAIR_SUBMISSION_COMPLETE"

[[ -f "${EXP_ROOT}/SUBMISSION_COMPLETE" ]] || {
    echo "Original submission is not complete." >&2
    exit 1
}
[[ ! -f "$DONE_MARKER" ]] || {
    echo "Repair submission already completed."
    exit 0
}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cellpose_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$BASE_EXP_ROOT"

submitted_movies=0
for repair_ids in "${EXP_ROOT}"/3_*/repair_cell_ids.txt; do
    [[ -s "$repair_ids" ]] || continue
    movie="$(basename "$(dirname "$repair_ids")")"
    if [[ "$movie" == 3_BF* ]]; then
        track_channel="bf"
    else
        track_channel="gfp"
    fi
    echo "Repairing $movie with $(wc -l < "$repair_ids") corrected cell IDs."
    python generate_cell_jobs.py \
        -w "$EXP_ROOT/$movie/repair_sb_scripts/" \
        -s "$BASE_EXP_ROOT/one_cell_quantification_1CH.py" \
        -i "$repair_ids" \
        -e "$MOVIE_ROOT" \
        -f "$movie" \
        -c "$track_channel" \
        -n 9 \
        -d 10 \
        -a 2000 \
        --direction forward \
        --job-name-prefix "M156_REPAIR_" \
        --job-id-file "$JOB_ID_FILE" \
        --submit slurm
    submitted_movies=$((submitted_movies + 1))
done

date -Is > "$DONE_MARKER"
echo "M156 repair submission complete for ${submitted_movies} movies."
