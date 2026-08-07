#!/usr/bin/env bash
set -euo pipefail

# Retrack ONLY the M156 FL/GFP films with the deterministic overlap+area-penalty
# tracker (track_one_direction / get_cell_mask_area_aware), replacing the ai_tracker
# GFP pass that used an out-of-domain M93-trained checkpoint. See project memory
# m156_fl_tracking_domain_mismatch.md for the diagnosis.
#
# Deliberately separate from submit_array_M156.sh (which still owns the BF films
# and already has a SUBMISSION_COMPLETE marker from the original full run) so this
# retrack doesn't collide with or get blocked by that run's guard files.
# Passes --update_existing so existing cell_*_masks.csv (written by ai_tracker) are
# overwritten rather than skipped.

BASE_EXP_ROOT="/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC"
MOVIE_ROOT="/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156"
EXP_ROOT="${BASE_EXP_ROOT}/2026_07_16_M156_fl_retrack"
JOB_ID_FILE="${EXP_ROOT}/slurm_job_ids.tsv"
DONE_MARKER="${EXP_ROOT}/SUBMISSION_COMPLETE"

cd "$BASE_EXP_ROOT"
mkdir -p "$EXP_ROOT" logs

if [[ -f "$DONE_MARKER" ]]; then
    echo "M156 FL retrack submission already completed: $DONE_MARKER"
    exit 0
fi
if [[ -s "$JOB_ID_FILE" ]]; then
    echo "Refusing to resubmit: partial job manifest exists at $JOB_ID_FILE" >&2
    exit 2
fi

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cellpose_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

required_files=(
    "$BASE_EXP_ROOT/one_cell_quantification_1CH.py"
    "$BASE_EXP_ROOT/Cell_tracking_functions.py"
    "$BASE_EXP_ROOT/quant_helpers.py"
    "$BASE_EXP_ROOT/bf_pattern.py"
    "$BASE_EXP_ROOT/Image_quantification_functions.py"
)
for required_file in "${required_files[@]}"; do
    [[ -s "$required_file" ]] || {
        echo "Missing prerequisite: $required_file" >&2
        exit 7
    }
done

python -c \
    "import Cell_tracking_functions, quant_helpers, bf_pattern, Image_quantification_functions"
echo "Deterministic tracker dependency preflight passed (no torch/model load needed)."

mapfile -t MOVIES < <(
    find "$MOVIE_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -name '3_FL*' -printf '%f\n' | sort
)

if [[ "${#MOVIES[@]}" -ne 18 ]]; then
    echo "Expected 18 M156 FL movie directories; found ${#MOVIES[@]}." >&2
    exit 3
fi

: > "$JOB_ID_FILE"

for file_name in "${MOVIES[@]}"; do
    echo "Preparing $file_name (track_channel=gfp, deterministic retrack)"
    python generate_cell_ids_1CH.py \
        --movie_root "$MOVIE_ROOT" \
        --file_name "$file_name" \
        --output_base_dir "$EXP_ROOT" \
        --z_index 0 \
        --min_area 2000

    cell_ids_path="$EXP_ROOT/$file_name/cell_ids.txt"
    [[ -f "$cell_ids_path" ]] || {
        echo "Missing cell ID file: $cell_ids_path" >&2
        exit 5
    }

    python generate_cell_jobs.py \
        -w "$EXP_ROOT/$file_name/sb_scripts/" \
        -s "$BASE_EXP_ROOT/one_cell_quantification_1CH.py" \
        -i "$cell_ids_path" \
        -e "$MOVIE_ROOT" \
        -f "$file_name" \
        -c gfp \
        -n 9 \
        -d 10 \
        -z 0 \
        -a 2000 \
        --direction forward \
        --update_existing \
        --job-name-prefix "M156FLretrack_" \
        --job-id-file "$JOB_ID_FILE" \
        --submit slurm
done

[[ -s "$JOB_ID_FILE" ]] || {
    echo "No SLURM jobs were submitted." >&2
    exit 6
}

date -Is > "$DONE_MARKER"
echo "M156 FL retrack submission complete. Jobs recorded: $(wc -l < "$JOB_ID_FILE")"
