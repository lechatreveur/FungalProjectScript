#!/usr/bin/env bash
set -euo pipefail

SEG_BACKEND="${1:-legacy}"
BASE_EXP_ROOT="/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC"
MOVIE_ROOT="/RAID1/working/R402/hsushen/FungalProject/Movies/2026_07_16_M156"
if [[ "$SEG_BACKEND" == "cellpose_overexposed" ]]; then
    EXP_ROOT="${BASE_EXP_ROOT}/2026_07_16_M156_cpsam_overexp"
else
    EXP_ROOT="${BASE_EXP_ROOT}/2026_07_16_M156"
fi
JOB_ID_FILE="${EXP_ROOT}/slurm_job_ids.tsv"
DONE_MARKER="${EXP_ROOT}/SUBMISSION_COMPLETE"

cd "$BASE_EXP_ROOT"
mkdir -p "$EXP_ROOT" logs

if [[ -f "$DONE_MARKER" ]]; then
    echo "M156 submission already completed: $DONE_MARKER"
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
    "$BASE_EXP_ROOT/tracker_model.py"
    "$BASE_EXP_ROOT/tracker_dataset.py"
    "$BASE_EXP_ROOT/ai_tracking_inference.py"
    "$BASE_EXP_ROOT/Cell_tracking_functions.py"
    "$BASE_EXP_ROOT/quant_helpers.py"
    "$BASE_EXP_ROOT/bf_pattern.py"
    "$BASE_EXP_ROOT/Image_quantification_functions.py"
    "$BASE_EXP_ROOT/tracker_checkpoints/model_latest.pt"
    "$BASE_EXP_ROOT/tracker_checkpoints_m93_gfp/model_latest.pt"
)
for required_file in "${required_files[@]}"; do
    [[ -s "$required_file" ]] || {
        echo "Missing tracker prerequisite: $required_file" >&2
        exit 7
    }
done

python -c \
    "import tracker_dataset, ai_tracking_inference, Cell_tracking_functions, quant_helpers, bf_pattern, Image_quantification_functions; from tracker_model import load_tracker; load_tracker('tracker_checkpoints/model_latest.pt', device='cpu'); load_tracker('tracker_checkpoints_m93_gfp/model_latest.pt', device='cpu')"
echo "Tracker dependency preflight passed."

mapfile -t MOVIES < <(
    find "$MOVIE_ROOT" -mindepth 1 -maxdepth 1 -type d \
        \( -name '3_BF*' -o -name '3_FL*' \) -printf '%f\n' | sort
)

if [[ "${#MOVIES[@]}" -ne 33 ]]; then
    echo "Expected 33 selected M156 movie directories; found ${#MOVIES[@]}." >&2
    exit 3
fi

: > "$JOB_ID_FILE"

for file_name in "${MOVIES[@]}"; do
    if [[ "$file_name" == 3_BF* ]]; then
        track_channel="bf"
    elif [[ "$file_name" == 3_FL* ]]; then
        track_channel="gfp"
    else
        echo "Unexpected movie name: $file_name" >&2
        exit 4
    fi

    echo "Preparing $file_name (track_channel=$track_channel)"
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
        -c "$track_channel" \
        -n 9 \
        -d 10 \
        -z 0 \
        -a 2000 \
        --direction forward \
        --seg-backend "$SEG_BACKEND" \
        --job-name-prefix "M156_" \
        --job-id-file "$JOB_ID_FILE" \
        --submit slurm
done

[[ -s "$JOB_ID_FILE" ]] || {
    echo "No SLURM jobs were submitted." >&2
    exit 6
}

date -Is > "$DONE_MARKER"
echo "M156 submission complete. Jobs recorded: $(wc -l < "$JOB_ID_FILE")"
