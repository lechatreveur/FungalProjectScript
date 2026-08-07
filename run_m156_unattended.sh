#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Users/user/Documents/Python_Scripts/FungalProjectScript"
EXPERIMENT="2026_07_16_M156"
LOCAL_ROOT="/Volumes/X10 Pro/Movies/${EXPERIMENT}"
NAS_ROOT="/Volumes/Movies/${EXPERIMENT}"
HPC_HOST="hsushen@172.20.97.21"
REMOTE_MOVIE_ROOT="/RAID1/working/R402/hsushen/FungalProject/Movies/${EXPERIMENT}"
REMOTE_CODE_ROOT="/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC"
REMOTE_EXP_ROOT="${REMOTE_CODE_ROOT}/${EXPERIMENT}"
IMS_PYTHON="${PROJECT_ROOT}/ims_env/bin/python3"
CELLPOSE_PYTHON="/Users/user/miniforge3/envs/cellpose-sam/bin/python"
RSYNC="/opt/homebrew/bin/rsync"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/m156_unattended_pipeline.log"
STATE_FILE="${LOCAL_ROOT}/M156_PIPELINE_STATE.txt"
SELECTED_LIST="${LOG_DIR}/m156_selected_ims.txt"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

set_state() {
    printf '%s\t%s\n' "$(timestamp)" "$1" | tee "$STATE_FILE"
}

fail() {
    set_state "FAILED: $1"
    exit 1
}

ssh_hpc() {
    ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$HPC_HOST" "$@"
}

build_selection() {
    find "$LOCAL_ROOT" -maxdepth 1 -type f \
        \( -name '3_BF*.ims' -o -name '3_FL*.ims' \) \
        ! -name '3_snap*.ims' | sort > "$SELECTED_LIST"
    local count
    count=$(wc -l < "$SELECTED_LIST" | tr -d ' ')
    [[ "$count" -eq 33 ]] || fail "Expected 33 selected IMS files; found $count."
}

validate_segmentation() {
    local ims stem film frames masks
    while IFS= read -r ims; do
        stem=$(basename "$ims" .ims)
        film="${LOCAL_ROOT}/${stem}"
        [[ -f "${film}/DONE_segmentation.txt" ]] || return 1
        [[ -f "${film}/TrackedCells_${stem}/${stem}_c_0.mp4" ]] || return 1
        frames=$(find "${film}/Frames_${stem}" -maxdepth 1 -type f -name '*.tif' | wc -l | tr -d ' ')
        masks=$(find "${film}/Masks_${stem}" -maxdepth 1 -type f -name '*_seg.tif' | wc -l | tr -d ' ')
        [[ "$frames" -gt 0 && "$frames" -eq "$masks" ]] || return 1
    done < "$SELECTED_LIST"
}

count_markers() {
    local marker_name="$1"
    find "$LOCAL_ROOT" -mindepth 2 -maxdepth 2 -type f -name "$marker_name" \
        \( -path '*/3_BF*/*' -o -path '*/3_FL*/*' \) | wc -l | tr -d ' '
}

echo "[$(timestamp)] M156 unattended pipeline launched."
build_selection

set_state "WAITING_FOR_SEGMENTATION"
while ! validate_segmentation; do
    done_count=$(count_markers "DONE_segmentation.txt")
    echo "[$(timestamp)] Segmentation complete for ${done_count}/33 movies; waiting."
    sleep 30
done
set_state "SEGMENTATION_VALIDATED"

STABILIZE_ARGS=(
    "$PROJECT_ROOT/stabilize_in_place.py"
    --align-by-masks
    --include '3_BF*'
    --include '3_FL*'
    --exclude '3_snap*'
    --max-shift 250
    "$LOCAL_ROOT"
)

set_state "STABILIZATION_DRY_RUN"
"$IMS_PYTHON" "${STABILIZE_ARGS[@]}" --dry-run

set_state "STABILIZING"
"$IMS_PYTHON" "${STABILIZE_ARGS[@]}"
[[ "$(count_markers "DONE_stabilization.txt")" -eq 33 ]] \
    || fail "Stabilization markers did not reach 33."
validate_segmentation || fail "Frame/mask validation failed after stabilization."

set_state "REGENERATING_PREVIEWS"
"$IMS_PYTHON" "$PROJECT_ROOT/regenerate_previews.py" "$LOCAL_ROOT" \
    --include '3_BF*' --include '3_FL*' --exclude '3_snap*'

set_state "UPLOADING_DERIVED_DATA"
"$RSYNC" -az --partial \
    --exclude='._*' \
    --include='3_BF*/' --include='3_BF*/***' \
    --include='3_FL*/' --include='3_FL*/***' \
    --exclude='*' \
    "${LOCAL_ROOT}/" "${HPC_HOST}:${REMOTE_MOVIE_ROOT}/"

set_state "DEPLOYING_HPC_PIPELINE"
"$RSYNC" -az \
    "$PROJECT_ROOT/Cell_tracking_functions.py" \
    "$PROJECT_ROOT/quant_helpers.py" \
    "$PROJECT_ROOT/bf_pattern.py" \
    "$PROJECT_ROOT/Image_quantification_functions.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/generate_cell_ids_1CH.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/generate_cell_jobs.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/one_cell_quantification_1CH.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/tracker_model.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/tracker_dataset.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/ai_tracking_inference.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/submit_array_M156.sh" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/check_slurm_manifest.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/validate_tracking_movie.py" \
    "${HPC_HOST}:${REMOTE_CODE_ROOT}/"
"$RSYNC" -az \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/tracker_checkpoints/" \
    "${HPC_HOST}:${REMOTE_CODE_ROOT}/tracker_checkpoints/"
"$RSYNC" -az \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/tracker_checkpoints_m93_gfp/" \
    "${HPC_HOST}:${REMOTE_CODE_ROOT}/tracker_checkpoints_m93_gfp/"

set_state "SUBMITTING_HPC_JOBS"
ssh_hpc "mkdir -p '${REMOTE_EXP_ROOT}' '${REMOTE_CODE_ROOT}/logs'; \
    if [ ! -f '${REMOTE_EXP_ROOT}/SUBMISSION_COMPLETE' ]; then \
      cd '${REMOTE_CODE_ROOT}'; \
      nohup bash submit_array_M156.sh > logs/submit_M156.log 2>&1 < /dev/null & \
      echo \$! > '${REMOTE_EXP_ROOT}/SUBMISSION_PID'; \
    fi"

submission_health_failures=0
while true; do
    if ssh_hpc "test -f '${REMOTE_EXP_ROOT}/SUBMISSION_COMPLETE'"; then
        break
    fi
    if ssh_hpc "pid=\$(cat '${REMOTE_EXP_ROOT}/SUBMISSION_PID' 2>/dev/null || true); \
        test -n \"\$pid\" && kill -0 \"\$pid\" 2>/dev/null"; then
        submission_health_failures=0
        echo "[$(timestamp)] HPC submission generator is still active."
        sleep 60
    else
        submission_health_failures=$((submission_health_failures + 1))
        echo "[$(timestamp)] Submission health check failed " \
            "(${submission_health_failures}/5); retrying."
        if [[ "$submission_health_failures" -ge 5 ]]; then
            ssh_hpc "tail -n 100 '${REMOTE_CODE_ROOT}/logs/submit_M156.log'" || true
            fail "HPC submission generator stopped before writing its completion marker."
        fi
        sleep 30
    fi
done

set_state "MONITORING_SLURM"
while true; do
    active=$(ssh_hpc "squeue -h -u hsushen -o '%j' | awk '/^M156_/{n++} END{print n+0}'")
    echo "[$(timestamp)] Active M156 SLURM jobs: $active"
    [[ "$active" -eq 0 ]] && break
    sleep 60
done

accounting_attempt=0
while true; do
    set +e
    ssh_hpc "cd '${REMOTE_CODE_ROOT}' && python check_slurm_manifest.py '${REMOTE_EXP_ROOT}/slurm_job_ids.tsv'"
    check_status=$?
    set -e
    if [[ "$check_status" -eq 0 ]]; then
        break
    elif [[ "$check_status" -eq 2 && "$accounting_attempt" -lt 20 ]]; then
        accounting_attempt=$((accounting_attempt + 1))
        echo "[$(timestamp)] Waiting for SLURM accounting records (${accounting_attempt}/20)."
        sleep 30
    else
        fail "One or more SLURM jobs failed or accounting never became complete."
    fi
done

set_state "PULLING_RESULTS"
"$RSYNC" -az --partial --prune-empty-dirs \
    --include='*/' \
    --include='*/TrackedCells_*/' \
    --include='*/TrackedCells_*/*cell_*_data.csv' \
    --include='*/TrackedCells_*/*cell_*_masks.csv' \
    --exclude='*' \
    "${HPC_HOST}:${REMOTE_MOVIE_ROOT}/" "${LOCAL_ROOT}/"

"$RSYNC" -az "${HPC_HOST}:${REMOTE_EXP_ROOT}/slurm_job_ids.tsv" "${LOG_DIR}/"
expected_jobs=$(wc -l < "${LOG_DIR}/slurm_job_ids.tsv" | tr -d ' ')
data_files=$(find "$LOCAL_ROOT" -type f -path '*/TrackedCells_3_*/*cell_*_data.csv' | wc -l | tr -d ' ')
mask_files=$(find "$LOCAL_ROOT" -type f -path '*/TrackedCells_3_*/*cell_*_masks.csv' | wc -l | tr -d ' ')
[[ "$data_files" -eq "$expected_jobs" && "$mask_files" -eq "$expected_jobs" ]] \
    || fail "Result count mismatch: jobs=$expected_jobs data=$data_files masks=$mask_files."

set_state "GENERATING_POPULATION_MOVIES"
(
    cd "$PROJECT_ROOT"
    env KMP_DUPLICATE_LIB_OK=TRUE \
        PATH="/Users/user/miniforge3/envs/cellpose-sam/bin:${PATH}" \
        bash make_all_population_movies.sh "$EXPERIMENT"
)

set_state "BACKING_UP_DERIVED_RESULTS"
mkdir -p "$NAS_ROOT"
"$RSYNC" -az --update --exclude='._*' --exclude='*.ims' \
    --exclude='CellCrops_*' --exclude='PopulationFrames_*' \
    "${LOCAL_ROOT}/" "${NAS_ROOT}/"

set_state "COMPLETE"
echo "[$(timestamp)] M156 unattended pipeline completed successfully."
