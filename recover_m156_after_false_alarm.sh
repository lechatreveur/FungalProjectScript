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
RSYNC="/opt/homebrew/bin/rsync"
LOG_DIR="${PROJECT_ROOT}/logs"
LOG_FILE="${LOG_DIR}/m156_recovery.log"
STATE_FILE="${LOCAL_ROOT}/M156_PIPELINE_STATE.txt"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() {
    date '+%Y-%m-%d %H:%M:%S'
}

set_state() {
    printf '%s\t%s\n' "$(timestamp)" "$1" | tee "$STATE_FILE"
}

fail() {
    set_state "RECOVERY_FAILED: $1"
    exit 1
}

ssh_hpc() {
    ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
        "$HPC_HOST" "$@"
}

wait_for_process_marker() {
    local marker="$1"
    local pid_file="$2"
    local label="$3"
    local health_failures=0
    while true; do
        if ssh_hpc "test -f '$marker'"; then
            return 0
        fi
        if ssh_hpc "pid=\$(cat '$pid_file' 2>/dev/null || true); \
            test -n \"\$pid\" && kill -0 \"\$pid\" 2>/dev/null"; then
            health_failures=0
            echo "[$(timestamp)] ${label} is active."
        else
            health_failures=$((health_failures + 1))
            echo "[$(timestamp)] ${label} health check failed " \
                "(${health_failures}/5)."
            if [[ "$health_failures" -ge 5 ]]; then
                fail "${label} stopped before writing ${marker}."
            fi
        fi
        sleep 60
    done
}

wait_for_jobs() {
    local job_pattern="$1"
    local label="$2"
    local active
    while true; do
        if active=$(ssh_hpc \
            "squeue -h -u hsushen -o '%j' | awk '/${job_pattern}/{n++} END{print n+0}'"); then
            echo "[$(timestamp)] Active ${label} jobs: ${active}"
            [[ "$active" -eq 0 ]] && return 0
        else
            echo "[$(timestamp)] Could not query ${label} jobs; retrying."
        fi
        sleep 60
    done
}

echo "[$(timestamp)] M156 recovery watcher launched."
set_state "RECOVERY_WAITING_FOR_ORIGINAL_SUBMISSION"
wait_for_process_marker \
    "${REMOTE_EXP_ROOT}/SUBMISSION_COMPLETE" \
    "${REMOTE_EXP_ROOT}/SUBMISSION_PID" \
    "Original submission generator"

set_state "RECOVERY_WAITING_FOR_ORIGINAL_JOBS"
wait_for_jobs '^M156_cell_' "original M156"

set_state "RECOVERY_DEPLOYING_FIXES"
"$RSYNC" -az \
    "$PROJECT_ROOT/Image_quantification_functions.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/generate_cell_ids_1CH.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/check_slurm_manifest.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/audit_manifest_cell_ids.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/apply_manifest_cell_id_repair.py" \
    "$PROJECT_ROOT/SingleCellQuantificationHPC/submit_repair_M156.sh" \
    "${HPC_HOST}:${REMOTE_CODE_ROOT}/"

set_state "RECOVERY_AUDITING_CELL_IDS"
set +e
ssh_hpc "source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"; \
    conda activate cellpose_env; \
    python '${REMOTE_CODE_ROOT}/audit_manifest_cell_ids.py' \
      --movie-root '${REMOTE_MOVIE_ROOT}' \
      --experiment-root '${REMOTE_EXP_ROOT}' \
      --manifest '${REMOTE_EXP_ROOT}/slurm_job_ids.tsv' \
      --min-area 2000 \
      > '${REMOTE_EXP_ROOT}/cell_id_audit.final.json'"
audit_status=$?
set -e
[[ "$audit_status" -eq 0 || "$audit_status" -eq 1 ]] \
    || fail "Cell-ID audit could not run (exit ${audit_status})."

set_state "RECOVERY_APPLYING_CELL_ID_REPAIR"
ssh_hpc "source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"; \
    conda activate cellpose_env; \
    python '${REMOTE_CODE_ROOT}/apply_manifest_cell_id_repair.py' \
      --movie-root '${REMOTE_MOVIE_ROOT}' \
      --experiment-root '${REMOTE_EXP_ROOT}' \
      --manifest '${REMOTE_EXP_ROOT}/slurm_job_ids.tsv' \
      --audit-json '${REMOTE_EXP_ROOT}/cell_id_audit.final.json' \
      --apply"

set_state "RECOVERY_SUBMITTING_CORRECTED_CELLS"
ssh_hpc "nohup bash '${REMOTE_CODE_ROOT}/submit_repair_M156.sh' \
    > '${REMOTE_CODE_ROOT}/logs/repair_M156.log' 2>&1 < /dev/null & \
    echo \$! > '${REMOTE_EXP_ROOT}/REPAIR_PID'"
wait_for_process_marker \
    "${REMOTE_EXP_ROOT}/REPAIR_SUBMISSION_COMPLETE" \
    "${REMOTE_EXP_ROOT}/REPAIR_PID" \
    "Repair submission generator"

set_state "RECOVERY_MONITORING_REPAIR_JOBS"
wait_for_jobs '^M156_REPAIR_' "M156 repair"

set_state "RECOVERY_CHECKING_ACCOUNTING"
accounting_attempt=0
while true; do
    set +e
    ssh_hpc "source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"; \
        conda activate cellpose_env; \
        python '${REMOTE_CODE_ROOT}/check_slurm_manifest.py' \
          '${REMOTE_EXP_ROOT}/slurm_job_ids.tsv'"
    check_status=$?
    set -e
    if [[ "$check_status" -eq 0 ]]; then
        break
    elif [[ "$check_status" -eq 2 && "$accounting_attempt" -lt 20 ]]; then
        accounting_attempt=$((accounting_attempt + 1))
        sleep 30
    else
        fail "Canonical repaired manifest contains failed jobs."
    fi
done

set_state "RECOVERY_PULLING_RESULTS"
"$RSYNC" -az --partial --prune-empty-dirs \
    --include='*/' \
    --include='*/TrackedCells_*/' \
    --include='*/TrackedCells_*/*cell_*_data.csv' \
    --include='*/TrackedCells_*/*cell_*_masks.csv' \
    --exclude='*' \
    "${HPC_HOST}:${REMOTE_MOVIE_ROOT}/" "${LOCAL_ROOT}/"

"$RSYNC" -az \
    "${HPC_HOST}:${REMOTE_EXP_ROOT}/slurm_job_ids.tsv" \
    "${LOG_DIR}/"
expected_jobs=$(wc -l < "${LOG_DIR}/slurm_job_ids.tsv" | tr -d ' ')
data_files=$(find "$LOCAL_ROOT" -type f \
    -path '*/TrackedCells_3_*/*cell_*_data.csv' | wc -l | tr -d ' ')
mask_files=$(find "$LOCAL_ROOT" -type f \
    -path '*/TrackedCells_3_*/*cell_*_masks.csv' | wc -l | tr -d ' ')
[[ "$data_files" -eq "$expected_jobs" && "$mask_files" -eq "$expected_jobs" ]] \
    || fail "Result count mismatch: jobs=$expected_jobs data=$data_files masks=$mask_files."

set_state "RECOVERY_GENERATING_POPULATION_MOVIES"
(
    cd "$PROJECT_ROOT"
    env KMP_DUPLICATE_LIB_OK=TRUE \
        PATH="/Users/user/miniforge3/envs/cellpose-sam/bin:${PATH}" \
        bash make_all_population_movies.sh "$EXPERIMENT"
)

set_state "RECOVERY_BACKING_UP_DERIVED_RESULTS"
mkdir -p "$NAS_ROOT"
"$RSYNC" -az --update --exclude='._*' --exclude='*.ims' \
    --exclude='CellCrops_*' --exclude='PopulationFrames_*' \
    "${LOCAL_ROOT}/" "${NAS_ROOT}/"

set_state "COMPLETE"
echo "[$(timestamp)] M156 recovery completed successfully."
