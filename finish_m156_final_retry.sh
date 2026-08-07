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
LOG_FILE="${LOG_DIR}/m156_final_retry.log"
STATE_FILE="${LOCAL_ROOT}/M156_PIPELINE_STATE.txt"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
set_state() { printf '%s\t%s\n' "$(timestamp)" "$1" | tee "$STATE_FILE"; }
fail() { set_state "FINAL_RETRY_FAILED: $1"; exit 1; }
ssh_hpc() {
    ssh -o BatchMode=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
        "$HPC_HOST" "$@"
}

set_state "FINAL_RETRY_WAITING_FOR_SUBMISSION"
health_failures=0
while true; do
    if ssh_hpc "test -f '${REMOTE_EXP_ROOT}/FAILED_RETRY_SUBMISSION_COMPLETE'"; then
        break
    fi
    if ssh_hpc "pid=\$(cat '${REMOTE_EXP_ROOT}/FAILED_RETRY_PID' 2>/dev/null || true); \
        test -n \"\$pid\" && kill -0 \"\$pid\" 2>/dev/null"; then
        health_failures=0
        echo "[$(timestamp)] Final retry submission generator is active."
    else
        health_failures=$((health_failures + 1))
        [[ "$health_failures" -lt 5 ]] \
            || fail "Final retry submission generator stopped."
    fi
    sleep 60
done

set_state "FINAL_RETRY_MONITORING_JOBS"
while true; do
    if active=$(ssh_hpc \
        "squeue -h -u hsushen -o '%j' | awk '/^M156_FINALRETRY_/{n++} END{print n+0}'"); then
        echo "[$(timestamp)] Active final-retry jobs: $active"
        [[ "$active" -eq 0 ]] && break
    fi
    sleep 60
done

set_state "FINAL_RETRY_CHECKING_ACCOUNTING"
accounting_attempt=0
while true; do
    set +e
    ssh_hpc "source \"\$HOME/miniconda3/etc/profile.d/conda.sh\"; \
        conda activate cellpose_env; \
        python '${REMOTE_CODE_ROOT}/check_slurm_manifest.py' \
          '${REMOTE_EXP_ROOT}/slurm_job_ids.tsv'"
    status=$?
    set -e
    if [[ "$status" -eq 0 ]]; then
        break
    elif [[ "$status" -eq 2 && "$accounting_attempt" -lt 20 ]]; then
        accounting_attempt=$((accounting_attempt + 1))
        sleep 30
    else
        fail "Canonical manifest still contains failed jobs."
    fi
done

set_state "FINAL_RETRY_PULLING_RESULTS"
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

set_state "FINAL_RETRY_GENERATING_POPULATION_MOVIES"
(
    cd "$PROJECT_ROOT"
    env KMP_DUPLICATE_LIB_OK=TRUE \
        PATH="/Users/user/miniforge3/envs/cellpose-sam/bin:${PATH}" \
        bash make_all_population_movies.sh "$EXPERIMENT"
)

set_state "FINAL_RETRY_BACKING_UP_TO_NAS"
mkdir -p "$NAS_ROOT"
"$RSYNC" -az --update --exclude='._*' --exclude='*.ims' \
    --exclude='CellCrops_*' --exclude='PopulationFrames_*' \
    "${LOCAL_ROOT}/" "${NAS_ROOT}/"

set_state "COMPLETE"
echo "[$(timestamp)] M156 final retry and downstream workflow completed."
