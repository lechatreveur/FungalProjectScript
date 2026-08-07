#!/bin/bash
# run_unattended_pipeline.sh
# ---------------------------
# Automatically orchestrates remaining steps:
# 1. Waits for PID 22903 (A14_1TP1_BF_F1 tracking) to finish.
# 2. Runs selective SAM2 tracking on A14_1TP2_BF_F1 pending cells.
# 3. Regenerates population frames and calculates suspicious jumps.

set -e

# Target PID of the active tracking process
TARGET_PID=22972
LOG_FILE="/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/unattended_pipeline.log"

echo "=== UNATTENDED PIPELINE ORCHESTRATOR STARTED ===" | tee -a "$LOG_FILE"
echo "Active tracking process PID to wait for: $TARGET_PID" | tee -a "$LOG_FILE"
date | tee -a "$LOG_FILE"

# 1. Wait for target PID to exit
if kill -0 $TARGET_PID 2>/dev/null; then
    echo "Waiting for process $TARGET_PID to complete tracking A14_1TP1_BF_F1..." | tee -a "$LOG_FILE"
    while kill -0 $TARGET_PID 2>/dev/null; do
        sleep 30
    done
    echo "Process $TARGET_PID has completed!" | tee -a "$LOG_FILE"
else
    echo "Warning: PID $TARGET_PID is not running or already exited. Proceeding directly." | tee -a "$LOG_FILE"
fi

date | tee -a "$LOG_FILE"

# 2. Run selective tracking on A14_1TP2_BF_F1
echo "Starting selective SAM2 tracking on A14_1TP2_BF_F1 pending cells..." | tee -a "$LOG_FILE"
conda run -n cellpose-sam --no-capture-output env PYTHONUNBUFFERED=1 python SingleCellQuantificationHPC/track_pending_only.py --experiment 2025_09_17 --sequence F1 --film A14_1TP2_BF_F1 --device mps >> "$LOG_FILE" 2>&1

date | tee -a "$LOG_FILE"

# 3. Run pregeneration and jumps
echo "Starting population frames pregeneration and suspicious jumps calculation..." | tee -a "$LOG_FILE"
conda run -n cellpose-sam --no-capture-output env PYTHONUNBUFFERED=1 python SingleCellQuantificationHPC/pregenerate_and_jumps.py >> "$LOG_FILE" 2>&1

date | tee -a "$LOG_FILE"
echo "=== UNATTENDED PIPELINE COMPLETED SUCCESSFULLY! ===" | tee -a "$LOG_FILE"
