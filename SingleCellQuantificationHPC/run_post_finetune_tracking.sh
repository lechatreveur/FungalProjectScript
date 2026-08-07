#!/bin/bash
# run_post_finetune_tracking.sh
# ---------------------------
# Automatically runs selective SAM2 tracking with refined weights and regenerates
# overlays/suspicious jumps for 2026_04_29_M133 (YES_Scd1_D_F1) first, followed by
# 2025_09_17 (F1) and 2026_04_30_M135 (A14_F0).

set -e

LOG_FILE="/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/post_finetune_tracking.log"
py_exec="/Users/user/miniforge3/envs/cellpose-sam/bin/python"

# Clear log file
> "$LOG_FILE"

echo "=== POST FINETUNE TRACKING PIPELINE STARTED ===" | tee -a "$LOG_FILE"
date | tee -a "$LOG_FILE"

# 1. 2026_04_29_M133 Sequence YES_Scd1_D_F1
echo "=== 1/3: Processing 2026_04_29_M133 (YES_Scd1_D_F1) ===" | tee -a "$LOG_FILE"

echo "Generating population overlays and suspicious jumps for M133 YES_Scd1_D_F1..." | tee -a "$LOG_FILE"
$py_exec SingleCellQuantificationHPC/pregenerate_and_jumps.py --experiment 2026_04_29_M133 --sequence YES_Scd1_D_F1 --films YES_Scd1_D_F1,YES_Scd1_D_1_F1,YES_Scd1_D_2_F1,YES_Scd1_D_3_F1,YES_Scd1_D_4_F1,YES_Scd1_D_5_F1 >> "$LOG_FILE" 2>&1


# 2. 2025_09_17 Sequence F1
echo "=== 2/3: Processing 2025_09_17 (F1) ===" | tee -a "$LOG_FILE"
echo "Tracking A14_1TP1_BF_F1..." | tee -a "$LOG_FILE"
$py_exec SingleCellQuantificationHPC/track_pending_only.py --experiment 2025_09_17 --sequence F1 --film A14_1TP1_BF_F1 --device mps >> "$LOG_FILE" 2>&1

echo "Tracking A14_1TP2_BF_F1..." | tee -a "$LOG_FILE"
$py_exec SingleCellQuantificationHPC/track_pending_only.py --experiment 2025_09_17 --sequence F1 --film A14_1TP2_BF_F1 --device mps >> "$LOG_FILE" 2>&1

echo "Generating population overlays and suspicious jumps for 2025_09_17 F1..." | tee -a "$LOG_FILE"
$py_exec SingleCellQuantificationHPC/pregenerate_and_jumps.py --experiment 2025_09_17 --sequence F1 --films A14_1TP1_BF_F1,A14_1TP2_BF_F1 >> "$LOG_FILE" 2>&1


# 3. 2026_04_30_M135 Sequence A14_F0
echo "=== 3/3: Processing 2026_04_30_M135 (A14_F0) ===" | tee -a "$LOG_FILE"
echo "Tracking A14_BF1_F0..." | tee -a "$LOG_FILE"
$py_exec SingleCellQuantificationHPC/track_pending_only.py --experiment 2026_04_30_M135 --sequence A14_F0 --film A14_BF1_F0 --device mps >> "$LOG_FILE" 2>&1

echo "Tracking A14_BF2_F0..." | tee -a "$LOG_FILE"
$py_exec SingleCellQuantificationHPC/track_pending_only.py --experiment 2026_04_30_M135 --sequence A14_F0 --film A14_BF2_F0 --device mps >> "$LOG_FILE" 2>&1

echo "Generating population overlays and suspicious jumps for M135 A14_F0..." | tee -a "$LOG_FILE"
$py_exec SingleCellQuantificationHPC/pregenerate_and_jumps.py --experiment 2026_04_30_M135 --sequence A14_F0 --films A14_BF1_F0,A14_BF2_F0 >> "$LOG_FILE" 2>&1


date | tee -a "$LOG_FILE"
echo "=== POST FINETUNE TRACKING PIPELINE COMPLETED SUCCESSFULLY! ===" | tee -a "$LOG_FILE"
