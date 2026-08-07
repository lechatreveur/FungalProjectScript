#!/bin/bash
# run_phase2.sh
# -------------
# Tracks remaining cells (index 50 to end) for both films and updates overlays.

set -e

LOG_FILE="/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/phase2_pipeline.log"

echo "=== PHASE 2 PIPELINE STARTED ===" | tee -a "$LOG_FILE"
date | tee -a "$LOG_FILE"

# 1. Track remaining cells in A14_1TP1_BF_F1
echo -e "\n[PHASE 2] Starting tracking for remaining cells (50 to end) (A14_1TP1_BF_F1)..." | tee -a "$LOG_FILE"
conda run -n cellpose-sam --no-capture-output env PYTHONUNBUFFERED=1 python SingleCellQuantificationHPC/track_by_ui_order.py --experiment 2025_09_17 --sequence F1 --film A14_1TP1_BF_F1 --device mps --range-start 50 --range-end 9999 >> "$LOG_FILE" 2>&1

# 2. Track remaining cells in A14_1TP2_BF_F1
echo -e "\n[PHASE 2] Starting tracking for remaining cells (50 to end) (A14_1TP2_BF_F1)..." | tee -a "$LOG_FILE"
conda run -n cellpose-sam --no-capture-output env PYTHONUNBUFFERED=1 python SingleCellQuantificationHPC/track_by_ui_order.py --experiment 2025_09_17 --sequence F1 --film A14_1TP2_BF_F1 --device mps --range-start 50 --range-end 9999 >> "$LOG_FILE" 2>&1

# 3. Update overlays and jumps mathematically
echo -e "\n[PHASE 2] Updating final population frames & suspicious jumps cache for F1..." | tee -a "$LOG_FILE"
python SingleCellQuantificationHPC/pregenerate_and_jumps.py >> "$LOG_FILE" 2>&1

echo -e "\n🎉 === [PHASE 2 COMPLETED] ALL REMAINING CELLS READY! === 🎉" | tee -a "$LOG_FILE"
date | tee -a "$LOG_FILE"
