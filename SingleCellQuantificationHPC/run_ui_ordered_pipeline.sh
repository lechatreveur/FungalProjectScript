#!/bin/bash
# run_ui_ordered_pipeline.sh
# ---------------------------
# Coordinates tracking by UI order:
# 1. Tracks first 50 global cell IDs (indices 0 to 50) for A14_1TP1_BF_F1.
# 2. Tracks first 50 global cell IDs (indices 0 to 50) for A14_1TP2_BF_F1.
# 3. Pregenerates population frames and updates suspicious jumps immediately so the user can start manual correction.
# 4. Continues tracking remaining cells (indices 50 to end) for both films.
# 5. Runs a final pregeneration and jump calculation update.

set -e

LOG_FILE="/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/unattended_ui_pipeline.log"

echo "=== UNATTENDED UI-ORDERED PIPELINE STARTED ===" | tee -a "$LOG_FILE"
date | tee -a "$LOG_FILE"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: FIRST 50 CELLS
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n[PHASE 1] Starting tracking for first 50 global cells (A14_1TP1_BF_F1)..." | tee -a "$LOG_FILE"
conda run -n cellpose-sam --no-capture-output env PYTHONUNBUFFERED=1 python SingleCellQuantificationHPC/track_by_ui_order.py --experiment 2025_09_17 --sequence F1 --film A14_1TP1_BF_F1 --device mps --range-start 0 --range-end 50 >> "$LOG_FILE" 2>&1

echo -e "\n[PHASE 1] Starting tracking for first 50 global cells (A14_1TP2_BF_F1)..." | tee -a "$LOG_FILE"
conda run -n cellpose-sam --no-capture-output env PYTHONUNBUFFERED=1 python SingleCellQuantificationHPC/track_by_ui_order.py --experiment 2025_09_17 --sequence F1 --film A14_1TP2_BF_F1 --device mps --range-start 0 --range-end 50 >> "$LOG_FILE" 2>&1

echo -e "\n[PHASE 1] Updating population frames & suspicious jumps cache for F1..." | tee -a "$LOG_FILE"
python SingleCellQuantificationHPC/pregenerate_and_jumps.py >> "$LOG_FILE" 2>&1

echo -e "\n🎉 === [PHASE 1 COMPLETED] FIRST 50 CELLS READY FOR MANUAL CORRECTION! ===" | tee -a "$LOG_FILE"
date | tee -a "$LOG_FILE"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: REMAINING CELLS
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n[PHASE 2] Starting tracking for remaining cells (50 to end) (A14_1TP1_BF_F1)..." | tee -a "$LOG_FILE"
conda run -n cellpose-sam --no-capture-output env PYTHONUNBUFFERED=1 python SingleCellQuantificationHPC/track_by_ui_order.py --experiment 2025_09_17 --sequence F1 --film A14_1TP1_BF_F1 --device mps --range-start 50 --range-end 9999 >> "$LOG_FILE" 2>&1

echo -e "\n[PHASE 2] Starting tracking for remaining cells (50 to end) (A14_1TP2_BF_F1)..." | tee -a "$LOG_FILE"
conda run -n cellpose-sam --no-capture-output env PYTHONUNBUFFERED=1 python SingleCellQuantificationHPC/track_by_ui_order.py --experiment 2025_09_17 --sequence F1 --film A14_1TP2_BF_F1 --device mps --range-start 50 --range-end 9999 >> "$LOG_FILE" 2>&1

echo -e "\n[PHASE 2] Updating final population frames & suspicious jumps cache for F1..." | tee -a "$LOG_FILE"
python SingleCellQuantificationHPC/pregenerate_and_jumps.py >> "$LOG_FILE" 2>&1

echo -e "\n🎉 === [PHASE 2 COMPLETED] ALL REMAINING CELLS READY! ===" | tee -a "$LOG_FILE"
date | tee -a "$LOG_FILE"
