#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Users/user/Documents/Python_Scripts/FungalProjectScript"
SCRIPT_ROOT="${PROJECT_ROOT}/SingleCellQuantificationHPC"
LOCAL_MOVIE_ROOT="/Volumes/X10 Pro/Movies"
PYTHON_EXEC="/Users/user/miniforge3/envs/OCR_env/bin/python"
READY_MARKER="${PROJECT_ROOT}/logs/M156_FL_RETRACK_RESULTS_READY"
DONE_MARKER="${PROJECT_ROOT}/logs/M156_FL_PREGEN_COMPLETE"
FAILED_MARKER="${PROJECT_ROOT}/logs/M156_FL_PREGEN_FAILED"

mkdir -p "${PROJECT_ROOT}/logs"

if [[ -f "$DONE_MARKER" ]]; then
    echo "M156 FL pregeneration already completed: $DONE_MARKER"
    exit 0
fi

trap 'status=$?; if [[ $status -ne 0 ]]; then date "+%Y-%m-%dT%H:%M:%S%z" > "$FAILED_MARKER"; fi' EXIT

echo "Waiting for verified M156 FL retrack pullback marker: $READY_MARKER"
while [[ ! -f "$READY_MARKER" ]]; do
    sleep 60
done

run_sequence() {
    local sequence="$1"
    local films="$2"
    echo "Pregenerating M156 ${sequence}: ${films}"
    env \
        KMP_DUPLICATE_LIB_OK=TRUE \
        PYTHONUNBUFFERED=1 \
        LOCAL_MOVIE_ROOT="$LOCAL_MOVIE_ROOT" \
        "$PYTHON_EXEC" "$SCRIPT_ROOT/pregenerate_and_jumps.py" \
        --experiment 2026_07_16_M156 \
        --sequence "$sequence" \
        --films "$films" \
        --force
}

run_sequence 3_F0 "3_FL1_F0,3_FL2_F0,3_FL3_F0,3_FL4_F0,3_FL5_F0,3_FL6_F0"
run_sequence 3_F1 "3_FL1_F1,3_FL2_F1,3_FL3_F1,3_FL4_F1,3_FL5_F1,3_FL6_F1"
run_sequence 3_F2 "3_FL1_F2,3_FL2_F2,3_FL3_F2,3_FL4_F2,3_FL5_F2,3_FL6_F2"

date "+%Y-%m-%dT%H:%M:%S%z" > "$DONE_MARKER"
echo "M156 FL population-frame and gallery pregeneration complete."
