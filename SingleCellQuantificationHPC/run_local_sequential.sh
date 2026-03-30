#!/bin/bash
set -euo pipefail

# ---- 1) Go to the folder that contains your python script ----
cd "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/"

# ---- 2) Create a logs folder ----
mkdir -p logs

# ---- 3) CPU threading (tuneable) ----
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

# ---- 4) Activate conda env (Miniforge) ----
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate cellpose-sam
# Alternatively (most robust):
# conda activate "/Users/user/miniforge3/envs/cellpose-sam"

# ---- 5) Working directories ----
WORKDIRS=(
  "/Volumes/Movies/2025_12_31_M92"
  "/Volumes/Movies/2026_01_08_M93"
  "/Volumes/Movies/2026_01_16_M96"
  "/Volumes/Movies/2026_01_18_M97"
)

# ---- 6) Debug: confirm python + cellpose version ----
python -c "import sys, importlib.metadata as m; print('python:', sys.executable); print('cellpose', m.version('cellpose'))"

# ---- 7) Run sequentially ----
for WD in "${WORKDIRS[@]}"; do
  if [[ ! -d "$WD" ]]; then
    echo "ERROR: working_dir not found: $WD"
    exit 2
  fi

  tag="$(basename "$WD")"
  echo "============================================================"
  echo "[$(date)] Starting: $WD"
  echo "Logging to: logs/seg_${tag}.out and logs/seg_${tag}.err"
  echo "Threads: OMP=$OMP_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS"
  echo "============================================================"

  python batch_segment_ims_1CH.py "$WD" > "logs/seg_${tag}.out" 2> "logs/seg_${tag}.err"

  echo "[$(date)] Finished: $WD"
done

echo "All experiments finished."
