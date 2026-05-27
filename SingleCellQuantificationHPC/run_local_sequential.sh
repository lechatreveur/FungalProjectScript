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

# ---- 5) Working directories on the SSD ----
WORKDIRS=(
  "/Volumes/X10 Pro/Movies/2026_04_23_M130"
  "/Volumes/X10 Pro/Movies/2026_04_29_M133"
  "/Volumes/X10 Pro/Movies/2026_04_30_M135"
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
  echo "============================================================"

  # Run the 1-channel segmentation script locally
  python batch_segment_ims_1CH.py "$WD" > "logs/seg_${tag}.out" 2> "logs/seg_${tag}.err"

  echo "[$(date)] Finished: $WD"
done

echo "All experiments finished."
