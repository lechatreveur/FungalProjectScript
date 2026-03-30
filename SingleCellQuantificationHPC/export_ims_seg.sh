#!/bin/bash
#SBATCH --job-name=ims_seg
#SBATCH --output=/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/logs/seg_%A_%a.out
#SBATCH --error=/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/logs/seg_%A_%a.err
#SBATCH --array=0-3
#SBATCH --cpus-per-task=24
#SBATCH --mem=40G
#SBATCH --time=72:00:00

set -euo pipefail

cd /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC
mkdir -p logs  # still useful, but now output paths are absolute so SLURM won't fail

# threads (optional but recommended on CPU)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose_env

# define the working directories
WORKDIRS=(
  "/RAID1/working/R402/hsushen/FungalProject/Movies/2025_12_31_M92"
  "/RAID1/working/R402/hsushen/FungalProject/Movies/2026_01_08_M93"
  "/RAID1/working/R402/hsushen/FungalProject/Movies/2026_01_16_M96"
  "/RAID1/working/R402/hsushen/FungalProject/Movies/2026_01_18_M97"
)

# guard
if [[ $SLURM_ARRAY_TASK_ID -ge ${#WORKDIRS[@]} ]]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range (n=${#WORKDIRS[@]})"
  exit 2
fi

WD="${WORKDIRS[$SLURM_ARRAY_TASK_ID]}"

echo "[$(date)] SLURM_JOB_ID=$SLURM_JOB_ID TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Working dir: $WD"

# debug (optional, but very helpful)
python -c "import sys, importlib.metadata as m; print('python:', sys.executable); print('cellpose', m.version('cellpose'))"

python batch_segment_ims_1CH.py "$WD"
