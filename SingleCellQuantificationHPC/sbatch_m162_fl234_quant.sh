#!/bin/bash
#SBATCH --job-name=m162_fl234_quant
#SBATCH --partition=compute
#SBATCH --array=0-11
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/RAID1/working/R402/hsushen/FungalProject/Outputs/model_based_dense_tracking/2026_09_09_M162/logs/quant_%A_%a.out
#SBATCH --error=/RAID1/working/R402/hsushen/FungalProject/Outputs/model_based_dense_tracking/2026_09_09_M162/logs/quant_%A_%a.err
#
# Stage 4 quantification for M162 FL2-FL4 (P14 stage 4, P17 compute placement).
#
# One array task per film, which is the natural unit: stage 4 reads that film's
# frames and computes its intensity scale once, so a per-cell array would redo
# that ~190 times per film.
#
# Resumable by design — no --force. Every completed cell CSV is valid, so a
# timeout or preemption costs only the cells in flight. Do NOT add --force:
# a kill mid-run would then leave a mixture of fresh and stale outputs that a
# later resume accepts silently (see the 2026-09-22 development report).
#
# Moved here from the workstation after three OOM kills on 18 GB of RAM against
# a 2,270-cell job whose data was already on RAID1. P17 exists because of it.

set -u

CODE=/home/hsushen/FungalProjectScript
HPC_ROOT=/RAID1/working/R402/hsushen/FungalProject
EXP="$HPC_ROOT/Movies/2026_09_09_M162"
BASE="$HPC_ROOT/Outputs/model_based_dense_tracking/2026_09_09_M162"

# NOTE: SLURM creates the --output/--error files BEFORE this script runs, so
# the logs directory must already exist at submit time. Creating it here is
# too late — the first submission (job 2476229) failed instantly on all 12
# tasks with no logs at all for exactly that reason. Kept as a safety net.
mkdir -p "$BASE/logs"

FILMS=(
  NeonG_YES_FL2_F0 NeonG_YES_FL2_F1 NeonG_YES_FL2_F2 NeonG_YES_FL2_F3
  NeonG_YES_FL3_F0 NeonG_YES_FL3_F1 NeonG_YES_FL3_F2 NeonG_YES_FL3_F3
  NeonG_YES_FL4_F0 NeonG_YES_FL4_F1 NeonG_YES_FL4_F2 NeonG_YES_FL4_F3
)
FILM="${FILMS[$SLURM_ARRAY_TASK_ID]}"

echo "host      : $(hostname)"
echo "film      : $FILM"
echo "task      : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "started   : $(date -Is)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cellpose_env

cd "$CODE"
python3 SingleCellQuantificationHPC/quantify_model_based_dense.py \
  --exp "$EXP" \
  --dense "$BASE/dense_masks" \
  --out "$BASE/quant" \
  --films "$FILM" \
  --channel FL \
  --workers 8
rc=$?

echo "finished  : $(date -Is)  rc=$rc"
exit $rc
