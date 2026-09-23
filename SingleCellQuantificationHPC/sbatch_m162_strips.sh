#!/bin/bash
#SBATCH --job-name=m162_strips
#SBATCH --partition=compute
#SBATCH --array=0-15
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/RAID1/working/R402/hsushen/FungalProject/Outputs/model_based_dense_tracking/2026_09_09_M162/logs/strips_%A_%a.out
#SBATCH --error=/RAID1/working/R402/hsushen/FungalProject/Outputs/model_based_dense_tracking/2026_09_09_M162/logs/strips_%A_%a.err
#
# Vertical strips for M162, one array task per FL film (P14 stage 4b, P17).
#
# THE CONTRAST SCALE IS PINNED, and that is the whole point of this job.
# M162's signal collapses across FL1-FL4 (pol1_mid 20.9 -> 3.7, pole/cytoplasm
# excess 11.8% -> 1.7%). If each task computed its own "global" scale it would
# pool only its own film, which is per-film scaling by another name, and two
# datapoints of the SAME cell would render equally bright while the underlying
# signal fell by 85%. So the pooled values are computed once over all 16 FL
# films and passed to every task:
#
#     POOLED_SCALE c1max=135.0 c1min=95.0     (13,855,968 sampled pixels)
#
# FL1 alone would give c1max=148, so FL1's brightest pixels clip slightly under
# the pooled scale. That is the price of comparability and is deliberate.
#
# Resumable: existing strips are skipped unless --force (P17).
#
# The logs directory must exist BEFORE submitting — SLURM creates the
# --output file before the script runs. Job 2476229 died instantly on all
# tasks for exactly that reason.

set -u

CODE=/home/hsushen/FungalProjectScript
R=/RAID1/working/R402/hsushen/FungalProject
EXP="$R/Movies/2026_09_09_M162"
BASE="$R/Outputs/model_based_dense_tracking/2026_09_09_M162"

C1MAX=135.0
C1MIN=95.0

FILMS=(
  NeonG_YES_FL1_F0 NeonG_YES_FL1_F1 NeonG_YES_FL1_F2 NeonG_YES_FL1_F3
  NeonG_YES_FL2_F0 NeonG_YES_FL2_F1 NeonG_YES_FL2_F2 NeonG_YES_FL2_F3
  NeonG_YES_FL3_F0 NeonG_YES_FL3_F1 NeonG_YES_FL3_F2 NeonG_YES_FL3_F3
  NeonG_YES_FL4_F0 NeonG_YES_FL4_F1 NeonG_YES_FL4_F2 NeonG_YES_FL4_F3
)
FILM="${FILMS[$SLURM_ARRAY_TASK_ID]}"

echo "host   : $(hostname)"
echo "film   : $FILM"
echo "scale  : c1max=$C1MAX c1min=$C1MIN (pooled over all 16 FL films)"
echo "started: $(date -Is)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cellpose_env

cd "$CODE"
python3 SingleCellQuantificationHPC/build_strips_m162.py \
  --exp "$EXP" \
  --dense "$BASE/dense_masks" \
  --out "$BASE/strips" \
  --films "$FILM" \
  --c1max "$C1MAX" --c1min "$C1MIN"
rc=$?

echo "finished: $(date -Is)  rc=$rc"
exit $rc
