#!/bin/bash
#SBATCH --job-name=gen_cell_jobs_single
#SBATCH --output=logs/generator_single_%j.out
#SBATCH --error=logs/generator_single_%j.err
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8

set -euo pipefail

cd /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose_env

EXP_ROOT="/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/2026_06_03_M143/"
MOVIE_ROOT="/RAID1/working/R402/hsushen/FungalProject/Movies/2026_06_03_M143/"
file_name="Scd1S573A_2_F0"
track_channel="gfp"

echo "🔎 Processing movie: $file_name  (track_channel=$track_channel)"

python generate_cell_ids_1CH.py \
  --movie_root "$MOVIE_ROOT" \
  --file_name "$file_name" \
  --output_base_dir "$EXP_ROOT" \
  --z_index 0 \
  --min_area 2000

cell_ids_path="$EXP_ROOT/$file_name/cell_ids.txt"

python generate_cell_jobs.py \
  -w "$EXP_ROOT/$file_name/sb_scripts/" \
  -s /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/one_cell_quantification_1CH.py \
  -i "$cell_ids_path" \
  -e "$MOVIE_ROOT" \
  -f "$file_name" \
  -c "$track_channel" \
  -n 9 \
  -d 10 \
  -z 0 \
  -a 2000 \
  --direction forward
