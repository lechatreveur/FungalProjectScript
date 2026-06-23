#!/bin/bash
#SBATCH --job-name=gen_cell_jobs_M143
#SBATCH --output=logs/generator_M143_%j.out
#SBATCH --error=logs/generator_M143_%j.err
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8

set -euo pipefail

cd /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose_env

WORKDIRS=(
  "2026_06_03_M143"
)

BASE_EXP_ROOT="/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC"
BASE_MOVIE_ROOT="/RAID1/working/R402/hsushen/FungalProject/Movies"

MOVIES=(
  # "Scd1S573A_1_F0"
  # "Scd1S573A_1_F1"
  # "Scd1S573A_1_F2"
  # "Scd1S573A_2_F0"
  "Scd1S573A_2_F1"
  "Scd1S573A_2_F2"
  "Scd1S573A_3_F0"
  "Scd1S573A_3_F1"
  "Scd1S573A_3_F2"
  "Scd1S573A_4_F0"
  "Scd1S573A_4_F1"
  "Scd1S573A_4_F2"
  "Scd1S573A_F0"
  "Scd1S573A_F1"
  "Scd1S573A_F2"
  "Scd1S573D_1_F0"
  "Scd1S573D_1_F1"
  "Scd1S573D_1_F2"
  "Scd1S573D_2_F0"
  "Scd1S573D_2_F1"
  "Scd1S573D_2_F2"
  "Scd1S573D_3_F0"
  "Scd1S573D_3_F1"
  "Scd1S573D_3_F2"
  "Scd1S573D_4_F0"
  "Scd1S573D_4_F1"
  "Scd1S573D_4_F2"
  "Scd1S573D_F0"
  "Scd1S573D_F1"
  "Scd1S573D_F2"
)

is_bf_movie() {
  local name="$1"
  # Based on pixel intensity analysis, '_1_F' and '_3_F' are brightfield.
  if [[ "$name" == *"_1_F"* || "$name" == *"_3_F"* ]]; then
    return 0
  else
    return 1
  fi
}

for wd in "${WORKDIRS[@]}"; do
  EXP_ROOT="${BASE_EXP_ROOT}/${wd}/"
  MOVIE_ROOT="${BASE_MOVIE_ROOT}/${wd}/"

  echo "=============================="
  echo "📁 WORKDIR:  $wd"
  echo "EXP_ROOT:   $EXP_ROOT"
  echo "MOVIE_ROOT: $MOVIE_ROOT"
  echo "=============================="

  mkdir -p "$EXP_ROOT"

  if [ ! -d "$MOVIE_ROOT" ]; then
    echo "⚠️  Skipping workdir $wd (MOVIE_ROOT not found: $MOVIE_ROOT)"
    continue
  fi

  for file_name in "${MOVIES[@]}"; do
    folder="${MOVIE_ROOT}/${file_name}"

    if [ ! -d "$folder" ]; then
      echo "⏭️  Skipping $file_name (folder not found: $folder)"
      continue
    fi

    if is_bf_movie "$file_name"; then
      track_channel="bf"
    else
      track_channel="gfp"
    fi

    echo "🔎 Processing movie: $file_name  (track_channel=$track_channel)"

    python generate_cell_ids_1CH.py \
      --movie_root "$MOVIE_ROOT" \
      --file_name "$file_name" \
      --output_base_dir "$EXP_ROOT" \
      --z_index 0 \
      --min_area 2000

    cell_ids_path="$EXP_ROOT/$file_name/cell_ids.txt"
    if [ ! -f "$cell_ids_path" ]; then
      echo "⏭️  Skipping $file_name (no cell_ids.txt at $cell_ids_path)"
      continue
    fi

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
  done
done
