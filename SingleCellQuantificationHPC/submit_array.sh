#!/bin/bash
#SBATCH --job-name=gen_cell_jobs
#SBATCH --output=logs/generator_%j.out
#SBATCH --error=logs/generator_%j.err
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8

set -euo pipefail

cd /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose_env

WORKDIRS=(
  "2026_01_18_M97"
)

BASE_EXP_ROOT="/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC"
BASE_MOVIE_ROOT="/RAID1/working/R402/hsushen/FungalProject/Movies"

MOVIES=(
  "A14-YES-t-0_F0"
  "A14-YES-t-0_F1"
  "A14-YES-t-0_F2"
  "A14-YES-t-1_F0"
  "A14-YES-t-1_F1"
  "A14-YES-t-1_F2"
  "A14-YES-t-2_F0"
  "A14-YES-t-2_F1"
  "A14-YES-t-2_F2"
  "A14-YES-t-3_F0"
  "A14-YES-t-3_F1"
  "A14-YES-t-3_F2"
  "A14-YES-t-4_F0"
  "A14-YES-t-4_F1"
  "A14-YES-t-4_F2"
  "A14-YES-t-5_F0"
  "A14-YES-t-5_F1"
  "A14-YES-t-5_F2"
  "A14-YES-t-6_F0"
  "A14-YES-t-6_F1"
  "A14-YES-t-6_F2"
)

is_bf_movie() {
  local name="$1"
  # BF movies are -0, -2, -4, -6, -8 (underscore avoids matching -20, etc.)
  [[ "$name" == *"-0_"* || "$name" == *"-2_"* || "$name" == *"-4_"* || "$name" == *"-6_"* || "$name" == *"-8_"* ]]
}

#is_bf_movie() {
#  local name="$1"
#  # BF movies are identified by the token "_BF_"
#  [[ "$name" == *"_BF_"* ]]
#}

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
      --min_area 2500

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
      -a 2500 \
      --update_existing
  done
done