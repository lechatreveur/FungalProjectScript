#!/bin/bash
set -euo pipefail

cd /Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC

EXP_ROOT="/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/2025_12_31_M92/"
MOVIE_ROOT="/Volumes/Movies/2025_12_31_M92/"
TARGETS=("A14-YES-1t-FBFBF_F2")

mkdir -p "$EXP_ROOT"

for file_name in "${TARGETS[@]}"; do
  folder="${MOVIE_ROOT%/}/$file_name"
  if [ ! -d "$folder" ]; then
    echo "⚠️  Skipping $file_name (folder not found at $folder)"
    continue
  fi

  echo "🔎 Processing movie: $file_name"

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

  echo "🏃 Running all cells locally for: $file_name"

  while read -r cell_id; do
    [[ -z "$cell_id" ]] && continue
    echo "➡️  cell_id=$cell_id"

    python one_cell_quantification_1CH.py \
      --cell_id "$cell_id" \
      --track_channel gfp \
      --direction both \
      --experiment_path "$MOVIE_ROOT" \
      --file_name "$file_name" \
      --xcorr_select off \
      --xcorr_debug

  done < "$cell_ids_path"

done
