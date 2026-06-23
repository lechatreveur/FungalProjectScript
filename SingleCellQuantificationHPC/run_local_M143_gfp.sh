#!/bin/bash
set -euo pipefail

# 1) Go to the directory containing our scripts
cd "/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC"

# 2) Create local logs directory
mkdir -p logs

# 3) Activate conda environment
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate cellpose-sam

# 4) Define directories
MOVIE_ROOT="/Volumes/X10 Pro/Movies/2026_06_03_M143"
EXP_ROOT="/Volumes/X10 Pro/Movies/2026_06_03_M143"

# 18 GFP movies of M143
MOVIES=(
  "Scd1S573A_2_F0"
  "Scd1S573A_2_F1"
  "Scd1S573A_2_F2"
  "Scd1S573A_4_F0"
  "Scd1S573A_4_F1"
  "Scd1S573A_4_F2"
  "Scd1S573A_F0"
  "Scd1S573A_F1"
  "Scd1S573A_F2"
  "Scd1S573D_2_F0"
  "Scd1S573D_2_F1"
  "Scd1S573D_2_F2"
  "Scd1S573D_4_F0"
  "Scd1S573D_4_F1"
  "Scd1S573D_4_F2"
  "Scd1S573D_F0"
  "Scd1S573D_F1"
  "Scd1S573D_F2"
)

# Optional argument to run a single test cell for verification
TEST_MODE=false
if [[ "${1:-}" == "--test" ]]; then
  TEST_MODE=true
  echo "🧪 TEST MODE: Will only run a single cell for the first movie."
fi

for file_name in "${MOVIES[@]}"; do
  folder="${MOVIE_ROOT}/${file_name}"
  if [ ! -d "$folder" ]; then
    echo "⚠️ Skipping $file_name (folder not found at $folder)"
    continue
  fi

  echo "============================================================"
  echo "[$(date)] Starting Movie: $file_name"
  echo "============================================================"

  # Generate cell IDs
  python generate_cell_ids_1CH.py \
    --movie_root "$MOVIE_ROOT" \
    --file_name "$file_name" \
    --output_base_dir "$EXP_ROOT" \
    --z_index 0 \
    --min_area 2000

  cell_ids_path="$EXP_ROOT/$file_name/cell_ids.txt"
  if [ ! -f "$cell_ids_path" ]; then
    echo "⏭️ Skipping $file_name (no cell_ids.txt generated)"
    continue
  fi

  # Clean old bad tracking data if not in test mode
  tracked_cells_dir="$EXP_ROOT/$file_name/TrackedCells_$file_name"
  if [ "$TEST_MODE" = false ] && [ -d "$tracked_cells_dir" ]; then
    echo "🧹 Cleaning existing tracking folder: $tracked_cells_dir"
    rm -rf "$tracked_cells_dir"
  fi
  mkdir -p "$tracked_cells_dir"

  # Run sequentially for all cell IDs
  while read -r cell_id; do
    [[ -z "$cell_id" ]] && continue
    echo "➡️ Processing cell_id=$cell_id in $file_name"
    
    python one_cell_quantification_1CH.py \
      --cell_id "$cell_id" \
      --track_channel gfp \
      --direction both \
      --experiment_path "$EXP_ROOT" \
      --file_name "$file_name"
      
    if [ "$TEST_MODE" = true ]; then
      echo "🧪 Test mode single-cell run complete. Exiting successfully."
      exit 0
    fi

  done < "$cell_ids_path"

  echo "🎉 Finished Movie: $file_name"
done

echo "✅ All M143 GFP tracking completed successfully!"
