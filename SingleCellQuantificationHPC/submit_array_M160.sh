#!/bin/bash
#SBATCH --job-name=gen_cell_jobs_M160
#SBATCH --output=logs/generator_M160_%j.out
#SBATCH --error=logs/generator_M160_%j.err
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
# ^ explicit walltime (was previously unset -> ran under the 3-00:00:00
#   partition default). With the resume/skip logic below only the FL3_F1
#   tail + ~13 untouched films remain, so 2 days is generous and safely
#   under the compute-short/himem-short 3-day cap.

set -euo pipefail

cd /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC
mkdir -p logs

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate cellpose-sam
else
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate cellpose_env
fi

WORKDIRS=(
  "2026_08_28_M160"
)

BASE_EXP_ROOT="/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC"
BASE_MOVIE_ROOT="/RAID1/working/R402/hsushen/FungalProject/Movies"

# BF1-BF6 and FL1-FL7, fields F0/F1/F2 -- 42 films total.
# NOTE: '5_1_N1_2_F#' and '5_1_N1_snap_F#' are intentionally excluded per
# 2026-08-31 decision: N1_2/snap are single-timepoint reference stills, not
# tracked time series, so they're not run through per-cell quantification.
MOVIES=(
  "5_1_N1_BF1_F0" "5_1_N1_BF1_F1" "5_1_N1_BF1_F2"
  "5_1_N1_BF2_F0" "5_1_N1_BF2_F1" "5_1_N1_BF2_F2"
  "5_1_N1_BF3_F0" "5_1_N1_BF3_F1" "5_1_N1_BF3_F2"
  "5_1_N1_BF4_F0" "5_1_N1_BF4_F1" "5_1_N1_BF4_F2"
  "5_1_N1_BF5_F0" "5_1_N1_BF5_F1" "5_1_N1_BF5_F2"
  "5_1_N1_BF6_F0" "5_1_N1_BF6_F1" "5_1_N1_BF6_F2"
  "5_1_N1_FL1_F0" "5_1_N1_FL1_F1" "5_1_N1_FL1_F2"
  "5_1_N1_FL2_F0" "5_1_N1_FL2_F1" "5_1_N1_FL2_F2"
  "5_1_N1_FL3_F0" "5_1_N1_FL3_F1" "5_1_N1_FL3_F2"
  "5_1_N1_FL4_F0" "5_1_N1_FL4_F1" "5_1_N1_FL4_F2"
  "5_1_N1_FL5_F0" "5_1_N1_FL5_F1" "5_1_N1_FL5_F2"
  "5_1_N1_FL6_F0" "5_1_N1_FL6_F1" "5_1_N1_FL6_F2"
  "5_1_N1_FL7_F0" "5_1_N1_FL7_F1" "5_1_N1_FL7_F2"
)

is_bf_movie() {
  local name="$1"
  if [[ "$name" == *"_BF"* ]]; then
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

    # Resume support (added 2026-09-02 after the generator hit a walltime
    # limit partway through the MOVIES loop): skip re-submitting a film
    # whose per-cell CSVs are already all present. one_cell_quantification_1CH.py
    # itself also early-exits per-cell if its CSV already exists, so this is
    # just an optimization to avoid re-queuing thousands of already-done cells
    # (which would otherwise burn through the -n/-d throttle for no reason).
    expected_count=$(wc -l < "$cell_ids_path" | tr -d ' ')
    tracked_dir="${MOVIE_ROOT}/${file_name}/TrackedCells_${file_name}"
    done_count=0
    if [ -d "$tracked_dir" ]; then
      done_count=$(find "$tracked_dir" -maxdepth 1 -name 'cell_*_data.csv' -type f | wc -l | tr -d ' ')
    fi
    if [ "$expected_count" -gt 0 ] && [ "$done_count" -ge "$expected_count" ]; then
      echo "✅ Skipping $file_name — already fully quantified ($done_count/$expected_count cells)."
      continue
    fi
    echo "   -> $file_name: $done_count/$expected_count cells done so far, submitting remaining."

    python generate_cell_jobs.py \
      -w "$EXP_ROOT/$file_name/sb_scripts/" \
      -s /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/one_cell_quantification_1CH.py \
      -i "$cell_ids_path" \
      -e "$MOVIE_ROOT" \
      -f "$file_name" \
      -c "$track_channel" \
      -n 30 \
      -d 10 \
      -z 0 \
      -a 2000 \
      --direction forward \
      --make_strips  # per 2026-08-31 request: build the vertical strip PNGs for M160
  done
done
