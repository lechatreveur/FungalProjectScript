#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <movie_folder_1> [<movie_folder_2> ...]"
    echo "Example: $0 2026_04_23_M130 2026_04_29_M133"
    exit 1
fi

HPC_HOST="hsushen@172.20.97.21"

for MOVIE in "$@"; do
    echo "=================================================="
    echo "Pulling data for $MOVIE..."
    echo "=================================================="
    
    SRC="/RAID1/working/R402/hsushen/FungalProject/Movies/${MOVIE}/"
    DST="/Volumes/X10 Pro/Movies/${MOVIE}/"

    rsync -avP \
      --prune-empty-dirs \
      --include='*/' \
      --include='*/TrackedCells_*/' \
      --include='*/TrackedCells_*/*cell_*_data.csv' \
      --include='*/TrackedCells_*/*cell_*_masks.csv' \
      --exclude='*' \
      "${HPC_HOST}:${SRC}" "${DST}"
done
