#!/usr/bin/env bash
set -euo pipefail

HPC_HOST="hsushen@172.20.97.21"
SRC="/RAID1/working/R402/hsushen/FungalProject/Movies/2026_01_18_M97/"
DST="/Volumes/X10 Pro/Movies/2026_01_18_M97/"

# rsync include/exclude rules:
# - include directories so rsync can walk the tree
# - include only the TrackedCells_<movie>/ path and the two CSV file patterns
# - exclude everything else
rsync -avP \
  --prune-empty-dirs \
  --include='*/' \
  --include='*/TrackedCells_*/' \
  --include='*/TrackedCells_*/*cell_*_data.csv' \
  --include='*/TrackedCells_*/*cell_*_masks.csv' \
  --exclude='*' \
  "${HPC_HOST}:${SRC}" "${DST}"
