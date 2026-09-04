#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <movie_folder_1> [<movie_folder_2> ...]"
    echo "Example: $0 2026_06_03_M143"
    exit 1
fi

REMOTE_USER="hsushen"
REMOTE_HOST="172.20.97.21"
REMOTE_BASE="/RAID1/working/R402/hsushen/FungalProject/Movies"
LOCAL_BASE="/Volumes/X10 Pro/Movies"

echo "🚀 Starting upload of Frames and Masks to HPC..."

for exp in "$@"; do
    echo "📁 Syncing experiment: $exp"
    
    rsync -avz --progress \
        --exclude="._*" \
        --include="*/" \
        --include="Frames_**" \
        --include="Masks_**" \
        --include="TrackedCells_**" \
        --include="DONE_segmentation.txt" \
        --exclude="*" \
        "${LOCAL_BASE}/${exp}/" \
        "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/${exp}/"
        
done

echo "✅ Upload complete."
