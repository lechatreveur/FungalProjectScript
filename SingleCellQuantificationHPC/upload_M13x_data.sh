#!/bin/bash

# Configuration
REMOTE_USER="hsushen"
REMOTE_HOST="172.20.97.21"
REMOTE_BASE="/RAID1/working/R402/hsushen/FungalProject/Movies"
LOCAL_BASE="/Volumes/X10 Pro/Movies"

EXPERIMENTS=("2026_04_23_M130" "2026_04_29_M133" "2026_04_30_M135")

echo "🚀 Starting upload of Frames and Masks to HPC..."

for exp in "${EXPERIMENTS[@]}"; do
    echo "📁 Syncing experiment: $exp"
    
    # We want to sync all subdirectories containing Frames_ or Masks_
    # Using --include and --exclude patterns to target exactly what we need
    # This avoids syncing large raw files if they are already there
    
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
