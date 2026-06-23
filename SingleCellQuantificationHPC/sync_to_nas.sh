#!/bin/bash

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <movie_folder_1> [<movie_folder_2> ...]"
    echo "Example: $0 2026_06_03_M143"
    exit 1
fi

LOCAL_BASE="/Volumes/X10 Pro/Movies"
NAS_BASE="/Volumes/Movies"

# Check if NAS is mounted
if [ ! -d "$NAS_BASE" ] || [ -z "$(ls -A "$NAS_BASE")" ]; then
    echo "❌ Error: NAS is not mounted or empty at $NAS_BASE."
    echo "Please make sure the NAS is mounted (e.g. by opening Finder -> Go -> Connect to Server -> smb://hsushen@R402-NAS._smb._tcp.local/Movies)"
    exit 1
fi

echo "🚀 Syncing Local SSD to NAS..."

for exp in "$@"; do
    if [ ! -d "${LOCAL_BASE}/${exp}" ]; then
        echo "❌ Error: Local directory ${LOCAL_BASE}/${exp} does not exist."
        continue
    fi

    echo "📁 Syncing experiment: $exp"
    
    # We do NOT use --update so that it forces overwrite of older files on the NAS
    rsync -avz --progress \
        --exclude="._*" \
        --include="*/" \
        --include="Frames_**" \
        --include="Masks_**" \
        --include="TrackedCells_**" \
        --include="DONE_segmentation.txt" \
        --exclude="*" \
        "${LOCAL_BASE}/${exp}/" \
        "${NAS_BASE}/${exp}/"
        
done

echo "✅ Sync to NAS complete."
