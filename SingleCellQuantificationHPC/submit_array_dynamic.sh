#!/bin/bash
#SBATCH --job-name=gen_cell_jobs
#SBATCH --output=logs/generator_%j.out
#SBATCH --error=logs/generator_%j.err
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8

set -euo pipefail

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <experiment_folder_1> [<experiment_folder_2> ...]"
    exit 1
fi

# HPC paths (defaults)
BASE_EXP_ROOT="/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC"
BASE_MOVIE_ROOT="/RAID1/working/R402/hsushen/FungalProject/Movies"

# Detect if running on local Mac or HPC
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "💻 Local Mac detected. Using local movie paths for ID generation."
    BASE_MOVIE_ROOT="/Volumes/X10 Pro/Movies"
    BASE_EXP_ROOT="$(pwd)/SingleCellQuantificationHPC"
fi

cd "$BASE_EXP_ROOT"
mkdir -p logs

# Local check for script execution
if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate cellpose-sam
else
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate cellpose_env
fi

get_track_channel() {
    local name="$1"
    if [[ "$name" == *"YES_Scd1_D_1"* || "$name" == *"YES_Scd1_D_3"* || "$name" == *"YES_Scd1_D_5"* ]]; then
        echo "bf"
    elif [[ "$name" == *"YES_Scd1_D_2"* || "$name" == *"YES_Scd1_D_4"* || "$name" == "YES_Scd1_D_F"* ]]; then
        echo "gfp"
    elif [[ "$name" == *"_BF"* ]]; then
        echo "bf"
    elif [[ "$name" == *"_FL"* ]]; then
        echo "gfp"
    else
        echo "bf"
    fi
}

EXPERIMENTS=("$@")

for wd in "${EXPERIMENTS[@]}"; do
    EXP_ROOT="${BASE_EXP_ROOT}/${wd}"
    MOVIE_ROOT="${BASE_MOVIE_ROOT}/${wd}"
    
    echo "========================================"
    echo "📁 Processing Experiment: $wd"
    echo "========================================"
    
    mkdir -p "$EXP_ROOT"
    
    # Dynamically find movies
    MOVIES=()
    for d in "$MOVIE_ROOT"/*/; do
        if [ -d "$d" ]; then
            movie_name=$(basename "$d")
            MOVIES+=("$movie_name")
        fi
    done

    for file_name in "${MOVIES[@]}"; do
        track_channel=$(get_track_channel "$file_name")
        echo "🔎 Movie: $file_name -> channel: $track_channel"
        
        # Generate Cell IDs
        python generate_cell_ids_1CH.py \
            --movie_root "$MOVIE_ROOT" \
            --file_name "$file_name" \
            --output_base_dir "$EXP_ROOT" \
            --z_index 0 \
            --min_area 2500
            
        cell_ids_path="$EXP_ROOT/$file_name/cell_ids.txt"
        if [ ! -f "$cell_ids_path" ]; then
            echo "⚠️  No cell_ids.txt for $file_name, skipping job generation."
            continue
        fi
        
        # Generate SLURM Jobs
        python generate_cell_jobs.py \
            -w "$EXP_ROOT/$file_name/sb_scripts/" \
            -s /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/one_cell_quantification_1CH.py \
            -i "$cell_ids_path" \
            -e "$MOVIE_ROOT" \
            -f "$file_name" \
            -c "$track_channel" \
            -n 10 \
            -d 10 \
            -z 0 \
            -a 2500 \
            --update_existing
            
        # Submit the master job directly
        echo "🚀 Submitting SLURM array job for $file_name..."
        cd "$EXP_ROOT/$file_name/sb_scripts/"
        sbatch cell_job_1.sh || echo "⚠️  Failed to submit job for $file_name"
        cd "$BASE_EXP_ROOT"
    done
done

echo "✅ All jobs generated and submitted for ${EXPERIMENTS[*]}."
