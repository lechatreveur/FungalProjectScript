#!/bin/bash
#SBATCH --job-name=gen_cell_jobs_M13x
#SBATCH --output=logs/generator_M13x_%j.out
#SBATCH --error=logs/generator_M13x_%j.err
#SBATCH --mem=4G
#SBATCH --cpus-per-task=8

set -euo pipefail

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

EXPERIMENTS=("2026_04_23_M130" "2026_04_29_M133" "2026_04_30_M135")

for wd in "${EXPERIMENTS[@]}"; do
    EXP_ROOT="${BASE_EXP_ROOT}/${wd}"
    MOVIE_ROOT="${BASE_MOVIE_ROOT}/${wd}"
    
    echo "========================================"
    echo "📁 Processing Experiment: $wd"
    echo "========================================"
    
    mkdir -p "$EXP_ROOT"
    
    # Get movies for this experiment
    if [ "$wd" == "2026_04_23_M130" ]; then
        MOVIES=(A14_BF1_F0 A14_BF1_F1 A14_BF1_F2 A14_BF2_F0 A14_BF2_F1 A14_BF2_F2 A14_FL1_F0 A14_FL1_F1 A14_FL1_F2 A14_FL2_F0 A14_FL2_F1 A14_FL2_F2 A14_test_1_F0 A14_test_1_F1 A14_test_1_F2 A14_test_2_F0 A14_test_2_F1 A14_test_2_F2 A14_test_F0 A14_test_F1 A14_test_F2 Scd1S573D_BF1_F0 Scd1S573D_BF1_F1 Scd1S573D_BF1_F2 Scd1S573D_BF2_F0 Scd1S573D_BF2_F1 Scd1S573D_BF2_F2 Scd1S573D_FL1_F0 Scd1S573D_FL1_F1 Scd1S573D_FL1_F2 Scd1S573D_FL2_F0 Scd1S573D_FL2_F1 Scd1S573D_FL2_F2)
    elif [ "$wd" == "2026_04_29_M133" ]; then
        MOVIES=(YES_Scd1_D_1_F0 YES_Scd1_D_1_F1 YES_Scd1_D_1_F2 YES_Scd1_D_2_F0 YES_Scd1_D_2_F1 YES_Scd1_D_2_F2 YES_Scd1_D_3_F0 YES_Scd1_D_3_F1 YES_Scd1_D_3_F2 YES_Scd1_D_4_F0 YES_Scd1_D_4_F1 YES_Scd1_D_4_F2 YES_Scd1_D_5_F0 YES_Scd1_D_5_F1 YES_Scd1_D_5_F2 YES_Scd1_D_F0 YES_Scd1_D_F1 YES_Scd1_D_F2)
    elif [ "$wd" == "2026_04_30_M135" ]; then
        MOVIES=(A14_BF1_F0 A14_BF1_F1 A14_BF1_F2 A14_BF2_F0 A14_BF2_F1 A14_BF2_F2 A14_BF3_F0 A14_BF3_F1 A14_BF3_F2 A14_FL1_F0 A14_FL1_F1 A14_FL1_F2 A14_FL2_F0 A14_FL2_F1 A14_FL2_F2 A14_FL3_F0 A14_FL3_F1 A14_FL3_F2 A14_test_1_F0 A14_test_1_F1 A14_test_1_F2 A14_test_F0 A14_test_F1 A14_test_F2)
    fi

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
    done
done

echo "✅ All jobs generated for M130, M133, and M135."
