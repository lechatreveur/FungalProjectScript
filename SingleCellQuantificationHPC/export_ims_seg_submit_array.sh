#!/bin/bash
#SBATCH --job-name=ims_seg_gen_cell_jobs
#SBATCH --output=logs/seg_generator%j.out
#SBATCH --error=logs/seg_generator%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G


# Change to your working directory
cd /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC

# Proper initialization
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cellpose_env

# Run the segmentation script
#python batch_segment_ims.py



EXP_ROOT="/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/2025_06_25/"
MOVIE_ROOT="/RAID1/working/R402/hsushen/FungalProject/Movies/2025_06_25/"




for folder in "$MOVIE_ROOT"/*; do
  if [ -d "$folder" ]; then
    file_name=$(basename "$folder")
    
    
    echo "🔎 Processing movie: $file_name"

    python generate_cell_ids.py \
      --movie_root "$MOVIE_ROOT" \
      --file_name "$file_name" \
      --output_base_dir "$EXP_ROOT" \
      --z_index 1 \
      --min_area 2500
	

    cell_ids_path="$EXP_ROOT/$file_name/cell_ids.txt"
    if [ ! -f "$cell_ids_path" ]; then
      echo "⏭️ Skipping $file_name (no cell_ids.txt)"
      continue
    fi

    python generate_cell_jobs.py \
      -w "$EXP_ROOT/$file_name/sb_scripts/" \
      -s /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/quantify_cell.py \
      -i "$cell_ids_path" \
      -e "$MOVIE_ROOT" \
      -f "$file_name" \
      -n 9 \
      -d 10 \
      -z 1 \
      -a 2500 
      
  fi
done
