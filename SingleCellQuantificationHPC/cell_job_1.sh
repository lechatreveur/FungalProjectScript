#!/bin/bash
#SBATCH --job-name=cell_2
#SBATCH --output=/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/2025_05_15_M63//A14_4_F0/sb_scripts/logs/cell_2.out
#SBATCH --error=/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/2025_05_15_M63//A14_4_F0/sb_scripts/logs/cell_2.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4


source ~/.bashrc
conda activate cellpose_env
cd /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/2025_05_15_M63//A14_4_F0/sb_scripts/

python /home/hsushen/FungalProjectScript/SingleCellQuantificationHPC/quantify_cell.py --cell_id 2 --experiment_path /RAID1/working/R402/hsushen/FungalProject/Movies/2025_05_15_M63/ --file_name A14_4_F0 

