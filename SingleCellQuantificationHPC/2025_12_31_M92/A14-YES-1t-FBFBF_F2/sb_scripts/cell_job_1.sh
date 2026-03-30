#!/bin/bash
#SBATCH --job-name=cell_1
#SBATCH --output=/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/2025_12_31_M92//A14-YES-1t-FBFBF_F2/sb_scripts/logs/cell_1.out
#SBATCH --error=/Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/2025_12_31_M92//A14-YES-1t-FBFBF_F2/sb_scripts/logs/cell_1.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4


source ~/.bashrc
conda activate cellpose_env
cd /Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/2025_12_31_M92//A14-YES-1t-FBFBF_F2/sb_scripts/




python /Users/user/Documents/Python_Scripts/FungalProjectScript/SingleCellQuantificationHPC/one_cell_quantification_1CH.py --xcorr_select off --xcorr_debug --cell_id 1 --experiment_path /Volumes/Movies/2025_12_31_M92/ --file_name A14-YES-1t-FBFBF_F2 --track_channel gfp



