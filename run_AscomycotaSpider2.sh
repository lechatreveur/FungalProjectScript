#!/bin/bash
#SBATCH --job-name=AscoSpider
#SBATCH --partition=himem
#SBATCH --output=AS_out_%j.log
#SBATCH --error=AS_err_%j.log

# Load required modules
module load Biopython/1.79-foss-2021b

python /home/hsushen/FungalProjectScript/AscomycotaSpider2.py
