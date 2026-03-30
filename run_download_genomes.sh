#!/bin/bash
#SBATCH --job-name=run_download_genomes
#SBATCH --partition=himem
#SBATCH --output=Fv2_output_%j.log
#SBATCH --error=Fv2_error_%j.log

module load Biopython/1.79-foss-2021b


python /home/hsushen/FungalProjectScript/download_genomes.py

