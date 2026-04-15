#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 18:31:47 2025
Modified for Cell Quantification HPC Batch Submission
"""

import os
import time
import subprocess
import argparse
import shutil


# Function to create a SLURM job script for a single cell ID
def create_cell_job_script(job_id, cell_id, working_dir, script_path,experiment_path,file_name,z_index, min_area, channel, update_existing=False, direction='both'):
    logs_dir = os.path.join(working_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    job_script_path = os.path.join(working_dir, f"cell_job_{job_id}.sh")
    update_flag = "--update_existing" if update_existing else ""
    with open(job_script_path, 'w') as job_file:
        job_file.write(f"""#!/bin/bash
#SBATCH --job-name=cell_{cell_id}
#SBATCH --output={logs_dir}/cell_{cell_id}.out
#SBATCH --error={logs_dir}/cell_{cell_id}.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4


source ~/.bashrc
conda activate cellpose_env
cd {working_dir}




python {script_path} --xcorr_select off --xcorr_debug --cell_id {cell_id} --experiment_path {experiment_path} --file_name {file_name} --track_channel {channel} --direction {direction} {update_flag}



""")
#python {script_path} --cell_id {cell_id} --experiment_path {experiment_path} --file_name {file_name} --z_index {z_index} --min_area {min_area} --track_channel gfp {update_flag}
#python {script_path} --cell_id {cell_id} --experiment_path {experiment_path} --file_name {file_name} --z_index {z_index} --min_area {min_area} {update_flag}
    return job_script_path

# Helper to run shell commands and capture output
def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    return process.stdout.read().decode('utf-8').split('\n')

# Limit number of concurrent jobs on SLURM
def check_slurm_queue(max_jobs, delay):
    if shutil.which("squeue") is None:
        return
    check_cmd = "squeue -u $(whoami)"
    while len(run_command(check_cmd)) > max_jobs + 1:
        print(f"Too many jobs in queue. Sleeping {delay} seconds...")
        time.sleep(delay)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and submit SLURM jobs for cell quantification.")
    parser.add_argument('-w', '--working_dir', type=str, required=True, help="Working directory where scripts and logs will be written.")
    parser.add_argument('-s', '--script_path', type=str, required=True, help="Path to quantify_cell.py.")
    parser.add_argument('-i', '--cell_ids_file', type=str, required=True, help="Text file with one cell ID per line.")
    parser.add_argument('-e', '--experiment_path', type=str, required=True, help="Path to an experiment.")
    parser.add_argument('-f', '--file_name', type=str, required=True, help="File name of an .ims file.")
    parser.add_argument('-c', '--channel', type=str, choices=['bf', 'gfp'], default='bf',
                        help='Which single channel to process (controls quantification labels/paths only).')
    parser.add_argument('-n', '--num_jobs', type=int, default=5, help="Max number of jobs running simultaneously.")
    parser.add_argument('-d', '--delay', type=int, default=60, help="Delay in seconds between job queue checks.")
    parser.add_argument('-z', '--z_index', type=int, default=1, help="Z-slice index to load for segmentation.")
    parser.add_argument('-a', '--min_area', type=int, default=2500, help="Minimal cell area.")
    parser.add_argument('--update_existing', action='store_true', help="Retrack cells with masks files.")
    parser.add_argument('--direction', choices=['forward', 'backward', 'both'], default='both', help="Tracking direction.")
    parser.add_argument(
        '--submit',
        choices=['auto', 'slurm', 'none', 'local'],
        default='auto',
        help="Submission mode: auto=use slurm if sbatch exists else none; "
             "slurm=always sbatch; none=only generate scripts; local=run scripts locally with bash."
    )

    
    args = parser.parse_args()

    with open(args.cell_ids_file) as f:
        cell_ids = [int(line.strip()) for line in f if line.strip()]

    os.makedirs(args.working_dir, exist_ok=True)

    for i, cell_id in enumerate(cell_ids, start=1):
        job_script = create_cell_job_script(i, cell_id, args.working_dir, args.script_path, args.experiment_path, args.file_name, args.z_index, args.min_area, args.channel, update_existing=args.update_existing, direction=args.direction)
        mode = args.submit
        has_sbatch = shutil.which("sbatch") is not None
        
        if mode == 'auto':
            mode = 'slurm' if has_sbatch else 'none'
        
        if mode == 'slurm':
            if not has_sbatch:
                raise FileNotFoundError("sbatch not found. Use --submit none or --submit local on non-SLURM systems.")
            subprocess.run(["sbatch", job_script], check=True)
            print(f"Submitted job for cell {cell_id}.")
            check_slurm_queue(args.num_jobs - 1, args.delay)
        
        elif mode == 'local':
            # Run the generated script locally (sequentially)
            # Note: your job script contains #SBATCH lines; bash will ignore them.
            subprocess.run(["bash", job_script], check=True)
            print(f"Ran locally for cell {cell_id}: {job_script}")
        
        else:  # 'none'
            print(f"[gen-only] Wrote: {job_script}")


    #check_slurm_queue(0, args.delay)
    print(f"All {len(cell_ids)} cell quantification jobs have been submitted.")
