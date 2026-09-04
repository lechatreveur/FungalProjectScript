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
def create_cell_job_script(job_id, cell_id, working_dir, script_path,experiment_path,file_name,z_index, min_area, channel, update_existing=False, direction='both', job_name_prefix='', seg_backend='legacy', do_plot=False, make_strips=False):
    logs_dir = os.path.join(working_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    job_script_path = os.path.join(working_dir, f"cell_job_{job_id}.sh")
    update_flag = "--update_existing" if update_existing else ""
    do_plot_flag = "--do_plot" if do_plot else ""
    make_strips_flag = "--make_strips" if make_strips else ""
    with open(job_script_path, 'w') as job_file:
        job_file.write(f"""#!/bin/bash
#SBATCH --job-name={job_name_prefix}cell_{cell_id}
#SBATCH --output={logs_dir}/cell_{cell_id}.out
#SBATCH --error={logs_dir}/cell_{cell_id}.err
#SBATCH --mem=8G
#SBATCH --cpus-per-task=8

set -euo pipefail

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate cellpose-sam
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate cellpose_env
else
    source ~/.bashrc
    conda activate cellpose_env
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}"
cd {working_dir}




python {script_path} --cell_id {cell_id} --experiment_path {experiment_path} --file_name {file_name} --track_channel {channel} --min_area {min_area} --direction {direction} --seg-backend {seg_backend} {update_flag} {do_plot_flag} {make_strips_flag}



""")
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
    parser.add_argument('--direction', choices=['forward', 'backward', 'both'], default='forward', help="Tracking direction.")
    parser.add_argument('--job-name-prefix', default='', help="Prefix for generated SLURM job names.")
    parser.add_argument('--job-id-file', default='', help="Append submitted file_name, cell_id, and SLURM job ID here.")
    parser.add_argument('--seg-backend', choices=['legacy', 'cellpose_overexposed'], default='legacy',
                        help='Segmentation & tracking backend: "legacy" (default) or "cellpose_overexposed".')
    parser.add_argument('--make_strips', action='store_true',
                        help="Pass --make_strips through to one_cell_quantification_1CH.py "
                             "to build a vertical strip PNG per quantified cell.")
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

    failed_cells = []

    for i, cell_id in enumerate(cell_ids, start=1):
        job_script = create_cell_job_script(
            i,
            cell_id,
            args.working_dir,
            args.script_path,
            args.experiment_path,
            args.file_name,
            args.z_index,
            args.min_area,
            args.channel,
            update_existing=args.update_existing,
            direction=args.direction,
            job_name_prefix=args.job_name_prefix,
            seg_backend=args.seg_backend,
            make_strips=args.make_strips,
        )
        mode = args.submit
        has_sbatch = shutil.which("sbatch") is not None
        
        if mode == 'auto':
            mode = 'slurm' if has_sbatch else 'none'
        
        if mode == 'slurm':
            if not has_sbatch:
                raise FileNotFoundError("sbatch not found. Use --submit none or --submit local on non-SLURM systems.")
            # Resilient submit: a single transient slurmctld hiccup (socket
            # timeout / "Slurmctld temporarily unavailable") must not abort a
            # multi-hour generator run the way `check=True` did (job 2389331
            # died at cell 47 of 5_1_N1_FL3_F1 after ~2 days). Retry with
            # backoff; if it still fails, surface sbatch's own stderr, record
            # the cell, and move on. one_cell_quantification_1CH.py early-exits
            # per-cell and submit_array_*.sh skips fully-done films, so a
            # re-run cleanly picks up anything skipped here.
            max_attempts = 5
            job_id = None
            for attempt in range(1, max_attempts + 1):
                result = subprocess.run(["sbatch", "--parsable", job_script], capture_output=True, text=True)
                if result.returncode == 0:
                    job_id = result.stdout.strip().split(";", 1)[0]
                    break
                print(
                    f"WARNING: sbatch failed for cell {cell_id} "
                    f"(attempt {attempt}/{max_attempts}, rc={result.returncode}). "
                    f"stderr={result.stderr.strip()!r} stdout={result.stdout.strip()!r}",
                    flush=True,
                )
                if attempt < max_attempts:
                    time.sleep(min(60, 5 * 2 ** (attempt - 1)))
            if job_id is None:
                print(
                    f"ERROR: giving up on cell {cell_id} after {max_attempts} "
                    f"sbatch attempts; continuing with remaining cells.",
                    flush=True,
                )
                failed_cells.append(cell_id)
                continue
            print(f"Submitted job {job_id} for cell {cell_id}.")
            if args.job_id_file:
                with open(args.job_id_file, "a", encoding="utf-8") as job_ids:
                    job_ids.write(f"{args.file_name}\t{cell_id}\t{job_id}\n")
            time.sleep(0.2)  # Fast submit since masks are precomputed on disk
            check_slurm_queue(args.num_jobs - 1, args.delay)
        
        elif mode == 'local':
            # Run the generated script locally (sequentially)
            # Note: your job script contains #SBATCH lines; bash will ignore them.
            subprocess.run(["bash", job_script], check=True)
            print(f"Ran locally for cell {cell_id}: {job_script}")
        
        else:  # 'none'
            print(f"[gen-only] Wrote: {job_script}")


    #check_slurm_queue(0, args.delay)
    if failed_cells:
        print(
            f"WARNING: {len(failed_cells)}/{len(cell_ids)} cells could not be "
            f"submitted after retries: {failed_cells}. Re-run this script to "
            f"pick them up (already-done cells are skipped)."
        )
    print(f"All {len(cell_ids)} cell quantification jobs have been submitted.")
