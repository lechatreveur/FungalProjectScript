#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the SLURM array for the M160 model-based pipeline (stages 3 and 4).

One array task per FLUORESCENCE FILM, 21 in total. A film is the right unit
here because both stages cache per film: stage 3 reads that film's `_seg.tif`
series, stage 4 reads its frames and computes the film intensity scale once. A
per-cell array, which is what the older quantification path uses, would redo all
of that 6,246 times.

Cohort: every cell whose QC status is good, corrected or unreviewed, in **every
film it appears in** — taken from `sequence_linkage.json`, not from the work
queue's `sus_pos`. See the comment in `run_model_based_dense_tracking.build_tasks`
for why that distinction cost a rebuild.

    python SingleCellQuantificationHPC/generate_model_based_jobs_m160.py
    # then, on the HPC:
    sbatch <out>/model_based_m160_array.sh

Both stages are resumable, so a timed-out or requeued task picks up where it
stopped. Stage 4 additionally re-runs any cell whose output is not a complete
101-frame table.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXP_NAME = "2026_08_28_M160"
SEQS = ["5_1_N1_F0", "5_1_N1_F1", "5_1_N1_F2"]

# HPC layout, per PIPELINE_PROTOCOL.md and submit_array_M160.sh
HPC_CODE = "/home/hsushen/FungalProjectScript/SingleCellQuantificationHPC"
HPC_MOVIES = f"/RAID1/working/R402/hsushen/FungalProject/Movies/{EXP_NAME}"
HPC_OUTPUTS = (f"/RAID1/working/R402/hsushen/FungalProject/Outputs/"
               f"model_based_dense_tracking/{EXP_NAME}")
HPC_SCRATCH = f"{HPC_CODE}/scratch"

LOCAL_EXP = Path("/Volumes/X10 Pro/Movies") / EXP_NAME
DEFAULT_OUT = (Path("/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking")
               / EXP_NAME / "hpc_jobs")

ARRAY = """#!/bin/bash
#SBATCH --job-name=mbdt_m160
#SBATCH --output={logs}/mbdt_%A_%a.out
#SBATCH --error={logs}/mbdt_%A_%a.err
#SBATCH --array=0-{last}%{throttle}
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00

# Model-based dense tracking (stage 3) + polarity quantification (stage 4)
# for M160. One task per fluorescence film; {ncells} (film, cell) pairs total.
#
# Both stages are resumable, so requeueing a timed-out task is safe.

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
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONPATH="{code}:{repo}${{PYTHONPATH:+:$PYTHONPATH}}"

cd {code}

FILMS=({films})
FILM="${{FILMS[$SLURM_ARRAY_TASK_ID]}}"
echo "=== task $SLURM_ARRAY_TASK_ID : $FILM ==="
date

echo "--- stage 3: model-based dense tracking ---"
python -u run_model_based_dense_tracking.py \\
    --exp "{movies}" \\
    --out "{outputs}/dense_masks" \\
    --work-queue "{scratch}/work_queue_all_statuses.csv" \\
    --seed "{scratch}/bayesian_tracker_benchmark_307.csv" \\
    --status good corrected unreviewed \\
    --channel FL \\
    --films "$FILM"

echo "--- stage 4: polarity quantification ---"
python -u quantify_model_based_dense.py \\
    --exp "{movies}" \\
    --dense "{outputs}/dense_masks" \\
    --out "{outputs}/quant" \\
    --channel FL \\
    --films "$FILM" \\
    --workers 4

echo "--- stage 4b: vertical strips ---"
python -u build_strips_m160.py \\
    --exp "{movies}" \\
    --dense "{outputs}/dense_masks" \\
    --out "{outputs}/strips" \\
    --films "$FILM"

date
echo "=== done $FILM ==="
"""

AFTER = """#!/bin/bash
#SBATCH --job-name=mbdt_m160_post
#SBATCH --output={logs}/post_%j.out
#SBATCH --error={logs}/post_%j.err
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# Stages 5 and 6, once every film's array task has finished.
# Submit with a dependency:  sbatch --dependency=afterok:<ARRAY_JOB_ID> {name}

set -euo pipefail

if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
    conda activate cellpose-sam
else
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate cellpose_env
fi
export OMP_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONPATH="{code}:{repo}${{PYTHONPATH:+:$PYTHONPATH}}"
cd {code}

# Stages 5 and 6 need scikit-learn, umap-learn and the SingleCellDataAnalysis
# package, none of which are on the cluster as of 2026-09-15 (cellpose_env has
# torch, skimage, tifffile, pandas and scipy only). Fail here with a clear
# message rather than part-way through a long job.
python - <<'PYCHK'
import importlib, sys
missing = [m for m in ("sklearn", "umap") if not importlib.util.find_spec(m)]
if missing:
    sys.exit("stages 5-6 need " + ", ".join(missing) +
             "; install them or run these stages on the workstation instead")
PYCHK

echo "--- stage 5: feature extraction ---"
python -u build_features_m160.py \\
    --exp "{movies}" --quant "{outputs}/quant" --out "{outputs}/features"

echo "--- stage 6a: train the M160 autoencoder ---"
python -u train_fc_ae_m160.py \\
    --features-dir "{outputs}/features" --model "{outputs}/fc_ae_3d_m160.pth"

echo "--- stage 6b: build the explorer ---"
python -u build_umap_html_m160.py \\
    --features-dir "{outputs}/features" --model "{outputs}/fc_ae_3d_m160.pth" \\
    --exp "{movies}" --strips "{outputs}/strips" \\
    --out "{outputs}/umap_m160_standalone.html"
"""


def fl_films(exp_dir):
    linkage = json.load(open(Path(exp_dir) / "sequence_linkage.json"))
    out = []
    for seq in SEQS:
        if seq in linkage:
            out += [f for f in linkage[seq]["films"] if "FL" in f]
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", type=Path, default=LOCAL_EXP,
                    help="local experiment dir, read only to list the films")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--throttle", type=int, default=7,
                    help="max array tasks running at once")
    ap.add_argument("--hpc-code", default=HPC_CODE)
    ap.add_argument("--hpc-movies", default=HPC_MOVIES)
    ap.add_argument("--hpc-outputs", default=HPC_OUTPUTS)
    ap.add_argument("--hpc-scratch", default=HPC_SCRATCH)
    a = ap.parse_args()

    films = fl_films(a.exp)
    if not films:
        raise SystemExit("no fluorescence films found in sequence_linkage.json")

    # cohort size, for the header only
    ncells = "?"
    try:
        import run_model_based_dense_tracking as R
        tasks = R.build_tasks(a.exp, R.DEFAULT_WQ, R.DEFAULT_SEED,
                              ["good", "corrected", "unreviewed"], "FL")
        ncells = len(tasks)
    except Exception as exc:
        print(f"(could not size the cohort locally: {exc})")

    a.out.mkdir(parents=True, exist_ok=True)
    logs = f"{a.hpc_code}/logs"
    repo = str(Path(a.hpc_code).parent)

    array_path = a.out / "model_based_m160_array.sh"
    array_path.write_text(ARRAY.format(
        logs=logs, last=len(films) - 1, throttle=a.throttle, ncells=ncells,
        code=a.hpc_code, repo=repo, movies=a.hpc_movies,
        outputs=a.hpc_outputs, scratch=a.hpc_scratch,
        films=" ".join(f'"{f}"' for f in films)))
    os.chmod(array_path, 0o755)

    post_path = a.out / "model_based_m160_post.sh"
    post_path.write_text(AFTER.format(
        logs=logs, code=a.hpc_code, repo=repo, movies=a.hpc_movies,
        outputs=a.hpc_outputs, name=post_path.name))
    os.chmod(post_path, 0o755)

    (a.out / "films.txt").write_text("\n".join(films) + "\n")

    readme = a.out / "README.md"
    readme.write_text(f"""# M160 model-based pipeline on the HPC

{len(films)} fluorescence films, {ncells} (film, cell) pairs, statuses good /
corrected / unreviewed, every film each cell appears in.

## Sync

    rsync -av {a.out}/ hsushen@<hpc>:{a.hpc_code}/sb_scripts_mbdt_m160/
    # the work queue and ABBT benchmark must also be present under
    #   {a.hpc_scratch}/

## Submit

    cd {a.hpc_code}/sb_scripts_mbdt_m160
    sbatch model_based_m160_array.sh

The array runs stages 3, 4 and the strips, one task per film, at most
{a.throttle} at once.

**Stages 5 and 6 do not run here.** `cellpose_env` has torch, skimage,
tifffile, pandas and scipy, but not scikit-learn or umap-learn, and the
`SingleCellDataAnalysis` package is not deployed on the cluster. Retrieve the
output and run them on the workstation:

    python SingleCellQuantificationHPC/build_features_m160.py
    python SingleCellQuantificationHPC/train_fc_ae_m160.py
    python SingleCellQuantificationHPC/build_umap_html_m160.py

`model_based_m160_post.sh` is kept for the day those packages exist there; it
checks for them and exits early with a message if they do not.

## Resume

Both stages skip work that is already complete, so a timed-out or requeued task
picks up where it stopped. Stage 4 also re-runs any cell whose table is not a
full 101 frames, so a partial earlier run is repaired rather than kept.

## Retrieve

    rsync -av hsushen@<hpc>:{a.hpc_outputs}/ \\
        "/Volumes/X10 Pro/FungalProject_Outputs/model_based_dense_tracking/{EXP_NAME}/"
""")

    print(f"films: {len(films)}   cohort: {ncells} (film, cell) pairs")
    print(f"array : {array_path}")
    print(f"post  : {post_path}")
    print(f"readme: {readme}")


if __name__ == "__main__":
    main()
