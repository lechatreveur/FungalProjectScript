#!/bin/bash

# Train the AI tracker on the GFP dataset

python train_tracker.py \
    --curated-csv "/Volumes/X10 Pro/Movies/2026_01_08_M93/curated_training_samples_gfp.csv" \
    --out_dir ./tracker_checkpoints_m93_gfp \
    --epochs 60 \
    --batch_size 16 \
    --lr 3e-4

echo "Training complete!"
