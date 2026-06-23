#!/bin/bash
echo "Generating AI tracked movie..."
KMP_DUPLICATE_LIB_OK=TRUE /Users/user/miniforge3/envs/cellpose-sam/bin/python ../make_population_movie.py \
  --base_dir "/Volumes/X10 Pro/Movies/2026_01_08_M93/A14_BF_1_F0" \
  --frames_dir "Frames_A14_BF_1_F0" \
  --cells_dir "TrackedCells_A14_BF_1_F0_AI" \
  --out "A14_BF_1_F0_AI_population.mp4" \
  --fps 10 \
  --alpha 0.4
echo "Done"
