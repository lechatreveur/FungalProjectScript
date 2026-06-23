#!/bin/bash
for i in {1..10}; do
  echo "Running AI tracking for cell $i..."
  KMP_DUPLICATE_LIB_OK=TRUE /Users/user/miniforge3/envs/cellpose-sam/bin/python -u one_cell_quantification_1CH.py --cell_id $i --experiment_path "/Volumes/X10 Pro/Movies/2026_01_08_M93" --file_name "A14_BF_1_F0" --use_ai_tracker --update_existing --no_plot
done
