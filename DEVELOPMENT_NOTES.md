# Fungal Septum Classifier: Development History & Lessons Learned

This document serves as a guide for future developers maintaining the Fungal Septum Classifier. It outlines the major architectural shifts, design decisions, and critical "traps" encountered during the development of version 2.0.

## 1. Multi-Film Architecture
### The Problem
Originally, the GUI used a single `global_interval` variable and a simplified indexing system. When the project scaled to multiple experiments (M92, M93, M96), state interference occurred—setting a red interval in one film would erroneously appear in others.

### The Solution (Multi-Film State)
- **Composite Keys**: Most internal state dictionaries (offsets, cell intervals) now use a `(film_name, cell_id)` tuple as the primary key.
- **Global Interval Map**: We implemented a `global_intervals_map` to store intervals per-film.
- **Forward-Filling Logic**: During dataset export, if a film hasn't been explicitly touched, it inherits the global interval from the previous film (chronological persistence).

## 2. The "White Septum" Trap (Critical)
### The Mistake
Early in development, "white septums" (highly contrastive, bright ridges) were labeled with the `i` key. However, the training script deterministically inverted these cells (`1.0 - x`). 
- **Result**: The AI *never* saw a bright septum during training. It became purely a "dark valley detector."
- **Failure**: When encountering white septums in the GUI during real-time inference, the model predicted 0% confidence because it was never trained on the original bright polarity.

### The Fix: Global Polarity Invariance
We removed the deterministic labeling-based flip and replaced it with **50% Random Polarity Inversion** during training.
- **Why**: This forces the CNN to detect the *structure* of the septum (the gradient and shape) regardless of whether it is pixel-bright or pixel-dark. 
- **Lesson**: Do not use hard-coded label-based inversions; let the data augmentation teach the model invariance.

## 3. Data Integrity & Export Pipeline
### The Baseline Bug
A critical bug existed where cells without explicit start/end alignments were being skipped during export, even if they had a valid global interval. This resulted in severely under-sampled training sets (e.g., M96 originally exported 18 samples instead of the expected 54).

### Key Logic
- The `septum_training_utils.py` script now correctly falls back: 
  1. Per-cell labels (most specific).
  2. Global film interval (if cell is not specifically labeled).
  3. Skips only if neither exists.

## 4. GUI Rendering & Visualization
- **Saliency (AI Vision)**: We integrated autograd-based saliency heatmaps. 
- **Transparency Fix**: When overlaying heatmaps, do not use simple `alpha` on a 2D array, or the "black/low" values will darken the biology. Instead, use a **4-channel RGBA map** where the intensity of the saliency is mapped directly to the **Alpha** channel. 
- **Axis Bounds**: When adding new `imshow` artists for overlays, always initialize them with the same dimensions as the main sheet (`np.zeros_like(sheet)`) to avoid Matplotlib's auto-scaling squishing the window.

## Summary Checklist for Updates
- [ ] If changing the neural network, update `FungalInferenceCore` in `inference_core.py`.
- [ ] If adding human labels, verify the keybinding doesn't conflict in `alignment_board_gui.py`.
- [ ] Always check `manifest.csv` after an export to ensure the `has_septum` count matches your GUI expectations.
