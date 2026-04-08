# 🍄 Fungal Project Image Classifier

Welcome to the **Fungal Project Single Cell Data Analysis Pipeline**. This repository houses the complete deep-learning workflow utilized for semantic segmentation, cell crop extraction, and binary temporal classification of fungal structural checkpoints (e.g. tracking Start/End Septums during growth processes).

## 🔬 Scientific Context
This pipeline aims to systematically convert raw high-throughput multichannel movies of dividing fungal cells into analyzable, predictable mathematical matrices. 
By utilizing Multi-Instance Learning (MIL) built on PyTorch, this package can consume variable-sized `[L, 1, H, W]` strips of bounding-box tracked fungal cells and temporally predict whether specific division artifacts (start and end endpoints) exist within entirely isolated crops.

---

## 💻 Tech Stack
- **Deep Learning Architecture:** PyTorch (Backbone: 1D Temporal CNN + 2D Tile Encoding)
- **Data Engineering:** Numpy / Pandas
- **Visualization:** Matplotlib / Imaris Interfacing

## 🔧 Installation & Setup
To ensure total hardware and computational reproducibility, all dependencies should mimic the original pipeline exactly as tracked.

### 1. Clone the repository
```bash
git clone https://github.com/lechatreveur/FungalProjectScript.git
cd FungalProjectScript
```

### 2. Prepare Virtual Environment
We strongly advise utilizing an isolated python environment.
```bash
python3 -m venv cellpose_3011_env
source cellpose_3011_env/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Execution Guide (Model Training)

The main pipeline script `septum_train_binary.py` is capable of dynamically loading, fusing, and learning across an infinitely scalable array of `.npz` experimental datasets!

### Starting a Model Run (or Combining Datasets)
When training for the first time or aggressively improving model weights across new experiment samples, you can dynamically link the `training_dataset` folders:
```bash
python SingleCellDataAnalysis/septum_train_binary.py \
    "/Volumes/X10 Pro/Movies/2025_12_31_M92" \
    "/Volumes/X10 Pro/Movies/2026_01_08_M93" \
    --epochs 100 \
    --batch_size 32 \
    --L_max 81
```

### Resuming Incomplete Training Loops
We checkpoint our weights automatically. If you desire to scale your Epoch counts or resume an externally interrupted run:
```bash
python SingleCellDataAnalysis/septum_train_binary.py \
    "/Volumes/X10 Pro/Movies/2025_12_31_M92" \
    --epochs 50 \
    --resume_from "/Volumes/X10 Pro/Movies/2025_12_31_M92/training_dataset/checkpoints_binary/model_ep050.pt"
```

---

## 📂 Project Directory Structure 
- `/SingleCellDataAnalysis`: Core package for temporal filtering, tracking GUIs, spatial alignment, and PyTorch mathematical models.
  - `septum_train_binary.py`: Production-ready script driving the multi-instance binary learning neural network.
- `requirements.txt`: The completely frozen state of all project computational dependencies at time of development.
