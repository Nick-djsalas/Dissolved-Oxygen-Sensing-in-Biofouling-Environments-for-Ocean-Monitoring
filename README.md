# Physics-Informed Neural Networks for Robust Dissolved Oxygen Sensing in Biofouling Environments

**Companion code for:**
> Salaris N., Desjardins A., Tiwari M.K., *"A New Paradigm for Robust, Low-Cost Dissolved Oxygen Sensing in Biofouling Environments"*,

---

## Overview

This repository contains the complete computational framework for camera-based dissolved oxygen (DO) sensing via quenching of phosphorescence under biofouling conditions. Two scripts together reproduce every quantitative result in the manuscript.

| Script | Purpose | Paper sections |
|--------|---------|----------------|
| `classical_and_physics_reinforced_baselines.py` | Classical SV calibration, "Best Pixels" strategies, physics-reinforced LightGBM | Results §1, SI §S1 |
| `pinn_vit_framework.py` | Physics-Informed Neural Networks (CNN & ViT) with deep ensembles | Results §2–§5, SI §S2–S5 |

Both scripts share an identical data-processing pipeline to ensure fair comparison.

---

## Models Implemented

### Classical & Physics-Reinforced Baselines

| Model | Description | Figures |
|-------|-------------|---------|
| **GA** | Global Average: single linear SV fit to mean pixel intensity | Fig. 3A |
| **Best Pixels** (R2-10-NL, IO-1000-L, etc.) | Top-N pixels ranked by physics metrics, averaged into a "super-pixel" | Fig. 3E |
| **LGA** | Physics-reinforced LightGBM with aggregated SV parameters as features | Fig. 3B |
| **LRGBT, LRGBTSV, LRGBSVTP** | Position-agnostic LGBM variants | SI Figs. S1–S2 |
| **LSSV, LSSVP** | Position-aware LGBM variants | SI Fig. S1 |

> **Key distinction:** Physics-reinforced models use physical quantities as *input features* but do **not** enforce them through the loss function.

### PINN & Vision Transformer Framework

| Model | Architecture | Loss | Figures |
|-------|--------------|------|---------|
| **CNN** | ResNet-18 + CBAM | Data only | Fig. 4 |
| **PCNN / PCNNB** | ResNet-18 + CBAM + SV parameter head | Data + Physics (± Biofouling) | Figs. 6–7 |
| **PViT-O / PViT-EA / PViT-EB** | ViT + physics loss + deep ensemble | Data + Physics | Figs. 8–9 |

The PINN architecture comprises four heads (O₂ regression, biofouling mask, confidence map, SV parameter estimation) described in the Methods and SI §S4–S5.

---

## Directory Structure

```text
├── data/
│   ├── raw/                              # Experiment sub-directories
│   │   ├── 01-01-2024/                   # Must contain: *_ROI.mp4, *_arduino_*.txt, *temperature*.csv
│   │   └── ...
│   ├── cache_features/                   # (Auto-generated) Cached .parquet dataframes
│   └── cache_hpo/                        # (Auto-generated) Optuna databases
│
├── outputs/
│   ├── Classical_ML_Analysis_Report/     # (Auto-generated) Baseline outputs
│   ├── heatmaps/                         # (Auto-generated) PINN diagnostic maps
│   ├── loocv_folds/                      # (Auto-generated) PINN checkpoints
│   └── detailed_analysis_report/         # (Auto-generated) PINN results & uncertainty
│
├── classical_and_physics_reinforced_baselines.py
├── pinn_vit_framework.py
├── requirements.txt
├── LICENSE
└── README.md
```

Each experiment directory must contain: a cropped UV-excitation video (`*_ROI.mp4`), a Pyroscience DO sensor log (`*_arduino_*.txt`), and a temperature log (`*temperature*.csv`).

---

## Code → Paper Figure Mapping

| Figure | Script |
|--------|--------|
| Fig. 2C-D (spatial R² heatmaps) | `classical_...baselines.py` |
| Fig. 3 (GA, LGA, Best Pixels) | `classical_...baselines.py` |
| Fig. 4 (ML baseline hierarchy) | `classical_...baselines.py` |
| Fig. 5 (PINN residual & attention maps) | `pinn_vit_framework.py` |
| Figs. 6–7 (PCNN results) | `pinn_vit_framework.py` |
| Figs. 8–9 (PViT, uncertainty, scalability) | `pinn_vit_framework.py` |
| Fig. 10 (temporal extrapolation) | `pinn_vit_framework.py` |
| SI Figs. S1–S2 (LGBM ablation) | `classical_...baselines.py` |

---

## Installation

```bash
git clone https://github.com/[username]/do-sensing-pinn.git
cd do-sensing-pinn
pip install -r requirements.txt
```

**Dependencies:** numpy, pandas, scipy, scikit-learn, lightgbm, opencv-python, matplotlib, seaborn, tqdm, joblib, torch, torchvision, timm, optuna

A CUDA-compatible GPU is recommended for PINN training but not required.

---

## Usage

```bash
# 1. Classical baselines (reproduces Figs. 2-4, SI Figs. S1-S2)
python classical_and_physics_reinforced_baselines.py

# 2. PINN framework (reproduces Figs. 5-10)
python pinn_vit_framework.py
```

Edit `BASE_PROJECT_DIR` at the top of each script to point to your data directory.

---

## Cross-Validation Strategy

All models use Leave-One-(Day-)Out Cross-Validation. Each fold holds out an entire experimental day rather than randomly sampling observations, guarding against temporal autocorrelation. A strict chronological split (train days 1–11, validate day 12, test day 13) is also included for temporal forecasting validation.

---

## Data Availability

Raw image data, processed datasets, and ground-truth measurements are available from the corresponding author upon request.

## Citation

```bibtex
@article{salaris2025do_pinn,
  title   = {A New Paradigm for Robust, Low-Cost Dissolved Oxygen Sensing in Biofouling Environments},
  author  = {Salaris, Nikolaos and Desjardins, Adrien and Tiwari, Manish K.},
  journal = {},
  year    = {},
  doi     = {}
}
```

## Contact

**Corresponding author:** Manish K. Tiwari ([m.tiwari@ucl.ac.uk](mailto:m.tiwari@ucl.ac.uk)) — Nanoengineered Systems Laboratory, UCL Mechanical Engineering & UCL Hawkes Institute
