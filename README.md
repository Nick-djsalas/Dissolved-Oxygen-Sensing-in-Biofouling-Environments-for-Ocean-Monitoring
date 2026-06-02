# Deep Learning-Enabled Dissolved Oxygen Sensing in Biofouling Environments for Ocean Monitoring

**Companion code for:**
> Salaris N., Desjardins A., Tiwari M.K., *"Deep Learning-Enabled Dissolved Oxygen Sensing in Biofouling Environments for Ocean Monitoring"*,

---

## Overview

This repository contains the complete computational framework for camera-based dissolved oxygen (DO) sensing via quenching of phosphorescence under biofouling conditions. Three main scripts reproduce the quantitative results in the manuscript and provide a robust framework for real-world deployment simulations, including demonstrations of model scalability.

| Script | Purpose | Paper sections |
|--------|---------|----------------|
| `classical_and_physics_reinforced_baselines.py` | Classical SV calibration, "Best Pixels" strategies, physics-reinforced LightGBM | Results §1, SI §S1 |
| `pinn_vit_framework.py` | Physics-Informed Neural Networks (CNN & ViT) with deep ensembles via LOOCV | Results §2–§5, SI §S2–S5 |
| `pinn_chronological_forward_chaining.py` | Physics-Informed ViT with strict Walk-Forward validation for real-world deployment simulation and scalability analysis | Methods (Scalability) |

All scripts share an identical data-processing pipeline to ensure fair comparison.

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
| **PViT-WF** (Walk-Forward) | ViT + physics loss + deep ensemble | Data + Physics | Figs. 7E-F, 10 |

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
│   ├── detailed_analysis_report/         # (Auto-generated) PINN results & uncertainty
│   └── PINN_Analysis_V8_3_Chronological/ # (Auto-generated) Walk-forward validation results & split reports
│
├── classical_and_physics_reinforced_baselines.py
├── pinn_vit_framework.py
├── pinn_chronological_forward_chaining.py
├── requirements.txt
├── LICENSE
└── README.md
