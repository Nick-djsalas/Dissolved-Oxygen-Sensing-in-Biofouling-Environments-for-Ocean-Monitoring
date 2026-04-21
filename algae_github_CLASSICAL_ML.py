#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
classical_and_physics_reinforced_baselines.py
================================================================================

COMPANION CODE FOR:
    "A New Paradigm for Robust, Low-Cost Dissolved Oxygen Sensing
     in Biofouling Environments"
    Salaris N., Desjardins A., Tiwari M.K.
    Published in: [Journal Name, Year]

--------------------------------------------------------------------------------
PURPOSE
--------------------------------------------------------------------------------
This script implements and evaluates all CLASSICAL and PHYSICS-REINFORCED
baseline models described in the paper. It serves as the quantitative control
group against which the Physics-Informed Neural Network (PINN) and Vision
Transformer (ViT-PINN) architectures are compared.

Specifically, this script reproduces the results presented in:

    MAIN TEXT:
    - "Limitation of Typical Averaging and Physics-Based Methods"
      (Section: Results, Figures 2-3 in the main manuscript)
    - "Spatial Information with Machine Learning: Improvement Without
      Understanding" (summarised in the main text; full ablation in SI)

    SUPPLEMENTARY INFORMATION:
    - Section S1: "ML with Feature Engineering using Spatial Information"
      (Figures S1-S2)

--------------------------------------------------------------------------------
MODELS IMPLEMENTED (with paper nomenclature)
--------------------------------------------------------------------------------
The script evaluates a systematic hierarchy of models of increasing complexity.
All models are assessed using Leave-One-(Day-)Out Cross-Validation (LOOCV),
where an entire experimental day is held out as the test set in each fold.

  MODEL NAME   | TYPE                        | PAPER SECTION      | KEY RESULT
  (in paper)   |                             |                    | (Test MAE)
  -------------|-----------------------------|--------------------|------------
  GA           | Global Average: fits a      | Results, Fig. 3    | ~24.8 µmol/L
               | single linear Stern-Volmer  |                    |
               | (SV) equation to the mean   |                    |
               | red-channel intensity of    |                    |
               | all pixels per frame.       |                    |
               | Equivalent to industry-     |                    |
               | standard single-point       |                    |
               | calibration.                |                    |
  -------------|-----------------------------|--------------------|------------
  Best Pixels  | Selects top-N pixels ranked | Results, Fig. 3    | ~24.6 µmol/L
  (R2-10-NL,   | by physics-derived metrics  |                    | (best case)
   IO-1000-L,  | (R², I₀, K_SV, DR, LOD).   |                    |
   DR-1000-NL, | Their intensities are       |                    |
   LOD-1000-L, | averaged into a "super-     |                    |
   KSV-1000-L) | pixel" signal fitted to     |                    |
               | linear or non-linear SV.    |                    |
  -------------|-----------------------------|--------------------|------------
  LGA          | Physics-Reinforced LightGBM:| Results, Fig. 3    | ~18.5 µmol/L
               | A gradient-boosted tree     |                    |
               | model supplied with red-    |                    |
               | channel intensity statistics|                    |
               | AND aggregated physics      |                    |
               | parameters (mean I₀, mean   |                    |
               | K_SV, mean LOD, mean DR,    |                    |
               | mean R²). Distinct from a   |                    |
               | PINN: physics quantities    |                    |
               | are used as INPUT FEATURES, |                    |
               | not enforced through the    |                    |
               | loss function.              |                    |
  -------------|-----------------------------|--------------------|------------

  The following models appear in Supplementary Section S1 (Figs. S1-S2):

  MODEL NAME   | TYPE                        | PAPER SECTION      | KEY RESULT
  -------------|-----------------------------|--------------------|------------
  LRGBT        | Position-Agnostic LGBM:     | SI Section S1,     | 41.8 µmol/L
               | per-pixel RGB + temperature.| Fig. S1A           |
  -------------|-----------------------------|--------------------|------------
  LRGBTSV      | + per-pixel SV parameters.  | SI Section S1,     | 34.6 µmol/L
               |                             | Fig. S1A           |
  -------------|-----------------------------|--------------------|------------
  LRGBSVTP     | + all physics-derived stats.| SI Section S1,     | Similar
               |                             | Figs. S1A, S2A-C   |
  -------------|-----------------------------|--------------------|------------
  LSSV         | Position-Aware LGBM:        | SI Section S1,     | 13.2 µmol/L
               | flattened 48×48×3 image +   | Fig. S1A           |
               | SV parameters.              |                    |
  -------------|-----------------------------|--------------------|------------
  LSSVP        | + all physics-derived stats.| SI Section S1,     | 13.2 µmol/L
               |                             | Figs. S1A-D        |
  -------------|-----------------------------|--------------------|------------

--------------------------------------------------------------------------------
OUTPUTS GENERATED (mapped to paper figures)
--------------------------------------------------------------------------------
All outputs are saved to `Classical_ML_Analysis_Report/` with subfolders:

  OUTPUT                              | CORRESPONDS TO
  ------------------------------------|------------------------------------------
  Parity plots (GA, LGA)              | Main text Fig. 3A-B
  Residual & error distribution plots  | Main text Fig. 3C-D
  MAE bar chart (all classical)       | Main text Fig. 3E
  MAE by DO range (LGA)               | Main text Fig. 3F
  Spatial R² heatmaps (linear, NL)    | Main text Fig. 2C-D
  LGBM feature importance (LSSVP)     | SI Fig. S1C
  LGBM feature importance (LRGBSVTP)  | SI Fig. S1D
  Pixel-wise MAE heatmaps             | SI Fig. S2A-C

--------------------------------------------------------------------------------
PHYSICS EQUATIONS USED (references to paper Methods)
--------------------------------------------------------------------------------
This script implements:
  - Linear Stern-Volmer:        I₀/I = 1 + K_SV·[O₂]         (Eq. 4a)
  - Non-linear two-site SV:     see Eq. 4b-4c in Methods
  - Dynamic Range (DR):         Eq. 5
  - Limit of Detection (LOD):   Eq. 7
  - Coefficient of Determination (R²): Eq. 8

--------------------------------------------------------------------------------
CROSS-VALIDATION STRATEGY
--------------------------------------------------------------------------------
All models use Leave-One-(Day-)Out Cross-Validation (LOOCV). Each fold holds
out an ENTIRE experimental day as the test set, rather than randomly sampling
individual observations. This guards against temporal autocorrelation and
covariate leakage, since measurements acquired on the same day share
environmental and biofouling conditions.

--------------------------------------------------------------------------------
DATA REQUIREMENTS
--------------------------------------------------------------------------------
The script expects the following directory structure under BASE_PROJECT_DIR:
  <date_folder>/          (e.g., "01-01-2024")
    ├── *_ROI.mp4         (cropped video of the sensor under UV excitation)
    ├── *_arduino_*.txt   (Pyroscience DO sensor log)
    └── *temperature*.csv (temperature sensor log)

Processed data is cached as a Parquet file to avoid re-extraction on
subsequent runs. Set FORCE_RECREATE_DATAFRAME = True to regenerate.

--------------------------------------------------------------------------------
COMPUTATIONAL NOTES
--------------------------------------------------------------------------------
  - Per-pixel SV fitting is parallelised via joblib across available CPU cores.
  - LightGBM training uses GPU acceleration when available (auto-detected),
    with automatic fallback to CPU.
  - On a system with 12 CPU cores, the full pipeline (13-day dataset) completes
    in approximately [X] minutes.

--------------------------------------------------------------------------------
DEPENDENCIES
--------------------------------------------------------------------------------
  Python >= 3.9
  numpy, pandas, scipy, scikit-learn, lightgbm, opencv-python,
  matplotlib, seaborn, tqdm, joblib

  Install via:  pip install numpy pandas scipy scikit-learn lightgbm \
                           opencv-python matplotlib seaborn tqdm joblib

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
  1. Set BASE_PROJECT_DIR to the root of your experimental data.
  2. Adjust RESIZE_DIM, FRAME_SKIP, and other configuration parameters
     as needed (defaults match the paper).
  3. Run:  python classical_and_physics_reinforced_baselines.py
  4. Outputs are saved to Classical_ML_Analysis_Report/ under the project dir.

--------------------------------------------------------------------------------
RELATED SCRIPTS IN THIS REPOSITORY
--------------------------------------------------------------------------------
  - pinn_cnn.py         : PCNN and PCNNB models (Main text, Figs. 6-7)
  - pinn_vit.py         : PViT models with deep ensemble (Main text, Figs. 8-10)
  - spatial_lgbm.py     : Position-aware/agnostic LGBM models (SI Section S1)

--------------------------------------------------------------------------------
LICENSE & CITATION
--------------------------------------------------------------------------------
  If you use this code, please cite:
    Salaris N., Desjardins A., Tiwari M.K., "A New Paradigm for Robust,
    Low-Cost Dissolved Oxygen Sensing in Biofouling Environments",
    [Journal], [Year]. DOI: [XXX]

================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import os
import sys
import glob
import re
import traceback
import warnings
import functools
import shutil
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import cv2
from scipy.optimize import curve_fit, root_scalar
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from joblib import Parallel, delayed


# ==============================================================================
# CONFIGURATION
# ==============================================================================
# All parameters below correspond to those reported in the Methods section.
# Modify BASE_PROJECT_DIR to point to your local copy of the experimental data.

BASE_PROJECT_DIR = r"F:\UNI papers - ALL data\algae\tests\algae_new\exp_1_algae\video\5-algae_tests"
OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "Classical_ML_Analysis_Report")
FEATURE_CACHE_DIR = os.path.join(BASE_PROJECT_DIR, "PINN_features")
FEATURE_DATAFRAME_PATH = os.path.join(FEATURE_CACHE_DIR, "pinn_features_resampled_red_channel.parquet")
CLASSICAL_CACHE_DIR = os.path.join(BASE_PROJECT_DIR, "Classical_Analysis_Cache")

# --- Data Processing Parameters (must match PINN scripts for fair comparison) ---
FORCE_RECREATE_DATAFRAME = True   # Set True to regenerate feature Parquet from raw videos
FORCE_REFIT_PIXELS = True         # Set True to refit all per-pixel SV models
NUM_EXPERIMENT_DAYS_TO_USE = None  # None = use all available days (13 in the paper)
RESIZE_DIM = (96, 96)             # Spatial resolution after downsampling (paper: 48x48)
FRAME_SKIP = 15                   # Process every Nth frame (paper: yields ~2 fps from 30 fps)
FORCED_VIDEO_FPS = 30             # Override video FPS metadata
FRAME_CHUNK_SIZE = 3000           # Frames per processing chunk (memory management)
SENSOR_RESAMPLE_WINDOW_S = 10     # Resample ground-truth sensor data (10s step averaging)

# --- Stable DO concentration intervals for per-pixel fitting (seconds) ---
# These correspond to the 5 discrete O₂ concentrations used during calibration
# (0%, 5%, 10%, 15%, 20% V/V O₂ via mass flow controllers; see Methods).
ANALYSIS_INTERVALS_S = [
    (1450, 1650),  # Interval 1
    (2300, 2500),  # Interval 2
    (3100, 3300),  # Interval 3
    (3900, 4100),  # Interval 4
    (4700, 4900),  # Interval 5
]

# --- Best Pixels strategy: cohort sizes to evaluate ---
BEST_PIXELS_N = [10, 100, 1000]

# --- Parallelisation ---
try:
    NUM_WORKERS = min(max(1, cpu_count() - 1), 12)
except NotImplementedError:
    NUM_WORKERS = 1

# --- LightGBM device detection ---
LGBM_DEVICE = 'cpu'
try:
    lgb.LGBMRegressor(device='gpu').fit(np.array([[0]]), np.array([0]))
    LGBM_DEVICE = 'gpu'
    print(f"\n>>> GPU detected. LightGBM will use GPU acceleration. <<<\n")
except Exception:
    print(f"\n>>> No GPU detected. Using CPU for LightGBM ({NUM_WORKERS} cores). <<<\n")


# ==============================================================================
# STAGE 1: DATA PROCESSING & UTILITY FUNCTIONS
# ==============================================================================
# This stage handles parsing raw sensor logs (Pyroscience DO sensor, temperature
# probe) and extracting red-channel pixel intensities from UV-excited video
# frames. The pipeline is identical to that used by the PINN scripts to ensure
# a fair comparison.


def parse_arduino_log(file_path):
    """Parse Pyroscience DO sensor log (Arduino serial output).

    Returns a DataFrame with columns ['timestamp', 'oxygen_umol_L'].
    Ground-truth values are resampled using SENSOR_RESAMPLE_WINDOW_S to
    reduce random systematic error from the industry sensor (see Methods).
    """
    timestamps, oxygen_values = [], []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                if "MEA" not in line:
                    continue
                parts = line.split()
                timestamp_str = f"{parts[0]} {parts[1]}"
                dt_object = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                mea_index = parts.index("MEA")
                raw_oxygen_value = int(parts[mea_index + 5])
                timestamps.append(dt_object)
                oxygen_values.append(raw_oxygen_value / 1000.0)
            except (ValueError, IndexError, TypeError):
                pass
    if not timestamps:
        return pd.DataFrame()
    df = pd.DataFrame({'timestamp': timestamps, 'oxygen_umol_L': oxygen_values})
    if df.timestamp.duplicated().any():
        df = df.groupby('timestamp')['oxygen_umol_L'].mean().reset_index()
    if not df.empty and SENSOR_RESAMPLE_WINDOW_S > 0:
        df = df.set_index('timestamp').resample(
            f'{SENSOR_RESAMPLE_WINDOW_S}S'
        ).mean().dropna().reset_index()
    return df


def parse_temperature_log(file_path):
    """Parse DB9 temperature sensor CSV log.

    Returns a DataFrame with columns ['timestamp', 'temperature_C'].
    Includes spike removal (>100% change between consecutive readings).
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = ['timestamp', 'temperature_C']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['temperature_C'] = pd.to_numeric(df['temperature_C'], errors='coerce')
        df.dropna(inplace=True)
        if df.empty:
            return pd.DataFrame()
        if df.timestamp.duplicated().any():
            df = df.groupby('timestamp')['temperature_C'].mean().reset_index()
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        # Remove temperature spikes (sensor artefacts)
        pct_change = df['temperature_C'].pct_change().abs()
        spike_mask = pct_change >= 1.0
        if spike_mask.sum() > 0:
            df = df[~spike_mask].copy()
        if not df.empty and SENSOR_RESAMPLE_WINDOW_S > 0:
            df = df.set_index('timestamp').resample(
                f'{SENSOR_RESAMPLE_WINDOW_S}S'
            ).mean().dropna().reset_index()
        return df
    except Exception:
        return pd.DataFrame()


def find_experiment_files(root_dir, num_days_to_use=None):
    """Discover and sort experiment directories by date.

    Each valid experiment directory must contain:
      - An ROI-cropped video (*_ROI.mp4)
      - A Pyroscience DO sensor log (*_arduino_*.txt)
      - A temperature log (*temperature*.csv or .txt)
    """
    all_dirs = [d for d in glob.glob(os.path.join(root_dir, '*-*')) if os.path.isdir(d)]
    dated_dirs = []
    for d in all_dirs:
        dir_name = os.path.basename(d)
        match = re.search(r'(\d{1,2}[-_]\d{2}[-_]\d{4})', dir_name.replace('_', '-'))
        if match:
            try:
                date_str = match.group(1).replace('_', '-')
                dated_dirs.append({
                    'path': d,
                    'date': datetime.strptime(date_str, '%d-%m-%Y')
                })
            except ValueError:
                pass
    dated_dirs.sort(key=lambda x: x['date'])
    experiment_dirs = [
        d['path'] for d in (dated_dirs[:num_days_to_use] if num_days_to_use else dated_dirs)
    ]
    valid_experiments = []
    for exp_dir in tqdm(experiment_dirs, desc="Scanning experiments"):
        exp_id = os.path.basename(exp_dir)
        roi_video_list = glob.glob(os.path.join(exp_dir, "**", "*_ROI.mp4"), recursive=True)
        arduino_data_list = glob.glob(os.path.join(exp_dir, "**", "*_arduino_*.txt"), recursive=True)
        temp_data_list = (
            glob.glob(os.path.join(exp_dir, "**", "*temperature*.csv"), recursive=True)
            + glob.glob(os.path.join(exp_dir, "**", "*temperature*.txt"), recursive=True)
        )
        if roi_video_list and arduino_data_list and temp_data_list:
            valid_experiments.append({
                "id": exp_id,
                "video_path": roi_video_list[0],
                "raw_arduino_path": arduino_data_list[0],
                "temperature_path": temp_data_list[0],
            })
    if not valid_experiments:
        raise FileNotFoundError("FATAL: No valid experiments found in the specified directory.")
    return valid_experiments


def extract_timestamp_from_filename(filename):
    """Extract recording start time from video filename (format: YYYYMMDD_HHMMSS)."""
    match = re.search(r'_(\d{8}_\d{6})', filename)
    if not match:
        return None
    return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')


def process_experiment_chunked(args):
    """Extract red-channel pixel intensities from a single experiment video.

    For each valid frame (after frame skipping), the function:
      1. Resizes the frame to RESIZE_DIM
      2. Extracts only the red channel (closest to PtOEP emission spectrum)
      3. Matches the frame timestamp to the nearest DO and temperature readings
      4. Saves results in chunks as Parquet files for memory efficiency

    Returns True on success, False on failure.
    """
    exp = args['exp']
    frame_skip = args['frame_skip']
    resize_dim = args['resize_dim']
    temp_dir = args['temp_dir']
    exp_id = exp['id']

    try:
        df_arduino = parse_arduino_log(exp['raw_arduino_path'])
        if df_arduino.empty:
            return False
        df_temp = parse_temperature_log(exp['temperature_path'])
        if df_temp.empty:
            return False
        df_arduino.set_index('timestamp', inplace=True)
        df_temp.set_index('timestamp', inplace=True)

        cap = cv2.VideoCapture(exp['video_path'])
        if not cap.isOpened():
            return False
        fps = FORCED_VIDEO_FPS if FORCED_VIDEO_FPS else cap.get(cv2.CAP_PROP_FPS)
        video_start_time = extract_timestamp_from_filename(os.path.basename(exp['video_path']))
        if not fps or not video_start_time:
            cap.release()
            return False

        chunk_index = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc=f"Processing {exp_id}", leave=False,
                  position=args.get('worker_id', 0)) as pbar:
            while True:
                chunk_data = []
                frames_in_chunk = 0
                while frames_in_chunk < FRAME_CHUNK_SIZE:
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    ret, frame = cap.read()
                    if not ret:
                        break
                    pbar.update(1)
                    if frame_number % frame_skip == 0:
                        elapsed_seconds = frame_number / fps
                        frame_timestamp = video_start_time + timedelta(seconds=elapsed_seconds)
                        try:
                            nearest_o2_idx = df_arduino.index.get_indexer(
                                [frame_timestamp], method='nearest'
                            )[0]
                            nearest_temp_idx = df_temp.index.get_indexer(
                                [frame_timestamp], method='nearest'
                            )[0]
                            if nearest_o2_idx != -1 and nearest_temp_idx != -1:
                                oxygen_val = df_arduino.iloc[nearest_o2_idx]['oxygen_umol_L']
                                temp_val = df_temp.iloc[nearest_temp_idx]['temperature_C']
                                resized_frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
                                red_channel_pixels = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)[:, :, 0]
                                data_row = [
                                    exp_id, frame_timestamp, elapsed_seconds, temp_val, oxygen_val
                                ] + red_channel_pixels.flatten().tolist()
                                chunk_data.append(data_row)
                        except (KeyError, IndexError):
                            pass
                    frames_in_chunk += 1

                if not chunk_data:
                    break
                pd.DataFrame(chunk_data).to_parquet(
                    os.path.join(temp_dir, f"{exp_id}_chunk_{chunk_index}.parquet")
                )
                chunk_index += 1
                if not ret:
                    break
        cap.release()
    except Exception:
        traceback.print_exc()
        return False
    return True


def create_pinn_dataframe(experiments, output_path, frame_skip, resize_dim):
    """Build or load the master feature DataFrame.

    The resulting DataFrame has one row per processed frame, with columns:
      - experiment_id, timestamp, elapsed_seconds, temperature_C, oxygen_umol_L
      - pixel_0 ... pixel_N (red-channel intensities for each spatial pixel)

    This is the same feature file used by the PINN scripts, ensuring identical
    input data for all model comparisons reported in the paper.
    """
    if os.path.exists(output_path) and not FORCE_RECREATE_DATAFRAME:
        print(f"Loading cached feature dataframe from {output_path}")
        return pd.read_parquet(output_path)
    if os.path.exists(output_path) and FORCE_RECREATE_DATAFRAME:
        os.remove(output_path)

    temp_dir = os.path.join(os.path.dirname(output_path), "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    tasks = [{
        'exp': exp, 'frame_skip': frame_skip, 'resize_dim': resize_dim,
        'temp_dir': temp_dir, 'worker_id': i
    } for i, exp in enumerate(experiments)]

    if NUM_WORKERS > 0:
        with Pool(processes=NUM_WORKERS) as pool:
            list(tqdm(
                pool.imap_unordered(process_experiment_chunked, tasks),
                total=len(tasks), desc="Processing experiments"
            ))
    else:
        for task in tqdm(tasks, desc="Processing experiments (sequential)"):
            process_experiment_chunked(task)

    chunk_files = glob.glob(os.path.join(temp_dir, "*.parquet"))
    if not chunk_files:
        raise ValueError("FATAL: No data chunks generated. Check input paths.")
    master_df = pd.concat(
        [pd.read_parquet(f) for f in tqdm(chunk_files, desc="Assembling chunks")],
        ignore_index=True
    )
    shutil.rmtree(temp_dir)

    num_red_pixels = resize_dim[0] * resize_dim[1]
    pixel_cols = [f'pixel_{i}' for i in range(num_red_pixels)]
    columns = ['experiment_id', 'timestamp', 'elapsed_seconds',
               'temperature_C', 'oxygen_umol_L'] + pixel_cols
    master_df.columns = columns

    for col in pixel_cols:
        master_df[col] = pd.to_numeric(master_df[col], downcast='unsigned')
    master_df['elapsed_seconds'] = master_df['elapsed_seconds'].astype('float32')
    master_df['temperature_C'] = master_df['temperature_C'].astype('float32')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master_df.to_parquet(output_path, index=False)
    print(f"Saved feature dataframe ({len(master_df)} samples) to {output_path}")
    return master_df


# ==============================================================================
# STAGE 2: STERN-VOLMER MODEL FITTING
# ==============================================================================
# Implements the linear (Eq. 4a) and non-linear two-site (Eq. 4c) SV equations,
# their inverse functions for O₂ prediction, and per-pixel fitting with
# physics-derived metrics (R², DR, I₀, K_SV, LOD).


def sv_linear(o2, i0, ksv):
    """Linear Stern-Volmer model (Eq. 4a): I = I₀ / (1 + K_SV · [O₂])."""
    return i0 / (1 + ksv * o2)


def sv_nonlinear(o2, i0, ksv1, ksv2, a):
    """Non-linear two-site SV model (Eq. 4c):
    I = I₀ · [a/(1 + K_SV1·[O₂]) + (1-a)/(1 + K_SV2·[O₂])]
    """
    return i0 * (a / (1 + ksv1 * o2) + (1 - a) / (1 + ksv2 * o2))


def predict_o2_linear(intensity, i0, ksv):
    """Invert the linear SV equation to predict [O₂] from measured intensity."""
    intensity = np.maximum(intensity, 1e-9)
    i0 = np.maximum(i0, 1e-9)
    ksv = np.maximum(ksv, 1e-9)
    return (i0 / intensity - 1) / ksv


def _nonlinear_inverse_solver(intensity, o2_guess, i0, ksv1, ksv2, a):
    """Numerically invert the non-linear SV equation for a single intensity value."""
    try:
        func = lambda o2: sv_nonlinear(o2, i0, ksv1, ksv2, a) - intensity
        sol = root_scalar(func, bracket=[0, 1000], method='brentq')
        return sol.root
    except (ValueError, RuntimeError):
        return np.nan


def predict_o2_nonlinear(intensities, i0, ksv1, ksv2, a):
    """Invert the non-linear SV equation for an array of intensity values."""
    return np.vectorize(_nonlinear_inverse_solver)(
        intensities, o2_guess=100.0, i0=i0, ksv1=ksv1, ksv2=ksv2, a=a
    )


def fit_sv_model(o2_data, intensity_data, model_type='linear'):
    """Fit a Stern-Volmer model to intensity vs. [O₂] data.

    Args:
        o2_data: array of oxygen concentrations (µmol/L)
        intensity_data: array of corresponding red-channel intensities
        model_type: 'linear' (Eq. 4a) or 'nonlinear' (Eq. 4c)

    Returns:
        Fitted parameters, or None if fitting fails.
    """
    try:
        if model_type == 'linear':
            p0 = [np.max(intensity_data), 0.01]
            params, _ = curve_fit(sv_linear, o2_data, intensity_data,
                                  p0=p0, maxfev=10000)
        else:
            p0 = [np.max(intensity_data), 0.01, 0.001, 0.5]
            params, _ = curve_fit(sv_nonlinear, o2_data, intensity_data,
                                  p0=p0,
                                  bounds=([0, 0, 0, 0], [np.inf, np.inf, np.inf, 1]),
                                  maxfev=10000)
        return params
    except RuntimeError:
        return None


def _fit_single_pixel_worker(args):
    """Worker function for parallel per-pixel SV fitting (called by joblib)."""
    pixel_idx, o2_data, intensity_data, model_type = args
    params = fit_sv_model(o2_data, intensity_data, model_type)
    if params is not None:
        pred_intensity = (sv_linear(o2_data, *params) if model_type == 'linear'
                          else sv_nonlinear(o2_data, *params))
        r2 = r2_score(intensity_data, pred_intensity)
        dr = np.max(intensity_data) - np.min(intensity_data)
        return pixel_idx, params, r2, dr
    else:
        num_params = 2 if model_type == 'linear' else 4
        return pixel_idx, [np.nan] * num_params, np.nan, np.nan


def calculate_pixel_metrics(train_df, train_pixel_data, model_type='linear'):
    """Fit per-pixel SV models and compute physics-derived metrics.

    For each pixel, this function computes (see Methods, Eqs. 4-8):
      - Fitted SV parameters (I₀, K_SV for linear; I₀, K_SV1, K_SV2, a for NL)
      - Goodness-of-fit R² (Eq. 8)
      - Dynamic Range DR (Eq. 5)
      - Limit of Detection LOD (Eq. 7, linear model only)

    These metrics are used both for the "Best Pixels" ranking strategy and
    as input features for the physics-reinforced LGA model.

    Parallelised via joblib for performance.
    """
    num_pixels = train_pixel_data.shape[1]
    train_o2_data = train_df['oxygen_umol_L'].values

    tasks = [(i, train_o2_data, train_pixel_data[:, i], model_type)
             for i in range(num_pixels)]

    results = Parallel(n_jobs=NUM_WORKERS)(
        delayed(_fit_single_pixel_worker)(task)
        for task in tqdm(tasks, desc=f"Per-pixel SV fitting ({model_type})")
    )

    num_params = 2 if model_type == 'linear' else 4
    all_params = np.full((num_pixels, num_params), np.nan)
    all_r2 = np.full(num_pixels, np.nan)
    all_dr = np.full(num_pixels, np.nan)

    for res in results:
        pixel_idx, params, r2, dr = res
        all_params[pixel_idx, :] = params
        all_r2[pixel_idx] = r2
        all_dr[pixel_idx] = dr

    metrics_df = pd.DataFrame(index=range(num_pixels))
    if model_type == 'linear':
        metrics_df['i0'] = all_params[:, 0]
        metrics_df['ksv'] = all_params[:, 1]
    else:
        metrics_df['i0'] = all_params[:, 0]
        metrics_df['ksv1'] = all_params[:, 1]
        metrics_df['ksv2'] = all_params[:, 2]
        metrics_df['a'] = all_params[:, 3]

    metrics_df['r_squared'] = all_r2
    metrics_df['dynamic_range'] = all_dr

    # LOD calculation (linear model only; Eq. 7 in Methods)
    if model_type == 'linear':
        stds_per_interval = []
        for start, end in ANALYSIS_INTERVALS_S:
            interval_mask = (
                (train_df['elapsed_seconds'] >= start)
                & (train_df['elapsed_seconds'] <= end)
            )
            stds_per_interval.append(np.std(train_pixel_data[interval_mask], axis=0))
        avg_std = np.nanmean(stds_per_interval, axis=0)
        metrics_df['lod'] = metrics_df['ksv'] * avg_std

    return metrics_df


# ==============================================================================
# STAGE 3: PHYSICS-REINFORCED LGBM FEATURE ENGINEERING
# ==============================================================================
# The LGA model (Main text, Fig. 3) uses aggregated statistics from red-channel
# intensities AND physics-derived parameters as input features. This is termed
# "physics-reinforced" because physical quantities inform the feature space but
# are NOT enforced through the loss function (contrast with the PINN approach
# described in the main text).


def generate_lgbm_features(pixel_intensities, pixel_metrics_df):
    """Generate the feature set for the physics-reinforced LGA model.

    Features include:
      - Intensity statistics: mean, std, Q25, Q50, Q75 across selected pixels
      - Aggregated physics parameters: mean and std of each metric (I₀, K_SV,
        R², DR, LOD) computed across the selected pixel cohort

    This corresponds to the "rich input feature set" described in the Results
    section for the LGA model.
    """
    features = pd.DataFrame()
    features['mean_intensity'] = np.mean(pixel_intensities, axis=1)
    features['std_intensity'] = np.std(pixel_intensities, axis=1)
    features['q25_intensity'] = np.percentile(pixel_intensities, 25, axis=1)
    features['q50_intensity'] = np.percentile(pixel_intensities, 50, axis=1)
    features['q75_intensity'] = np.percentile(pixel_intensities, 75, axis=1)

    for col in pixel_metrics_df.columns:
        features[f'mean_{col}'] = pixel_metrics_df[col].mean()
        features[f'std_{col}'] = pixel_metrics_df[col].std()

    return features.fillna(0)


# ==============================================================================
# STAGE 4: VISUALISATION & REPORTING
# ==============================================================================
# Functions for generating publication figures. Output filenames are annotated
# with the corresponding figure number in the manuscript.


def save_plot_and_data(fig, plot_name, data_df, output_dir):
    """Save a figure as PNG (300 DPI) and its underlying data as CSV."""
    os.makedirs(output_dir, exist_ok=True)
    if fig:
        fig.savefig(os.path.join(output_dir, f"{plot_name}.png"),
                    dpi=300, bbox_inches='tight')
    if data_df is not None and not data_df.empty:
        data_df.to_csv(os.path.join(output_dir, f"{plot_name}_data.csv"), index=False)
    if fig:
        plt.close(fig)


def plot_best_pixels_summary(mae_summary_df, output_dir):
    """Bar chart: MAE for all 'Best Pixels' strategies (→ Main text Fig. 3E)."""
    plot_dir = os.path.join(output_dir, "4_best_pixel_strategy_comparison")
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.barplot(data=mae_summary_df, x='MAE (umol/L)', y='Strategy',
                hue='Model Type', ax=ax, dodge=True)
    ax.set_title('Performance of "Best Pixels" Strategies (Cross-Validated MAE)',
                 fontsize=16, weight='bold')
    ax.set_xlabel('Mean Absolute Error (µmol/L)', fontsize=12)
    ax.set_ylabel('Pixel Selection Strategy', fontsize=12)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot_and_data(fig, 'best_pixels_strategy_comparison', mae_summary_df, plot_dir)


def plot_lgbm_summary(lgbm_mae_df, best_model_results, feature_importances, output_dir):
    """Summary plots for the LGA model (→ Main text Fig. 3)."""
    plot_dir = os.path.join(output_dir, "5_lgbm_analysis")

    # MAE vs. number of input pixels
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    sns.barplot(data=lgbm_mae_df, x='N_Pixels', y='MAE (umol/L)', ax=ax1, color='c')
    ax1.set_title('LGBM Model Performance vs. Number of Input Pixels', fontsize=16)
    ax1.set_xlabel('Number of Top Pixels Used (Ranked by R²)', fontsize=12)
    save_plot_and_data(fig1, 'lgbm_mae_vs_n_pixels', lgbm_mae_df, plot_dir)

    # Parity plot for best LGBM (→ Main text Fig. 3B)
    mae = mean_absolute_error(best_model_results['o2_true'], best_model_results['o2_pred'])
    r2 = r2_score(best_model_results['o2_true'], best_model_results['o2_pred'])
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    sns.scatterplot(data=best_model_results, x='o2_true', y='o2_pred',
                    alpha=0.5, ax=ax2, s=15, edgecolor=None)
    min_val = min(best_model_results['o2_true'].min(), best_model_results['o2_pred'].min())
    max_val = max(best_model_results['o2_true'].max(), best_model_results['o2_pred'].max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal (y=x)')
    ax2.set_title(f'Parity Plot: Best LGBM Model\nMAE = {mae:.3f} µmol/L, R² = {r2:.3f}')
    ax2.set_xlabel('True Oxygen (µmol/L)')
    ax2.set_ylabel('Predicted Oxygen (µmol/L)')
    ax2.legend()
    save_plot_and_data(fig2, 'lgbm_best_model_parity_plot', best_model_results, plot_dir)

    # Feature importance (→ Main text Fig. 3 / SI Fig. S1C-D)
    fig3, ax3 = plt.subplots(figsize=(12, 10))
    sns.barplot(data=feature_importances.head(25), x='Importance', y='Feature',
                ax=ax3, palette='viridis')
    ax3.set_title('Top 25 Feature Importances for Best LGBM Model', fontsize=16)
    save_plot_and_data(fig3, 'lgbm_feature_importance', feature_importances, plot_dir)


def generate_spatial_maps(pixel_fits_np, model_type, resize_dim, output_dir):
    """Spatial heatmaps of fitted SV parameters (→ Main text Fig. 2C-D).

    Uses two-step NaN imputation:
      1. Per-pixel mean across all LOOCV folds (ignoring NaN folds)
      2. Global mean imputation for pixels that failed in ALL folds
    """
    print(f"\n--- Generating Spatial Heterogeneity Maps ({model_type}) ---")
    spatial_dir = os.path.join(output_dir, "2_spatial_heterogeneity_maps")

    if model_type == 'linear':
        params = {'i0': 0, 'ksv': 1, 'r_squared': 2}
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
        fig.suptitle("Spatial Distribution of Fitted Linear SV Parameters "
                     "(Averaged Across Folds)", fontsize=16)
    else:
        params = {'i0': 0, 'ksv1': 1, 'ksv2': 2, 'a': 3, 'r_squared': 4}
        fig, axes = plt.subplots(1, 5, figsize=(35, 7))
        fig.suptitle("Spatial Distribution of Fitted Non-Linear SV Parameters "
                     "(Averaged Across Folds)", fontsize=16)

    for i, (name, idx) in enumerate(params.items()):
        metric_data_all_folds = pixel_fits_np[:, :, idx]
        per_pixel_mean = np.nanmean(metric_data_all_folds, axis=0)
        global_mean = np.nanmean(per_pixel_mean)
        if np.isnan(global_mean):
            global_mean = 0
        nan_mask = np.isnan(per_pixel_mean)
        if np.any(nan_mask):
            print(f"  '{name}': imputing {np.sum(nan_mask)} all-NaN pixels "
                  f"with global mean ({global_mean:.4f})")
            per_pixel_mean[nan_mask] = global_mean

        param_map = per_pixel_mean.reshape(resize_dim)
        ax = axes.flat[i]
        sns.heatmap(param_map, cmap='viridis', ax=ax, cbar_kws={'label': name})
        ax.set_title(f'Heatmap of {name}')
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        save_plot_and_data(None, f'spatial_map_{model_type}_{name}',
                          pd.DataFrame(param_map), spatial_dir)

    save_plot_and_data(fig, f'spatial_maps_combined_{model_type}', None, spatial_dir)


def analyze_pixel_performance_correlation(all_pixel_fits_np, all_pixel_maes_np,
                                          model_type, output_dir):
    """Scatter plot: training R² vs. test MAE per pixel (diagnostic analysis)."""
    print(f"\n--- Pixel Performance Correlation ({model_type}) ---")
    corr_dir = os.path.join(output_dir, "3_pixel_performance_correlation")

    r2_idx = 2 if model_type == 'linear' else 4
    r2_train = all_pixel_fits_np[:, :, r2_idx].flatten()
    mae_test = all_pixel_maes_np.flatten()

    df_corr = pd.DataFrame({'training_r2': r2_train, 'test_mae': mae_test}).dropna()
    plot_df = df_corr.sample(n=min(50000, len(df_corr)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.regplot(data=plot_df, x='training_r2', y='test_mae', ax=ax,
                scatter_kws={'alpha': 0.1, 's': 10}, line_kws={'color': 'red'})
    correlation = df_corr['training_r2'].corr(df_corr['test_mae'])
    ax.set_title(f'Pixel Performance Correlation ({model_type})\n'
                 f'Pearson r = {correlation:.3f}')
    ax.set_xlabel('Pixel R² on Training Data (Fit Quality)')
    ax.set_ylabel('Pixel MAE on Unseen Test Data (Prediction Quality)')
    ax.set_ylim(bottom=0, top=np.percentile(plot_df['test_mae'], 99))
    save_plot_and_data(fig, f'pixel_performance_correlation_{model_type}',
                       df_corr, corr_dir)


# ==============================================================================
# STAGE 5: PUBLICATION FIGURE GENERATION
# ==============================================================================
# Generates the multi-panel figures as they appear in the manuscript.


def generate_publication_figures(results_dict, resize_dim, output_dir):
    """Generate all publication-specific composite figures.

    Outputs:
      Fig_a → Main text Fig. 3E (performance comparison bar chart)
      Fig_b → Main text Fig. 3A-B (parity plots: GA vs LGA)
      Fig_c → Main text Fig. 2C-D (spatial R² and I₀ heatmaps)
      Fig_d → Main text Fig. 3D (error distribution)
      Fig_e → Main text Fig. 3 / SI Fig. S1C (feature importance)
    """
    print("\n\n--- Generating Publication Figures ---")
    pub_dir = os.path.join(output_dir, "6_publication_figures")
    os.makedirs(pub_dir, exist_ok=True)

    # --- Fig. 3E: Performance Comparison Bar Chart ---
    print("  Fig. 3E: Performance comparison")
    mae_global = results_dict['mae_global']
    best_pixels_df = results_dict['best_pixels_summary']
    mae_best_pixels_lod100 = best_pixels_df[
        (best_pixels_df['Strategy'] == 'Best 100 by lod')
        & (best_pixels_df['Model Type'] == 'Linear')
    ]['MAE (umol/L)'].iloc[0]
    lgbm_df = results_dict['lgbm_summary']
    mae_lgbm_1000 = lgbm_df[lgbm_df['N_Pixels'] == '1000']['MAE (umol/L)'].iloc[0]

    df_a = pd.DataFrame({
        'Method': ['Global Average', 'Best Pixels (Top 100 by LOD)', 'LGBM (Top 1000 Pixels)'],
        'MAE (umol/L)': [mae_global, mae_best_pixels_lod100, mae_lgbm_1000]
    })
    fig_a, ax_a = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df_a, x='Method', y='MAE (umol/L)', ax=ax_a, palette='muted')
    ax_a.set_title('Performance Comparison of Calibration Methods',
                   fontsize=16, weight='bold')
    ax_a.set_ylabel('Cross-Validated MAE (µmol/L)', fontsize=12)
    ax_a.set_xlabel('')
    ax_a.tick_params(axis='x', rotation=15)
    for container in ax_a.containers:
        ax_a.bar_label(container, fmt='%.2f')
    plt.tight_layout()
    save_plot_and_data(fig_a, 'Fig_3E_Performance_Comparison', df_a, pub_dir)

    # --- Fig. 3A-B: Parity Plots (GA vs LGA) ---
    print("  Fig. 3A-B: Parity plots")
    oof_global = results_dict['oof_global']
    oof_lgbm_best = results_dict['oof_lgbm_best']

    fig_b, (ax_b1, ax_b2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True, sharex=True)
    fig_b.suptitle('Model Performance: Global Average vs. Best LGBM',
                   fontsize=18, weight='bold')

    mae_b1 = mean_absolute_error(oof_global['o2_true'], oof_global['o2_pred'])
    r2_b1 = r2_score(oof_global['o2_true'], oof_global['o2_pred'])
    sns.scatterplot(data=oof_global, x='o2_true', y='o2_pred',
                    ax=ax_b1, alpha=0.3, s=15, edgecolor=None)
    min_val, max_val = oof_global['o2_true'].min(), oof_global['o2_true'].max()
    ax_b1.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal (y=x)')
    ax_b1.set_title(f'Global Average (GA)\nMAE = {mae_b1:.2f} µmol/L, R² = {r2_b1:.2f}')
    ax_b1.set_xlabel('True Oxygen (µmol/L)')
    ax_b1.set_ylabel('Predicted Oxygen (µmol/L)')
    ax_b1.legend()
    ax_b1.set_aspect('equal', adjustable='box')

    mae_b2 = mean_absolute_error(oof_lgbm_best['o2_true'], oof_lgbm_best['o2_pred'])
    r2_b2 = r2_score(oof_lgbm_best['o2_true'], oof_lgbm_best['o2_pred'])
    sns.scatterplot(data=oof_lgbm_best, x='o2_true', y='o2_pred',
                    ax=ax_b2, alpha=0.3, s=15, edgecolor=None)
    ax_b2.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal (y=x)')
    ax_b2.set_title(f'Physics-Reinforced LGBM (LGA)\nMAE = {mae_b2:.2f} µmol/L, R² = {r2_b2:.2f}')
    ax_b2.set_xlabel('True Oxygen (µmol/L)')
    ax_b2.set_ylabel('')
    ax_b2.legend()
    ax_b2.set_aspect('equal', adjustable='box')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    df_b = pd.concat([
        oof_global.rename(columns={'o2_pred': 'o2_pred_global'}),
        oof_lgbm_best.rename(columns={'o2_pred': 'o2_pred_lgbm_best'})['o2_pred_lgbm_best']
    ], axis=1)
    save_plot_and_data(fig_b, 'Fig_3AB_Parity_Plots', df_b, pub_dir)

    # --- Fig. 2C-D: Spatial Heterogeneity Heatmaps ---
    print("  Fig. 2C-D: Spatial heterogeneity")
    np_fits_linear = results_dict['np_fits_linear']

    r2_map = np.nanmean(np_fits_linear[:, :, 2], axis=0)
    r2_map[np.isnan(r2_map)] = np.nanmean(r2_map)
    r2_map = r2_map.reshape(resize_dim)

    i0_map = np.nanmean(np_fits_linear[:, :, 0], axis=0)
    i0_map[np.isnan(i0_map)] = np.nanmean(i0_map)
    i0_map = i0_map.reshape(resize_dim)

    fig_c, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(16, 7))
    fig_c.suptitle('Spatial Heterogeneity of Sensor Response (Averaged Across Folds)',
                   fontsize=18, weight='bold')
    sns.heatmap(r2_map, cmap='viridis', ax=ax_c1,
                cbar_kws={'label': 'R² (Goodness-of-Fit)'})
    ax_c1.set_title('Goodness-of-Fit (R²)', fontsize=14)
    ax_c1.set_xlabel('Pixel X')
    ax_c1.set_ylabel('Pixel Y')
    sns.heatmap(i0_map, cmap='inferno', ax=ax_c2,
                cbar_kws={'label': 'Intensity at 0% O₂'})
    ax_c2.set_title('Zero-Oxygen Intensity (I₀)', fontsize=14)
    ax_c2.set_xlabel('Pixel X')
    ax_c2.set_ylabel('')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    df_c = pd.DataFrame({'r2': r2_map.flatten(), 'i0': i0_map.flatten()})
    save_plot_and_data(fig_c, 'Fig_2CD_Spatial_Heterogeneity', df_c, pub_dir)

    # --- Fig. 3D: Error Distribution ---
    print("  Fig. 3D: Error distribution")
    df_d = oof_lgbm_best.copy()
    df_d['error'] = df_d['o2_pred'] - df_d['o2_true']
    df_d['o2_bin'] = pd.cut(
        df_d['o2_true'], bins=8,
        labels=[f'{int(b.mid)}' for b in pd.cut(df_d['o2_true'], bins=8).categories]
    )
    fig_d, ax_d = plt.subplots(figsize=(14, 8))
    sns.violinplot(data=df_d, x='o2_bin', y='error', ax=ax_d,
                   palette='coolwarm', inner='quartile', cut=0)
    ax_d.axhline(0, color='k', linestyle='--', lw=2)
    ax_d.set_title('Distribution of Prediction Errors for Best LGBM Model',
                   fontsize=16, weight='bold')
    ax_d.set_xlabel('Ground Truth Oxygen Bin (µmol/L)', fontsize=12)
    ax_d.set_ylabel('Prediction Error (Predicted − True) (µmol/L)', fontsize=12)
    save_plot_and_data(fig_d, 'Fig_3D_Error_Distribution',
                       df_d[['o2_true', 'o2_pred', 'error', 'o2_bin']], pub_dir)

    # --- Fig. 3 / SI Fig. S1C: Feature Importance ---
    print("  Feature importance")
    df_e = results_dict['feature_importances'].head(15)
    fig_e, ax_e = plt.subplots(figsize=(12, 8))
    sns.barplot(data=df_e, x='Importance', y='Feature', ax=ax_e, palette='viridis')
    ax_e.set_title('Top 15 Feature Importances for Physics-Reinforced LGBM',
                   fontsize=16, weight='bold')
    ax_e.set_xlabel('Importance Score', fontsize=12)
    ax_e.set_ylabel('Feature', fontsize=12)
    plt.tight_layout()
    save_plot_and_data(fig_e, 'Fig_Feature_Importance', df_e, pub_dir)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Orchestrate the full baseline analysis pipeline.

    Workflow:
      Phase 1 — Load and preprocess raw experimental data
      Phase 2 — Run LOOCV for all baseline models (GA, Best Pixels, LGA)
      Phase 3 — Generate reports, spatial maps, and publication figures
    """
    start_time = datetime.now()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CLASSICAL_CACHE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Data Loading & Preparation
    # ------------------------------------------------------------------
    print("=" * 70)
    print("PHASE 1: Data Loading & Preparation")
    print("=" * 70)

    experiments = find_experiment_files(BASE_PROJECT_DIR, NUM_EXPERIMENT_DAYS_TO_USE)
    master_df = create_pinn_dataframe(
        experiments, FEATURE_DATAFRAME_PATH, FRAME_SKIP, RESIZE_DIM
    )

    # Filter to stable DO concentration intervals only
    conditions = [
        (master_df['elapsed_seconds'] >= s) & (master_df['elapsed_seconds'] <= e)
        for s, e in ANALYSIS_INTERVALS_S
    ]
    filtered_df = master_df[functools.reduce(np.logical_or, conditions)].reset_index(drop=True)
    pixel_cols = [c for c in filtered_df.columns if c.startswith('pixel_')]
    red_channel_pixels = filtered_df[pixel_cols].values.astype(np.uint8)
    print(f"Data prepared. Red channel shape: {red_channel_pixels.shape}")

    # ------------------------------------------------------------------
    # Phase 2: Leave-One-(Day-)Out Cross-Validation
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 2: Cross-Validation (LOOCV by Experiment Day)")
    print("=" * 70)

    groups = filtered_df['experiment_id']
    logo = LeaveOneGroupOut()

    oof_results_global = pd.DataFrame()
    oof_results_best_pixels = []
    oof_results_lgbm = {f'LGBM_{n}': pd.DataFrame() for n in BEST_PIXELS_N + ['All']}
    all_folds_pixel_metrics = {'linear': [], 'nonlinear': []}
    all_folds_pixel_maes_linear = []

    n_folds = logo.get_n_splits(groups=groups)

    for fold, (train_idx, val_idx) in enumerate(logo.split(filtered_df, groups=groups)):
        train_df = filtered_df.iloc[train_idx]
        val_df = filtered_df.iloc[val_idx]
        val_group = val_df['experiment_id'].unique()[0]
        print(f"\n--- Fold {fold + 1}/{n_folds}: Holding out {val_group} ---")

        train_pixels = red_channel_pixels[train_idx]
        val_pixels = red_channel_pixels[val_idx]
        train_o2 = train_df['oxygen_umol_L'].values
        val_o2_true = val_df['oxygen_umol_L'].values

        # -- Model: Global Average (GA) --
        train_global_intensity = np.mean(train_pixels, axis=1)
        global_params = fit_sv_model(train_o2, train_global_intensity, 'linear')
        if global_params is not None:
            val_global_intensity = np.mean(val_pixels, axis=1)
            o2_pred_global = predict_o2_linear(val_global_intensity, *global_params)
            temp_df_global = pd.DataFrame({
                'o2_true': val_o2_true, 'o2_pred': o2_pred_global
            })
            oof_results_global = pd.concat(
                [oof_results_global, temp_df_global], ignore_index=True
            )

        # -- Per-pixel SV fitting (for Best Pixels and LGA) --
        pixel_metrics = {}
        for model_type in ['linear', 'nonlinear']:
            cache_path = os.path.join(
                CLASSICAL_CACHE_DIR, f'pixel_metrics_fold_{fold}_{model_type}.parquet'
            )
            if os.path.exists(cache_path) and not FORCE_REFIT_PIXELS:
                pixel_metrics[model_type] = pd.read_parquet(cache_path)
            else:
                pixel_metrics[model_type] = calculate_pixel_metrics(
                    train_df, train_pixels, model_type
                )
                pixel_metrics[model_type].to_parquet(cache_path)
        all_folds_pixel_metrics['linear'].append(pixel_metrics['linear'])
        all_folds_pixel_metrics['nonlinear'].append(pixel_metrics['nonlinear'])

        # -- Model: Best Pixels strategies --
        metrics_to_rank = {
            'r_squared': False,     # Higher is better
            'dynamic_range': False,
            'i0': False,
            'ksv': False,
            'lod': True,            # Lower is better
        }
        for mt_label in ['Linear', 'NonLinear']:
            mt_lower = mt_label.lower()
            current_metrics = pixel_metrics[mt_lower].dropna()
            for metric, ascending in metrics_to_rank.items():
                if metric not in current_metrics.columns:
                    continue
                for n in BEST_PIXELS_N:
                    top_n_indices = current_metrics.sort_values(
                        metric, ascending=ascending
                    ).head(n).index
                    train_super_pixel = np.mean(train_pixels[:, top_n_indices], axis=1)
                    val_super_pixel = np.mean(val_pixels[:, top_n_indices], axis=1)
                    params = fit_sv_model(train_o2, train_super_pixel, mt_lower)
                    if params is not None:
                        pred_func = (predict_o2_linear if mt_lower == 'linear'
                                     else predict_o2_nonlinear)
                        o2_pred = pred_func(val_super_pixel, *params)
                        valid_preds = ~np.isnan(o2_pred)
                        if np.any(valid_preds):
                            mae = mean_absolute_error(
                                val_o2_true[valid_preds], o2_pred[valid_preds]
                            )
                            oof_results_best_pixels.append({
                                'Strategy': f'Best {n} by {metric}',
                                'MAE (umol/L)': mae,
                                'Model Type': mt_label,
                            })

        # -- Per-pixel prediction error (for spatial heatmaps, SI Fig. S2) --
        pm_linear = pixel_metrics['linear']
        pixel_preds_linear = predict_o2_linear(
            val_pixels, pm_linear['i0'].values, pm_linear['ksv'].values
        )
        pixel_abs_errors = np.abs(pixel_preds_linear - val_o2_true[:, np.newaxis])
        all_folds_pixel_maes_linear.append(np.nanmean(pixel_abs_errors, axis=0))

        # -- Model: Physics-reinforced LGBM (LGA) --
        pixel_rank_indices = (pixel_metrics['linear'].dropna()
                              .sort_values('r_squared', ascending=False).index)
        n_pixel_options = {
            'LGBM_10': 10,
            'LGBM_100': 100,
            'LGBM_1000': 1000,
            'LGBM_All': len(pixel_rank_indices),
        }
        for model_name, n in n_pixel_options.items():
            top_n_indices = pixel_rank_indices[:n]
            X_train = generate_lgbm_features(
                train_pixels[:, top_n_indices],
                pixel_metrics['linear'].loc[top_n_indices]
            )
            X_val = generate_lgbm_features(
                val_pixels[:, top_n_indices],
                pixel_metrics['linear'].loc[top_n_indices]
            )
            model = lgb.LGBMRegressor(
                random_state=42, n_jobs=-1, verbosity=-1, device=LGBM_DEVICE
            )
            model.fit(X_train, train_o2)
            o2_pred = model.predict(X_val)
            temp_df = pd.DataFrame({'o2_true': val_o2_true, 'o2_pred': o2_pred})
            oof_results_lgbm[model_name] = pd.concat(
                [oof_results_lgbm[model_name], temp_df], ignore_index=True
            )

    # ------------------------------------------------------------------
    # Phase 3: Reporting & Publication Figures
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PHASE 3: Generating Reports & Publication Figures")
    print("=" * 70)

    # Best Pixels summary
    best_pixels_summary_df = (
        pd.DataFrame(oof_results_best_pixels)
        .groupby(['Strategy', 'Model Type']).mean().reset_index()
    )
    plot_best_pixels_summary(best_pixels_summary_df, OUTPUT_DIR)

    # LGBM summary
    lgbm_mae_summary = [{
        'N_Pixels': name.split('_')[1],
        'MAE (umol/L)': mean_absolute_error(df['o2_true'], df['o2_pred'])
    } for name, df in oof_results_lgbm.items() if not df.empty]
    lgbm_mae_df = pd.DataFrame(lgbm_mae_summary)
    best_lgbm_model_name = lgbm_mae_df.loc[lgbm_mae_df['MAE (umol/L)'].idxmin()]['N_Pixels']
    best_lgbm_results_df = oof_results_lgbm[f'LGBM_{best_lgbm_model_name}']

    # Train final LGBM on all data for feature importance analysis
    print("Training final LGBM model for feature importance...")
    final_pixel_metrics = calculate_pixel_metrics(filtered_df, red_channel_pixels, 'linear')
    final_rank_indices = (final_pixel_metrics.dropna()
                          .sort_values('r_squared', ascending=False).index)
    best_n = (int(best_lgbm_model_name) if best_lgbm_model_name != 'All'
              else len(final_rank_indices))
    top_indices_final = final_rank_indices[:best_n]
    X_final = generate_lgbm_features(
        red_channel_pixels[:, top_indices_final],
        final_pixel_metrics.loc[top_indices_final]
    )
    y_final = filtered_df['oxygen_umol_L'].values
    final_lgbm = lgb.LGBMRegressor(
        random_state=42, verbosity=-1, device=LGBM_DEVICE
    ).fit(X_final, y_final)
    feature_importances_df = (
        pd.DataFrame({
            'Feature': X_final.columns,
            'Importance': final_lgbm.feature_importances_,
        })
        .sort_values('Importance', ascending=False)
    )
    plot_lgbm_summary(lgbm_mae_df, best_lgbm_results_df, feature_importances_df, OUTPUT_DIR)

    # Spatial heterogeneity maps
    def metrics_to_numpy(metrics_list, param_cols):
        num_pixels = RESIZE_DIM[0] * RESIZE_DIM[1]
        return np.array([
            df.reindex(range(num_pixels))[param_cols].values for df in metrics_list
        ])

    np_fits_linear = metrics_to_numpy(
        all_folds_pixel_metrics['linear'], ['i0', 'ksv', 'r_squared']
    )
    np_fits_nonlinear = metrics_to_numpy(
        all_folds_pixel_metrics['nonlinear'], ['i0', 'ksv1', 'ksv2', 'a', 'r_squared']
    )
    generate_spatial_maps(np_fits_linear, 'linear', RESIZE_DIM, OUTPUT_DIR)
    generate_spatial_maps(np_fits_nonlinear, 'nonlinear', RESIZE_DIM, OUTPUT_DIR)

    # Pixel performance correlation
    np_maes_linear = np.array(all_folds_pixel_maes_linear)
    analyze_pixel_performance_correlation(
        np_fits_linear, np_maes_linear, 'linear', OUTPUT_DIR
    )

    # Publication figures
    publication_results = {
        'mae_global': mean_absolute_error(
            oof_results_global['o2_true'], oof_results_global['o2_pred']
        ),
        'oof_global': oof_results_global,
        'best_pixels_summary': best_pixels_summary_df,
        'lgbm_summary': lgbm_mae_df,
        'oof_lgbm_best': best_lgbm_results_df,
        'np_fits_linear': np_fits_linear,
        'feature_importances': feature_importances_df,
    }
    generate_publication_figures(publication_results, RESIZE_DIM, OUTPUT_DIR)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    elapsed = datetime.now() - start_time
    print(f"\n{'=' * 70}")
    print(f"COMPLETE — Total execution time: {elapsed}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    plt.style.use('seaborn-v0_8-whitegrid')
    try:
        main()
    except Exception as e:
        print(f"\n--- FATAL ERROR ---\n{e}")
        traceback.print_exc()
        sys.exit(1)