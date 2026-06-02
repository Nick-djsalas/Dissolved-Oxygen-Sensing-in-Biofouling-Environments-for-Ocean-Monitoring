# ==============================================================================
# SCRIPT OVERVIEW: Physics-Informed Vision Transformer Framework for DO Sensing
# ==============================================================================
# This script provides a comprehensive, end-to-end computational framework for
# training and evaluating a Physics-Informed Neural Network (PINN). The primary
# scientific objective is to estimate dissolved oxygen (DO) concentrations directly
# from video frames of a luminescent sensor, while simultaneously quantifying
# the degree of surface biofouling and the model's predictive uncertainty.
#
# This code serves as the companion framework for robust, real-world deployment
# simulations, adhering to strict data validation protocols to prevent temporal
# leakage and ensure scientific rigor.
#
# --- METHODOLOGICAL BREAKDOWN ---
#
# 1. DATA PREPROCESSING & SYNCHRONIZATION:
#    - Automated discovery and chronological sorting of experimental datasets.
#    - Synchronizes high-frequency video frames with low-frequency sensor telemetry
#      using non-overlapping time windows to ensure target stability.
#    - Caches extracted features to accelerate subsequent executions.
#
# 2. MODEL ARCHITECTURE (ViT-PINN):
#    - VISION TRANSFORMER (ViT) BACKBONE: Leverages self-attention mechanisms to
#      weigh the importance of every image patch globally. This enables the model
#      to capture the diffuse, large-scale spatial heterogeneities characteristic
#      of biofouling better than localized convolutional approaches.
#    - MULTI-HEAD DESIGN: Simultaneously predicts oxygen concentration, a biofouling
#      segmentation mask, experiment-specific Stern-Volmer parameters, and a
#      spatial confidence map.
#
# 3. PHYSICS-INFORMED LOSS FUNCTION:
#    - Integrates a composite loss function that balances standard empirical
#      supervision (data loss) with a physics-based regularization term derived
#      from the Stern-Volmer quenching equation.
#    - NORMALIZED CONFIDENCE WEIGHTING: The physics residual is spatially weighted
#      by the learned confidence map, allowing the model to dynamically down-weight
#      regions corrupted by severe biofouling or optical occlusion.
#
# 4. RIGOROUS VALIDATION (Chronological Forward-Chaining):
#    - To simulate a realistic deployment and prevent future data leakage, the
#      framework employs a strict "Walk-Forward" temporal validation strategy:
#        * TRAIN SET: Historical data (Days 1 to t).
#        * VALIDATION SET: Immediate next phase (Day t+1) for hyperparameter tuning
#          and early stopping.
#        * TEST SET: All subsequent future data (Days t+2 onward) strictly reserved
#          for final performance evaluation.
#
# 5. UNCERTAINTY QUANTIFICATION (Deep Ensembles):
#    - Trains an ensemble of models with diverse initializations for each temporal
#      split. Final predictions utilize the ensemble mean, while the standard
#      deviation serves as a quantifiable metric of model confidence.
#
# --- AUTOMATED OUTPUT STRUCTURE ---
# The framework automatically generates comprehensive diagnostic and results directories:
# - `detailed_analysis_report/`: Hub for all analytical plots and underlying CSV data.
#   - `0_data_preprocessing_reports/`: Alignment and signal quality diagnostics.
#   - `1_data_split_reports/`: Train/Val/Test temporal distribution overlaps.
#   - `split_train_[N]_days/`: Dedicated results for each chronological split, including:
#     - Convergence trajectories and loss component analysis.
#     - Spatial heatmaps of physics residuals and attention/confidence maps.
#     - Parity plots and uncertainty correlations on the unseen future test set.
# - `hpo_analysis/`: Hyperparameter optimization databases and importance plots.
# - `cv_split_cache/`: Saved ensemble checkpoints for reproducible inference.
# ==============================================================================

# --- IMPORTS ---
import os, sys, glob, re, traceback, warnings, random, json
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
import functools
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import timm
from torchvision import transforms
import shutil

# ==============================================================================
# --- REPRODUCIBILITY CONFIGURATION ---
# ==============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --- GLOBAL SETTINGS ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
# --- Primary Paths ---
# MODIFY THIS PATH TO POINT TO YOUR DATASET ROOT
BASE_PROJECT_DIR = r"F:\UNI papers - ALL data\algae\tests\algae_new\exp_1_algae\video\5-algae_tests"
EXPERIMENTS_ROOT_DIR = BASE_PROJECT_DIR
# Versioned output directory
OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "PINN_Analysis_V8_3_Chronological")
ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "detailed_analysis_report")

# --- Cache Paths (Stored outside output dir for reusability across versions) ---
FEATURE_CACHE_DIR = os.path.join(BASE_PROJECT_DIR, "PINN_features")
FEATURE_DATAFRAME_PATH = os.path.join(FEATURE_CACHE_DIR, "pinn_features_resampled.parquet")
HPO_CACHE_DIR = os.path.join(BASE_PROJECT_DIR, "PINN_hpo_cache")
HPO_PARAMS_PATH = os.path.join(HPO_CACHE_DIR, "best_hpo_params_v83.json")  # Versioned HPO params
HPO_STUDY_DB_PATH = os.path.join(HPO_CACHE_DIR, "hpo_study_v83.db")

# --- Data Processing Settings ---
FORCE_RECREATE_DATAFRAME = False
FORCE_RECREATE_HEATMAPS = False

NUM_EXPERIMENT_DAYS_TO_USE = None  # Set to None to use all available, or integer to limit
FRAME_SKIP = 15
STORAGE_RESIZE_DIM = (48, 48)  # How frames are stored on disk (Features)
MODEL_INPUT_DIM = (224, 224)  # How frames are fed to the ViT model
FORCED_VIDEO_FPS = 30
FRAME_CHUNK_SIZE = 2500
# Analysis intervals in seconds (stable measurement periods)
ANALYSIS_INTERVALS_S = [(1450, 1650), (2300, 2500), (3100, 3300), (3900, 4100), (4700, 4900)]
SENSOR_RESAMPLE_WINDOW_S = 10

# --- HPO & Training Settings ---
FORCE_RERUN_HPO = False  # If True, deletes the HPO database and starts over.
FORCE_RERUN_SPLITS = False  # If True, ignores all cached models and split results.

PHYSICS_MODEL = 'nonlinear'
N_TRIALS_FOR_TUNING = 10
N_EPOCHS_FOR_TUNING = 20
N_EPOCHS_FOR_FINAL_TRAINING = 35  # Increased slightly for forward chaining robustness
ENSEMBLE_SIZE = 1  # Number of models to train per split for uncertainty (Increase for better UQ)
HPO_DATA_SUBSET_FRACTION = 0.35  # Fraction of data to use for HPO
BATCH_SIZE = 24
GRAD_ACCUMULATION_STEPS = 24
EARLY_STOPPING_PATIENCE = 15  # Stop if validation loss doesn't improve
GRADIENT_CLIP_VALUE = 1.0
NUM_WORKERS = 0  # Set to 0 for Windows compatibility, >0 for Linux/Mac

# --- Model & Training Strategy Toggles ---
USE_DATA_AUGMENTATION = True
USE_TRANSFER_LEARNING = True
TRANSFER_MODEL_NAME = 'vit_small_patch16_224.augreg_in21k_ft_in1k'
USE_LR_SCHEDULER = True
USE_CURRICULUM_LEARNING = True
LAMBDA_CURRICULUM_START = 0.01
LAMBDA_CURRICULUM_EPOCHS = 10
USE_CONFIDENCE_LOSS = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == 'cuda')
DEVICE_TYPE = DEVICE.type


# ==============================================================================
# --- STAGE 1: DATA DISCOVERY AND PREPROCESSING ---
# ==============================================================================

def parse_arduino_log(file_path):
    """
    Parses a raw Arduino oxygen log, handles duplicate timestamps by averaging,
    and then resamples the data into discrete time chunks (e.g., 10 seconds).
    """
    timestamps, oxygen_values = [], []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                if "MEA" not in line: continue
                parts = line.split()
                timestamp_str = f"{parts[0]} {parts[1]}"
                dt_object = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                mea_index = parts.index("MEA")
                raw_oxygen_value = int(parts[mea_index + 5])
                timestamps.append(dt_object)
                oxygen_values.append(raw_oxygen_value / 1000.0)
            except (ValueError, IndexError, TypeError):
                pass
    if not timestamps: return pd.DataFrame()
    df = pd.DataFrame({'timestamp': timestamps, 'oxygen_umol_L': oxygen_values})
    if df.timestamp.duplicated().any():
        df = df.groupby('timestamp')['oxygen_umol_L'].mean().reset_index()
    if not df.empty and SENSOR_RESAMPLE_WINDOW_S > 0:
        df = df.set_index('timestamp').resample(f'{SENSOR_RESAMPLE_WINDOW_S}S').mean().dropna().reset_index()
    return df


def parse_temperature_log(file_path):
    """
    Parses a temperature log, handles errors and duplicates, filters spikes,
    and then resamples the data into discrete time chunks (e.g., 10 seconds).
    """
    try:
        df = pd.read_csv(file_path)
        df.columns = ['timestamp', 'temperature_C']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['temperature_C'] = pd.to_numeric(df['temperature_C'], errors='coerce')
        df.dropna(inplace=True)
        if df.empty: return pd.DataFrame()
        if df.timestamp.duplicated().any():
            df = df.groupby('timestamp')['temperature_C'].mean().reset_index()
        df.sort_values('timestamp', inplace=True)
        df.reset_index(drop=True, inplace=True)
        pct_change = df['temperature_C'].pct_change().abs()
        spike_mask = pct_change >= 1.0
        if spike_mask.sum() > 0:
            df = df[~spike_mask].copy()
        if not df.empty and SENSOR_RESAMPLE_WINDOW_S > 0:
            df = df.set_index('timestamp').resample(f'{SENSOR_RESAMPLE_WINDOW_S}S').mean().dropna().reset_index()
        return df
    except Exception as e:
        print(f"Warning: Could not parse temperature file {os.path.basename(file_path)}. Error: {e}")
        return pd.DataFrame()


def find_experiment_files(root_dir, num_days_to_use=None):
    """
    Scans for valid experiments.
    CRITICAL: This function extracts dates and sorts experiments CHRONOLOGICALLY.
    This is essential for the Walk-Forward Validation strategy.
    """
    print("--- Discovering Experiment Files ---")
    all_dirs = [d for d in glob.glob(os.path.join(root_dir, '*-*')) if os.path.isdir(d)]
    dated_dirs = []
    for d in all_dirs:
        dir_name = os.path.basename(d)
        # Attempt to parse date from folder name (e.g., 20_05_2023)
        match = re.search(r'(\d{1,2}[-_]\d{2}[-_]\d{4})', dir_name.replace('_', '-'))
        if match:
            try:
                date_str = match.group(1).replace('_', '-')
                dated_dirs.append({'path': d, 'date': datetime.strptime(date_str, '%d-%m-%Y')})
            except ValueError:
                pass

    # --- CRITICAL STEP: SORT CHRONOLOGICALLY ---
    dated_dirs.sort(key=lambda x: x['date'])

    experiment_dirs = [d['path'] for d in (dated_dirs[:num_days_to_use] if num_days_to_use else dated_dirs)]
    valid_experiments = []
    for exp_dir in tqdm(experiment_dirs, desc="Scanning selected experiments"):
        exp_id = os.path.basename(exp_dir)
        roi_video_list = glob.glob(os.path.join(exp_dir, "**", "*ROI_Output*", "*.mp4"), recursive=True)
        arduino_data_list = glob.glob(os.path.join(exp_dir, "**", "*_arduino_*.txt"), recursive=True)
        temp_data_list = glob.glob(os.path.join(exp_dir, "**", "*temperature*.csv"), recursive=True) + \
                         glob.glob(os.path.join(exp_dir, "**", "*temperature*.txt"), recursive=True)
        if roi_video_list and arduino_data_list and temp_data_list:
            valid_experiments.append(
                {"id": exp_id, "video_path": roi_video_list[0], "raw_arduino_path": arduino_data_list[0],
                 "temperature_path": temp_data_list[0],
                 "date": [d['date'] for d in dated_dirs if d['path'] == exp_dir][0]}
            )
    if not valid_experiments: raise FileNotFoundError(
        "FATAL: No valid experiments found with all required files (video, oxygen, temperature).")

    print(f"Found {len(valid_experiments)} valid experiments (Sorted Chronologically).")
    for i, exp in enumerate(valid_experiments):
        print(f"  {i + 1}: {exp['date'].strftime('%Y-%m-%d')} - {exp['id']}")

    return valid_experiments


def extract_timestamp_from_filename(filename):
    """Extracts the start timestamp from a video filename."""
    match = re.search(r'_(\d{8}_\d{6})', filename)
    if not match: return None
    return datetime.strptime(match.group(1), '%Y%m%d_%H%M%S')


def process_experiment_chunked(args):
    """
    Worker function to process a single experiment. Always uses nearest-neighbor
    matching against the pre-resampled sensor data.
    """
    exp, frame_skip, resize_dim, temp_dir = args['exp'], args['frame_skip'], args['resize_dim'], args[
        'temp_dir']
    exp_id = exp['id']
    try:
        df_arduino = parse_arduino_log(exp['raw_arduino_path'])
        if df_arduino.empty: return False
        df_temp = parse_temperature_log(exp['temperature_path'])
        if df_temp.empty: return False

        df_arduino.set_index('timestamp', inplace=True)
        df_temp.set_index('timestamp', inplace=True)
        cap = cv2.VideoCapture(exp['video_path'])
        if not cap.isOpened(): return False
        fps = FORCED_VIDEO_FPS if FORCED_VIDEO_FPS else cap.get(cv2.CAP_PROP_FPS)
        video_start_time = extract_timestamp_from_filename(os.path.basename(exp['video_path']))
        if not fps or not video_start_time:
            cap.release()
            return False

        chunk_index = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc=f"Worker for {exp_id}", leave=False,
                  position=args.get('worker_id', 0)) as pbar:
            while True:
                chunk_data = []
                frames_in_chunk = 0
                while frames_in_chunk < FRAME_CHUNK_SIZE:
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    ret, frame = cap.read()
                    if not ret: break
                    pbar.update(1)
                    if frame_number % frame_skip == 0:
                        elapsed_seconds = frame_number / fps
                        frame_timestamp = video_start_time + timedelta(seconds=elapsed_seconds)
                        try:
                            nearest_o2_idx = df_arduino.index.get_indexer([frame_timestamp], method='nearest')[0]
                            nearest_temp_idx = df_temp.index.get_indexer([frame_timestamp], method='nearest')[0]
                            if nearest_o2_idx != -1 and nearest_temp_idx != -1:
                                oxygen_val = df_arduino.iloc[nearest_o2_idx]['oxygen_umol_L']
                                temp_val = df_temp.iloc[nearest_temp_idx]['temperature_C']
                                resized_frame = cv2.resize(frame, resize_dim, interpolation=cv2.INTER_AREA)
                                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                                chunk_data.append([exp_id, frame_timestamp, elapsed_seconds, temp_val,
                                                   oxygen_val] + rgb_frame.flatten().tolist())
                        except (KeyError, IndexError):
                            pass
                    frames_in_chunk += 1
                if not chunk_data: break
                pd.DataFrame(chunk_data).to_parquet(os.path.join(temp_dir, f"{exp_id}_chunk_{chunk_index}.parquet"))
                chunk_index += 1
                if not ret: break
        cap.release()
    except Exception:
        traceback.print_exc();
        return False
    return True


def save_plot_and_data(fig, plot_name, data_df, output_dir):
    """Saves a matplotlib figure and its underlying data to a CSV file in a specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    if fig:
        fig.savefig(os.path.join(output_dir, f"{plot_name}.png"), dpi=300, bbox_inches='tight')
    if data_df is not None and not data_df.empty:
        data_df.to_csv(os.path.join(output_dir, f"{plot_name}_data.csv"), index=False);
    if fig:
        plt.close(fig)


def visualize_alignment(master_df, experiments, analysis_dir):
    """
    Plots raw sensor data vs. the resampled and synced data for a sample experiment.
    """
    print("--- Generating Synchronization Alignment Visualization ---")
    sync_method = f"Resampled ({SENSOR_RESAMPLE_WINDOW_S}s Chunks) + Nearest Neighbor"
    exp_to_plot = master_df['experiment_id'].unique()[0]
    exp_meta = next((exp for exp in experiments if exp['id'] == exp_to_plot), None)
    if not exp_meta: return
    resampled_o2_df = parse_arduino_log(exp_meta['raw_arduino_path'])
    resampled_temp_df = parse_temperature_log(exp_meta['temperature_path'])
    processed_df = master_df[master_df['experiment_id'] == exp_to_plot].copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(f'Data Synchronization Check for Experiment: {exp_to_plot}\n(Method: {sync_method})',
                 fontsize=16, weight='bold')
    ax1.plot(resampled_o2_df['timestamp'], resampled_o2_df['oxygen_umol_L'],
             label=f'Resampled O₂ ({SENSOR_RESAMPLE_WINDOW_S}s Mean)',
             color='gray', alpha=0.9, marker='o', linestyle='--', drawstyle='steps-post')
    ax1.plot(processed_df['timestamp'], processed_df['oxygen_umol_L'], label=f'Synced to Frames', color='blue',
             marker='x', linestyle='None', markersize=4)
    ax1.set_ylabel('Oxygen (μmol/L)');
    ax1.legend();
    ax1.set_title('Oxygen Data Alignment')
    ax2.plot(resampled_temp_df['timestamp'], resampled_temp_df['temperature_C'],
             label=f'Resampled Temp ({SENSOR_RESAMPLE_WINDOW_S}s Mean)',
             color='gray', alpha=0.9, marker='o', linestyle='--', drawstyle='steps-post')
    ax2.plot(processed_df['timestamp'], processed_df['temperature_C'], label=f'Synced to Frames', color='red',
             marker='x', linestyle='None', markersize=4)
    ax2.set_ylabel('Temperature (°C)');
    ax2.legend();
    ax2.set_title('Temperature Data Alignment')
    plt.xlabel('Timestamp');
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    processed_df.rename(columns={'oxygen_umol_L': 'synced_oxygen_umol_L', 'temperature_C': 'synced_temperature_C'},
                        inplace=True)
    plot_data = pd.merge(resampled_o2_df, resampled_temp_df, on='timestamp', how='outer')
    plot_data = pd.merge(plot_data, processed_df[['timestamp', 'synced_oxygen_umol_L', 'synced_temperature_C']],
                         on='timestamp', how='outer').sort_values('timestamp')
    target_dir = os.path.join(analysis_dir, "0_data_preprocessing_reports")
    save_plot_and_data(fig, f'alignment_check_{exp_to_plot}', plot_data, target_dir)
    print(f"Saved alignment plot for '{exp_to_plot}' to reports directory.")


def create_pinn_dataframe(experiments, output_path, frame_skip, resize_dim, analysis_dir):
    """Orchestrates creation of the master feature dataframe, using a cache if available."""
    if os.path.exists(output_path) and not FORCE_RECREATE_DATAFRAME:
        print(f"Feature dataframe found. Loading from {output_path}")
        df = pd.read_parquet(output_path)
        visualize_alignment(df, experiments, analysis_dir)
        return df
    if os.path.exists(output_path) and FORCE_RECREATE_DATAFRAME:
        os.remove(output_path)
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)
    tasks = [{'exp': exp, 'frame_skip': frame_skip, 'resize_dim': resize_dim, 'temp_dir': temp_dir, 'worker_id': i} for
             i, exp in enumerate(experiments)]
    if NUM_WORKERS > 0:
        with Pool(processes=NUM_WORKERS) as pool:
            list(tqdm(pool.imap_unordered(process_experiment_chunked, tasks), total=len(tasks),
                      desc="Processing Experiments"))
    else:
        print("NUM_WORKERS is 0, processing experiments sequentially...")
        for task in tqdm(tasks, desc="Processing Experiments"):
            process_experiment_chunked(task)

    chunk_files = glob.glob(os.path.join(temp_dir, "*.parquet"))
    if not chunk_files: raise ValueError("FATAL: No data chunks were generated.")
    master_df = pd.concat([pd.read_parquet(f) for f in tqdm(chunk_files, desc="Loading chunks")], ignore_index=True)
    shutil.rmtree(temp_dir)
    pixel_cols = [f'pixel_{i}' for i in range(resize_dim[0] * resize_dim[1] * 3)]
    columns = ['experiment_id', 'timestamp', 'elapsed_seconds', 'temperature_C', 'oxygen_umol_L'] + pixel_cols
    master_df.columns = columns
    for col in pixel_cols: master_df[col] = pd.to_numeric(master_df[col], downcast='unsigned')
    master_df['elapsed_seconds'] = master_df['elapsed_seconds'].astype('float32')
    master_df['temperature_C'] = master_df['temperature_C'].astype('float32')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    master_df.to_parquet(output_path, index=False)
    print(f"Successfully created and saved PINN dataframe with {len(master_df)} samples to {output_path}.")
    visualize_alignment(master_df, experiments, analysis_dir)
    return master_df


# ==============================================================================
# --- STAGE 2: PYTORCH DATASET AND MODEL ARCHITECTURE ---
# ==============================================================================

class VideoFrameDataset(Dataset):
    def __init__(self, dataframe, exp_id_map, storage_dim, model_input_dim, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.exp_id_map = exp_id_map
        self.storage_dim = storage_dim
        self.model_input_dim = model_input_dim
        self.pixel_cols = [col for col in self.df.columns if col.startswith('pixel_')]
        self.base_transform = transforms.ToTensor()
        self.resize_transform = transforms.Resize(self.model_input_dim, antialias=True)
        self.user_transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = row[self.pixel_cols].values.astype(np.uint8).reshape(self.storage_dim[0], self.storage_dim[1], 3)
        image_tensor = self.base_transform(image)
        image_tensor = self.resize_transform(image_tensor)
        if self.user_transform:
            image_tensor = self.user_transform(image_tensor)

        oxygen_true = torch.tensor(row['oxygen_umol_L'], dtype=torch.float32)
        exp_id_index = torch.tensor(self.exp_id_map[row['experiment_id']], dtype=torch.long)
        temperature = torch.tensor(row['temperature_C'], dtype=torch.float32)
        return image_tensor, oxygen_true, exp_id_index, temperature


class AlgaePINN(nn.Module):
    def __init__(self, num_experiments, embedding_dim=8, model_name=TRANSFER_MODEL_NAME):
        super().__init__()
        self.vit_base = timm.create_model(model_name, pretrained=USE_TRANSFER_LEARNING, num_classes=0)
        self.vit_embed_dim = self.vit_base.embed_dim
        patch_size = self.vit_base.patch_embed.patch_size[0]
        self.grid_size = MODEL_INPUT_DIM[0] // patch_size
        self.experiment_embedding = nn.Embedding(num_experiments, embedding_dim)
        self.sv_param_head = nn.Sequential(nn.Linear(embedding_dim, 16), nn.GELU(), nn.Linear(16, 4))
        self.oxygen_head = nn.Sequential(nn.Linear(self.vit_embed_dim + 1, 128), nn.GELU(), nn.Linear(128, 1))
        decoder_channels = 256
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.vit_embed_dim, decoder_channels, 1), nn.GELU(),
            nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
            nn.Conv2d(decoder_channels, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )
        if USE_CONFIDENCE_LOSS:
            self.confidence_head = nn.Sequential(
                nn.Conv2d(self.vit_embed_dim, decoder_channels, 1), nn.GELU(),
                nn.Upsample(scale_factor=patch_size, mode='bilinear', align_corners=False),
                nn.Conv2d(decoder_channels, 64, 3, 1, 1), nn.GELU(),
                nn.Conv2d(64, 1, 1), nn.Sigmoid()
            )

    def forward(self, image_tensor, exp_id_index, temperature):
        features = self.vit_base.forward_features(image_tensor)
        cls_token = features[:, 0]
        temp_reshaped = temperature.view(-1, 1)
        combined_oxygen_features = torch.cat([cls_token, temp_reshaped], dim=1)
        oxygen_pred = self.oxygen_head(combined_oxygen_features)
        patch_tokens = features[:, 1:]
        B, N, C = patch_tokens.shape
        patch_tokens_grid = patch_tokens.reshape(B, self.grid_size, self.grid_size, C).permute(0, 3, 1, 2).contiguous()
        mask_pred = self.mask_head(patch_tokens_grid)
        embedding = self.experiment_embedding(exp_id_index)
        sv_params_raw = self.sv_param_head(embedding)
        sv_params = {'i0': nn.functional.softplus(sv_params_raw[:, 0]),
                     'ksv1': nn.functional.softplus(sv_params_raw[:, 1]),
                     'ksv2': nn.functional.softplus(sv_params_raw[:, 2]), 'a': torch.sigmoid(sv_params_raw[:, 3])}
        confidence_map = self.confidence_head(patch_tokens_grid).squeeze(1) if USE_CONFIDENCE_LOSS else None
        if confidence_map is not None:
            batch_size = confidence_map.size(0)
            map_sums = torch.sum(confidence_map.view(batch_size, -1), dim=1, keepdim=True).view(batch_size, 1, 1)
            confidence_map = confidence_map / (map_sums + 1e-8)

        return {'oxygen_pred': oxygen_pred.squeeze(-1), 'mask_pred': mask_pred.squeeze(1), 'sv_params': sv_params,
                'confidence_map': confidence_map, 'attention_map': None}


# ==============================================================================
# --- STAGE 3: LOSS FUNCTION AND TRAINING/EVALUATION LOGIC ---
# ==============================================================================
def calculate_pinn_loss(model_outputs, batch_data, lambda_physics, return_residual=False):
    """
    Calculates the composite loss function, returning both relative and absolute residuals if needed.
    """
    image_tensor, oxygen_true, _, _ = batch_data
    oxygen_pred, mask, sv_params, confidence_map = model_outputs['oxygen_pred'], model_outputs['mask_pred'], \
        model_outputs['sv_params'], model_outputs['confidence_map']

    loss_data = nn.functional.mse_loss(oxygen_pred, oxygen_true)
    resized_image_tensor = transforms.functional.resize(image_tensor, mask.shape[-2:], antialias=True)
    red_channel = resized_image_tensor[:, 0, :, :] * 255.0
    oxygen_pred_r = oxygen_pred.view(-1, 1, 1)
    i0_r = sv_params['i0'].view(-1, 1, 1)

    if PHYSICS_MODEL == 'nonlinear':
        ksv1_r, ksv2_r, a_r = sv_params['ksv1'].view(-1, 1, 1), sv_params['ksv2'].view(-1, 1, 1), sv_params['a'].view(
            -1, 1, 1)
        predicted_intensity = i0_r * ((a_r / (1 + ksv1_r * oxygen_pred_r)) + ((1 - a_r) / (1 + ksv2_r * oxygen_pred_r)))
        sv_residual_absolute = torch.abs(red_channel - predicted_intensity)
        sv_residual_relative = sv_residual_absolute / (red_channel + 1.0)
    else:  # Linear Model
        ksv_r = sv_params['ksv1'].view(-1, 1, 1)
        sv_residual_relative = torch.abs(i0_r / (red_channel + 1e-6) - (1 + ksv_r * oxygen_pred_r))
        sv_residual_absolute = sv_residual_relative  # In linear case, they are equivalent for our purpose

    squared_residual = (mask * sv_residual_relative) ** 2
    loss_physics = torch.mean(
        confidence_map * squared_residual) if USE_CONFIDENCE_LOSS and confidence_map is not None else torch.mean(
        squared_residual)
    total_loss = loss_data + (lambda_physics * loss_physics)
    loss_dict = {'total': total_loss, 'data': loss_data.detach(), 'physics': loss_physics.detach()}
    return (loss_dict, sv_residual_relative.detach(), sv_residual_absolute.detach()) if return_residual else loss_dict


def train_one_epoch(model, loader, optimizer, lambda_physics, device, scaler, clip_value, grad_accumulation_steps):
    model.train();
    loss_sums = {'total': 0.0, 'data': 0.0, 'physics': 0.0};
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        batch_data = [d.to(device) for d in batch]
        with torch.amp.autocast(device_type=DEVICE_TYPE, dtype=torch.float16, enabled=USE_AMP):
            model_outputs = model(batch_data[0], batch_data[2], batch_data[3])
            loss_dict = calculate_pinn_loss(model_outputs, batch_data, lambda_physics)
            loss = loss_dict['total'] / grad_accumulation_steps
        if torch.isnan(loss):
            warnings.warn(f"NaN total loss detected at batch {i}. Skipping batch update.");
            continue
        scaler.scale(loss).backward() if USE_AMP else loss.backward()
        if (i + 1) % grad_accumulation_steps == 0:
            if USE_AMP: scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            if USE_AMP:
                scaler.step(optimizer);
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        for key in loss_sums:
            if not torch.isnan(loss_dict[key]): loss_sums[key] += loss_dict[key].item()
    return {key: val / len(loader) for key, val in loss_sums.items()}


def evaluate_model(model, loader, device):
    model.eval();
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch_data = [d.to(device) for d in batch]
            with torch.amp.autocast(device_type=DEVICE_TYPE, dtype=torch.float16, enabled=USE_AMP):
                preds_cpu = model(batch_data[0], batch_data[2], batch_data[3])['oxygen_pred'].cpu().numpy()
            if np.isnan(preds_cpu).any(): return np.nan
            all_preds.extend(preds_cpu);
            all_trues.extend(batch_data[1].cpu().numpy())
    return mean_absolute_error(all_trues, all_preds)


# ==============================================================================
# --- STAGE 4: HPO, REPORTING, AND ANALYSIS ---
# ==============================================================================
def plot_train_val_test_distribution(train_df, val_df, test_df, n_train_days, analysis_dir):
    """
    *** NEW in V8.3 ***
    Plots Train vs Validation vs Test distributions.
    This helps visually confirm that the Validation day is a reasonable step
    between the Training past and the Testing future.
    """
    print(f"--- Generating distribution plot for split: Train({n_train_days}) | Val(1) | Test(Rest) ---")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.kdeplot(data=train_df, x='oxygen_umol_L', fill=True, alpha=0.3, label=f'Train ({n_train_days} days)',
                color='blue', ax=ax)
    sns.kdeplot(data=val_df, x='oxygen_umol_L', fill=True, alpha=0.3, label='Validation (1 day)', color='orange', ax=ax)
    sns.kdeplot(data=test_df, x='oxygen_umol_L', fill=True, alpha=0.3, label='Test (Future days)', color='green', ax=ax)

    ax.set_title(f'Ground Truth Oxygen Distribution\nWalk-Forward Split: {n_train_days} Train Days',
                 fontsize=16, weight='bold')
    ax.set_xlabel('Oxygen (μmol/L)')
    ax.set_ylabel('Density')
    ax.legend()
    fig.tight_layout()

    # Prepare data for CSV
    t_d = train_df[['oxygen_umol_L']].copy();
    t_d['dataset'] = 'train'
    v_d = val_df[['oxygen_umol_L']].copy();
    v_d['dataset'] = 'validation'
    te_d = test_df[['oxygen_umol_L']].copy();
    te_d['dataset'] = 'test'
    dist_data = pd.concat([t_d, v_d, te_d], ignore_index=True)

    plot_name = f'distribution_split_train_{n_train_days}_days'
    target_dir = os.path.join(analysis_dir, "1_data_split_reports")
    save_plot_and_data(fig, plot_name, dist_data, target_dir)


def generate_fold_training_plots(history_df, split_name, analysis_dir, ensemble_member=None):
    member_str = f" (Ensemble Member {ensemble_member})" if ensemble_member is not None else ""
    fig1, ax1a = plt.subplots(figsize=(12, 7));
    fig1.suptitle(f'Split: {split_name}{member_str} - Validation MAE vs. Physics Loss', fontsize=16, weight='bold')
    ax1a.set_xlabel('Epoch');
    ax1a.set_ylabel('Validation MAE (μmol/L)', color='tab:orange');
    ax1a.plot(history_df['epoch'], history_df['val_mae'], color='tab:orange', marker='o', label='Validation MAE');
    ax1a.tick_params(axis='y', labelcolor='tab:orange');
    ax1a.legend(loc='upper left')
    ax1b = ax1a.twinx();
    ax1b.set_ylabel('Train Physics Loss (Log Scale)', color='tab:green');
    ax1b.plot(history_df['epoch'], history_df['train_loss_physics'], color='tab:green', linestyle='--',
              label='Physics Loss');
    ax1b.tick_params(axis='y', labelcolor='tab:green');
    ax1b.set_yscale('log');
    ax1b.legend(loc='upper right')
    plot_name = f'pinn_split_{split_name}_member_{ensemble_member}_mae_vs_physics' if ensemble_member is not None else f'pinn_split_{split_name}_mae_vs_physics'
    save_plot_and_data(fig1, plot_name, history_df[['epoch', 'val_mae', 'train_loss_physics']],
                       os.path.join(analysis_dir, "fold_convergence_plots"))


def plot_pinn_loss_components(history_df, split_name, analysis_dir, ensemble_member=None):
    member_str = f" (Ensemble Member {ensemble_member})" if ensemble_member is not None else ""
    fig, ax = plt.subplots(figsize=(12, 7));
    fig.suptitle(f'PINN - Split: {split_name}{member_str} - Loss Components', fontsize=16, weight='bold')
    plot_df = history_df.melt(id_vars=['epoch'], value_vars=['train_loss_data', 'train_loss_physics'],
                              var_name='Metric', value_name='Loss')
    sns.lineplot(data=plot_df, x='epoch', y='Loss', hue='Metric', style='Metric', ax=ax);
    ax.set_yscale('log');
    ax.set_xlabel('Epoch');
    ax.set_ylabel('Loss (Log Scale)');
    ax.legend(title='Loss Component')
    plot_name = f'pinn_split_{split_name}_member_{ensemble_member}_loss_components' if ensemble_member is not None else f'pinn_split_{split_name}_loss_components'
    save_plot_and_data(fig, plot_name, history_df, os.path.join(analysis_dir, "pinn_loss_component_plots"))


def plot_physics_residual_heatmap(model, loader, device, split_name, analysis_dir, ensemble_member=None):
    """
    Generates and saves a heatmap of the ABSOLUTE physics residual for a given model.
    """
    model.eval();
    all_residuals = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5: break
            batch_data = [d.to(device) for d in batch]
            model_outputs = model(batch_data[0], batch_data[2], batch_data[3])
            _, _, sv_residual_absolute = calculate_pinn_loss(model_outputs, batch_data, 1.0, return_residual=True)
            all_residuals.append(sv_residual_absolute.cpu().numpy())
    if not all_residuals: return None
    avg_residual = np.mean(np.concatenate(all_residuals, axis=0), axis=0)
    member_str = f" (Ensemble Member {ensemble_member})" if ensemble_member is not None else ""
    plot_title = f'PINN - Split: {split_name}{member_str} - Avg Absolute Physics Residual'
    plot_name = f'pinn_split_{split_name}_member_{ensemble_member}_physics_residual' if ensemble_member else f'pinn_split_{split_name}_physics_residual'
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(avg_residual, cmap='viridis', ax=ax, cbar_kws={'label': 'Absolute Residual |I_obs - I_pred|'})
    ax.set_title(plot_title);
    ax.set_xlabel('Pixel X');
    ax.set_ylabel('Pixel Y')
    save_plot_and_data(fig, plot_name, pd.DataFrame(avg_residual), os.path.join(analysis_dir, "pinn_physics_heatmaps"))
    return avg_residual


def plot_attention_and_confidence_maps(model, loader, device, title_prefix, analysis_dir):
    model.eval();
    avg_confidence_map = None
    with torch.no_grad():
        batch_data = [d.to(device) for d in next(iter(loader))]
        outputs = model(batch_data[0], batch_data[2], batch_data[3])
        if USE_CONFIDENCE_LOSS and outputs['confidence_map'] is not None:
            avg_confidence_map = outputs['confidence_map'].mean(dim=0).cpu().numpy()
    if avg_confidence_map is None: return None, None
    fig, ax = plt.subplots(1, 1, figsize=(8, 7));
    fig.suptitle(f'{title_prefix} - Physics Confidence Map', fontsize=16, weight='bold')
    sns.heatmap(avg_confidence_map, cmap='viridis', ax=ax, cbar_kws={'label': 'Confidence Weight (Normalized)'});
    ax.set_title('Average Physics Confidence Map')
    save_plot_and_data(fig, f'{title_prefix}_physics_confidence_map', pd.DataFrame(avg_confidence_map),
                       os.path.join(analysis_dir, "attention_confidence_maps"))
    return None, avg_confidence_map


def plot_biofouling_and_correlation_maps(model, loader, device, exp_id_map, red_heatmap_dir, title_prefix,
                                         analysis_dir):
    output_path = os.path.join(analysis_dir, "biofouling_analysis");
    os.makedirs(output_path, exist_ok=True)
    model.eval();
    all_masks, all_exp_indices = [], []
    with torch.no_grad():
        for batch in loader:
            batch_data = [d.to(device) for d in batch]
            outputs = model(batch_data[0], batch_data[2], batch_data[3])
            all_masks.append(outputs['mask_pred'].cpu().numpy());
            all_exp_indices.append(batch_data[2].cpu().numpy())
    if not all_masks: return None
    all_masks = np.concatenate(all_masks, axis=0);
    all_exp_indices = np.concatenate(all_exp_indices, axis=0)
    idx_to_exp_id = {v: k for k, v in exp_id_map.items()};
    all_exp_ids = [idx_to_exp_id[idx] for idx in all_exp_indices]
    unique_exp_ids_in_fold = np.unique(all_exp_ids);
    fold_avg_biofouling_maps = []
    for exp_id in unique_exp_ids_in_fold:
        exp_mask_indices = [i for i, eid in enumerate(all_exp_ids) if eid == exp_id]
        if not exp_mask_indices: continue
        avg_pred_mask = all_masks[exp_mask_indices].mean(axis=0);
        avg_pred_biofouling = 1.0 - avg_pred_mask
        fold_avg_biofouling_maps.append(avg_pred_biofouling)
        red_intensity_csv_path = os.path.join(red_heatmap_dir, f'heatmap_{exp_id}_overall_data.csv')
        if not os.path.exists(red_intensity_csv_path): continue
        red_intensity_map_original = pd.read_csv(red_intensity_csv_path).values
        target_h, target_w = avg_pred_biofouling.shape
        red_intensity_map = cv2.resize(red_intensity_map_original, (target_w, target_h),
                                       interpolation=cv2.INTER_LINEAR) if red_intensity_map_original.shape != (
            target_h, target_w) else red_intensity_map_original
        pred_flat, red_flat = avg_pred_biofouling.flatten(), red_intensity_map.flatten()
        corr, p_val = scipy.stats.pearsonr(pred_flat, red_flat) if np.std(pred_flat) > 1e-6 and np.std(
            red_flat) > 1e-6 else (0.0, 1.0)
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        fig.suptitle(
            f'Biofouling Analysis for {exp_id} ({title_prefix})\nPearson Correlation: {corr:.3f} (p={p_val:.2e})',
            fontsize=16, weight='bold')
        sns.heatmap(avg_pred_biofouling, cmap='YlGnBu', ax=axes[0], cbar_kws={'label': 'Predicted Biofouling Level'});
        axes[0].set_title('Predicted Biofouling Map\n(1 - Mask)')
        sns.heatmap(red_intensity_map, cmap='Reds_r', ax=axes[1], cbar_kws={'label': 'Avg. Red Channel Intensity'});
        axes[1].set_title('Ground Truth Proxy\n(Avg. Red Intensity)')
        sns.regplot(x=red_flat, y=pred_flat, ax=axes[2], scatter_kws={'alpha': 0.1, 's': 10});
        axes[2].set_xlabel('Red Channel Intensity');
        axes[2].set_ylabel('Predicted Biofouling');
        axes[2].set_title('Pixel-wise Correlation')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        correlation_data_df = pd.DataFrame({'predicted_biofouling': pred_flat, 'red_channel_intensity': red_flat})
        save_plot_and_data(fig, f'{title_prefix}_{exp_id}_biofouling_correlation', correlation_data_df, output_path)
    return np.mean(fold_avg_biofouling_maps, axis=0) if fold_avg_biofouling_maps else None


def generate_all_intensity_proxy_reports(master_df, experiments, resize_dim, base_project_dir):
    report_dir = os.path.join(base_project_dir, "Intensity_Proxy_Reports");
    os.makedirs(report_dir, exist_ok=True)
    print(f"\n--- Generating Pre-flight Data Quality Checks for all experiments ---")
    pixel_cols = [c for c in master_df.columns if c.startswith('pixel_')]
    for exp_meta in tqdm(experiments, desc="Creating Intensity Proxy Reports"):
        exp_id = exp_meta['id'];
        exp_full_df = master_df[master_df['experiment_id'] == exp_id].copy()
        if exp_full_df.empty: continue
        images = exp_full_df[pixel_cols].values.astype(np.uint8).reshape(-1, resize_dim[0], resize_dim[1], 3)
        exp_full_df['inv_red_intensity_proxy'] = 1.0 / (images[:, :, :, 0].mean(axis=(1, 2)) + 1e-6)
        raw_o2_df = parse_arduino_log(exp_meta['raw_arduino_path'])
        if raw_o2_df.empty: continue
        fig1, ax1 = plt.subplots(figsize=(18, 8));
        fig1.suptitle(f'Data Quality Check vs. Timestamp\nExperiment: {exp_id}', fontsize=16, weight='bold')
        ax1.set_xlabel('Timestamp');
        ax1.set_ylabel('Ground Truth O₂ (μmol/L)', color='dodgerblue', weight='bold');
        ax1.plot(raw_o2_df['timestamp'], raw_o2_df['oxygen_umol_L'], color='dodgerblue', label='Arduino O₂ (GT)',
                 drawstyle='steps-post');
        ax1.tick_params(axis='y', labelcolor='dodgerblue');
        ax2 = ax1.twinx();
        ax2.set_ylabel('Inverse Mean Red Intensity (Proxy)', color='red', weight='bold');
        ax2.plot(exp_full_df['timestamp'], exp_full_df['inv_red_intensity_proxy'], color='red', marker='.',
                 linestyle='None', alpha=0.6, label='Inverse Red Intensity (Proxy)');
        ax2.tick_params(axis='y', labelcolor='red');
        lines1, labels1 = ax1.get_legend_handles_labels();
        lines2, labels2 = ax2.get_legend_handles_labels();
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right');
        fig1.tight_layout()
        proxy_data_df_ts = exp_full_df[['timestamp', 'inv_red_intensity_proxy']].copy();
        plot_data_df_ts = pd.merge_asof(raw_o2_df.sort_values('timestamp'), proxy_data_df_ts.sort_values('timestamp'),
                                        on='timestamp', direction='nearest', tolerance=pd.Timedelta('30s'))
        save_plot_and_data(fig1, f'intensity_proxy_check_vs_timestamp_{exp_id}', plot_data_df_ts, report_dir)


def generate_red_intensity_heatmaps(master_df, experiments, resize_dim, base_project_dir):
    report_dir = os.path.join(base_project_dir, "Red_Intensity_Heatmaps")
    if os.path.exists(report_dir) and not FORCE_RECREATE_HEATMAPS: print(
        f"\n--- Red intensity heatmaps exist. Skipping. ---"); return
    os.makedirs(report_dir, exist_ok=True);
    print(f"\n--- Generating Red Intensity Heatmaps ---")
    pixel_cols = [c for c in master_df.columns if c.startswith('pixel_')]
    for exp_meta in tqdm(experiments, desc="Generating Red Intensity Heatmaps"):
        exp_id = exp_meta['id'];
        exp_df = master_df[master_df['experiment_id'] == exp_id].copy()
        if exp_df.empty: continue
        all_interval_maps = []
        for start, end in ANALYSIS_INTERVALS_S:
            interval_df = exp_df[(exp_df['elapsed_seconds'] >= start) & (exp_df['elapsed_seconds'] <= end)]
            if interval_df.empty: continue
            images = interval_df[pixel_cols].values.astype(np.uint8).reshape(-1, resize_dim[0], resize_dim[1], 3)
            avg_red_map = images[:, :, :, 0].mean(axis=0);
            all_interval_maps.append(avg_red_map)
            fig, ax = plt.subplots(figsize=(8, 7));
            sns.heatmap(avg_red_map, cmap='Reds_r', ax=ax, cbar_kws={'label': 'Average Red Channel Intensity'});
            ax.set_title(f'Avg. Red Intensity: {exp_id}\nInterval: {start}-{end} seconds');
            save_plot_and_data(fig, f'heatmap_{exp_id}_{start}-{end}s', pd.DataFrame(avg_red_map), report_dir)
        if all_interval_maps:
            overall_map = np.mean(all_interval_maps, axis=0);
            fig, ax = plt.subplots(figsize=(8, 7));
            sns.heatmap(overall_map, cmap='Reds_r', ax=ax, cbar_kws={'label': 'Average Red Channel Intensity'});
            ax.set_title(f'Overall Avg. Red Intensity: {exp_id}');
            save_plot_and_data(fig, f'heatmap_{exp_id}_overall', pd.DataFrame(overall_map), report_dir)


def run_optuna_tuning(train_dataset, val_dataset, num_experiments, output_dir, hpo_params_path, hpo_db_path):
    print(f"\n--- Phase 2: Hyperparameter Tuning (Physics: {PHYSICS_MODEL}) ---")
    hpo_analysis_dir = os.path.join(output_dir, "hpo_analysis");
    os.makedirs(hpo_analysis_dir, exist_ok=True)
    if FORCE_RERUN_HPO and os.path.exists(hpo_db_path):
        print(f"FORCE_RERUN_HPO is True. Deleting existing study database: {hpo_db_path}");
        os.remove(hpo_db_path)
    if os.path.exists(hpo_params_path) and not FORCE_RERUN_HPO:
        print(f"Loading cached HPO parameters from: {hpo_params_path}")
        with open(hpo_params_path, 'r') as f: return json.load(f)

    # Note on HPO Strategy for V8.3:
    # While strictly chronological splits should use nested CV for HPO, that is computationally prohibitive.
    # Here we use a random subset of the data for HPO to find structural parameters.
    # This assumes that optimal hyperparameters (LR, Lambda Physics) are relatively stable across time.
    def objective(trial):
        try:
            params = {'lr': trial.suggest_float("lr", 1e-5, 1e-3, log=True),
                      'lambda_physics': trial.suggest_float("lambda_physics", 0.1, 10.0, log=True),
                      'embedding_dim': trial.suggest_categorical("embedding_dim", [8, 16]),
                      'weight_decay': trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)}

            model = AlgaePINN(num_experiments, params['embedding_dim']).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
            scaler = torch.amp.GradScaler(enabled=USE_AMP)
            train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                      worker_init_fn=seed_worker)
            val_loader = DataLoader(val_dataset, BATCH_SIZE, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
            for epoch in range(N_EPOCHS_FOR_TUNING):
                train_one_epoch(model, train_loader, optimizer, params['lambda_physics'], DEVICE, scaler,
                                GRADIENT_CLIP_VALUE, GRAD_ACCUMULATION_STEPS)
                val_mae = evaluate_model(model, val_loader, DEVICE)
                if np.isnan(val_mae): return 1e9
                trial.report(val_mae, epoch)
                if trial.should_prune(): raise optuna.exceptions.TrialPruned()
            return val_mae
        except Exception as e:
            print(f"Trial failed with exception: {e}")
            traceback.print_exc()
            return 1e9
        finally:
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    storage = optuna.storages.RDBStorage(url=f"sqlite:///{hpo_db_path}")
    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(),
                                sampler=optuna.samplers.TPESampler(seed=SEED), storage=storage,
                                study_name="pinn_v83_study", load_if_exists=True)
    n_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"Found {n_completed_trials} completed trials in the HPO database.")
    if n_completed_trials < N_TRIALS_FOR_TUNING:
        print(f"Running {N_TRIALS_FOR_TUNING - n_completed_trials} more trials...")
        study.optimize(objective, n_trials=(N_TRIALS_FOR_TUNING - n_completed_trials), show_progress_bar=True)
    else:
        print("Required number of HPO trials are already complete.")
    best_params = study.best_params if study.best_value < 1e9 else None
    if best_params:
        os.makedirs(os.path.dirname(hpo_params_path), exist_ok=True)
        with open(hpo_params_path, 'w') as f:
            json.dump(best_params, f, indent=4)
        try:
            importances = optuna.importance.get_param_importances(study)
            importance_df = pd.DataFrame.from_dict(importances, orient='index', columns=['importance']).sort_values(
                'importance', ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6));
            sns.barplot(x=importance_df.importance, y=importance_df.index, ax=ax)
            ax.set_title('Hyperparameter Importances for HPO Study');
            ax.set_xlabel('Importance');
            ax.set_ylabel('Parameter');
            fig.tight_layout()
            save_plot_and_data(fig, 'hpo_importances',
                               importance_df.reset_index().rename(columns={'index': 'parameter'}), hpo_analysis_dir)
        except Exception as e:
            print(f"Warning: Could not generate HPO importance plot. Error: {e}")
        study.trials_dataframe().to_csv(os.path.join(hpo_analysis_dir, "hpo_trials_history.csv"), index=False)
    return best_params


def get_inference_results(models, loader, device):
    for model in models: model.eval()
    results = {'o2_true': [], 'o2_pred_mean': [], 'o2_pred_std': [], 'biofouling_score': [], 'temperature_C': []}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating Ensemble Analysis Data"):
            batch_data = [d.to(device) for d in batch]
            image_tensor, oxygen_true, exp_id_index, temperature = batch_data
            ensemble_preds, ensemble_masks = [], []
            for model in models:
                with torch.amp.autocast(device_type=DEVICE_TYPE, dtype=torch.float16, enabled=USE_AMP):
                    outputs = model(image_tensor, exp_id_index, temperature)
                    ensemble_preds.append(outputs['oxygen_pred']);
                    ensemble_masks.append(outputs['mask_pred'].mean(dim=[1, 2]))
            stacked_preds = torch.stack(ensemble_preds, dim=0);
            stacked_masks = torch.stack(ensemble_masks, dim=0)
            results['o2_true'].extend(oxygen_true.cpu().numpy())
            results['o2_pred_mean'].extend(stacked_preds.mean(dim=0).cpu().numpy())
            results['o2_pred_std'].extend(stacked_preds.std(dim=0).cpu().numpy())
            results['biofouling_score'].extend((1.0 - stacked_masks.mean(dim=0)).cpu().numpy())
            results['temperature_C'].extend(temperature.cpu().numpy())
    return {k: np.array(v) for k, v in results.items()}


def generate_final_report_for_split(df_analysis, analysis_dir, split_name):
    """
    Generates a final report for a *single* train/test split.
    """
    print(f"--- Generating Final Analysis Report for Split: {split_name} ---")
    output_dir = os.path.join(analysis_dir, "final_test_set_plots")

    df_analysis['error'] = df_analysis['o2_pred_mean'] - df_analysis['o2_true']
    df_analysis['abs_error'] = abs(df_analysis['error'])
    mae = df_analysis['abs_error'].mean()
    r2 = r2_score(df_analysis['o2_true'], df_analysis['o2_pred_mean'])

    fig, ax = plt.subplots(figsize=(8, 8));
    sns.scatterplot(data=df_analysis, x='o2_true', y='o2_pred_mean', alpha=0.5, ax=ax)
    min_val, max_val = df_analysis['o2_true'].min(), df_analysis['o2_true'].max();
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal (y=x)')
    ax.set_title(f'Parity Plot for Split: {split_name}\nOverall MAE = {mae:.3f} μmol/L | R² = {r2:.3f}');
    ax.set_xlabel('True Oxygen (μmol/L)');
    ax.set_ylabel('Mean Predicted Oxygen (μmol/L)');
    ax.legend()
    save_plot_and_data(fig, f'{split_name}_final_parity_plot', df_analysis[['o2_true', 'o2_pred_mean']], output_dir)

    fig, ax = plt.subplots(figsize=(10, 6));
    sns.scatterplot(data=df_analysis, x='o2_pred_std', y='abs_error', alpha=0.3, s=20, ax=ax)
    corr = df_analysis[['o2_pred_std', 'abs_error']].corr().iloc[0, 1];
    ax.set_title(f'Model Uncertainty vs. Absolute Error for Split: {split_name}\n(Pearson Correlation: {corr:.3f})')
    ax.set_xlabel('Prediction Standard Deviation (Uncertainty)');
    ax.set_ylabel('Absolute Error (μmol/L)')
    save_plot_and_data(fig, f'{split_name}_final_uncertainty_vs_error', df_analysis[['o2_pred_std', 'abs_error']],
                       output_dir)

    fig, ax = plt.subplots(figsize=(10, 6));
    sns.histplot(df_analysis['error'], kde=True, bins=50, ax=ax);
    ax.set_title(
        f'Error Distribution for Split: {split_name}\n(Mean: {df_analysis["error"].mean():.2f}, Std: {df_analysis["error"].std():.2f})');
    ax.set_xlabel('Prediction Error (μmol/L)')
    save_plot_and_data(fig, f'{split_name}_final_error_histogram', df_analysis[['error']], output_dir)
    return mae


def plot_performance_vs_training_size(summary_df, analysis_dir):
    """
    Plots the final test MAE as a function of the number of training days used.
    """
    print("\n--- Generating Overall Performance Summary ---")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=summary_df, x='n_train_days', y='mae', marker='o', ax=ax)
    ax.set_title('Model Performance (on Future Data) vs. Training Data Size\n(Chronological Forward Chaining)',
                 fontsize=16, weight='bold')
    ax.set_xlabel('Number of Days in Training Set (Past)')
    ax.set_ylabel('Test MAE (Future) (μmol/L)')
    ax.grid(True, which='both', linestyle='--')
    fig.tight_layout()
    target_dir = os.path.join(analysis_dir, "_overall_performance_summary")
    save_plot_and_data(fig, 'overall_performance_vs_training_size', summary_df, target_dir)


# ==============================================================================
# --- STAGE 5: MAIN EXECUTION ORCHESTRATOR (V8.3 - CHRONOLOGICAL) ---
# ==============================================================================

def main():
    start_time = datetime.now()
    os.makedirs(OUTPUT_DIR, exist_ok=True);
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True);
    os.makedirs(HPO_CACHE_DIR, exist_ok=True)
    SPLIT_CACHE_DIR = os.path.join(OUTPUT_DIR, "cv_split_cache");
    os.makedirs(SPLIT_CACHE_DIR, exist_ok=True)

    # 1. Discovery and Loading
    experiments = find_experiment_files(EXPERIMENTS_ROOT_DIR, NUM_EXPERIMENT_DAYS_TO_USE)
    master_df = create_pinn_dataframe(experiments, FEATURE_DATAFRAME_PATH, FRAME_SKIP, STORAGE_RESIZE_DIM, ANALYSIS_DIR)

    # 2. Pre-flight Checks
    generate_all_intensity_proxy_reports(master_df, experiments, STORAGE_RESIZE_DIM, BASE_PROJECT_DIR)
    generate_red_intensity_heatmaps(master_df, experiments, STORAGE_RESIZE_DIM, BASE_PROJECT_DIR)
    red_intensity_heatmap_dir = os.path.join(BASE_PROJECT_DIR, "Red_Intensity_Heatmaps")

    # 3. Filtering and Setup
    conditions = [(master_df['elapsed_seconds'] >= s) & (master_df['elapsed_seconds'] <= e) for s, e in
                  ANALYSIS_INTERVALS_S]
    filtered_df = master_df[functools.reduce(np.logical_or, conditions)].reset_index(drop=True)
    if filtered_df.empty: raise RuntimeError("FATAL: No data after filtering for analysis intervals.")

    # 4. Chronological Sorting of IDs
    # Since experiments were already sorted by date in find_experiment_files, we assume list order is chronological.
    # We verify this by using the `experiments` list order to define `exp_ids`.
    sorted_exp_ids = [exp['id'] for exp in experiments]
    exp_id_map = {name: i for i, name in enumerate(sorted_exp_ids)}
    num_experiments = len(sorted_exp_ids)

    print("\n--- Experiment Chronological Order for Training ---")
    for i, eid in enumerate(sorted_exp_ids):
        print(f"  Day {i + 1}: {eid}")

    full_dataset = VideoFrameDataset(filtered_df, exp_id_map, STORAGE_RESIZE_DIM, MODEL_INPUT_DIM)

    # 5. HPO (Using random subset to establish architectural hyperparameters)
    groups = filtered_df['experiment_id'].values
    hpo_subset_indices, _ = train_test_split(np.arange(len(full_dataset)), train_size=HPO_DATA_SUBSET_FRACTION,
                                             stratify=groups, random_state=SEED)
    hpo_dataset = Subset(full_dataset, hpo_subset_indices)
    hpo_train_indices, hpo_val_indices = train_test_split(np.arange(len(hpo_dataset)), test_size=0.2, random_state=SEED)
    best_params = run_optuna_tuning(Subset(hpo_dataset, hpo_train_indices), Subset(hpo_dataset, hpo_val_indices),
                                    num_experiments, OUTPUT_DIR, HPO_PARAMS_PATH, HPO_STUDY_DB_PATH)
    if not best_params: raise RuntimeError("FATAL: HPO failed to find any valid parameters.")

    # 6. Chronological Forward Chaining Loop
    # We need at least: 1 Train Day + 1 Validation Day + 1 Test Day = 3 experiments minimum.
    if num_experiments < 3:
        raise ValueError("FATAL: Not enough experiments for Walk-Forward Validation (Need at least 3).")

    overall_performance_summary = []

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_user_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), normalize])
    val_user_transform = normalize

    # Loop logic:
    # Iteration 1: Train [Day 1], Val [Day 2], Test [Days 3...N]
    # Iteration 2: Train [Day 1, 2], Val [Day 3], Test [Days 4...N]
    # ...
    # Stop when we run out of test days.

    max_train_idx = num_experiments - 2  # If N=5, max train index is 3 (indices 0,1,2 used for train)
    # Val is index 3, Test is index 4.

    for train_end_idx in range(0, max_train_idx):
        # Indices are 0-based.
        # Training set: 0 to train_end_idx (inclusive)
        # Validation set: train_end_idx + 1 (single day)
        # Test set: train_end_idx + 2 to End (rest)

        train_exp_ids = sorted_exp_ids[0: train_end_idx + 1]
        val_exp_ids = [sorted_exp_ids[train_end_idx + 1]]
        test_exp_ids = sorted_exp_ids[train_end_idx + 2:]

        n_train_days = len(train_exp_ids)
        split_name = f"split_train_{n_train_days}_days"

        print(
            f"\n{'=' * 80}\n--- Processing {split_name} ---\nTRAIN: {len(train_exp_ids)} Days | VAL: {len(val_exp_ids)} Day | TEST: {len(test_exp_ids)} Days\n{'=' * 80}")
        print(f"  Training IDs: {train_exp_ids}")
        print(f"  Validation ID: {val_exp_ids[0]} (Used for Model Selection)")
        print(f"  Testing IDs: {test_exp_ids} (Used for Performance)")

        split_analysis_dir = os.path.join(ANALYSIS_DIR, split_name)
        split_cache_dir = os.path.join(SPLIT_CACHE_DIR, split_name);
        os.makedirs(split_cache_dir, exist_ok=True)
        test_results_cache_path = os.path.join(split_cache_dir, "test_results.parquet")

        if os.path.exists(test_results_cache_path) and not FORCE_RERUN_SPLITS:
            print(f"  Found cached test results for {split_name}. Loading and skipping.")
            df_res = pd.read_parquet(test_results_cache_path)
            mae = mean_absolute_error(df_res['o2_true'], df_res['o2_pred_mean'])
            overall_performance_summary.append({'n_train_days': n_train_days, 'mae': mae})
            continue

        train_idx = filtered_df.index[filtered_df['experiment_id'].isin(train_exp_ids)]
        val_idx = filtered_df.index[filtered_df['experiment_id'].isin(val_exp_ids)]
        test_idx = filtered_df.index[filtered_df['experiment_id'].isin(test_exp_ids)]

        # --- Visual Check of Distributions ---
        plot_train_val_test_distribution(filtered_df.loc[train_idx], filtered_df.loc[val_idx],
                                         filtered_df.loc[test_idx], n_train_days, ANALYSIS_DIR)

        train_subset = Subset(full_dataset, train_idx);
        val_subset = Subset(full_dataset, val_idx)
        test_subset = Subset(full_dataset, test_idx)

        split_ensemble_models = []

        for member in range(ENSEMBLE_SIZE):
            print(f"  Processing Ensemble Member {member + 1}/{ENSEMBLE_SIZE}...")
            model_cache_path = os.path.join(split_cache_dir, f"model_member_{member}.pth")
            model = AlgaePINN(num_experiments, best_params['embedding_dim']).to(DEVICE)

            is_training_required = not (os.path.exists(model_cache_path) and not FORCE_RERUN_SPLITS)

            if is_training_required:
                # Compile for speed if possible
                if torch.__version__ >= "2.0.0" and sys.platform != "win32":
                    try:
                        print("    Compiling model with torch.compile()...")
                        model = torch.compile(model)
                    except Exception as e:
                        print(f"    WARNING: torch.compile() failed: {e}")
                elif sys.platform == "win32":
                    print("    Skipping torch.compile() on Windows.")

            if not is_training_required:
                print(f"    Found cached model weights. Loading.");
                model.load_state_dict(torch.load(model_cache_path, map_location=DEVICE));
                split_ensemble_models.append(model)
            else:
                print(f"    No cached model found. Starting training...")
                train_subset.dataset.user_transform = train_user_transform if USE_DATA_AUGMENTATION else val_user_transform
                val_subset.dataset.user_transform = val_user_transform  # Validation always clean

                train_loader = DataLoader(train_subset, BATCH_SIZE, True, num_workers=NUM_WORKERS,
                                          worker_init_fn=seed_worker, pin_memory=True)
                val_loader = DataLoader(val_subset, BATCH_SIZE, False, num_workers=NUM_WORKERS,
                                        worker_init_fn=seed_worker, pin_memory=True)

                optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'],
                                        weight_decay=best_params.get('weight_decay', 1e-4))
                scaler = torch.amp.GradScaler(enabled=USE_AMP);
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                 T_max=N_EPOCHS_FOR_FINAL_TRAINING) if USE_LR_SCHEDULER else None

                best_val_mae = float('inf')
                epochs_no_improve = 0
                best_model_state = None
                split_history_log = []

                for epoch in range(1, N_EPOCHS_FOR_FINAL_TRAINING + 1):
                    lambda_p = best_params['lambda_physics']
                    if USE_CURRICULUM_LEARNING: lambda_p *= (
                                LAMBDA_CURRICULUM_START + (1 - LAMBDA_CURRICULUM_START) * min(1.0,
                                                                                              epoch / LAMBDA_CURRICULUM_EPOCHS))

                    avg_losses = train_one_epoch(model, train_loader, optimizer, lambda_p, DEVICE, scaler,
                                                 GRADIENT_CLIP_VALUE, GRAD_ACCUMULATION_STEPS)

                    # --- CRITICAL: Model Selection based on VALIDATION SET ---
                    val_mae = evaluate_model(model, val_loader, DEVICE);

                    print(
                        f"    Epoch {epoch}/{N_EPOCHS_FOR_FINAL_TRAINING}, Loss: {avg_losses['total']:.4f}, Val MAE: {val_mae:.4f}",
                        end='\r')

                    if np.isnan(val_mae): break
                    split_history_log.append(
                        {'epoch': epoch, **{f'train_loss_{k}': v for k, v in avg_losses.items()}, 'val_mae': val_mae})

                    if scheduler: scheduler.step()

                    # Checkpointing logic
                    if val_mae < best_val_mae:
                        best_val_mae = val_mae
                        epochs_no_improve = 0
                        best_model_state = model.state_dict()
                    else:
                        epochs_no_improve += 1

                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                        print(f"\n    Early stopping triggered at epoch {epoch}. Best Val MAE: {best_val_mae:.4f}");
                        break
                print()

                if split_history_log:
                    generate_fold_training_plots(pd.DataFrame(split_history_log), split_name, split_analysis_dir,
                                                 member + 1);
                    plot_pinn_loss_components(pd.DataFrame(split_history_log), split_name, split_analysis_dir,
                                              member + 1)

                if best_model_state:
                    torch.save(best_model_state, model_cache_path);
                    print(f"    Saved best model (by Val MAE) to: {model_cache_path}")
                    model.load_state_dict(best_model_state);
                    split_ensemble_models.append(model)
                else:
                    print(f"    Warning: Training failed for member. Excluding from ensemble.")

        if len(split_ensemble_models) > 0:
            # --- FINAL INFERENCE ON TEST SET ---
            print(f"  Running inference on TEST set (Days {len(train_exp_ids) + 2} to end)...")
            test_subset.dataset.user_transform = val_user_transform  # Ensure no aug on test
            test_loader = DataLoader(test_subset, BATCH_SIZE, False, num_workers=NUM_WORKERS,
                                     worker_init_fn=seed_worker)

            test_results_df = pd.DataFrame(get_inference_results(split_ensemble_models, test_loader, DEVICE));
            test_results_df.to_parquet(test_results_cache_path, index=False);
            print(f"  Saved TEST results for {split_name} to cache.")

            mae = generate_final_report_for_split(test_results_df, split_analysis_dir, split_name)
            overall_performance_summary.append({'n_train_days': n_train_days, 'mae': mae})

            # Generate Residuals and maps using the Test Set loader
            member_residuals = [
                plot_physics_residual_heatmap(m, test_loader, DEVICE, split_name, split_analysis_dir, member + 1) for
                m, member in zip(split_ensemble_models, range(len(split_ensemble_models)))]

            if any(r is not None for r in member_residuals):
                avg_ensemble_residual = np.mean([r for r in member_residuals if r is not None], axis=0)
                fig_res, ax_res = plt.subplots(figsize=(8, 7));
                sns.heatmap(avg_ensemble_residual, cmap='viridis', ax=ax_res, cbar_kws={'label': 'Absolute Residual'})
                ax_res.set_title(f'{split_name} - Average Ensemble Physics Residual (Test Set)');
                ax_res.set_xlabel('Pixel X');
                ax_res.set_ylabel('Pixel Y')
                save_plot_and_data(fig_res, f'pinn_{split_name}_avg_ensemble_physics_residual',
                                   pd.DataFrame(avg_ensemble_residual),
                                   os.path.join(split_analysis_dir, "pinn_physics_heatmaps"))

            plot_attention_and_confidence_maps(split_ensemble_models[0], test_loader, DEVICE, f'pinn_{split_name}',
                                               split_analysis_dir)
            plot_biofouling_and_correlation_maps(split_ensemble_models[0], test_loader, DEVICE, exp_id_map,
                                                 red_intensity_heatmap_dir, f'pinn_{split_name}', split_analysis_dir)

        else:
            print(f"  Warning: No valid models for split {split_name}. Skipping inference and analysis.")

    if not overall_performance_summary:
        raise RuntimeError("FATAL: No valid results were generated from any train/test split.")

    summary_df = pd.DataFrame(overall_performance_summary).sort_values('n_train_days')
    plot_performance_vs_training_size(summary_df, ANALYSIS_DIR)

    print(f"\n--- SCRIPT FINISHED ---\nTotal execution time: {datetime.now() - start_time}")


if __name__ == '__main__':
    if sys.platform == 'win32' and NUM_WORKERS != 0:
        print("Warning: Setting NUM_WORKERS to 0 for Windows compatibility.");
        NUM_WORKERS = 0
    if NUM_WORKERS > 0:
        matplotlib.freeze_support()
    try:
        main()
    except Exception as e:
        print(f"\n--- A FATAL SCRIPT-LEVEL ERROR OCCURRED ---\n{e}");
        traceback.print_exc();
        sys.exit(1)
