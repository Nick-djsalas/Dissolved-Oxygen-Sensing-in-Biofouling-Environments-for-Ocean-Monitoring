#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# Physics-Loss Ablation Study (Vision Transformer + Walk-Forward Validation)
# ==============================================================================
#
# Companion code for the dissolved-oxygen (DO) sensing study on physics-informed
# deep learning under algal biofouling.
#
# --- SCIENTIFIC OBJECTIVE ---
# This script isolates the contribution of the physics-informed loss term by
# training two otherwise-identical Vision Transformer (ViT) models on the same
# data with the same protocol:
#   1. ViT + Physics Loss   (lambda_physics > 0)  -- Stern-Volmer regularisation on
#   2. ViT Data-Only        (lambda_physics = 0)  -- MSE supervision only
#
# --- VALIDATION PROTOCOL (no data leakage) ---
# Experiments are ordered chronologically and split with a strict walk-forward
# scheme so that evaluation mimics real deployment on future, unseen acquisitions:
#   - TRAIN      : experiment days [1 .. N-2]
#   - VALIDATION : experiment day  [N-1]   (early stopping / checkpoint selection)
#   - TEST       : experiment day  [N]     (final performance, reported once)
# The test set never influences model weights or checkpoint selection.
#
# --- DATA & USAGE ---
# See README.md for the expected input directory layout, file-naming conventions,
# environment setup, and configuration. Raw experimental data are not distributed
# with this repository; point DATA_ROOT (below) at your own data laid out as
# described in the README, or at a pre-built feature cache.
#
# --- NOTE ON THE TRAINING LOOP ---
# Training uses standard gradient accumulation: gradients are accumulated over
# GRAD_ACCUMULATION_STEPS mini-batches and the optimiser steps once per window,
# giving an effective batch size of BATCH_SIZE * GRAD_ACCUMULATION_STEPS. Every
# mini-batch contributes to the update.
#
# ==============================================================================

import os
import sys
import glob
import re
import traceback
import warnings
import random
import json
import functools
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import timm
from torchvision import transforms
from sklearn.metrics import mean_absolute_error

# ==============================================================================
# --- REPRODUCIBILITY CONFIGURATION ---
# ==============================================================================
# NOTE: cudnn.benchmark is enabled and deterministic mode is off for speed, so
# run-to-run variation is expected even with a fixed seed. For bitwise-comparable
# runs set deterministic=True and benchmark=False (at some throughput cost).
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use('Agg')
plt.style.use('seaborn-v0_8-whitegrid')

# ==============================================================================
# --- USER CONFIGURATION ---
# ==============================================================================
# Root directory containing the chronological experiment folders (see README for
# the expected layout). Resolution order:
#   1. the DO_ABLATION_DATA_ROOT environment variable, if set;
#   2. otherwise a local ./data directory next to this script.
# Edit the fallback below or export the environment variable rather than
# hard-coding an absolute path.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("DO_ABLATION_DATA_ROOT", os.path.join(_SCRIPT_DIR, "data"))

EXPERIMENTS_ROOT_DIR = DATA_ROOT
# Outputs (results CSV + figure) are written here; this folder is git-ignored.
OUTPUT_DIR = os.path.join(_SCRIPT_DIR, "results", "ViT_Physics_Ablation")
# Cached, pre-processed feature table. Delete it (or set FORCE_RECREATE_DATAFRAME)
# to rebuild from raw experiment folders.
FEATURE_DATAFRAME_PATH = os.path.join(DATA_ROOT, "PINN_features", "pinn_features.parquet")
FORCE_RECREATE_DATAFRAME = False

# --- Data settings ---
STORAGE_RESIZE_DIM = (48, 48)          # on-disk frame size (memory-efficient cache)
MODEL_INPUT_DIM = (224, 224)           # ViT input size
FRAME_SKIP = 15                        # keep 1 frame in every FRAME_SKIP
FORCED_VIDEO_FPS = 30                  # set to None to read FPS from the video header
FRAME_CHUNK_SIZE = 3000                # rows per intermediate parquet chunk
SENSOR_RESAMPLE_WINDOW_S = 10          # resample cadence for O2 / temperature logs
ANALYSIS_INTERVALS_S = [(1450, 1650), (2300, 2500), (3100, 3300), (3900, 4100), (4700, 4900)]

# --- Training settings ---
EPOCHS = 35
BATCH_SIZE = 24
GRAD_ACCUMULATION_STEPS = 8            # effective batch size = BATCH_SIZE * this
EARLY_STOPPING_PATIENCE = 12
LR = 3e-4
WEIGHT_DECAY = 1e-4
LAMBDA_PHYSICS = 0.1                   # weight of the physics loss when enabled

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == 'cuda')
NUM_WORKERS = 0 if sys.platform == "win32" else min(8, cpu_count() - 2)


# ==============================================================================
# --- STAGE 1: DATA PREPROCESSING (end-to-end raw ingestion) ---
# ==============================================================================

def parse_arduino_log(file_path):
    timestamps, oxygen_values = [], []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if "MEA" not in line: continue
            try:
                parts = line.split()
                dt_object = datetime.strptime(f"{parts[0]} {parts[1]}", '%Y-%m-%d %H:%M:%S')
                mea_index = parts.index("MEA")
                timestamps.append(dt_object)
                oxygen_values.append(int(parts[mea_index + 5]) / 1000.0)
            except (ValueError, IndexError, TypeError):
                pass
    df = pd.DataFrame({'timestamp': timestamps, 'oxygen_umol_L': oxygen_values})
    if df.timestamp.duplicated().any():
        df = df.groupby('timestamp')['oxygen_umol_L'].mean().reset_index()
    if not df.empty:
        df = df.set_index('timestamp').resample(f'{SENSOR_RESAMPLE_WINDOW_S}S').mean().dropna().reset_index()
    return df


def parse_temperature_log(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = ['timestamp', 'temperature_C']
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['temperature_C'] = pd.to_numeric(df['temperature_C'], errors='coerce')
        df.dropna(inplace=True)
        if df.timestamp.duplicated().any():
            df = df.groupby('timestamp')['temperature_C'].mean().reset_index()
        pct_change = df['temperature_C'].pct_change().abs()
        df = df[~(pct_change >= 1.0)].copy()
        if not df.empty:
            df = df.set_index('timestamp').resample(f'{SENSOR_RESAMPLE_WINDOW_S}S').mean().dropna().reset_index()
        return df
    except Exception:
        return pd.DataFrame()


def experiment_date(exp_id):
    """
    Parse the acquisition date out of an experiment ID / folder name such as
    '2-09_06_2025' (run-DD_MM_YYYY) -> datetime(2025, 6, 9).
    """
    m = re.search(r'(\d{1,2}[-_]\d{2}[-_]\d{4})', str(exp_id).replace('_', '-'))
    if not m:
        raise ValueError(f"Cannot parse date from experiment ID: {exp_id!r}")
    return datetime.strptime(m.group(1), '%d-%m-%Y')


def find_experiment_files(root_dir):
    """Sorts experiments CHRONOLOGICALLY based on folder dates."""
    all_dirs = [d for d in glob.glob(os.path.join(root_dir, '*-*')) if os.path.isdir(d)]
    dated_dirs = []
    for d in all_dirs:
        try:
            dated_dirs.append({'path': d, 'date': experiment_date(os.path.basename(d))})
        except ValueError:
            pass

    dated_dirs.sort(key=lambda x: x['date'])

    valid_experiments = []
    for exp in dated_dirs:
        exp_id = os.path.basename(exp['path'])
        vids = glob.glob(os.path.join(exp['path'], "**", "*ROI_Output*", "*.mp4"), recursive=True)
        ard = glob.glob(os.path.join(exp['path'], "**", "*_arduino_*.txt"), recursive=True)
        temp = glob.glob(os.path.join(exp['path'], "**", "*temperature*.csv"), recursive=True)
        if vids and ard and temp:
            valid_experiments.append({
                "id": exp_id, "video_path": vids[0], "raw_arduino_path": ard[0], "temperature_path": temp[0]
            })
    return valid_experiments


def save_chunk(data, temp_dir, exp_id, chunk_idx):
    """Memory-optimized chunk saver. Forces heavy pixel arrays to 1-byte uint8."""
    cols = ['experiment_id', 'timestamp', 'elapsed_seconds', 'temperature_C', 'oxygen_umol_L'] + \
           [f'pixel_{i}' for i in range(STORAGE_RESIZE_DIM[0] * STORAGE_RESIZE_DIM[1] * 3)]

    df_chunk = pd.DataFrame(data, columns=cols)

    pixel_cols = [c for c in cols if c.startswith('pixel_')]
    df_chunk = df_chunk.astype({c: np.uint8 for c in pixel_cols})

    df_chunk.to_parquet(os.path.join(temp_dir, f"{exp_id}_{chunk_idx}.parquet"))


def process_experiment_chunked(args):
    exp, temp_dir = args['exp'], args['temp_dir']
    try:
        df_ard = parse_arduino_log(exp['raw_arduino_path']).set_index('timestamp')
        df_tmp = parse_temperature_log(exp['temperature_path']).set_index('timestamp')

        cap = cv2.VideoCapture(exp['video_path'])
        if not cap.isOpened(): return False

        fps = FORCED_VIDEO_FPS if FORCED_VIDEO_FPS else cap.get(cv2.CAP_PROP_FPS)
        start_time = datetime.strptime(re.search(r'_(\d{8}_\d{6})', os.path.basename(exp['video_path'])).group(1),
                                       '%Y%m%d_%H%M%S')

        chunk_idx, data = 0, []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with tqdm(total=total_frames, desc=f"Processing {exp['id']}", leave=False) as pbar:
            for frame_num in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                pbar.update(1)

                if frame_num % FRAME_SKIP == 0:
                    elapsed = frame_num / fps
                    ts = start_time + timedelta(seconds=elapsed)

                    nearest_o2 = df_ard.index.get_indexer([ts], method='nearest')[0]
                    nearest_temp = df_tmp.index.get_indexer([ts], method='nearest')[0]

                    if nearest_o2 != -1 and nearest_temp != -1:
                        o2 = df_ard.iloc[nearest_o2]['oxygen_umol_L']
                        tmp = df_tmp.iloc[nearest_temp]['temperature_C']
                        res = cv2.resize(frame, STORAGE_RESIZE_DIM, interpolation=cv2.INTER_AREA)
                        rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB).flatten().tolist()
                        data.append([exp['id'], ts, elapsed, tmp, o2] + rgb)

                        if len(data) >= FRAME_CHUNK_SIZE:
                            save_chunk(data, temp_dir, exp['id'], chunk_idx)
                            data = []
                            chunk_idx += 1

            if data:
                save_chunk(data, temp_dir, exp['id'], chunk_idx)

        cap.release()
        return True
    except Exception:
        return False


def create_dataframe():
    if os.path.exists(FEATURE_DATAFRAME_PATH) and not FORCE_RECREATE_DATAFRAME:
        print(f"Loading cached dataframe: {FEATURE_DATAFRAME_PATH}")
        return pd.read_parquet(FEATURE_DATAFRAME_PATH)

    exps = find_experiment_files(EXPERIMENTS_ROOT_DIR)
    if not exps:
        raise FileNotFoundError(
            f"No valid experiment folders found under {EXPERIMENTS_ROOT_DIR!r}. "
            f"Set DATA_ROOT / DO_ABLATION_DATA_ROOT and check the layout described in README.md."
        )
    os.makedirs(os.path.dirname(FEATURE_DATAFRAME_PATH), exist_ok=True)
    temp_dir = os.path.join(os.path.dirname(FEATURE_DATAFRAME_PATH), "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    for f in glob.glob(os.path.join(temp_dir, "*.parquet")):
        try:
            os.remove(f)
        except OSError:
            pass

    for i, exp in enumerate(exps):
        process_experiment_chunked({'exp': exp, 'temp_dir': temp_dir})

    chunks = glob.glob(os.path.join(temp_dir, "*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in chunks], ignore_index=True)

    df.to_parquet(FEATURE_DATAFRAME_PATH, index=False)
    return df


# ==============================================================================
# --- STAGE 2: PYTORCH ARCHITECTURE (Vision Transformer) ---
# ==============================================================================

class VideoFrameDataset(Dataset):
    def __init__(self, df, exp_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.pixels = self.df[[c for c in self.df.columns if c.startswith('pixel_')]].values.astype(np.uint8)
        self.o2 = self.df['oxygen_umol_L'].values.astype(np.float32)
        self.temps = self.df['temperature_C'].values.astype(np.float32)
        self.exps = self.df['experiment_id'].map(exp_map).values.astype(np.int64)

        self.base_tf = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(MODEL_INPUT_DIM, antialias=True)
        ])
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        img = self.pixels[i].reshape(STORAGE_RESIZE_DIM[0], STORAGE_RESIZE_DIM[1], 3)
        t = self.base_tf(img)
        if self.transform: t = self.transform(t)
        return t, torch.tensor(self.o2[i]), torch.tensor(self.exps[i]), torch.tensor(self.temps[i])


class SensorPINN(nn.Module):
    """Vision Transformer (ViT) with multi-head outputs (O2 regression, mask,
    confidence, and per-experiment Stern-Volmer parameters)."""

    def __init__(self, num_exps, embed_dim=8):
        super().__init__()
        self.vit = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=True, num_classes=0)
        patch_size = self.vit.patch_embed.patch_size[0]
        self.grid_size = MODEL_INPUT_DIM[0] // patch_size

        self.exp_emb = nn.Embedding(num_exps, embed_dim)
        self.sv_head = nn.Sequential(nn.Linear(embed_dim, 16), nn.GELU(), nn.Linear(16, 4))
        self.o2_head = nn.Sequential(nn.Linear(self.vit.embed_dim + 1, 128), nn.GELU(), nn.Linear(128, 1))

        dc = 256
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.vit.embed_dim, dc, 1), nn.GELU(),
            nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=False),
            nn.Conv2d(dc, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )
        self.conf_head = nn.Sequential(
            nn.Conv2d(self.vit.embed_dim, dc, 1), nn.GELU(),
            nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=False),
            nn.Conv2d(dc, 64, 3, 1, 1), nn.GELU(), nn.Conv2d(64, 1, 1), nn.Sigmoid()
        )

    def forward(self, img, exp_idx, temp):
        feats = self.vit.forward_features(img)
        cls_token = feats[:, 0]
        combined = torch.cat([cls_token, temp.view(-1, 1)], dim=1)
        o2_pred = self.o2_head(combined).squeeze(-1)

        patch_tokens = feats[:, 1:]
        B, N, C = patch_tokens.shape
        grid = patch_tokens.reshape(B, self.grid_size, self.grid_size, C).permute(0, 3, 1, 2).contiguous()

        mask = self.mask_head(grid).squeeze(1)
        conf = self.conf_head(grid).squeeze(1)
        conf = conf / (torch.sum(conf.view(B, -1), dim=1, keepdim=True).view(B, 1, 1) + 1e-8)

        raw_sv = self.sv_head(self.exp_emb(exp_idx))
        sv_params = {
            "i0": nn.functional.softplus(raw_sv[:, 0]),
            "ksv1": nn.functional.softplus(raw_sv[:, 1]),
            "ksv2": nn.functional.softplus(raw_sv[:, 2]),
            "a": torch.sigmoid(raw_sv[:, 3])
        }

        return {"o2_pred": o2_pred, "mask": mask, "conf": conf, "sv_params": sv_params}


# ==============================================================================
# --- STAGE 3: LOSS & STRICT VALIDATION CHECKPOINTING ---
# ==============================================================================

def calculate_pinn_loss(outputs, batch, lambda_phys):
    """Computes MSE data loss + 2-site Stern-Volmer physics loss (relative)."""
    img, o2_true, _, _ = batch
    o2_pred = outputs["o2_pred"]

    loss_data = nn.functional.mse_loss(o2_pred, o2_true)

    if lambda_phys > 0:
        red = transforms.functional.resize(img, outputs["mask"].shape[-2:], antialias=True)[:, 0, :, :] * 255.0
        sv = outputs["sv_params"]

        o2r = o2_pred.view(-1, 1, 1)
        i0 = sv["i0"].view(-1, 1, 1)
        k1 = sv["ksv1"].view(-1, 1, 1)
        k2 = sv["ksv2"].view(-1, 1, 1)
        a = sv["a"].view(-1, 1, 1)

        i_hat = i0 * ((a / (1 + k1 * o2r)) + ((1 - a) / (1 + k2 * o2r)))
        residual = torch.abs(red - i_hat) / (red + 1.0)

        squared = (outputs["mask"] * residual) ** 2
        loss_phys = torch.mean(outputs["conf"] * squared)
    else:
        loss_phys = torch.tensor(0.0, device=DEVICE)

    return loss_data + (lambda_phys * loss_phys)


def train_eval_run(model, train_ld, val_ld, test_ld, lambda_phys, run_name):
    """
    Train with strict validation-based model selection.

    The best checkpoint is chosen on validation MAE only; the test set is
    evaluated once at the end for reporting and never used for checkpointing.
    Training uses gradient accumulation (effective batch size =
    BATCH_SIZE * GRAD_ACCUMULATION_STEPS); every mini-batch contributes.
    """
    print(f"\n--- Training: {run_name} (Lambda Physics = {lambda_phys}) ---")
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    best_val_mae = float('inf')
    best_weights = None
    patience_counter = 0

    for ep in range(EPOCHS):
        # --- TRAINING (gradient accumulation) ---
        model.train()
        opt.zero_grad(set_to_none=True)
        accumulated = 0

        for i, batch in enumerate(train_ld):
            # Async host->device transfer (non-blocking works with pinned memory).
            batch = [d.to(DEVICE, non_blocking=True) for d in batch]

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                out = model(batch[0], batch[2], batch[3])
                # Scale by the accumulation factor so the summed gradient matches
                # the mean over the effective (large) batch.
                loss = calculate_pinn_loss(out, batch, lambda_phys) / GRAD_ACCUMULATION_STEPS

            if torch.isnan(loss):
                continue

            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accumulated += 1

            # Step once per accumulation window.
            if accumulated == GRAD_ACCUMULATION_STEPS:
                if USE_AMP:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                accumulated = 0

        # Flush gradients left in a final, partial accumulation window.
        if accumulated > 0:
            if USE_AMP:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            opt.zero_grad(set_to_none=True)

        # --- VALIDATION EVALUATION ---
        model.eval()
        val_preds, val_trues = [], []
        with torch.no_grad():
            for batch in val_ld:
                batch = [d.to(DEVICE, non_blocking=True) for d in batch]
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    val_preds.append(model(batch[0], batch[2], batch[3])["o2_pred"])
                val_trues.append(batch[1])

        val_preds = torch.cat(val_preds).cpu().numpy()
        val_trues = torch.cat(val_trues).cpu().numpy()

        val_mae = mean_absolute_error(val_trues, val_preds)
        print(f"Epoch {ep + 1}/{EPOCHS} | Val MAE: {val_mae:.3f}")

        # --- CHECKPOINT ON VALIDATION MAE ---
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {ep + 1}")
                break

    # --- FINAL TEST INFERENCE ON UNSEEN DATA ---
    print("Evaluating best validation checkpoint on the UNSEEN test set...")
    model.load_state_dict(best_weights)
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for batch in test_ld:
            batch = [d.to(DEVICE, non_blocking=True) for d in batch]
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                test_preds.append(model(batch[0], batch[2], batch[3])["o2_pred"])
            test_trues.append(batch[1])

    test_preds = torch.cat(test_preds).cpu().numpy()
    test_trues = torch.cat(test_trues).cpu().numpy()

    test_mae = mean_absolute_error(test_trues, test_preds)
    print(f"Final Test MAE ({run_name}): {test_mae:.3f} umol/L")
    return test_mae


# ==============================================================================
# --- STAGE 4: ORCHESTRATION ---
# ==============================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    df = create_dataframe()
    conds = [(df['elapsed_seconds'] >= s) & (df['elapsed_seconds'] <= e) for s, e in ANALYSIS_INTERVALS_S]
    df = df[functools.reduce(np.logical_or, conds)].reset_index(drop=True)

    # 2. Chronological walk-forward split
    exp_ids = sorted(df['experiment_id'].unique(), key=experiment_date)
    if len(exp_ids) < 3: raise ValueError("Need >= 3 experiments for a walk-forward split.")

    train_ids = exp_ids[:-2]
    val_id = [exp_ids[-2]]
    test_id = [exp_ids[-1]]

    print(f"\n--- Walk-Forward Split Configuration ---")
    for tag, ids in (("TRAIN", train_ids), ("VAL", val_id), ("TEST", test_id)):
        print(f"{tag:<5s} : " + ", ".join(f"{i} [{experiment_date(i):%d %b %Y}]" for i in ids))

    exp_map = {k: i for i, k in enumerate(exp_ids)}

    # 3. Dataloaders
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    aug = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), norm])

    ds_tr = VideoFrameDataset(df[df['experiment_id'].isin(train_ids)], exp_map, transform=aug)
    ds_va = VideoFrameDataset(df[df['experiment_id'].isin(val_id)], exp_map, transform=norm)
    ds_te = VideoFrameDataset(df[df['experiment_id'].isin(test_id)], exp_map, transform=norm)

    use_pin = (DEVICE.type == 'cuda')
    persist = (NUM_WORKERS > 0)

    ld_tr = DataLoader(ds_tr, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                       pin_memory=use_pin)
    ld_va = DataLoader(ds_va, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_pin,
                       persistent_workers=persist)
    ld_te = DataLoader(ds_te, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=use_pin,
                       persistent_workers=persist)

    # 4. Ablation study
    # Run 1: with physics loss
    model_physics = SensorPINN(len(exp_ids)).to(DEVICE)
    mae_physics = train_eval_run(model_physics, ld_tr, ld_va, ld_te, LAMBDA_PHYSICS, "ViT + Physics")

    # Run 2: data-only (no physics loss)
    model_data = SensorPINN(len(exp_ids)).to(DEVICE)
    mae_data = train_eval_run(model_data, ld_tr, ld_va, ld_te, 0.0, "ViT Data-Only")

    # 5. Output results
    results = pd.DataFrame([
        {"Model Variant": "ViT + Physics", "Test MAE (umol/L)": mae_physics},
        {"Model Variant": "ViT Data-Only", "Test MAE (umol/L)": mae_data}
    ])
    results.to_csv(os.path.join(OUTPUT_DIR, "Ablation_Results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=results, x="Model Variant", y="Test MAE (umol/L)", ax=ax, palette="Set2")
    ax.set_title("Physics Loss Ablation (Strict Walk-Forward Validation)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "Ablation_BarChart.png"), dpi=300)
    print(f"\n--- Ablation complete. Results saved to: {OUTPUT_DIR} ---")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)