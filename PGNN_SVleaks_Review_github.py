#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pgnn_pretrained_leakage.py

Physics-Guided Neural Network (PGNN) Pretrained Leakage Analysis
Nature Communications — Review Response (Point 11, SV-map leakage)

This script implements a pretrained PGNN to evaluate the effect of spatial SV-map
leakage. It reproduces the physics-guided model using a pretrained backbone and
measures the test-set leakage from pre-fitted per-pixel Stern-Volmer (SV) maps.

Protocols evaluated:
  - Leaked: Every experiment uses its own pre-fitted SV map as input.
  - Fixed: The held-out day uses the pixel-wise mean SV map from the training
    days (no test-day labels). This is the deployment-realistic baseline.

Note on Architecture:
    The model closely matches the baseline PCNN architecture (Pretrained ResNet-18,
    CBAM, identical oxygen/mask/confidence heads) but differs where the PGNN concept
    requires it (SV maps provided as input channels instead of learned embeddings).
"""

import os
import sys
import re
import json
import random
import warnings
import functools
import traceback
import logging
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import curve_fit
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

try:
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

# =============================================================================
#  SECTION 0 — REPRODUCIBILITY + CONFIGURATION
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.benchmark = True
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    pass


def _env(name, default=None):
    """Retrieve environment variables with fallback defaults."""
    v = os.environ.get(name)
    return v if (v is not None and str(v).strip() != "") else default


# --- Paths: Configured for GitHub structure; override via environment variables
DEFAULT_BASE_DIR = "./data"
BASE_PROJECT_DIR = _env("REBUTTAL_BASE_DIR", DEFAULT_BASE_DIR)

FEATURE_DATAFRAME_PATH = _env(
    "REBUTTAL_FEATURE_PARQUET",
    os.path.join(BASE_PROJECT_DIR, "PINN_features_new", "pinn_features_resampled_new.parquet")
)

SV_PARAMS_PATH = _env(
    "REBUTTAL_SV_PARAMS",
    os.path.join(BASE_PROJECT_DIR, "SV_parameters_new_48", "per_pixel_sv_params_new.parquet")
)

HPO_PARAMS_PATH = _env(
    "REBUTTAL_HPO_JSON",
    os.path.join(BASE_PROJECT_DIR, "PINN_hpo_cache_new", "best_hpo_params_v25_new.json")
)

OUTPUT_ROOT = _env("REBUTTAL_OUTPUT_DIR", "./results_pgnn_leakage")

CACHE_CKPT = os.path.join(OUTPUT_ROOT, "_cache", "ckpt")
OOF_DIR = os.path.join(OUTPUT_ROOT, "_cache", "oof")
OUT_DIR = os.path.join(OUTPUT_ROOT, "results")
LOG_DIR = os.path.join(OUTPUT_ROOT, "logs")

# --- Geometry / Physics Constants
RESIZE_DIM = (48, 48)
ANALYSIS_INTERVALS_S = [(1450, 1650), (2300, 2500), (3100, 3300),
                        (3900, 4100), (4700, 4900)]
DO_SETPOINTS_VV = [0, 5, 10, 15, 20]
SV_PARAM_SAMPLES = 1000

# --- Model / Training Settings
TRANSFER_MODEL_NAME = "resnet18"
USE_ATTENTION = True
USE_CONFIDENCE_LOSS = True
USE_DATA_AUGMENTATION = True
USE_CURRICULUM_LEARNING = True
LAMBDA_CURRICULUM_START = 0.01
LAMBDA_CURRICULUM_EPOCHS = 10
N_EPOCHS_FOR_FINAL_TRAINING = 40
BATCH_SIZE = 128
GRAD_ACCUMULATION_STEPS = 4
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP_VALUE = 1.0

# Default hyperparameters (used if HPO JSON is missing)
DEFAULT_HPO = {"lr": 3e-4, "lambda_physics": 0.5, "weight_decay": 1e-4}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")
DEVICE_TYPE = DEVICE.type
NUM_WORKERS = 0 if sys.platform == "win32" else max(1, (os.cpu_count() or 4) - 2)

# --- Protocol Toggles
RUN_LEAKED = True
RUN_FIXED = True
SELECTION = _env("REBUTTAL_SELECTION", "best_epoch")

SMOKE_TEST = _env("REBUTTAL_SMOKE", "0") == "1"
if SMOKE_TEST:
    N_EPOCHS_FOR_FINAL_TRAINING = 2
    SV_PARAM_SAMPLES = 60


# =============================================================================
#  SECTION 0.1 — LOGGING + HELPERS
# =============================================================================
def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def setup_logging():
    ensure_dirs(LOG_DIR)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger("pgnnpt")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(os.path.join(LOG_DIR, f"pgnn_pt_{ts}.log"), encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


LOG = logging.getLogger("pgnnpt")


def banner(msg):
    LOG.info("=" * 78)
    LOG.info(msg)
    LOG.info("=" * 78)


def safe_mae(y_true, y_pred):
    yt = np.asarray(y_true, float)
    yp = np.asarray(y_pred, float)
    m = np.isfinite(yt) & np.isfinite(yp)
    return float(mean_absolute_error(yt[m], yp[m])) if m.sum() else np.nan


def micro_macro_mae(df, true_col="o2_true", pred_col="o2_pred", group_col="experiment_id"):
    micro = safe_mae(df[true_col], df[pred_col])
    per_day = df.groupby(group_col).apply(lambda g: safe_mae(g[true_col], g[pred_col]))
    macro = float(np.nanmean(per_day.values)) if len(per_day) else np.nan
    return micro, macro


def assign_do_bin(elapsed_seconds):
    for i, (s, e) in enumerate(ANALYSIS_INTERVALS_S):
        if s <= elapsed_seconds <= e:
            return i
    return -1


def seed_worker(_):
    ws = torch.initial_seed() % 2 ** 32
    np.random.seed(ws)
    random.seed(ws)


# =============================================================================
#  SECTION 1 — DATA + SV MAPS
# =============================================================================
def load_feature_dataframe():
    banner("SECTION 1 — LOAD FEATURE DATAFRAME")
    if not os.path.exists(FEATURE_DATAFRAME_PATH):
        raise FileNotFoundError(f"Feature parquet not found: {FEATURE_DATAFRAME_PATH}")

    df = pd.read_parquet(FEATURE_DATAFRAME_PATH)
    need = {"experiment_id", "timestamp", "elapsed_seconds", "temperature_C", "oxygen_umol_L"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Parquet missing columns: {miss}")

    LOG.info("Loaded %d frames, %d experiments", len(df), df["experiment_id"].nunique())
    return df


def enrich(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    cond = [(df["elapsed_seconds"] >= s) & (df["elapsed_seconds"] <= e) for s, e in ANALYSIS_INTERVALS_S]
    df = df[functools.reduce(np.logical_or, cond)].reset_index(drop=True)
    df["do_bin"] = df["elapsed_seconds"].apply(assign_do_bin).astype(int)

    order = df.groupby("experiment_id")["timestamp"].min().sort_values().index.tolist()
    df["day_index"] = df["experiment_id"].map({e: i for i, e in enumerate(order)}).astype(int)

    bin_med = df[df["do_bin"] >= 0].groupby("do_bin")["oxygen_umol_L"].median().sort_values()
    rank_map = {b: r for r, b in enumerate(bin_med.index.tolist())}
    df["do_rank"] = df["do_bin"].map(lambda b: rank_map.get(b, -1)).astype(int)
    return df, order


def nonlinear_sv_model(O2, i0, ksv1, ksv2, a):
    """Two-site Stern-Volmer model."""
    return i0 * ((a / (1 + ksv1 * O2)) + ((1 - a) / (1 + ksv2 * O2)))


def _fit_one_pixel(o2, inten):
    valid = inten > 1
    if np.sum(valid) < 10:
        return [np.nan] * 4

    O2f, If = o2[valid], inten[valid]
    try:
        p0 = [np.max(If), 0.01, 0.001, 0.5]
        bounds = ([0, 0, 0, 0], [255, 1, 1, 1])
        popt, _ = curve_fit(nonlinear_sv_model, O2f, If, p0=p0, bounds=bounds, maxfev=2000)
        return list(popt)
    except (RuntimeError, ValueError):
        return [np.nan] * 4


def load_or_compute_sv_params(df):
    """Loads precomputed per-pixel SV maps or computes them if unavailable."""
    if os.path.exists(SV_PARAMS_PATH):
        LOG.info("Reusing saved SV parameters: %s", SV_PARAMS_PATH)
        return pd.read_parquet(SV_PARAMS_PATH)

    LOG.info("Saved SV params not found; recomputing...")
    num_pixels = RESIZE_DIM[0] * RESIZE_DIM[1]
    red_cols = [f"pixel_{i}" for i in range(0, num_pixels * 3, 3)]
    out = []

    for exp in tqdm(df["experiment_id"].unique(), desc="SV fit"):
        ex = df[df["experiment_id"] == exp]
        if len(ex) < 50:
            continue

        s = ex.sample(n=min(SV_PARAM_SAMPLES, len(ex)), random_state=SEED)
        o2, inten = s["oxygen_umol_L"].values, s[red_cols].values

        if _HAS_JOBLIB:
            res = Parallel(n_jobs=max(1, (os.cpu_count() or 2) - 1))(
                delayed(_fit_one_pixel)(o2, inten[:, i]) for i in range(num_pixels)
            )
        else:
            res = [_fit_one_pixel(o2, inten[:, i]) for i in range(num_pixels)]

        d = pd.DataFrame(np.array(res, float), columns=["i0", "ksv1", "ksv2", "a"])
        d["experiment_id"] = exp
        d["pixel_index"] = np.arange(num_pixels)
        out.append(d)

    sv = pd.concat(out, ignore_index=True)
    sv[["i0", "ksv1", "ksv2", "a"]] = sv.groupby("experiment_id")[["i0", "ksv1", "ksv2", "a"]].transform(
        lambda x: x.fillna(x.median())
    )

    ensure_dirs(os.path.dirname(SV_PARAMS_PATH))
    sv.to_parquet(SV_PARAMS_PATH, index=False)
    return sv


def sv_df_to_dict(sv_df):
    """Converts SV DataFrame to a dictionary of spatial Tensors (4, H, W)."""
    d = {}
    for exp in sv_df["experiment_id"].unique():
        g = sv_df[sv_df["experiment_id"] == exp].sort_values("pixel_index")
        maps = [torch.tensor(g[c].values.reshape(RESIZE_DIM), dtype=torch.float32)
                for c in ["i0", "ksv1", "ksv2", "a"]]
        d[exp] = torch.stack(maps, dim=0)
    return d


def imputed_test_map(sv_dict, train_exps):
    """Imputes SV maps for the held-out day using the mean of the training days."""
    present = [e for e in train_exps if e in sv_dict]
    return torch.stack([sv_dict[e] for e in present], dim=0).mean(dim=0)


def build_fold_sv_dict(sv_dict, protocol, train_exps, test_exp):
    if protocol == "leaked":
        return sv_dict
    fold = dict(sv_dict)
    fold[test_exp] = imputed_test_map(sv_dict, train_exps)
    return fold


# =============================================================================
#  SECTION 2 — DATASET
# =============================================================================
_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class PGNNDataset(Dataset):
    """
    Returns:
        img_norm : (3,H,W) ImageNet-normalised RGB           -> model input (ch 0-2)
        sv_input : (4,H,W) SV maps for input (i0/255, rest)  -> model input (ch 3-6)
        sv_raw   : (4,H,W) raw SV maps (i0 in 0-255)         -> physics loss
        red_raw  : (H,W)   raw red channel (0-255)           -> physics loss
        o2, temp

    Note: Spatial SV maps are flipped JOINTLY with the image during augmentation
    to maintain spatial alignment.
    """

    def __init__(self, dataframe, sv_map_dict, resize_dim=(48, 48), augment=False):
        self.df = dataframe.reset_index(drop=True)
        self.sv_map_dict = sv_map_dict
        self.resize_dim = resize_dim
        self.pixel_cols = [c for c in self.df.columns if c.startswith("pixel_")]
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = row[self.pixel_cols].values.astype(np.uint8).reshape(
            self.resize_dim[0], self.resize_dim[1], 3)

        img = transforms.functional.to_tensor(image)
        sv_raw = self.sv_map_dict[row["experiment_id"]].clone()

        # JOINT flips for spatial alignment
        if self.augment:
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])
                sv_raw = torch.flip(sv_raw, dims=[2])
            if random.random() < 0.5:
                img = torch.flip(img, dims=[1])
                sv_raw = torch.flip(sv_raw, dims=[1])

        red_raw = img[0] * 255.0
        img_norm = _IMAGENET(img)

        sv_input = sv_raw.clone()
        sv_input[0] = sv_input[0] / 255.0

        o2 = torch.tensor(row["oxygen_umol_L"], dtype=torch.float32)
        tp = torch.tensor(row["temperature_C"], dtype=torch.float32)

        return img_norm, o2, tp, sv_input, sv_raw, red_raw


# =============================================================================
#  SECTION 3 — MODEL
# =============================================================================
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        x = x * self.sigmoid(avg_out + max_out)

        avg_pool_spatial = torch.mean(x, 1, keepdim=True)
        max_pool_spatial = torch.max(x, 1, keepdim=True)[0]
        sp = self.sigmoid(self.conv(torch.cat([avg_pool_spatial, max_pool_spatial], 1)))

        return x * sp, sp


class AlgaePGNN(nn.Module):
    """
    PGNN built to mirror the PCNN architecture, but with the learned-SV head
    replaced by pre-fitted per-pixel SV maps supplied as input channels.
    """

    def __init__(self, resize_dim=(48, 48), backbone="resnet18"):
        super().__init__()
        self.resize_dim = resize_dim
        self.input_channels = 7  # 3 RGB + 4 SV maps

        self.cnn_base = timm.create_model(
            backbone, pretrained=True, in_chans=self.input_channels,
            num_classes=0, features_only=True
        )

        with torch.no_grad():
            feats = self.cnn_base(torch.zeros(1, self.input_channels, *resize_dim))
            num_final = feats[-1].shape[1]
            fdim = feats[-1].shape[1] * feats[-1].shape[2] * feats[-1].shape[3]

        self.attention = CBAM(num_final) if USE_ATTENTION else nn.Identity()

        # Flatten oxygen head (fdim + temperature -> 128 -> 1)
        self.oxygen_head = nn.Sequential(
            nn.Linear(fdim + 1, 128), nn.GELU(), nn.Linear(128, 1)
        )

        self.mask_head = nn.Sequential(
            nn.Upsample(size=resize_dim, mode="bilinear", align_corners=False),
            nn.Conv2d(num_final, 64, 3, 1, 1), nn.GELU(),
            nn.Conv2d(64, 16, 3, 1, 1), nn.GELU(),
            nn.Conv2d(16, 1, 1), nn.Sigmoid()
        )

        if USE_CONFIDENCE_LOSS:
            self.confidence_head = nn.Sequential(
                nn.Upsample(size=resize_dim, mode="bilinear", align_corners=False),
                nn.Conv2d(num_final, 64, 3, 1, 1), nn.GELU(),
                nn.Conv2d(64, 16, 3, 1, 1), nn.GELU(),
                nn.Conv2d(16, 1, 1), nn.Sigmoid()
            )

    def forward(self, image_norm, temperature, sv_input):
        x = torch.cat([image_norm, sv_input], dim=1)
        feats = self.cnn_base(x)
        cnn = feats[-1] if isinstance(feats, list) else feats

        if USE_ATTENTION:
            cnn, _ = self.attention(cnn)

        flat = cnn.view(cnn.size(0), -1)
        oxygen = self.oxygen_head(torch.cat([flat, temperature.view(-1, 1)], dim=1))
        mask = self.mask_head(cnn)

        conf = None
        if USE_CONFIDENCE_LOSS:
            conf = self.confidence_head(cnn).squeeze(1)
            b = conf.size(0)
            conf = conf / (torch.sum(conf.view(b, -1), 1, keepdim=True).view(b, 1, 1) + 1e-8)

        return {
            "oxygen_pred": oxygen.squeeze(-1),
            "mask_pred": mask.squeeze(1),
            "confidence_map": conf
        }


def calculate_pgnn_loss(outputs, batch, lambda_physics):
    """
    Physics-loss form matching the baseline, using pre-fitted per-pixel SV maps
    and the un-normalized red channel (0-255) to align scales.
    """
    _, o2_true, _, _, sv_raw, red_raw = batch
    o2_pred = outputs["oxygen_pred"]
    mask = outputs["mask_pred"]
    conf = outputs["confidence_map"]

    loss_data = nn.functional.mse_loss(o2_pred, o2_true)

    i0, k1, k2, a = sv_raw[:, 0], sv_raw[:, 1], sv_raw[:, 2], sv_raw[:, 3]
    o2 = o2_pred.view(-1, 1, 1)

    i_hat = i0 * (a / (1 + k1 * o2) + (1 - a) / (1 + k2 * o2))
    resid = torch.abs(red_raw - i_hat) / (red_raw + 1.0)

    sq = (mask * resid) ** 2
    if USE_CONFIDENCE_LOSS and conf is not None:
        loss_phys = torch.mean(conf * sq)
    else:
        loss_phys = torch.mean(sq)

    return loss_data + lambda_physics * loss_phys


# =============================================================================
#  SECTION 4 — TRAINING & EVALUATION
# =============================================================================
def _loader(sub_df, sv_dict, augment, shuffle):
    return DataLoader(
        PGNNDataset(sub_df, sv_dict, RESIZE_DIM, augment=augment),
        batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker, pin_memory=(DEVICE.type == "cuda")
    )


def _evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = [d.to(DEVICE, non_blocking=True) for d in batch]
            img_norm, o2_true, temp, sv_input, _, _ = batch

            with torch.amp.autocast(device_type=DEVICE_TYPE, dtype=torch.float16, enabled=USE_AMP):
                out = model(img_norm, temp, sv_input)

            preds.extend(out["oxygen_pred"].detach().cpu().numpy())
            trues.extend(o2_true.cpu().numpy())

    return np.array(trues), np.array(preds)


def train_fold(train_df, test_df, fold_sv_dict, hpo, run_key, val_df=None):
    ensure_dirs(CACHE_CKPT)
    summ_path = os.path.join(CACHE_CKPT, f"{run_key}.json")
    pred_path = os.path.join(CACHE_CKPT, f"{run_key}_pred.parquet")

    if os.path.exists(summ_path) and os.path.exists(pred_path):
        with open(summ_path) as f:
            s = json.load(f)
        s["_pred"] = pd.read_parquet(pred_path)
        return s

    torch.manual_seed(SEED)
    model = AlgaePGNN(RESIZE_DIM, backbone=TRANSFER_MODEL_NAME).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=hpo["lr"], weight_decay=hpo["weight_decay"])
    scaler = torch.amp.GradScaler(enabled=USE_AMP)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS_FOR_FINAL_TRAINING)

    tl = _loader(train_df, fold_sv_dict, USE_DATA_AUGMENTATION, True)
    xl = _loader(test_df, fold_sv_dict, False, False)
    sel_loader = _loader(val_df, fold_sv_dict, False, False) if (val_df is not None) else xl

    best_sel, best_state, best_ep, opt_best_test = float("inf"), None, -1, float("inf")

    for ep in range(1, N_EPOCHS_FOR_FINAL_TRAINING + 1):
        lam = hpo["lambda_physics"]
        if USE_CURRICULUM_LEARNING and ep < LAMBDA_CURRICULUM_EPOCHS:
            lam *= (LAMBDA_CURRICULUM_START + (1 - LAMBDA_CURRICULUM_START) * (ep / LAMBDA_CURRICULUM_EPOCHS))

        model.train()
        opt.zero_grad()

        for i, batch in enumerate(tl):
            batch = [d.to(DEVICE, non_blocking=True) for d in batch]
            img_norm, _, temp, sv_input, _, _ = batch

            with torch.amp.autocast(device_type=DEVICE_TYPE, dtype=torch.float16, enabled=USE_AMP):
                out = model(img_norm, temp, sv_input)
                loss = calculate_pgnn_loss(out, batch, lam) / GRAD_ACCUMULATION_STEPS

            if torch.isnan(loss):
                continue

            if USE_AMP:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
                if USE_AMP:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)

                if USE_AMP:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()

        sched.step()

        y_sel, p_sel = _evaluate(model, sel_loader)
        sel_mae = safe_mae(y_sel, p_sel)

        y_test, p_test = _evaluate(model, xl)
        test_mae = safe_mae(y_test, p_test)

        if np.isfinite(test_mae):
            opt_best_test = min(opt_best_test, test_mae)

        if np.isfinite(sel_mae) and sel_mae < best_sel:
            best_sel, best_ep = sel_mae, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        LOG.info("  [%s] ep %d/%d sel=%.3f test=%.3f (best_sel=%.3f@%d)",
                 run_key, ep, N_EPOCHS_FOR_FINAL_TRAINING, sel_mae, test_mae, best_sel, best_ep)

        if best_ep > 0 and (ep - best_ep) >= EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, p_test = _evaluate(model, xl)
    pred = pd.DataFrame({
        "o2_true": y_test,
        "o2_pred": p_test,
        "experiment_id": test_df["experiment_id"].values,
        "do_rank": test_df["do_rank"].values
    })

    pred.to_parquet(pred_path, index=False)
    s = {
        "run_key": run_key,
        "reported_test_mae": safe_mae(y_test, p_test),
        "optimistic_best_test_mae": (opt_best_test if np.isfinite(opt_best_test) else None),
        "best_epoch": best_ep
    }

    with open(summ_path, "w") as f:
        json.dump(s, f, indent=2)
    s["_pred"] = pred
    return s


# =============================================================================
#  SECTION 5 — LOOCV DRIVER
# =============================================================================
def run(df, sv_dict, days_sorted, hpo):
    banner(f"PRETRAINED PGNN LEAKAGE — Leaked vs Fixed (Selection = {SELECTION})")
    ensure_dirs(OUT_DIR, OOF_DIR)
    LOG.info("Hyperparameters: %s", hpo)

    logo = LeaveOneGroupOut()
    groups = df["experiment_id"].values
    protocols = ([("leaked", "leaked")] if RUN_LEAKED else []) + ([("fixed", "fixed")] if RUN_FIXED else [])
    summary_rows, stratified_rows = [], []

    for proto_label, proto in protocols:
        preds, optimistic = [], []

        for fold, (tr, te) in enumerate(logo.split(df, groups=groups)):
            train_df = df.iloc[tr].reset_index(drop=True)
            test_df = df.iloc[te].reset_index(drop=True)
            test_exp = test_df["experiment_id"].iloc[0]
            train_exps = list(pd.unique(train_df["experiment_id"]))

            val_df, train_use = None, train_df
            if SELECTION == "val_selected":
                ti = days_sorted.index(test_exp)
                val_exp = days_sorted[ti - 1] if ti - 1 >= 0 and days_sorted[ti - 1] != test_exp else train_exps[-1]
                val_df = df[df["experiment_id"] == val_exp].reset_index(drop=True)
                train_use = train_df[train_df["experiment_id"] != val_exp].reset_index(drop=True)
                train_exps = [e for e in train_exps if e != val_exp]

            fold_sv = build_fold_sv_dict(sv_dict, proto, train_exps, test_exp)
            run_key = f"pgnnpt_{proto}_{SELECTION}_{re.sub(r'[^A-Za-z0-9]', '_', str(test_exp))}"

            s = train_fold(train_use, test_df, fold_sv, hpo, run_key, val_df=val_df)
            preds.append(s["_pred"])

            if s["optimistic_best_test_mae"] is not None:
                optimistic.append(s["optimistic_best_test_mae"])

            LOG.info("PGNN[%s] fold %d test=%s reported=%.3f",
                     proto_label, fold + 1, test_exp, s["reported_test_mae"])

        oof = pd.concat(preds, ignore_index=True)
        oof.to_parquet(os.path.join(OOF_DIR, f"pgnn_{proto}_{SELECTION}.parquet"), index=False)

        micro, macro = micro_macro_mae(oof)
        best_test_meanfold = float(np.nanmean(optimistic)) if optimistic else np.nan

        summary_rows.append({
            "model": "PGNN(pretrained)",
            "protocol": proto_label,
            "selection": SELECTION,
            "micro_pooled_MAE": micro,
            "macro_mean_of_day_MAE": macro,
            "best_epoch_test_meanfold": best_test_meanfold,
            "n_frames": int(len(oof))
        })

        LOG.info("PGNN[%s] micro=%.2f macro=%.2f (best-epoch mean-fold=%.2f)",
                 proto_label, micro, macro, best_test_meanfold)

        if proto == "fixed":
            for rank, g in oof.groupby("do_rank"):
                if rank < 0:
                    continue
                err = np.abs(g["o2_pred"].values - g["o2_true"].values)
                err = err[np.isfinite(err)]

                stratified_rows.append({
                    "model": "PGNN (pretrained, fixed)",
                    "do_rank": int(rank),
                    "do_setpoint_vv": DO_SETPOINTS_VV[int(rank)] if 0 <= int(rank) < 5 else np.nan,
                    "n": int(len(err)),
                    "mae_umol": float(np.mean(err)) if len(err) else np.nan,
                    "mae_std_umol": float(np.std(err)) if len(err) else np.nan
                })

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(os.path.join(OUT_DIR, "PGNN_leaked_vs_fixed_summary.csv"), index=False)

    if stratified_rows:
        pd.DataFrame(stratified_rows).to_csv(
            os.path.join(OUT_DIR, "PGNN_per_do_range_mae_fixed.csv"), index=False
        )

    if {"leaked", "fixed"} <= set(summ["protocol"]):
        lk = float(summ[summ.protocol == "leaked"]["micro_pooled_MAE"].iloc[0])
        fx = float(summ[summ.protocol == "fixed"]["micro_pooled_MAE"].iloc[0])

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(["leaked", "fixed"], [lk, fx], color=["tab:red", "tab:green"])
        for i, v in enumerate([lk, fx]):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom")

        ax.set_ylabel("Pooled (micro) MAE (umol/L)")
        ax.set_title(f"Pretrained PGNN leakage effect ({SELECTION})")
        fig.savefig(os.path.join(OUT_DIR, "PGNN_leaked_vs_fixed_bar.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

        LOG.info("LEAKED -> FIXED (micro MAE): %.2f -> %.2f  (%+.1f%%)",
                 lk, fx, (fx - lk) / lk * 100 if lk else float("nan"))
        LOG.info("NOTE: 'Leaked' corresponds to own-day SV maps (acting as an upper bound baseline). "
                 "'Fixed' corresponds to deployment-realistic mean SV maps.")

    return summ


# =============================================================================
#  SECTION 6 — MAIN ENTRY POINT
# =============================================================================
def main():
    global LOG, TRANSFER_MODEL_NAME
    ensure_dirs(OUTPUT_ROOT, CACHE_CKPT, OOF_DIR, OUT_DIR, LOG_DIR)
    LOG = setup_logging()

    banner("PRETRAINED PGNN LEAKAGE EVALUATION")
    LOG.info("Device=%s | AMP=%s | Selection=%s | Smoke Test=%s",
             DEVICE, USE_AMP, SELECTION, SMOKE_TEST)

    if os.path.exists(HPO_PARAMS_PATH):
        with open(HPO_PARAMS_PATH) as f:
            j = json.load(f)
        hpo = {
            "lr": j.get("lr", DEFAULT_HPO["lr"]),
            "lambda_physics": j.get("lambda_physics", DEFAULT_HPO["lambda_physics"]),
            "weight_decay": j.get("weight_decay", DEFAULT_HPO["weight_decay"])
        }
        TRANSFER_MODEL_NAME = j.get("model_name", "resnet18")
        LOG.info("Loaded HPO params from %s; backbone=%s", HPO_PARAMS_PATH, TRANSFER_MODEL_NAME)
    else:
        hpo = dict(DEFAULT_HPO)
        LOG.warning("HPO JSON not found; using defaults: %s", hpo)

    raw_df = load_feature_dataframe()
    df, days_sorted = enrich(raw_df)

    sv_df = load_or_compute_sv_params(df)
    sv_dict = sv_df_to_dict(sv_df)

    run(df, sv_dict, days_sorted, hpo)
    banner("EVALUATION COMPLETE — See results in 'PGNN_leaked_vs_fixed_summary.csv'")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error("Fatal error during execution: %s", e)
        traceback.print_exc()
        sys.exit(1)