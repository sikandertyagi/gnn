"""
december_replicate.py
─────────────────────
Exact replication of the December dense autoencoder pipeline that achieved:
    Precision: 98.76%  Recall: 96.10%  F1: 0.9741  ROC-AUC: 0.9947

Replicates preprocessing.py → autoencoder.py from the original codebase:
  1. Same 15 input features (metadata only — no CommandLine, no Image paths)
  2. Same OneHotEncoder for categoricals
  3. Same MinMaxScaler for numericals
  4. Same architecture: 128 → 64 → 32 → 16 → 32 → 64 → 128 with LayerNorm
  5. Same sigmoid output activation
  6. Same p97.5 threshold on training reconstruction errors
  7. All event types (no EventID filter)
  8. Binary labels: label > 0 → malicious

Ported from TensorFlow/Keras to PyTorch for consistency with the rest of
the codebase.  The architecture, preprocessing, and evaluation logic are
kept identical to the original.

Usage:
    python december_replicate.py
    python december_replicate.py --data /path/to/sysmondataless.csv
    python december_replicate.py --data /path/to/elastic_data.csv
"""

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, precision_score, recall_score, f1_score,
)

from config import DATA_PATH, ARTIFACTS_DIR, RANDOM_SEED

# ── config (matching December exactly) ───────────────────────────────────────
BOTTLENECK_DIM = 16
EPOCHS         = 100
BATCH_SIZE     = 256
LR             = 5e-4
PATIENCE       = 15
LR_PATIENCE    = 8
LR_FACTOR      = 0.5
LR_MIN         = 1e-6
TEST_SIZE      = 0.15
THRESHOLD_PCT  = 97.5

SAVE_DIR = os.path.join(ARTIFACTS_DIR, "december_replicate")

# ── features (same 15 as December preprocessing.py) ──────────────────────────
RAW_FEATURES = [
    'Computer', 'DestinationPortName', 'EventID', 'EventRecordID',
    'Execution_ProcessID', 'Initiated', 'ProcessId', 'SourceIsIpv6',
    'SystemTime_year', 'SystemTime_month', 'SystemTime_week',
    'SystemTime_day', 'SystemTime_hour', 'SystemTime_minute',
    'SystemTime_day_of_week',
]

CATEGORICAL_COLS = [
    'Computer', 'DestinationPortName', 'EventID', 'Initiated',
    'SourceIsIpv6', 'SystemTime_year', 'SystemTime_month',
    'SystemTime_week', 'SystemTime_day_of_week',
]

NUMERICAL_COLS = [
    'EventRecordID', 'Execution_ProcessID', 'ProcessId',
    'SystemTime_day', 'SystemTime_hour', 'SystemTime_minute',
]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — exact replica of December preprocessing.py
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """Replicate December preprocessing: time features → OHE + MinMax.

    Returns
    -------
    X_processed : np.ndarray (N, F) float32  — all features in [0, 1]
    y           : np.ndarray (N,)   int       — binary labels (0/1)
    ohe         : fitted OneHotEncoder
    scaler      : fitted MinMaxScaler
    """
    print("\n── Preprocessing (December pipeline) ──")

    # binary labels: 0 = benign, >0 = malicious
    if 'Label' not in df.columns:
        raise ValueError("No 'Label' column found")
    y = (df['Label'].apply(lambda x: 0 if x == 0 else 1)).values

    # time features from SystemTime
    if 'SystemTime' in df.columns:
        ts = pd.to_datetime(df['SystemTime'], errors='coerce')
        df['SystemTime_year'] = ts.dt.year.fillna(0).astype(int)
        df['SystemTime_month'] = ts.dt.month.fillna(0).astype(int)
        df['SystemTime_week'] = ts.dt.isocalendar().week.fillna(0).astype(int)
        df['SystemTime_day'] = ts.dt.day.fillna(0).astype(int)
        df['SystemTime_hour'] = ts.dt.hour.fillna(0).astype(int)
        df['SystemTime_minute'] = ts.dt.minute.fillna(0).astype(int)
        df['SystemTime_day_of_week'] = ts.dt.dayofweek.fillna(0).astype(int)
        print("  Extracted 7 time features from SystemTime")
    else:
        for col in ['SystemTime_year', 'SystemTime_month', 'SystemTime_week',
                     'SystemTime_day', 'SystemTime_hour', 'SystemTime_minute',
                     'SystemTime_day_of_week']:
            if col not in df.columns:
                df[col] = 0

    # check available features
    available = [f for f in RAW_FEATURES if f in df.columns]
    missing = [f for f in RAW_FEATURES if f not in df.columns]
    if missing:
        print(f"  WARNING: missing features (will be zero-filled): {missing}")
        for col in missing:
            df[col] = 0

    X = df[available + [c for c in missing]].copy()

    # clean categoricals
    cat_cols = [c for c in CATEGORICAL_COLS if c in X.columns]
    num_cols = [c for c in NUMERICAL_COLS if c in X.columns]

    for col in cat_cols:
        X[col] = X[col].astype(str).replace(
            ['nan', 'NaN', 'NULL', 'null', 'None', '-'], 'Unknown'
        ).fillna('Unknown')

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    print(f"  Categorical: {len(cat_cols)} cols → OneHotEncoding")
    print(f"  Numerical:   {len(num_cols)} cols → MinMaxScaler")

    # OneHot encode categoricals
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = ohe.fit_transform(X[cat_cols])
    print(f"  After OHE: {X_cat.shape[1]} features")

    # MinMax scale numericals
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(X[num_cols].values)

    # combine
    X_processed = np.hstack([X_cat, X_num]).astype(np.float32)
    print(f"  Total features: {X_processed.shape[1]}")
    print(f"  Benign: {(y == 0).sum():,}  Malicious: {(y == 1).sum():,}")

    return X_processed, y, ohe, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Model — exact replica of December autoencoder.py
# ─────────────────────────────────────────────────────────────────────────────

class DecemberAutoencoder(nn.Module):
    """128 → 64 → 32 → 16 → 32 → 64 → 128 with LayerNorm + sigmoid output.

    Matches the December TF/Keras EnhancedAutoencoder exactly:
      - Encoder: Dense(128,relu) → LN → Dense(64,relu) → LN → Dense(32,relu) → Dense(16,relu)
      - Decoder: Dense(32,relu) → LN → Dense(64,relu) → LN → Dense(128,relu) → Dense(F,sigmoid)
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = BOTTLENECK_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64), nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, bottleneck_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 32), nn.ReLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 64), nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                lr=LR, patience=PATIENCE):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    loader_kw = dict(
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda,
    )

    train_ds = TensorDataset(torch.from_numpy(X_train).float())
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED), **loader_kw,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_FACTOR, patience=LR_PATIENCE,
        min_lr=LR_MIN,
    )
    criterion = nn.MSELoss()
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_cuda)

    print(f"\n  Training on {device}  ({len(X_train):,} samples)")
    print(f"  Architecture: {X_train.shape[1]} → 128 → 64 → 32 → {BOTTLENECK_DIM} "
          f"→ 32 → 64 → 128 → {X_train.shape[1]}")

    best_loss = float('inf')
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for (x,) in train_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_cuda):
                loss = criterion(model(x), x)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            total_loss += loss.item()
            n_batches += 1

        epoch_loss = total_loss / n_batches
        scheduler.step(epoch_loss)
        cur_lr = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>3}/{epochs}  loss={epoch_loss:.6f}  lr={cur_lr:.1e}")

        if epoch_loss < best_loss - 1e-4:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch+1} (best={best_loss:.6f})")
                break

    model.load_state_dict(best_state)
    print(f"  Training complete — best loss: {best_loss:.6f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Reconstruction error
# ─────────────────────────────────────────────────────────────────────────────

def reconstruction_error(model, X, batch_size=1024):
    """Per-sample MSE reconstruction error — matches December exactly."""
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    model.eval()
    errors = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i+batch_size].astype(np.float32)).to(
                device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                recon = model(x)
            mse = torch.mean((x - recon.float()) ** 2, dim=1)
            errors.append(mse.cpu().numpy())
    return np.concatenate(errors)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="December pipeline replication")
    p.add_argument("--data", type=str, default=DATA_PATH,
                   help="Path to raw Sysmon CSV")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 70)
    print("DECEMBER PIPELINE REPLICATION")
    print("Dense Autoencoder — OneHot + MinMax — All Events")
    print("=" * 70)

    # ── 1. load data ─────────────────────────────────────────────────────────
    print(f"\n[1/5] Loading data from {args.data}...")
    t0 = time.time()
    df = pd.read_csv(args.data, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns  ({time.time()-t0:.1f}s)")

    # ── 2. preprocess ────────────────────────────────────────────────────────
    print("\n[2/5] Preprocessing...")
    X_processed, y, ohe, scaler = preprocess(df)

    # separate normal / malicious (same as December)
    X_normal = X_processed[y == 0]
    X_malicious = X_processed[y == 1]

    # train/test split on normal only (same as December: test_size=0.15)
    X_train_normal, X_test_normal = train_test_split(
        X_normal, test_size=TEST_SIZE, random_state=RANDOM_SEED,
    )

    print(f"\n  Train normal:   {len(X_train_normal):,}")
    print(f"  Test normal:    {len(X_test_normal):,}")
    print(f"  Test malicious: {len(X_malicious):,}")

    # ── 3. train ─────────────────────────────────────────────────────────────
    print(f"\n[3/5] Training autoencoder...")
    input_dim = X_processed.shape[1]
    model = DecemberAutoencoder(input_dim)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    model = train_model(
        model, X_train_normal,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
    )

    # save model
    model_path = os.path.join(SAVE_DIR, "december_autoencoder.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  Saved → {model_path}")

    # ── 4. score ─────────────────────────────────────────────────────────────
    print(f"\n[4/5] Computing reconstruction errors...")
    train_err = reconstruction_error(model, X_train_normal)
    test_normal_err = reconstruction_error(model, X_test_normal)
    test_malicious_err = reconstruction_error(model, X_malicious)

    print(f"  Train normal     — mean={train_err.mean():.6f}  "
          f"p97.5={np.percentile(train_err, 97.5):.6f}")
    print(f"  Test normal      — mean={test_normal_err.mean():.6f}  "
          f"p97.5={np.percentile(test_normal_err, 97.5):.6f}")
    print(f"  Test malicious   — mean={test_malicious_err.mean():.6f}  "
          f"median={np.median(test_malicious_err):.6f}")

    # ── 5. evaluate (same as December) ───────────────────────────────────────
    print(f"\n[5/5] Evaluation...")

    # threshold: p97.5 of TRAINING errors (same as December)
    threshold = float(np.percentile(train_err, THRESHOLD_PCT))
    print(f"  Threshold (p{THRESHOLD_PCT} train): {threshold:.6f}")

    # combine test normal + test malicious (same as December)
    y_true = np.concatenate([
        np.zeros_like(test_normal_err),
        np.ones_like(test_malicious_err),
    ])
    y_scores = np.concatenate([test_normal_err, test_malicious_err])
    y_pred = (y_scores > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "Threshold": float(threshold),
        "Precision": float(precision_score(y_true, y_pred)),
        "Recall": float(recall_score(y_true, y_pred)),
        "F1-score": float(f1_score(y_true, y_pred)),
        "ROC-AUC": float(roc_auc_score(y_true, y_scores)),
        "FPR": float(fp / (fp + tn)),
        "TPR": float(tp / (tp + fn)),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }

    print(f"\n  {'='*50}")
    print(f"  RESULTS")
    print(f"  {'='*50}")
    print(f"  Threshold : {metrics['Threshold']:.6f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  F1-score  : {metrics['F1-score']:.4f}")
    print(f"  ROC-AUC   : {metrics['ROC-AUC']:.4f}")
    print(f"  FPR       : {metrics['FPR']:.4f}")
    print(f"  TPR       : {metrics['TPR']:.4f}")
    print(f"  TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}")

    # PR-AUC (extra metric not in December)
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(rec_arr, prec_arr)
    metrics["PR-AUC"] = float(pr_auc)
    print(f"  PR-AUC    : {pr_auc:.4f}")

    # save metrics
    metrics_path = os.path.join(SAVE_DIR, "metrics_summary.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n  Metrics saved → {metrics_path}")

    # save scores for further analysis
    scores_path = os.path.join(SAVE_DIR, "scores.csv")
    pd.DataFrame({
        "label": y_true.astype(int),
        "recon_error": y_scores,
        "predicted": y_pred,
    }).to_csv(scores_path, index=False)
    print(f"  Scores saved → {scores_path}")

    # save preprocessor objects
    import joblib
    joblib.dump(ohe, os.path.join(SAVE_DIR, "ohe_encoder.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "minmax_scaler.pkl"))
    print(f"  Preprocessors saved → {SAVE_DIR}/")

    # comparison with December target
    print(f"\n  {'='*50}")
    print(f"  COMPARISON WITH DECEMBER TARGET")
    print(f"  {'='*50}")
    dec = {"Precision": 0.9876, "Recall": 0.9610, "F1-score": 0.9741, "ROC-AUC": 0.9947}
    for key in ["Precision", "Recall", "F1-score", "ROC-AUC"]:
        curr = metrics[key]
        tgt = dec[key]
        diff = curr - tgt
        arrow = "▲" if diff >= 0 else "▼"
        print(f"  {key:>10}: {curr:.4f}  (target {tgt:.4f}  {arrow} {abs(diff):.4f})")

    print(f"\n{'='*70}")
    print("Done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
