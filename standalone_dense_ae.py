"""
standalone_dense_ae.py
──────────────────────
Standalone dense autoencoder pipeline — replicates the December run.

Differences from the composite pipeline:
  · Uses only handcrafted features (no cmd/chain embeddings → ~25 features)
  · Raw MSE reconstruction error (no percentile-clipped normalization layer)
  · p97.5 threshold on benign MSE scores directly
  · No composite blending — pure dense AE signal
  · All event types (no EventID filter)

Usage:
    python standalone_dense_ae.py
"""

import copy
import os
import random
import re
import warnings
import zlib
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH, DATA_DIR, ARTIFACTS_DIR,
    RANDOM_SEED, TRAIN_LABEL,
    DENSE_MODEL_PATH,
    ALERT_PERCENTILE, ALERT_WINDOW,
    HAS_GROUND_TRUTH,
    ANOMALY_REPORT_PATH, FLAGGED_EVENTS_PATH,
    SCORES_PATH, ALERTS_PATH,
)

# ── architecture ─────────────────────────────────────────────────────────────
HIDDEN_DIMS = (128, 64, 32)
EPOCHS      = 30
BATCH_SIZE  = 1024
LR          = 1e-3
VAL_RATIO   = 0.15
PATIENCE    = 5
DROPOUT     = 0.1

# ── RFC 1918 pattern ─────────────────────────────────────────────────────────
_RFC1918 = re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|127\.)")


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — handcrafted only, no embeddings
# ─────────────────────────────────────────────────────────────────────────────

def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = np.array(list(Counter(s).values()), dtype=np.float64)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def _crc32_series(series: pd.Series) -> pd.Series:
    return series.fillna("unknown").astype(str).apply(
        lambda x: (zlib.crc32(x.encode("utf-8")) & 0xFFFFFFFF) / 0xFFFFFFFF
    )


def handcrafted_features(df: pd.DataFrame):
    """Feature engineering with only handcrafted features — no embeddings."""
    df = df.copy()

    # ensure optional columns
    for col, default in [
        ("Image", "unknown"), ("ParentImage", "unknown"),
        ("CommandLine", ""), ("User", "unknown"),
        ("IntegrityLevel", "unknown"), ("Signed", "false"),
        ("Company", None), ("DestinationIp", None),
        ("DestinationPort", 0), ("EventID", 0), ("Label", 0),
    ]:
        if col not in df.columns:
            df[col] = default if default is not None else np.nan

    df["SystemTime"] = pd.to_datetime(df["SystemTime"], errors="coerce")

    # process names
    df["process_name"] = (
        df["Image"].fillna("unknown").astype(str)
        .str.split(r"[/\\]").str[-1].str.lower()
    )
    df["parent_process"] = (
        df["ParentImage"].fillna("unknown").astype(str)
        .str.split(r"[/\\]").str[-1].str.lower()
    )
    df["parent_child"] = df["parent_process"] + "->" + df["process_name"]

    # rare process score
    freq = df["process_name"].value_counts()
    df["rare_process_score"] = df["process_name"].map(lambda x: 1.0 / freq.get(x, 1))

    # CRC32 encode categoricals
    for col in ["process_name", "parent_process", "parent_child", "User", "IntegrityLevel"]:
        df[col] = _crc32_series(df[col].fillna("unknown").astype(str))

    # command-line features
    cmd = df["CommandLine"].fillna("").astype(str)
    df["cmd_length"] = cmd.str.len()
    df["cmd_token_count"] = cmd.str.split().str.len().fillna(0).astype(int)
    df["has_base64"] = cmd.str.contains(r"[A-Za-z0-9+/]{20,}={0,2}", regex=True, na=False).astype(int)
    df["has_http"] = cmd.str.contains("http", case=False, na=False).astype(int)
    df["has_ip"] = cmd.str.contains(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", regex=True, na=False).astype(int)
    df["has_download"] = cmd.str.contains(
        r"wget|curl|invoke-webrequest|\biwr\b|downloadstring|downloadfile"
        r"|bitsadmin|start-bitstransfer|fetch|aria2c|axel|nc\b|ncat\b|socat\b",
        case=False, regex=True, na=False,
    ).astype(int)
    df["has_encodedcommand"] = cmd.str.contains(
        r"-(?:enc|encodedcommand)\b|base64\s+-d|\beval\b.*\$\("
        r"|python[23]?\s+-c\s|perl\s+-e\s|ruby\s+-e\s",
        case=False, regex=True, na=False,
    ).astype(int)
    df["cmd_entropy"] = cmd.apply(_entropy)

    # path features
    img = df["Image"].fillna("").astype(str)
    df["path_depth"] = img.str.count(r"[/\\]")
    df["is_system_bin"] = img.str.contains(
        r"system32|/usr/bin/|/usr/sbin/|/bin/|/sbin/", case=False, na=False
    ).astype(int)
    df["is_users_dir"] = img.str.contains(
        r"[/\\]users[/\\]|/home/", case=False, na=False
    ).astype(int)
    df["is_temp_exec"] = img.str.contains(
        r"[/\\]temp[/\\]|/tmp/|/var/tmp/|/dev/shm/|appdata|downloads|programdata",
        case=False, na=False,
    ).astype(int)

    # binary / metadata
    df["is_signed"] = df["Signed"].fillna("false").astype(str).str.lower().eq("true").astype(int)
    df["missing_company"] = df["Company"].isna().astype(int)

    # network
    df["dest_port"] = pd.to_numeric(df["DestinationPort"], errors="coerce").fillna(0)
    df["dest_external"] = (
        ~df["DestinationIp"].fillna("").astype(str).apply(
            lambda ip: bool(_RFC1918.match(ip)) or ip == ""
        )
    ).astype(int)

    # temporal
    df["hour"] = df["SystemTime"].dt.hour.fillna(0).astype(int)
    df["is_after_hours"] = ((df["hour"] < 7) | (df["hour"] > 19)).astype(int)

    # event type
    df["eventid"] = pd.to_numeric(df["EventID"].fillna(0), errors="coerce").fillna(0).astype(int)

    # categorical cols (not z-scored)
    cat_cols = [
        "process_name", "parent_process", "parent_child",
        "User", "IntegrityLevel", "rare_process_score",
    ]
    # numeric cols (z-scored)
    num_cols = [
        "cmd_length", "cmd_token_count",
        "has_base64", "has_http", "has_ip", "has_download", "has_encodedcommand",
        "cmd_entropy",
        "path_depth", "is_system_bin", "is_users_dir", "is_temp_exec",
        "is_signed", "missing_company",
        "dest_port", "dest_external",
        "hour", "is_after_hours",
        "eventid",
    ]

    feature_cols = cat_cols + num_cols
    return df, feature_cols, len(cat_cols)


# ─────────────────────────────────────────────────────────────────────────────
# Dense autoencoder model
# ─────────────────────────────────────────────────────────────────────────────

class DenseAutoencoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_dims: tuple = HIDDEN_DIMS,
                 dropout: float = DROPOUT):
        super().__init__()
        # encoder
        enc = []
        d = feature_dim
        for h in hidden_dims:
            enc += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.encoder = nn.Sequential(*enc)
        # decoder (mirror)
        dec = []
        rev = list(reversed(hidden_dims))
        d = rev[0]
        for h in list(rev[1:]) + [feature_dim]:
            dec += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_ae(X_train, feature_dim):
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(X_train))
    X_shuffled = X_train[perm]

    n_val = max(1, int(len(X_shuffled) * VAL_RATIO))
    X_tr = X_shuffled[n_val:]
    X_val = X_shuffled[:n_val]

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = DenseAutoencoder(feature_dim).to(device)
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_cuda)

    _kw = dict(num_workers=4 if use_cuda else 0, pin_memory=use_cuda,
               persistent_workers=use_cuda)

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr).float()),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED), **_kw,
    )
    vl_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).float()),
        batch_size=BATCH_SIZE * 4, shuffle=False, **_kw,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5,
    )
    criterion = nn.MSELoss()

    print(f"  Architecture: {feature_dim} → {' → '.join(map(str, HIDDEN_DIMS))} "
          f"→ {' → '.join(map(str, reversed(HIDDEN_DIMS)))} → {feature_dim}")
    print(f"  Training on {device}  ({len(X_tr):,} train, {n_val:,} val)")

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for (x,) in tr_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_cuda):
                loss = criterion(model(x), x)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            total += loss.item()
        tr_loss = total / len(tr_loader)

        model.eval()
        vl_total = 0.0
        with torch.no_grad():
            for (xv,) in vl_loader:
                xv = xv.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_cuda):
                    vl_total += criterion(model(xv), xv).item()
        vl_loss = vl_total / len(vl_loader)

        scheduler.step(vl_loss)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1:>3}/{EPOCHS}  train={tr_loss:.6f}  "
              f"val={vl_loss:.6f}  lr={cur_lr:.1e}")

        if vl_loss < best_loss:
            best_loss = vl_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"  Early stopping at epoch {epoch+1} (best val={best_loss:.6f})")
                break

    model.load_state_dict(best_state)
    print(f"  Training complete — best val loss: {best_loss:.6f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference — raw MSE per event
# ─────────────────────────────────────────────────────────────────────────────

def score_events(model, X, batch_size=2048):
    device = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.from_numpy(X[i:i+batch_size].astype(np.float32)).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                recon = model(x)
            mse = torch.mean((x - recon.float()) ** 2, dim=1)
            scores.append(mse.cpu().numpy())
    return np.concatenate(scores).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_scores(labels, scores):
    """Compute ROC-AUC, PR-AUC, and best-F1 metrics."""
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve

    binary = (labels > 0).astype(int)

    roc_auc = roc_auc_score(binary, scores)

    prec_arr, rec_arr, pr_thresholds = precision_recall_curve(binary, scores)
    pr_auc = auc(rec_arr, prec_arr)

    fpr_arr, tpr_arr, roc_thresholds = roc_curve(binary, scores)

    # best F1 via threshold sweep
    f1_scores = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-12)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = pr_thresholds[best_idx] if best_idx < len(pr_thresholds) else pr_thresholds[-1]
    best_prec = prec_arr[best_idx]
    best_rec = rec_arr[best_idx]

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_f1": best_f1,
        "best_f1_threshold": best_thresh,
        "best_f1_precision": best_prec,
        "best_f1_recall": best_rec,
    }


def evaluate_at_threshold(labels, scores, threshold):
    """Precision/Recall/F1 at a specific threshold."""
    binary = (labels > 0).astype(int)
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (binary == 1)).sum())
    fp = int(((preds == 1) & (binary == 0)).sum())
    fn = int(((preds == 0) & (binary == 1)).sum())
    tn = int(((preds == 0) & (binary == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    _set_seeds(RANDOM_SEED)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # ── 1. load data ────────────────────────────────────────────────────────
    print("[1/5] Loading data...")
    df = pd.read_csv(DATA_PATH)
    if "Label" not in df.columns:
        df["Label"] = 0
    n_raw = len(df)

    # EventID filter: keep only high-signal event types
    FILTER_EVENTIDS = [1, 3]
    df["EventID"] = pd.to_numeric(df["EventID"], errors="coerce")
    df = df[df["EventID"].isin(FILTER_EVENTIDS)].reset_index(drop=True)
    print(f"  {n_raw:,} → {len(df):,} events (EventIDs {FILTER_EVENTIDS})")

    # ── 2. feature engineering (handcrafted only) ────────────────────────────
    print("\n[2/5] Feature engineering (handcrafted only, no embeddings)...")
    df, feature_cols, n_cat = handcrafted_features(df)
    print(f"  {len(feature_cols)} features ({n_cat} categorical + "
          f"{len(feature_cols) - n_cat} numeric)")

    # ── 3. normalize (StandardScaler on benign, numeric cols only) ───────────
    print("\n[3/5] Normalizing features...")
    num_cols = feature_cols[n_cat:]
    scaler = StandardScaler()
    benign_mask = df["Label"].values == TRAIN_LABEL
    scaler.fit(df.loc[benign_mask, num_cols].values.astype(np.float32))
    df[num_cols] = scaler.transform(df[num_cols].values.astype(np.float32))
    n_benign = int(benign_mask.sum())
    n_attack = int((df["Label"].values > 0).sum())
    print(f"  {n_benign:,} benign, {n_attack:,} positive (label > 0)")

    # ── 4. train dense AE ────────────────────────────────────────────────────
    print("\n[4/5] Training standalone dense autoencoder...")
    X_all = df[feature_cols].values.astype(np.float32)
    X_benign = X_all[benign_mask]
    model = train_ae(X_benign, feature_dim=len(feature_cols))

    save_path = os.path.join(ARTIFACTS_DIR, "standalone_dense_ae.pt")
    torch.save(model.state_dict(), save_path)
    print(f"  Saved → {save_path}")

    # ── 5. score & evaluate ──────────────────────────────────────────────────
    print("\n[5/5] Scoring & evaluation...")
    scores = score_events(model, X_all)
    labels = df["Label"].values

    # score summary
    for lbl, name in [(0, "Benign"), (1, "Attack"), (2, "Suspicious")]:
        mask = labels == lbl
        if mask.any():
            s = scores[mask]
            print(f"  {name:>10}: n={mask.sum():>8,}  "
                  f"mean={s.mean():.6f}  median={np.median(s):.6f}  "
                  f"p95={np.percentile(s, 95):.6f}  p99={np.percentile(s, 99):.6f}")

    # threshold: p97.5 of benign scores
    benign_scores = scores[benign_mask]
    threshold = float(np.percentile(benign_scores, ALERT_PERCENTILE))
    print(f"\n  Threshold (p{ALERT_PERCENTILE} benign): {threshold:.6f}")

    if labels.max() > 0:
        metrics = evaluate_scores(labels, scores)
        print(f"\n  ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC   : {metrics['pr_auc']:.4f}")
        print(f"  Best F1  : {metrics['best_f1']:.4f}  "
              f"@ threshold={metrics['best_f1_threshold']:.6f}  "
              f"P={metrics['best_f1_precision']:.4f}  "
              f"R={metrics['best_f1_recall']:.4f}")

        # evaluate at percentile threshold
        pct = evaluate_at_threshold(labels, scores, threshold)
        print(f"\n  @ p{ALERT_PERCENTILE} threshold ({threshold:.6f}):")
        print(f"    Precision: {pct['precision']:.4f}")
        print(f"    Recall   : {pct['recall']:.4f}")
        print(f"    F1       : {pct['f1']:.4f}")
        print(f"    TP={pct['tp']:,}  FP={pct['fp']:,}  "
              f"FN={pct['fn']:,}  TN={pct['tn']:,}")

        # evaluate at best-F1 threshold
        bf1 = evaluate_at_threshold(labels, scores, metrics["best_f1_threshold"])
        print(f"\n  @ best-F1 threshold ({metrics['best_f1_threshold']:.6f}):")
        print(f"    Precision: {bf1['precision']:.4f}")
        print(f"    Recall   : {bf1['recall']:.4f}")
        print(f"    F1       : {bf1['f1']:.4f}")
        print(f"    TP={bf1['tp']:,}  FP={bf1['fp']:,}  "
              f"FN={bf1['fn']:,}  TN={bf1['tn']:,}")

    # save scores
    out_path = os.path.join(ARTIFACTS_DIR, "standalone_scores.csv")
    pd.DataFrame({
        "label": labels,
        "dense_error": scores,
    }).to_csv(out_path, index=False)
    print(f"\n  Scores saved → {out_path}")


if __name__ == "__main__":
    main()
