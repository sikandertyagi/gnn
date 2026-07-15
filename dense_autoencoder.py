"""
dense_autoencoder.py
────────────────────
Single-event dense (feedforward) autoencoder — December architecture.

Architecture (validated at ROC-AUC 0.9950 standalone)
──────────────────────────────────────────────────────
  Encoder: F → 128 → 64 → 32 → 16  (ReLU + LayerNorm)
  Decoder: 16 → 32 → 64 → 128 → F  (ReLU + LayerNorm, final sigmoid)

Key design choices matching December:
  · LayerNorm (not BatchNorm) — more stable for anomaly detection
  · Sigmoid output — matches MinMax-scaled [0, 1] inputs exactly
  · No dropout — December pipeline had none
  · Adam (not AdamW) with lr=5e-4

Training: MSE(output, input) on benign events only.
Scoring:  per-event MSE reconstruction error.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    DENSE_HIDDEN_DIMS, DENSE_EPOCHS, DENSE_BATCH_SIZE,
    DENSE_LR, DENSE_VAL_RATIO, DENSE_EARLY_STOPPING_PATIENCE,
    DENSE_LR_PATIENCE, DENSE_LR_FACTOR, DENSE_LR_MIN,
    RANDOM_SEED, TRAIN_LABEL,
)


class DenseAutoencoder(nn.Module):

    def __init__(self, feature_dim: int,
                 hidden_dims: tuple[int, ...] = DENSE_HIDDEN_DIMS):
        super().__init__()
        self.feature_dim = feature_dim

        # encoder: F → 128 → LN → 64 → LN → 32 → 16 (all ReLU)
        enc_layers = []
        in_dim = feature_dim
        for i, h in enumerate(hidden_dims):
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if i < len(hidden_dims) - 2:
                enc_layers.append(nn.LayerNorm(h))
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)

        # decoder: 16 → 32 → LN → 64 → LN → 128 → F (sigmoid)
        dec_layers = []
        rev = list(reversed(hidden_dims))
        in_dim = rev[0]
        for i, h in enumerate(list(rev[1:]) + [feature_dim]):
            dec_layers.append(nn.Linear(in_dim, h))
            if h == feature_dim:
                dec_layers.append(nn.Sigmoid())
            else:
                dec_layers.append(nn.ReLU())
                if i < len(rev) - 2:
                    dec_layers.append(nn.LayerNorm(h))
            in_dim = h
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


# ── training ─────────────────────────────────────────────────────────────────

def train_dense_ae(
    X_train: np.ndarray,
    feature_dim: int,
    epochs:     int   = DENSE_EPOCHS,
    batch_size: int   = DENSE_BATCH_SIZE,
    lr:         float = DENSE_LR,
    val_ratio:  float = DENSE_VAL_RATIO,
    patience:   int   = DENSE_EARLY_STOPPING_PATIENCE,
) -> DenseAutoencoder:
    """
    Train a DenseAutoencoder on benign event vectors.

    Parameters
    ----------
    X_train     : (N, F) float32 — benign events only, already scaled
    feature_dim : number of input features

    Returns
    -------
    model : trained DenseAutoencoder (best val-loss checkpoint)
    """
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(X_train))
    X_shuffled = X_train[perm]

    n_val = max(1, int(len(X_shuffled) * val_ratio)) if val_ratio > 0 else 0
    X_tr  = X_shuffled[n_val:]
    X_val = X_shuffled[:n_val] if n_val > 0 else None

    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    model = DenseAutoencoder(feature_dim).to(device)
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    _loader_kw = dict(
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda,
    )

    tr_ds     = TensorDataset(torch.from_numpy(X_tr).float())
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           generator=torch.Generator().manual_seed(RANDOM_SEED),
                           **_loader_kw)

    vl_loader = None
    if X_val is not None:
        vl_ds     = TensorDataset(torch.from_numpy(X_val).float())
        vl_loader = DataLoader(vl_ds, batch_size=batch_size * 4,
                               shuffle=False, **_loader_kw)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=DENSE_LR_FACTOR,
        patience=DENSE_LR_PATIENCE, min_lr=DENSE_LR_MIN,
    )
    criterion = nn.MSELoss()

    print(f"      Dense AE training on {device}  "
          f"({len(X_tr):,} train, {n_val:,} val)")
    print(f"      Architecture: {feature_dim} → "
          f"{' → '.join(str(h) for h in DENSE_HIDDEN_DIMS)} → "
          f"{feature_dim} (sigmoid)")

    best_loss  = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait       = 0

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for (x,) in tr_loader:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_cuda):
                loss = criterion(model(x), x)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            total += loss.item()
        tr_loss = total / len(tr_loader)

        if vl_loader is not None:
            model.eval()
            vl_total = 0.0
            with torch.no_grad():
                for (xv,) in vl_loader:
                    xv = xv.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_cuda):
                        vl_total += criterion(model(xv), xv).item()
            vl      = vl_total / len(vl_loader)
            monitor = vl
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"      Epoch {epoch+1:>3}/{epochs}  "
                      f"train={tr_loss:.6f}  val={vl:.6f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.1e}")
        else:
            monitor = tr_loss
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"      Epoch {epoch+1:>3}/{epochs}  "
                      f"loss={tr_loss:.6f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.1e}")

        scheduler.step(monitor)

        if monitor < best_loss - 1e-6:
            best_loss  = monitor
            best_state = copy.deepcopy(model.state_dict())
            wait       = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                print(f"      Early stopping at epoch {epoch+1}  "
                      f"(best={best_loss:.6f})")
                break

    print(f"      Dense AE training complete  best_loss={best_loss:.6f}")
    model.load_state_dict(best_state)
    return model


# ── inference ────────────────────────────────────────────────────────────────

def dense_anomaly_scores(
    model: DenseAutoencoder,
    X: np.ndarray,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Per-event MSE reconstruction error.

    Parameters
    ----------
    model      : trained DenseAutoencoder
    X          : (N, F) float32 — all events (benign + attack)
    batch_size : events per forward pass

    Returns
    -------
    scores : (N,) float32 — MSE reconstruction error per event
    """
    device   = next(model.parameters()).device
    use_cuda = device.type == "cuda"
    model.eval()

    scores = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            chunk = np.array(X[start:start + batch_size], dtype=np.float32)
            x     = torch.from_numpy(chunk).to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                recon = model(x)
            mse = torch.mean((x - recon.float()) ** 2, dim=1)
            scores.append(mse.cpu().numpy())

    return np.concatenate(scores).astype(np.float32)
