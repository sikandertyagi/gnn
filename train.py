"""
train.py
────────
Training loop for the Transformer autoencoder.

Improvements over the original implementation
──────────────────────────────────────────────
  · AdamW with weight_decay=1e-4 instead of plain Adam — regularises the
    large transformer weight matrices.
  · Linear LR warmup (10% of steps) → cosine annealing decay — prevents
    unstable updates in the first epoch and improves final convergence.
  · Gradient clipping (max_norm=1.0) — prevents exploding gradients that
    are common in deep transformers without warmup.
  · Early stopping with patience — avoids overfitting when training on a
    relatively small benign subset.
  · Fixed random seed for reproducibility.
  · Best-model checkpoint: restores the lowest-validation-loss weights.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import RANDOM_SEED

_WARMUP_FRACTION = 0.1
_WEIGHT_DECAY    = 1e-4


def _build_scheduler(optimizer, total_steps: int):
    """Linear warmup for the first 10% of steps, then cosine decay to 0."""
    warmup_steps = max(1, int(total_steps * _WARMUP_FRACTION))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_model(
    model,
    X_train:    np.ndarray,
    epochs:     int,
    batch_size: int,
    lr:         float,
    val_ratio:  float = 0.2,
    patience:   int   = 5,
) -> nn.Module:
    """
    Parameters
    ----------
    X_train   : (N, seq_len, features) float32 ndarray — benign sequences
    val_ratio : fraction of X_train held out for validation
    patience  : early-stopping patience in epochs (0 to disable)
    """
    # Reproducible shuffle
    rng  = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(X_train))
    X_shuffled = X_train[perm]

    n_val   = max(1, int(len(X_shuffled) * val_ratio)) if val_ratio > 0 else 0
    X_tr    = X_shuffled[n_val:]
    X_val   = X_shuffled[:n_val] if n_val > 0 else None

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)

    tr_ds     = TensorDataset(torch.tensor(X_tr).float())
    tr_loader = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    optimizer   = torch.optim.AdamW(model.parameters(), lr=lr,
                                     weight_decay=_WEIGHT_DECAY)
    total_steps = epochs * len(tr_loader)
    scheduler   = _build_scheduler(optimizer, total_steps)
    criterion   = nn.MSELoss()

    best_loss  = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait       = 0

    for epoch in range(epochs):

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for (x,) in tr_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        train_loss /= len(tr_loader)

        # ── validate ──────────────────────────────────────────────────────────
        if X_val is not None:
            model.eval()
            val_ds     = TensorDataset(torch.tensor(X_val).float())
            val_loader = DataLoader(val_ds, batch_size=batch_size * 4,
                                    shuffle=False, num_workers=0)
            val_loss = 0.0
            with torch.no_grad():
                for (xv,) in val_loader:
                    xv = xv.to(device)
                    val_loss += criterion(model(xv), xv).item()
            val_loss /= max(len(val_loader), 1)
            monitor = val_loss
            print(
                f"Epoch {epoch+1:>3}/{epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )
        else:
            monitor = train_loss
            print(
                f"Epoch {epoch+1:>3}/{epochs}  "
                f"loss={train_loss:.4f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        # ── early stopping ────────────────────────────────────────────────────
        if monitor < best_loss:
            best_loss  = monitor
            best_state = copy.deepcopy(model.state_dict())
            wait       = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                print(f"  Early stopping at epoch {epoch+1}  (best={best_loss:.4f})")
                break

    print(f"  Training complete  best_loss={best_loss:.4f}")
    model.load_state_dict(best_state)
    return model
