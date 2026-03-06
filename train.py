"""
train.py
────────
Two training entry points:

  train_model(model, X_train, ...)
      In-memory path: X_train is a numpy ndarray already in RAM.
      Includes train/val split and early stopping.

  train_model_large(model, seq_path, seq_shape, train_indices, ...)
      Disk-backed path: reads sequences from a np.memmap file produced by
      sequence_builder.build_sequences_memmap().
      Only one batch is in RAM at a time.  Includes val split and early stopping.

Both functions
  · shuffle with a fixed random seed for reproducibility
  · monitor val loss (if val_ratio > 0) or train loss otherwise
  · restore the best-seen model weights before returning
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

from config import RANDOM_SEED


def _build_scheduler(optimizer, n_epochs: int, n_batches: int,
                     warmup_fraction: float = 0.1):
    """
    Linear warmup (warmup_fraction of total steps) followed by cosine
    annealing to zero.  Returned scheduler is stepped once per batch.
    """
    total_steps  = n_epochs * n_batches
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── in-memory training ────────────────────────────────────────────────────────

def train_model(
    model,
    X_train:    np.ndarray,
    epochs:     int,
    batch_size: int,
    lr:         float,
    val_ratio:  float = 0.0,
    patience:   int   = 5,
) -> nn.Module:
    """
    Parameters
    ----------
    X_train   : (N, seq_len, features) float32 ndarray – benign sequences
    val_ratio : fraction held out for validation (0 → train loss monitored)
    patience  : epochs without improvement before early stopping (0 → disabled)
    """
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(X_train))
    X_shuffled = X_train[perm]

    n_val  = max(1, int(len(X_shuffled) * val_ratio)) if val_ratio > 0 else 0
    X_tr   = X_shuffled[n_val:]
    X_val  = X_shuffled[:n_val] if n_val > 0 else None

    device    = torch.device("cpu")
    model     = model.to(device)
    tr_ds     = TensorDataset(torch.from_numpy(X_tr).float())
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=False,
                           generator=torch.Generator().manual_seed(RANDOM_SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = _build_scheduler(optimizer, epochs, len(tr_loader))
    criterion = nn.MSELoss()

    best_loss  = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait       = 0

    for epoch in range(epochs):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        total = 0.0
        for (x,) in tr_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            # Fix #10: gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        tr_loss = total / len(tr_loader)

        # ── validate ──────────────────────────────────────────────────────────
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                xv  = torch.from_numpy(X_val).float().to(device)
                vl  = criterion(model(xv), xv).item()
            monitor = vl
            print(f"Epoch {epoch+1:>3}/{epochs}  train={tr_loss:.4f}  val={vl:.4f}")
        else:
            monitor = tr_loss
            print(f"Epoch {epoch+1:>3}/{epochs}  loss={tr_loss:.4f}")

        # ── early stopping ────────────────────────────────────────────────────
        if monitor < best_loss:
            best_loss  = monitor
            best_state = copy.deepcopy(model.state_dict())
            wait       = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                print(f"      Early stopping at epoch {epoch+1}  "
                      f"(best={best_loss:.4f})")
                break

    # Fix #12: log final training metrics
    print(f"      Transformer training complete  best_loss={best_loss:.4f}")
    model.load_state_dict(best_state)
    return model


# ── disk-backed training (large datasets) ────────────────────────────────────

class MemmapDataset(Dataset):
    """
    PyTorch Dataset backed by a read-only np.memmap.
    An optional index array lets you select a subset (e.g. benign-only)
    without copying the full file into RAM.
    """

    def __init__(self, seq_path: str, seq_shape: tuple,
                 indices: np.ndarray | None = None):
        self._mm      = np.memmap(seq_path, dtype=np.float32,
                                  mode="r", shape=seq_shape)
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices) if self._indices is not None else self._mm.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        real = int(self._indices[idx]) if self._indices is not None else idx
        # .copy() required: PyTorch cannot own memmap-backed memory
        return torch.from_numpy(self._mm[real].copy())


def _val_loss_memmap(model, loader, criterion, device) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            total += criterion(model(x), x).item()
    return total / max(len(loader), 1)


def train_model_large(
    model,
    seq_path:      str,
    seq_shape:     tuple,
    train_indices: np.ndarray,
    epochs:        int,
    batch_size:    int,
    lr:            float,
    val_ratio:     float = 0.0,
    patience:      int   = 5,
) -> nn.Module:
    """
    Train on a memmap sequence file using only the rows in *train_indices*.
    Peak RAM = one batch of sequences at a time.

    Parameters
    ----------
    val_ratio : fraction of train_indices held out for validation
    patience  : early-stopping patience (0 → disabled)
    """
    rng  = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(train_indices))
    idx  = train_indices[perm]

    n_val  = max(1, int(len(idx) * val_ratio)) if val_ratio > 0 else 0
    tr_idx = idx[n_val:]
    vl_idx = idx[:n_val] if n_val > 0 else None

    device    = torch.device("cpu")
    model     = model.to(device)
    tr_ds     = MemmapDataset(seq_path, seq_shape, tr_idx)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=False,
                           generator=torch.Generator().manual_seed(RANDOM_SEED))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = _build_scheduler(optimizer, epochs, len(tr_loader))
    criterion = nn.MSELoss()

    vl_loader = None
    if vl_idx is not None:
        vl_ds     = MemmapDataset(seq_path, seq_shape, vl_idx)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size * 4,
                               shuffle=False, num_workers=0)

    n_train = len(tr_idx)
    print(f"      Training on {n_train:,} benign sequences "
          f"({'disk-backed + val split' if vl_loader else 'disk-backed'})")

    best_loss  = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait       = 0

    for epoch in range(epochs):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        total = 0.0
        for x in tr_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            # Fix #10: gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
        tr_loss = total / len(tr_loader)

        # ── validate ──────────────────────────────────────────────────────────
        if vl_loader is not None:
            vl      = _val_loss_memmap(model, vl_loader, criterion, device)
            monitor = vl
            print(f"Epoch {epoch+1:>3}/{epochs}  train={tr_loss:.4f}  val={vl:.4f}")
        else:
            monitor = tr_loss
            print(f"Epoch {epoch+1:>3}/{epochs}  loss={tr_loss:.4f}")

        # ── early stopping ────────────────────────────────────────────────────
        if monitor < best_loss:
            best_loss  = monitor
            best_state = copy.deepcopy(model.state_dict())
            wait       = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                print(f"      Early stopping at epoch {epoch+1}  "
                      f"(best={best_loss:.4f})")
                break

    # Fix #12: log final training metrics
    print(f"      Transformer (large) training complete  best_loss={best_loss:.4f}")
    model.load_state_dict(best_state)
    return model
