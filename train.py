"""
train.py
────────
Two training entry points:

  train_model(model, X_train, ...)
      In-memory path: X_train is a numpy ndarray already in RAM.
      Fine for datasets where sequences fit comfortably in memory.

  train_model_large(model, seq_path, seq_shape, train_indices, ...)
      Disk-backed path: reads sequences from a np.memmap file produced by
      sequence_builder.build_sequences_memmap().
      Only one batch is in RAM at a time — safe for millions of sequences.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset


# ── in-memory training ────────────────────────────────────────────────────────

def train_model(model, X_train: np.ndarray, epochs: int,
                batch_size: int, lr: float):
    device    = torch.device("cpu")
    model     = model.to(device)
    dataset   = TensorDataset(torch.from_numpy(X_train).float())
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:>3}/{epochs}  Loss {total_loss/len(loader):.4f}")

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
        self._indices = indices   # None → use all rows

    def __len__(self) -> int:
        return len(self._indices) if self._indices is not None else self._mm.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        real = int(self._indices[idx]) if self._indices is not None else idx
        # .copy() is required: PyTorch cannot own memmap-backed memory
        return torch.from_numpy(self._mm[real].copy())


def train_model_large(model, seq_path: str, seq_shape: tuple,
                      train_indices: np.ndarray,
                      epochs: int, batch_size: int, lr: float):
    """
    Train on a memmap sequence file using only the rows in *train_indices*.
    Peak RAM = one batch of sequences at a time.
    """
    device    = torch.device("cpu")
    model     = model.to(device)
    dataset   = MemmapDataset(seq_path, seq_shape, indices=train_indices)
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    n_train = len(train_indices)
    print(f"      Training on {n_train:,} benign sequences (disk-backed)")

    for epoch in range(epochs):
        total_loss = 0.0
        for x in loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:>3}/{epochs}  Loss {total_loss/len(loader):.4f}")

    return model
