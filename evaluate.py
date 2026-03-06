"""
evaluate.py
───────────
Batched reconstruction-error inference.

anomaly_scores(model, X, batch_size)
  X can be:
    · np.ndarray  – regular in-memory array
    · np.memmap   – disk-backed array from sequence_builder

One batch is moved to the model device at a time, so peak RAM is
O(batch_size × seq_len × feature_dim) regardless of total dataset size.
"""

import numpy as np
import torch


def anomaly_scores(model, X, batch_size: int = 512) -> np.ndarray:
    """
    Parameters
    ----------
    model      : TransformerAutoencoder (or any model with forward(x) → x)
    X          : (N, seq_len, feature_dim) ndarray or memmap
    batch_size : sequences processed per forward pass

    Returns
    -------
    scores : (N,) float32 ndarray  — MSE reconstruction error per sequence
    """
    device = next(model.parameters()).device
    model.eval()

    total   = len(X)
    n_batch = (total + batch_size - 1) // batch_size
    scores  = []
    with torch.no_grad():
        for i, start in enumerate(range(0, total, batch_size)):
            # np.array() materialises memmap slices into a contiguous buffer
            chunk = np.array(X[start : start + batch_size], dtype=np.float32)
            x     = torch.from_numpy(chunk).to(device)
            recon = model(x)
            # MSE averaged over time and feature dimensions → one scalar per seq
            mse   = torch.mean((x - recon) ** 2, dim=(1, 2))
            scores.append(mse.cpu().numpy())
            if (i + 1) % 100 == 0 or (i + 1) == n_batch:
                print(f"\r      Inference: {i+1}/{n_batch} batches "
                      f"({100*(i+1)/n_batch:.1f}%)  ", end="", flush=True)
    print()

    return np.concatenate(scores).astype(np.float32)
