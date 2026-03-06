"""
sequence_builder.py
───────────────────
Two modes:

  Small datasets (< LARGE_DATASET_THRESHOLD events)
    build_sequences(df, feature_cols, seq_len)
      → returns (X: ndarray, y: ndarray) fully in RAM

  Large datasets
    build_sequences_memmap(df, feature_cols, seq_len, seq_path, labels_path)
      → streams sequences host-by-host onto disk using np.memmap
      → returns (n_sequences, seq_shape) without loading data into RAM

The memmap approach avoids the OOM risk that arises when storing millions of
overlapping windows as a single numpy array in memory.

Sliding windows are constructed with numpy stride tricks (zero-copy view) so
peak per-host RAM = one host's event matrix, not all windows simultaneously.
"""

import numpy as np
import pandas as pd


# ── in-memory (small datasets) ────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int):
    """
    Returns
    -------
    X : ndarray (N_seq, seq_len, n_features)  float32
    y : ndarray (N_seq,)                       int
    """
    sequences = []
    labels    = []

    df = df.sort_values("SystemTime")

    for host in df["Computer"].unique():
        host_df = df[df["Computer"] == host]
        values  = host_df[feature_cols].values.astype(np.float32)
        labs    = host_df["Label"].values

        n = len(values) - seq_len + 1  # sliding_window_view yields exactly M-seq_len+1 windows
        if n <= 0:
            continue

        # stride-trick view: no copy until np.array() call below
        windows = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
        # shape: (n, n_features, seq_len) → transpose to (n, seq_len, n_features)
        sequences.append(windows[:n].transpose(0, 2, 1).copy())
        labels.append(labs[seq_len - 1 : seq_len - 1 + n])

    X = np.concatenate(sequences, axis=0)
    y = np.concatenate(labels,    axis=0).astype(np.int32)
    return X, y


# ── disk-backed (large datasets) ──────────────────────────────────────────────

def build_sequences_memmap(
    df:          pd.DataFrame,
    feature_cols: list,
    seq_len:     int,
    seq_path:    str,
    labels_path: str,
) -> tuple:
    """
    Write all sequences and labels to memmap files without holding more than
    one host's data in RAM at a time.

    Returns
    -------
    n_sequences : int
    seq_shape   : tuple  (n_sequences, seq_len, n_features)
    """
    df         = df.sort_values("SystemTime")
    n_features = len(feature_cols)
    hosts      = df["Computer"].unique()

    # ── pass 1: count total sequences (O(n_hosts), free) ─────────────────────
    n_sequences = 0
    for host in hosts:
        host_len = int((df["Computer"] == host).sum())
        n_sequences += max(0, host_len - seq_len + 1)

    if n_sequences == 0:
        return 0, (0, seq_len, n_features)

    seq_shape = (n_sequences, seq_len, n_features)

    # ── allocate memmap files ─────────────────────────────────────────────────
    seq_mm = np.memmap(seq_path,    dtype=np.float32, mode="w+", shape=seq_shape)
    lbl_mm = np.memmap(labels_path, dtype=np.int32,   mode="w+", shape=(n_sequences,))

    # ── pass 2: write host-by-host ────────────────────────────────────────────
    offset = 0
    for host in hosts:
        host_df = df[df["Computer"] == host]
        values  = host_df[feature_cols].values.astype(np.float32)  # (M, F)
        labs    = host_df["Label"].values.astype(np.int32)

        n = len(values) - seq_len + 1  # sliding_window_view yields exactly M-seq_len+1 windows
        if n <= 0:
            continue

        # stride-trick view — no RAM copy until the assignment to seq_mm
        windows = np.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)
        # shape: (n, n_features, seq_len) → (n, seq_len, n_features)
        seq_mm[offset : offset + n] = windows[:n].transpose(0, 2, 1)
        lbl_mm[offset : offset + n] = labs[seq_len - 1 : seq_len - 1 + n]

        offset += n

    # flush to disk before returning
    seq_mm.flush()
    lbl_mm.flush()
    del seq_mm, lbl_mm

    return n_sequences, seq_shape


def load_seq_labels(labels_path: str, n_sequences: int) -> np.ndarray:
    """Re-open a labels memmap produced by build_sequences_memmap."""
    return np.memmap(labels_path, dtype=np.int32, mode="r", shape=(n_sequences,))


def load_seq_memmap(seq_path: str, seq_shape: tuple) -> np.memmap:
    """Re-open a sequences memmap produced by build_sequences_memmap."""
    return np.memmap(seq_path, dtype=np.float32, mode="r", shape=seq_shape)
