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

Temporal features
─────────────────
Both build functions prepend a call to _add_temporal_features() which adds
three host-relative time features before windowing:

  time_delta_seconds  seconds elapsed since the previous event on the same host
  log_time_delta      log1p(time_delta_seconds) — compresses long idle gaps
  event_burst_count   number of events on the same host in the preceding 60 s
                      (O(n log n) per host via numpy searchsorted)

All three are MinMax-scaled on benign rows only (mirroring the MinMaxScaler
convention in normaliser.py) so they enter the transformer on the same
[0, 1] scale as the rest of the feature vector.
"""

import numpy as np
import pandas as pd

from config import TRAIN_LABEL

# Names of the three temporal columns injected before sequence construction.
_TEMPORAL_COLS = ["time_delta_seconds", "log_time_delta", "event_burst_count"]


# ── temporal feature helpers ──────────────────────────────────────────────────

def _burst_count(host_df: pd.DataFrame) -> np.ndarray:
    """
    For each event in a per-host DataFrame (sorted ascending by SystemTime)
    count the number of events on that host in the 60-second window that
    PRECEDES the event (i.e. the event itself is not counted).

    Uses numpy searchsorted → O(n log n), no DatetimeIndex required.
    """
    t_raw    = host_df["SystemTime"].values
    nat_mask = pd.isnull(t_raw)
    t        = t_raw.astype("datetime64[ns]").copy()
    if nat_mask.any():
        t[nat_mask] = np.datetime64(0, "ns")

    t_ns      = t.view(np.int64)
    window_ns = np.int64(60_000_000_000)   # 60 s in nanoseconds

    # events before t  −  events before (t − 60 s)  =  events in [t−60s, t)
    counts = (
        np.searchsorted(t_ns, t_ns,            side="left")
        - np.searchsorted(t_ns, t_ns - window_ns, side="left")
    )
    return counts.astype(np.float32)


def _add_temporal_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    """
    Add time_delta_seconds, log_time_delta, and event_burst_count to *df*
    and append them to *feature_cols*.

    All three are MinMax-scaled on benign rows only (TRAIN_LABEL) to mirror
    the MinMaxScaler convention used for the rest of the feature vector.

    Parameters
    ----------
    df           : event DataFrame (must contain SystemTime, Computer, Label)
    feature_cols : existing list of feature column names

    Returns
    -------
    df           : augmented copy, sorted by [Computer, SystemTime]
    feature_cols : extended list including the three temporal column names
    """
    df = df.copy()
    # Per-host sort so that diff() and searchsorted produce correct deltas.
    # build_sequences / build_sequences_memmap will re-sort by SystemTime
    # globally afterwards; the feature values per row remain correct.
    df = df.sort_values(["Computer", "SystemTime"]).reset_index(drop=True)

    # ── time_delta_seconds ────────────────────────────────────────────────────
    df["time_delta_seconds"] = (
        df.groupby("Computer")["SystemTime"]
        .diff()
        .dt.total_seconds()
        .fillna(0.0)
        .clip(lower=0.0)
        .astype(np.float32)
    )

    # ── log_time_delta ────────────────────────────────────────────────────────
    df["log_time_delta"] = np.log1p(df["time_delta_seconds"]).astype(np.float32)

    # ── event_burst_count ─────────────────────────────────────────────────────
    burst_parts = []
    for _, grp in df.groupby("Computer", sort=False):
        burst_parts.append(pd.Series(_burst_count(grp), index=grp.index))
    df["event_burst_count"] = (
        pd.concat(burst_parts).reindex(df.index).astype(np.float32)
    )

    # ── MinMax scale on benign rows only ─────────────────────────────────────
    benign = df["Label"] == TRAIN_LABEL
    for col in _TEMPORAL_COLS:
        mn = float(df.loc[benign, col].min())
        mx = float(df.loc[benign, col].max())
        if mx - mn > 1e-8:
            df[col] = ((df[col] - mn) / (mx - mn)).clip(0.0, 1.0).astype(np.float32)
        else:
            df[col] = np.float32(0.0)

    return df, feature_cols + _TEMPORAL_COLS


# ── in-memory (small datasets) ────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, feature_cols: list, seq_len: int):
    """
    Returns
    -------
    X : ndarray (N_seq, seq_len, n_features)  float32
    y : ndarray (N_seq,)                       int
    """
    df, feature_cols = _add_temporal_features(df, feature_cols)

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
    df, feature_cols = _add_temporal_features(df, feature_cols)

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
