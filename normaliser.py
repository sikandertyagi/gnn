"""
normaliser.py
─────────────
Feature normalisation for the transformer autoencoder.

Why this is necessary
─────────────────────
The numeric input features span very different ranges:
  · cmd_length  – 0 … 32 000+
  · dest_port   – 0 … 65 535
  · cmd_entropy – 0 … ~4.5  (bits)
  · is_signed   – {0, 1}

Without normalisation the transformer reconstruction loss is dominated by
high-variance features, making it blind to anomalies in low-variance ones.

Design
──────
  · StandardScaler fitted on benign (Label == 0) rows ONLY.
    This mirrors the anomaly-detection assumption: we model normality;
    attack statistics must not leak into the mean / std.
  · The first N_CATEGORICAL_FEATURES columns are CRC32-hashed categoricals
    already in [0, 1].  Z-scoring them is semantically meaningless (their
    values are arbitrary hashes, not measurements), so the scaler is fitted
    and applied only to the numeric columns that follow.
  · Scaler is saved to SCALER_PATH so it can be reloaded for inference.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import SCALER_PATH, TRAIN_LABEL
from feature_engineering import N_CATEGORICAL_FEATURES


def fit_scaler(df: pd.DataFrame, feature_cols: list) -> StandardScaler:
    """
    Fit a StandardScaler on the benign subset of *df*, skipping the first
    N_CATEGORICAL_FEATURES columns (CRC32-hashed, already in [0, 1]).

    Parameters
    ----------
    df           : full event DataFrame (with Label column)
    feature_cols : list of all feature column names (categorical + numeric)

    Returns
    -------
    scaler : fitted StandardScaler  (also saved to SCALER_PATH)
    """
    numeric_cols = feature_cols[N_CATEGORICAL_FEATURES:]
    benign = df[df["Label"] == TRAIN_LABEL]
    scaler = StandardScaler()
    scaler.fit(benign[numeric_cols].values.astype(np.float32))
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    feature_cols: list,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Return *df* with the numeric feature columns replaced by z-score
    normalised values.  The first N_CATEGORICAL_FEATURES columns (CRC32
    hashes, already in [0, 1]) are left untouched.
    Does not modify the original DataFrame.
    """
    numeric_cols = feature_cols[N_CATEGORICAL_FEATURES:]
    df = df.copy()
    df[numeric_cols] = scaler.transform(
        df[numeric_cols].values.astype(np.float32)
    )
    return df


def load_scaler() -> StandardScaler:
    """Reload a scaler saved by fit_scaler() for inference on new data."""
    return joblib.load(SCALER_PATH)
