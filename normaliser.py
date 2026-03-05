"""
normaliser.py
─────────────
Feature normalisation for the transformer autoencoder.

Why this is necessary
─────────────────────
The 20 input features span very different ranges:
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
  · Scaler is saved to SCALER_PATH so it can be reloaded for inference on
    new data without refitting.
  · The scaler is applied in-place on the feature columns before sequences
    are built — so both the transformer and the GNN graph aggregates see
    normalised values.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import SCALER_PATH, TRAIN_LABEL


def fit_scaler(df: pd.DataFrame, feature_cols: list) -> StandardScaler:
    """
    Fit a StandardScaler on the benign subset of *df*.

    Parameters
    ----------
    df           : full event DataFrame (with Label column)
    feature_cols : list of numeric column names to normalise

    Returns
    -------
    scaler : fitted StandardScaler  (also saved to SCALER_PATH)
    """
    benign = df[df["Label"] == TRAIN_LABEL]
    scaler = StandardScaler()
    scaler.fit(benign[feature_cols].values.astype(np.float32))
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    feature_cols: list,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Return *df* with *feature_cols* replaced by z-score normalised values.
    Does not modify the original DataFrame.
    """
    df = df.copy()
    df[feature_cols] = scaler.transform(
        df[feature_cols].values.astype(np.float32)
    )
    return df


def load_scaler() -> StandardScaler:
    """Reload a scaler saved by fit_scaler() for inference on new data."""
    return joblib.load(SCALER_PATH)
