"""
normaliser.py
─────────────
Feature normalisation using MinMaxScaler.

Scales numerical features to [0, 1] to match the sigmoid output activation
of the autoencoders.  OneHotEncoded columns (already in {0, 1}) are skipped.

MinMaxScaler is fitted on benign (Label == 0) rows ONLY so that attack
statistics do not leak into the scaling parameters.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import SCALER_PATH, TRAIN_LABEL


def fit_scaler(df: pd.DataFrame, feature_cols: list,
               n_ohe_cols: int = 0) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the benign subset of *df*, skipping the first
    *n_ohe_cols* columns (OneHotEncoded, already in {0, 1}).

    Parameters
    ----------
    df           : full event DataFrame (with Label column)
    feature_cols : list of all feature column names (OHE + numeric)
    n_ohe_cols   : number of OHE columns at the start of feature_cols

    Returns
    -------
    scaler : fitted MinMaxScaler  (also saved to SCALER_PATH)
    """
    numeric_cols = feature_cols[n_ohe_cols:]
    benign = df[df["Label"] == TRAIN_LABEL]
    scaler = MinMaxScaler()
    scaler.fit(benign[numeric_cols].values.astype(np.float32))
    joblib.dump(scaler, SCALER_PATH)
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    feature_cols: list,
    scaler: MinMaxScaler,
    n_ohe_cols: int = 0,
) -> pd.DataFrame:
    """
    Return *df* with the numeric feature columns replaced by MinMax-scaled
    values in [0, 1].  The first *n_ohe_cols* columns (OHE, already in
    {0, 1}) are left untouched.
    Does not modify the original DataFrame.
    """
    numeric_cols = feature_cols[n_ohe_cols:]
    df = df.copy()
    df[numeric_cols] = scaler.transform(
        df[numeric_cols].values.astype(np.float32)
    )
    return df


def load_scaler() -> MinMaxScaler:
    """Reload a scaler saved by fit_scaler() for inference on new data."""
    return joblib.load(SCALER_PATH)
