"""
feature_engineering.py
──────────────────────
Converts raw Sysmon CSV rows into a numeric feature matrix using the
December pipeline approach: OneHotEncoding for categoricals + MinMaxScaler
for numericals.

This approach was validated to achieve ROC-AUC 0.9950 as a standalone
dense autoencoder — significantly outperforming the previous CRC32 hashing
+ StandardScaler approach.

Feature set (15 metadata features)
──────────────────────────────────
  Categorical (OHE):
    Computer, DestinationPortName, EventID, Initiated, SourceIsIpv6,
    SystemTime_year, SystemTime_month, SystemTime_week, SystemTime_day_of_week

  Numerical (MinMaxScaler):
    EventRecordID, Execution_ProcessID, ProcessId,
    SystemTime_day, SystemTime_hour, SystemTime_minute

Rarity engine columns
─────────────────────
  process_name, parent_process, parent_child are still extracted from
  Image/ParentImage and written to the DataFrame for use by RarityEngine
  (which reads raw df columns, not the feature matrix).
"""

import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from config import ARTIFACTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions — same 15 as December preprocessing.py
# ─────────────────────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    'Computer', 'DestinationPortName', 'EventID', 'Initiated',
    'SourceIsIpv6', 'SystemTime_year', 'SystemTime_month',
    'SystemTime_week', 'SystemTime_day_of_week',
]

NUMERICAL_COLS = [
    'EventRecordID', 'Execution_ProcessID', 'ProcessId',
    'SystemTime_day', 'SystemTime_hour', 'SystemTime_minute',
]

OHE_PATH = os.path.join(ARTIFACTS_DIR, "ohe_encoder.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# Optional Sysmon columns and their safe defaults
# ─────────────────────────────────────────────────────────────────────────────

_OPTIONAL_COLS: dict = {
    "Label":            0,
    "Image":            "unknown",
    "ParentImage":      "unknown",
    "CommandLine":      "",
    "User":             "unknown",
    "IntegrityLevel":   "unknown",
    "Signed":           "false",
    "Company":          None,
    "DestinationIp":    None,
    "DestinationPort":  0,
    "DestinationPortName": "Unknown",
    "EventID":          0,
    "EventRecordID":    0,
    "Execution_ProcessID": 0,
    "ProcessId":        0,
    "Initiated":        "Unknown",
    "SourceIsIpv6":     "Unknown",
    "Computer":         "Unknown",
}


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any optional column missing from *df* with its safe default."""
    missing = [c for c in _OPTIONAL_COLS if c not in df.columns]
    if missing:
        warnings.warn(
            f"Columns not found in CSV (using defaults): {missing}",
            stacklevel=3,
        )
    for col in missing:
        default = _OPTIONAL_COLS[col]
        df[col] = default if default is not None else np.nan
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers (kept for graph_builder / rarity_engine imports)
# ─────────────────────────────────────────────────────────────────────────────

def extract_process_name(path) -> str:
    """Return lowercased filename from a Windows or Linux path string."""
    if pd.isna(path):
        return "unknown"
    return re.split(r"[/\\]", str(path))[-1].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame):
    """
    Parameters
    ----------
    df : raw Sysmon DataFrame (any subset of Sysmon columns is accepted)

    Returns
    -------
    df           : DataFrame with added columns (time features, process names,
                   OHE binary columns)
    feature_cols : list[str] — names of feature columns for the model
                   (OHE columns first, then numerical columns)
    n_ohe_cols   : int — number of OHE columns at the start of feature_cols
                   (these are already in {0,1} and should NOT be scaled)
    """
    df = df.copy()
    df = _ensure_columns(df)

    # ── parse timestamps ──────────────────────────────────────────────────────
    df["SystemTime"] = pd.to_datetime(df["SystemTime"], errors="coerce")

    # ── process name extraction (for rarity engine, NOT for feature matrix) ──
    df["process_name"] = (
        df["Image"].fillna("unknown").astype(str)
        .str.split(r"[/\\]").str[-1].str.lower()
    )
    df["parent_process"] = (
        df["ParentImage"].fillna("unknown").astype(str)
        .str.split(r"[/\\]").str[-1].str.lower()
    )
    df["parent_child"] = df["parent_process"] + "->" + df["process_name"]

    # ── time features from SystemTime ─────────────────────────────────────────
    ts = df["SystemTime"]
    df["SystemTime_year"] = ts.dt.year.fillna(0).astype(int)
    df["SystemTime_month"] = ts.dt.month.fillna(0).astype(int)
    df["SystemTime_week"] = ts.dt.isocalendar().week.fillna(0).astype(int)
    df["SystemTime_day"] = ts.dt.day.fillna(0).astype(int)
    df["SystemTime_hour"] = ts.dt.hour.fillna(0).astype(int)
    df["SystemTime_minute"] = ts.dt.minute.fillna(0).astype(int)
    df["SystemTime_day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)

    # ── clean categoricals ────────────────────────────────────────────────────
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    for col in cat_cols:
        df[col] = (
            df[col].astype(str)
            .replace(['nan', 'NaN', 'NULL', 'null', 'None', '-'], 'Unknown')
            .fillna('Unknown')
        )

    # ── OneHotEncode categoricals ─────────────────────────────────────────────
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = ohe.fit_transform(df[cat_cols])
    ohe_feature_names = list(ohe.get_feature_names_out(cat_cols))
    print(f"      OHE: {len(cat_cols)} categorical cols → {len(ohe_feature_names)} binary features")

    joblib.dump(ohe, OHE_PATH)

    for i, name in enumerate(ohe_feature_names):
        df[name] = X_cat[:, i].astype(np.float32)

    # ── clean numericals ──────────────────────────────────────────────────────
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ── feature column list (OHE first, then numericals) ──────────────────────
    feature_cols = ohe_feature_names + num_cols
    n_ohe_cols = len(ohe_feature_names)

    print(f"      Total features: {len(feature_cols)} "
          f"({n_ohe_cols} OHE + {len(num_cols)} numerical)")

    return df, feature_cols, n_ohe_cols


# Number of columns at the start of feature_cols that are already in {0, 1}
# and must NOT be scaled.  Set dynamically by feature_engineering(); this
# module-level default is only used if something imports it before calling
# feature_engineering.
N_CATEGORICAL_FEATURES = 0
