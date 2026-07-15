"""
feature_engineering.py
──────────────────────
Converts raw Sysmon CSV rows into a numeric feature matrix.

Encoding strategy
─────────────────
  High-cardinality columns (Computer, DestinationPortName):
    → frequency encoding  (proportion of events with that value)
    → hash-bucket encoding (deterministic fixed-width binary features)
    → well-known port flag (DestinationPortName only)

  Low-cardinality columns (EventID, Initiated, SourceIsIpv6, time parts):
    → OneHotEncoding  (small, bounded number of unique values)

  Numerical columns:
    → passed through as-is (MinMaxScaler applied later by normaliser.py)

Feature column ordering
───────────────────────
  Binary columns first (OHE + hash buckets + flags), then float columns
  (frequencies + numericals).  The count of leading binary columns is
  returned as n_binary_cols so the normaliser knows which to skip.

Rarity engine columns
─────────────────────
  process_name, parent_process, parent_child are still extracted from
  Image/ParentImage and written to the DataFrame for use by RarityEngine
  (which reads raw df columns, not the feature matrix).
"""

import os
import re
import warnings
import zlib

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from config import ARTIFACTS_DIR


# ─────────────────────────────────────────────────────────────────────────────
# Column classifications
# ─────────────────────────────────────────────────────────────────────────────

HIGH_CARDINALITY_COLS = ['Computer', 'DestinationPortName']

LOW_CARDINALITY_COLS = [
    'EventID', 'Initiated', 'SourceIsIpv6',
    'SystemTime_year', 'SystemTime_month',
    'SystemTime_week', 'SystemTime_day_of_week',
]

NUMERICAL_COLS = [
    'EventRecordID', 'Execution_ProcessID', 'ProcessId',
    'SystemTime_day', 'SystemTime_hour', 'SystemTime_minute',
]

# Hash bucket counts per high-cardinality column
HASH_BUCKETS = {
    'Computer': 16,
    'DestinationPortName': 8,
}

WELL_KNOWN_PORTS = frozenset({
    'HTTP', 'HTTPS', 'DNS', 'SSH', 'RDP', 'SMB', 'SMTP', 'IMAP', 'POP3',
    'FTP', 'LDAP', 'LDAPS', 'Kerberos', 'NTP', 'SNMP', 'DHCP',
    'WinRM', 'WMI', 'MSSQL', 'MySQL', 'PostgreSQL', 'Redis',
    'Syslog', 'TFTP', 'Telnet',
    'http', 'https', 'dns', 'ssh', 'rdp', 'smb', 'smtp', 'imap', 'pop3',
    'ftp', 'ldap', 'ldaps', 'kerberos', 'ntp', 'snmp', 'dhcp',
    'winrm', 'wmi', 'mssql', 'mysql', 'postgresql', 'redis',
    'syslog', 'tftp', 'telnet',
})

OHE_PATH  = os.path.join(ARTIFACTS_DIR, "ohe_encoder.pkl")
FREQ_PATH = os.path.join(ARTIFACTS_DIR, "freq_maps.pkl")

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
# Encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hash_bucket(value: str, n_buckets: int) -> int:
    """Deterministic hash → bucket index. Uses CRC32 (stable across sessions)."""
    return zlib.crc32(value.encode('utf-8', errors='replace')) % n_buckets


def _build_freq_map(series: pd.Series) -> dict:
    """Build a {value: proportion} mapping from a Series."""
    counts = series.value_counts(normalize=True)
    return counts.to_dict()


def _apply_freq_encoding(series: pd.Series, freq_map: dict) -> np.ndarray:
    """Map series values to their frequency proportion (0.0 for unseen)."""
    return series.map(freq_map).fillna(0.0).values.astype(np.float32)


def _apply_hash_encoding(series: pd.Series, n_buckets: int) -> np.ndarray:
    """Map series values to a fixed-width binary matrix via hash bucketing."""
    bucket_ids = series.apply(lambda v: _hash_bucket(str(v), n_buckets)).values
    result = np.zeros((len(series), n_buckets), dtype=np.float32)
    result[np.arange(len(series)), bucket_ids] = 1.0
    return result


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
                   encoded features)
    feature_cols : list[str] — names of feature columns for the model
                   (binary columns first, then float columns)
    n_ohe_cols   : int — number of binary columns at the start of feature_cols
                   (OHE + hash buckets + flags; already in {0,1}, skip scaling)
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
    all_cat_cols = HIGH_CARDINALITY_COLS + LOW_CARDINALITY_COLS
    for col in all_cat_cols:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                .replace(['nan', 'NaN', 'NULL', 'null', 'None', '-'], 'Unknown')
                .fillna('Unknown')
            )

    # ══════════════════════════════════════════════════════════════════════════
    # A. HIGH-CARDINALITY COLUMNS → frequency + hash encoding
    # ══════════════════════════════════════════════════════════════════════════
    freq_maps = {}
    hash_binary_names = []
    freq_col_names = []

    for col in HIGH_CARDINALITY_COLS:
        if col not in df.columns:
            continue

        n_unique = df[col].nunique()
        n_buckets = HASH_BUCKETS.get(col, 8)

        # frequency encoding
        fmap = _build_freq_map(df[col])
        freq_maps[col] = fmap
        freq_name = f"{col}_freq"
        df[freq_name] = _apply_freq_encoding(df[col], fmap)
        freq_col_names.append(freq_name)

        # hash-bucket encoding
        hash_matrix = _apply_hash_encoding(df[col], n_buckets)
        bucket_names = [f"{col}_hash_{i}" for i in range(n_buckets)]
        for i, bname in enumerate(bucket_names):
            df[bname] = hash_matrix[:, i]
        hash_binary_names.extend(bucket_names)

        print(f"      {col}: {n_unique:,} unique values → "
              f"1 freq + {n_buckets} hash buckets")

    joblib.dump(freq_maps, FREQ_PATH)

    # well-known port flag
    flag_names = []
    if 'DestinationPortName' in df.columns:
        df['port_is_wellknown'] = (
            df['DestinationPortName'].isin(WELL_KNOWN_PORTS).astype(np.float32)
        )
        flag_names.append('port_is_wellknown')

    # ══════════════════════════════════════════════════════════════════════════
    # B. LOW-CARDINALITY COLUMNS → OneHotEncoding
    # ══════════════════════════════════════════════════════════════════════════
    cat_cols = [c for c in LOW_CARDINALITY_COLS if c in df.columns]
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = ohe.fit_transform(df[cat_cols])
    ohe_feature_names = list(ohe.get_feature_names_out(cat_cols))
    print(f"      OHE: {len(cat_cols)} low-cardinality cols → "
          f"{len(ohe_feature_names)} binary features")

    joblib.dump(ohe, OHE_PATH)

    for i, name in enumerate(ohe_feature_names):
        df[name] = X_cat[:, i].astype(np.float32)

    # ── clean numericals ──────────────────────────────────────────────────────
    num_cols = [c for c in NUMERICAL_COLS if c in df.columns]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ══════════════════════════════════════════════════════════════════════════
    # C. ASSEMBLE feature_cols: binary first, then floats
    # ══════════════════════════════════════════════════════════════════════════
    binary_cols = ohe_feature_names + hash_binary_names + flag_names
    float_cols = freq_col_names + num_cols

    feature_cols = binary_cols + float_cols
    n_ohe_cols = len(binary_cols)

    print(f"      Total features: {len(feature_cols)} "
          f"({n_ohe_cols} binary + {len(float_cols)} float)")

    return df, feature_cols, n_ohe_cols


# Number of columns at the start of feature_cols that are already in {0, 1}
# and must NOT be scaled.  Set dynamically by feature_engineering(); this
# module-level default is only used if something imports it before calling
# feature_engineering.
N_CATEGORICAL_FEATURES = 0
