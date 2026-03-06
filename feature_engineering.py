"""
feature_engineering.py
──────────────────────
Converts raw Sysmon CSV rows into a numeric feature matrix.

Robustness
──────────
  · Any Sysmon-optional column that is absent from the CSV is created with a
    safe default before feature computation — so the pipeline works on CSVs
    from any Sysmon configuration (process-create only, network-only, mixed).
  · 'Label' defaults to 0 (benign) when absent, allowing label-free inference.

Encoding strategy
─────────────────
  Categorical columns (process_name, parent_process, parent_child, User,
  IntegrityLevel) are encoded using deterministic CRC32 hashing normalised
  to [0, 1]:

      hash_value = (zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF) / 0xFFFFFFFF

  Benefits over sklearn LabelEncoder:
    · Stable across runs (no PYTHONHASHSEED, no sort-order dependency)
    · No fit/transform mismatch when new categories appear at inference
    · Always produces values in [0, 1] — no extra scaling needed

Bug fixes
─────────
  · dest_external: inverted the bool Series before .astype(int) to get 0/1
    (original bitwise ~ on an int Series yielded -1/-2).
  · RFC 1918: 172.16.0.0/12 range now correctly matched (172.16–172.31.x.x)
    instead of the broad 172.* which captured public IPs.
"""

import re
import warnings
import zlib

import numpy as np
import pandas as pd
from collections import Counter


# ─────────────────────────────────────────────────────────────────────────────
# Optional Sysmon columns and their safe defaults (None → leave as NaN)
# ─────────────────────────────────────────────────────────────────────────────

_OPTIONAL_COLS: dict = {
    "Label":          0,
    "Image":          "unknown",
    "ParentImage":    "unknown",
    "CommandLine":    "",
    "User":           "unknown",
    "IntegrityLevel": "unknown",
    "Signed":         "false",
    "Company":        None,       # NaN → missing_company = 1
    "DestinationIp":  None,       # NaN → no network event
    "DestinationPort": 0,
}

# RFC 1918 + loopback pattern
_RFC1918 = re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|127\.)")


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add any optional Sysmon column missing from *df* with its safe default."""
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
# Shannon entropy (optimised)
# ─────────────────────────────────────────────────────────────────────────────

def _entropy(s: str) -> float:
    """Shannon entropy of character distribution in string *s*."""
    if not s:
        return 0.0
    counts = np.array(list(Counter(s).values()), dtype=np.float64)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic hashing
# ─────────────────────────────────────────────────────────────────────────────

def _crc32_series(series: pd.Series) -> pd.Series:
    """Deterministic CRC32 hash of string values, normalised to [0, 1]."""
    return series.fillna("unknown").astype(str).apply(
        lambda x: (zlib.crc32(x.encode("utf-8")) & 0xFFFFFFFF) / 0xFFFFFFFF
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers (kept for graph_builder imports)
# ─────────────────────────────────────────────────────────────────────────────

def extract_process_name(path) -> str:
    """Return lowercased filename from a Windows path string."""
    if pd.isna(path):
        return "unknown"
    return str(path).split("\\")[-1].lower()


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
    df           : DataFrame with added / encoded feature columns
    feature_cols : list[str]  – names of numeric feature columns for the model
    """
    df = df.copy()
    df = _ensure_columns(df)

    # ── parse timestamps ──────────────────────────────────────────────────────
    df["SystemTime"] = pd.to_datetime(df["SystemTime"], errors="coerce")

    # ── process name features (vectorised str ops) ────────────────────────────
    df["process_name"] = (
        df["Image"].fillna("unknown").astype(str)
        .str.split("\\").str[-1].str.lower()
    )
    df["parent_process"] = (
        df["ParentImage"].fillna("unknown").astype(str)
        .str.split("\\").str[-1].str.lower()
    )
    df["parent_child"] = df["parent_process"] + "->" + df["process_name"]

    # ── encode categoricals via deterministic CRC32 hash ─────────────────────
    for col in ["process_name", "parent_process", "parent_child",
                "User", "IntegrityLevel"]:
        df[col] = _crc32_series(df[col].fillna("unknown").astype(str))

    # ── command-line features (vectorised) ────────────────────────────────────
    cmd = df["CommandLine"].fillna("").astype(str)

    df["cmd_length"]      = cmd.str.len()
    df["cmd_token_count"] = cmd.str.split().str.len().fillna(0).astype(int)
    df["has_base64"]      = cmd.str.contains(
        r"[A-Za-z0-9+/]{20,}={0,2}", regex=True, na=False
    ).astype(int)
    df["has_http"]        = cmd.str.contains("http", case=False, na=False).astype(int)

    # Shannon entropy via optimised Python function (unavoidably per-row)
    df["cmd_entropy"] = cmd.apply(_entropy)

    # ── path features (vectorised) ────────────────────────────────────────────
    img = df["Image"].fillna("").astype(str)

    df["path_depth"]   = img.str.count(r"\\")
    df["is_system32"]  = img.str.contains("system32", case=False, na=False).astype(int)
    df["is_users_dir"] = img.str.contains("users",    case=False, na=False).astype(int)
    df["is_temp_exec"] = img.str.contains("temp",     case=False, na=False).astype(int)

    # ── binary / metadata features ────────────────────────────────────────────
    df["is_signed"]       = (
        df["Signed"].fillna("false").astype(str).str.lower().eq("true").astype(int)
    )
    df["missing_company"] = df["Company"].isna().astype(int)

    # ── network features ──────────────────────────────────────────────────────
    df["dest_port"] = pd.to_numeric(df["DestinationPort"], errors="coerce").fillna(0)

    # RFC 1918 fix: proper 172.16.0.0/12 detection; bool inversion before .astype(int)
    df["dest_external"] = (
        ~df["DestinationIp"].fillna("").astype(str).apply(
            lambda ip: bool(_RFC1918.match(ip)) or ip == ""
        )
    ).astype(int)

    # ── temporal features ─────────────────────────────────────────────────────
    df["hour"]           = df["SystemTime"].dt.hour.fillna(0).astype(int)
    df["is_after_hours"] = (
        (df["hour"] < 7) | (df["hour"] > 19)
    ).astype(int)

    # ── feature column lists ─────────────────────────────────────────────────
    # CRC32-hashed categoricals come first (already in [0,1]); numeric cols follow.
    # The normaliser skips the first N_CATEGORICAL_FEATURES columns.
    feature_cols = [
        # CRC32-hashed categoricals (already [0, 1] — not z-scored)
        "process_name",
        "parent_process",
        "parent_child",
        "User",
        "IntegrityLevel",
        # numeric features (z-scored by normaliser)
        "cmd_length",
        "cmd_token_count",
        "has_base64",
        "has_http",
        "cmd_entropy",
        "path_depth",
        "is_system32",
        "is_users_dir",
        "is_temp_exec",
        "is_signed",
        "missing_company",
        "dest_port",
        "dest_external",
        "hour",
        "is_after_hours",
    ]

    return df, feature_cols


# Number of CRC32-hashed categorical columns at the start of feature_cols.
# normaliser.py uses this to skip z-scoring them.
N_CATEGORICAL_FEATURES = 5
