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

Vectorisation
─────────────
  · All per-row operations use pandas str/vectorised ops (no apply() loops)
    except cmd_entropy, which requires per-string Shannon entropy and is
    optimised with numpy character counting instead of the original O(N²) loop.

Bug fix
───────
  · dest_external: original code applied bitwise ~ to an int Series, yielding
    -1/-2 instead of 1/0. Fixed by inverting the bool Series before .astype(int).
"""

import warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder


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
    df["process_name"]  = (
        df["Image"].fillna("unknown").astype(str)
        .str.split("\\").str[-1].str.lower()
    )
    df["parent_process"] = (
        df["ParentImage"].fillna("unknown").astype(str)
        .str.split("\\").str[-1].str.lower()
    )
    df["parent_child"] = df["parent_process"] + "->" + df["process_name"]

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

    # str.count() uses regex; r"\\" matches a single backslash
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

    # BUG FIX: invert bool Series *before* .astype(int) to get 0/1 not -2/-1
    df["dest_external"] = (
        ~df["DestinationIp"].fillna("").astype(str)
        .str.startswith(("10.", "192.168.", "172."))
    ).astype(int)

    # ── temporal features ─────────────────────────────────────────────────────
    df["hour"]          = df["SystemTime"].dt.hour.fillna(0).astype(int)
    df["is_after_hours"] = (
        (df["hour"] < 7) | (df["hour"] > 19)
    ).astype(int)

    # ── encode categoricals ───────────────────────────────────────────────────
    encoders: dict = {}
    for col in ["process_name", "parent_process", "parent_child",
                "User", "IntegrityLevel"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("unknown").astype(str))
        encoders[col] = le

    feature_cols = [
        "process_name",
        "parent_process",
        "parent_child",
        "User",
        "IntegrityLevel",
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
