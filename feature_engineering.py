"""
feature_engineering.py
──────────────────────
Converts raw Sysmon CSV rows into a numeric feature matrix.

Encoding strategy
─────────────────
  Categorical columns (process_name, parent_process, parent_child) are
  encoded with deterministic CRC32 hashing normalised to [0, 1]:

      hash_value = (zlib.crc32(s.encode()) & 0xFFFFFFFF) / 0xFFFFFFFF

  Benefits over sklearn LabelEncoder:
    · Stable across runs regardless of PYTHONHASHSEED
    · No fit/transform mismatch at inference time when new categories appear
    · Always produces values in [0, 1] — no extra scaling step needed

Bug fixes
─────────
  · dest_external: original code applied bitwise ~ to an int Series, yielding
    -1/-2 instead of 1/0. Fixed by inverting the bool Series before .astype(int).
  · RFC 1918: 172.16.0.0/12 range now correctly checked (172.16–172.31.x.x)
    instead of the overly broad 172.* prefix which caught public addresses.
"""

import re
import zlib

import numpy as np
import pandas as pd


# ── RFC 1918 + loopback private IP pattern ────────────────────────────────────
_RFC1918 = re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|127\.)")


# ── low-level helpers ──────────────────────────────────────────────────────────

def _basename(path):
    """Return lowercased filename from a Windows path string."""
    if pd.isna(path):
        return "unknown"
    return str(path).split("\\")[-1].lower()


def _crc32(s: str) -> float:
    """Deterministic CRC32 hash of *s*, normalised to [0, 1]."""
    return (zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF) / 0xFFFFFFFF


def _entropy(s):
    """Shannon entropy of string *s*."""
    if not s:
        return 0.0
    counts = np.frompyfunc(s.count, 1, 1)(list(set(s))).astype(float)
    probs  = counts / len(s)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _has_base64(cmd):
    if pd.isna(cmd):
        return 0
    return int(bool(re.search(r'[A-Za-z0-9+/]{20,}={0,2}', str(cmd))))


def _has_http(cmd):
    if pd.isna(cmd):
        return 0
    return int("http" in str(cmd).lower())


def _has_ip(cmd):
    """Detects bare IP addresses in command line — common in C2 / reverse shells."""
    if pd.isna(cmd):
        return 0
    return int(bool(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', str(cmd))))


def _has_download(cmd):
    """Detects download-related keywords — often used in dropper / stager commands."""
    if pd.isna(cmd):
        return 0
    keywords = [
        "wget", "curl", "invoke-webrequest", "iwr",
        "downloadstring", "downloadfile", "bitsadmin",
        "start-bitstransfer",
    ]
    cmd_lower = str(cmd).lower()
    return int(any(k in cmd_lower for k in keywords))


def _has_encodedcommand(cmd):
    """Detects PowerShell -EncodedCommand / -enc flag."""
    if pd.isna(cmd):
        return 0
    return int(bool(re.search(r'-(enc|encodedcommand)\b', str(cmd), re.IGNORECASE)))


# ── main function ──────────────────────────────────────────────────────────────

def feature_engineering(df):
    """
    Returns
    -------
    df           : DataFrame with added feature columns
    feature_cols : list[str] — names of feature columns for the model
    """
    df = df.copy()

    # ── 1. Process identity ───────────────────────────────────────────────────
    df["process_name"]  = df["Image"].apply(_basename)
    df["parent_process"] = df["ParentImage"].apply(_basename)
    df["parent_child"]  = df["parent_process"] + "->" + df["process_name"]

    # CRC32-hash categoricals to [0, 1] — deterministic and inference-stable
    for col in ["process_name", "parent_process", "parent_child"]:
        df[col] = df[col].fillna("unknown").apply(_crc32)

    # Rare-process score: 1 / frequency (high = rare = suspicious).
    # Computed from the full df; frequency-based only (not label-based).
    freq = df["Image"].apply(_basename).value_counts()
    df["rare_process_score"] = df["Image"].apply(_basename).map(
        lambda x: 1.0 / freq.get(x, 1)
    )

    # ── 2. Command-line behaviour ─────────────────────────────────────────────
    cmd = df["CommandLine"].fillna("")
    df["cmd_length"]         = cmd.apply(len)
    df["cmd_token_count"]    = cmd.apply(lambda x: len(x.split()))
    df["cmd_entropy"]        = cmd.apply(_entropy)
    df["has_base64"]         = df["CommandLine"].apply(_has_base64)
    df["has_http"]           = df["CommandLine"].apply(_has_http)
    df["has_ip"]             = df["CommandLine"].apply(_has_ip)
    df["has_download"]       = df["CommandLine"].apply(_has_download)
    df["has_encodedcommand"] = df["CommandLine"].apply(_has_encodedcommand)

    # ── 3. Binary execution path ──────────────────────────────────────────────
    img = df["Image"].fillna("")
    df["path_depth"]   = img.apply(lambda x: str(x).count("\\"))
    df["is_system32"]  = img.str.contains("system32", case=False, na=False).astype(int)
    df["is_users_dir"] = img.str.contains("users",    case=False, na=False).astype(int)
    df["is_temp_exec"] = img.str.contains("temp",     case=False, na=False).astype(int)

    # ── 4. Binary metadata ────────────────────────────────────────────────────
    df["is_signed"]       = (
        df["Signed"].fillna("false").astype(str).str.lower().eq("true").astype(int)
    )
    df["missing_company"] = df["Company"].isna().astype(int)

    # ── 5. Network behaviour ──────────────────────────────────────────────────
    df["dest_port"] = pd.to_numeric(
        df["DestinationPort"].fillna(0), errors="coerce"
    ).fillna(0)

    # Fix: invert bool Series before .astype(int); RFC 1918 correct 172.16/12
    df["dest_external"] = (
        ~df["DestinationIp"].fillna("").astype(str).apply(
            lambda ip: bool(_RFC1918.match(ip)) or ip == ""
        )
    ).astype(int)

    # ── 6. Temporal behaviour ─────────────────────────────────────────────────
    df["SystemTime"]     = pd.to_datetime(df["SystemTime"], errors="coerce")
    df["hour"]           = df["SystemTime"].dt.hour.fillna(0).astype(int)
    df["is_after_hours"] = ((df["hour"] < 7) | (df["hour"] > 19)).astype(int)

    # ── 7. Event type ─────────────────────────────────────────────────────────
    df["eventid"] = pd.to_numeric(
        df["EventID"].fillna(0), errors="coerce"
    ).fillna(0).astype(int)

    # ── Feature list ──────────────────────────────────────────────────────────
    # CRC32-hashed categoricals are already in [0, 1]; they come first so
    # main.py can skip them when fitting the StandardScaler.
    feature_cols = [
        # CRC32-hashed categoricals (already [0, 1] — not z-scored)
        "process_name",
        "parent_process",
        "parent_child",
        "rare_process_score",
        # command-line behaviour (z-scored)
        "cmd_length",
        "cmd_token_count",
        "cmd_entropy",
        "has_base64",
        "has_http",
        "has_ip",
        "has_download",
        "has_encodedcommand",
        # binary execution path (z-scored)
        "path_depth",
        "is_system32",
        "is_users_dir",
        "is_temp_exec",
        # binary metadata (z-scored)
        "is_signed",
        "missing_company",
        # network (z-scored)
        "dest_port",
        "dest_external",
        # temporal (z-scored)
        "hour",
        "is_after_hours",
        # event type (z-scored)
        "eventid",
    ]

    return df, feature_cols


# Number of CRC32-hashed categorical columns at the START of feature_cols.
# main.py uses this to skip z-scoring those columns.
N_CATEGORICAL_FEATURES = 4
