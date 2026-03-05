import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder


# ── low-level helpers ──────────────────────────────────────────────────────────

def _basename(path):
    if pd.isna(path):
        return "unknown"
    return str(path).split("\\")[-1].lower()


def _entropy(s):
    """Shannon entropy of a string."""
    if not s:
        return 0.0
    counts = np.frompyfunc(s.count, 1, 1)(list(set(s))).astype(float)
    probs  = counts / len(s)
    return float(-np.sum(probs * np.log2(probs + 1e-12)))


def _has_base64(cmd):
    if pd.isna(cmd):
        return 0
    return int(bool(re.search(r'[A-Za-z0-9+/]{20,}={0,2}', cmd)))


def _has_http(cmd):
    if pd.isna(cmd):
        return 0
    return int("http" in cmd.lower())


def _has_ip(cmd):
    """Detects bare IP addresses in command line — common in C2 / reverse shells."""
    if pd.isna(cmd):
        return 0
    return int(bool(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', cmd)))


def _has_download(cmd):
    """Detects download-related keywords — often used in dropper / stager commands."""
    if pd.isna(cmd):
        return 0
    keywords = [
        "wget", "curl", "invoke-webrequest", "iwr",
        "downloadstring", "downloadfile", "bitsadmin",
        "start-bitstransfer",
    ]
    cmd_lower = cmd.lower()
    return int(any(k in cmd_lower for k in keywords))


def _has_encodedcommand(cmd):
    """Detects PowerShell -EncodedCommand / -enc flag."""
    if pd.isna(cmd):
        return 0
    return int(bool(re.search(r'-(enc|encodedcommand)\b', cmd, re.IGNORECASE)))


# ── main function ──────────────────────────────────────────────────────────────

def feature_engineering(df):

    df = df.copy()

    # ── 1. Process identity ───────────────────────────────────────────────────
    df["process_name"]  = df["Image"].apply(_basename)
    df["parent_process"] = df["ParentImage"].apply(_basename)
    df["parent_child"]  = df["parent_process"] + "->" + df["process_name"]

    # Rare-process score: 1 / frequency (high = rare = suspicious).
    # Computed from the full df; this is frequency-based (not label-based)
    # so no meaningful leakage occurs.
    freq = df["process_name"].value_counts()
    df["rare_process_score"] = df["process_name"].map(lambda x: 1.0 / freq.get(x, 1))

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
    df["path_depth"]   = img.apply(lambda x: x.count("\\"))
    df["is_system32"]  = img.str.contains("system32", case=False).astype(int)
    df["is_users_dir"] = img.str.contains("users",    case=False).astype(int)
    df["is_temp_exec"] = img.str.contains("temp",     case=False).astype(int)

    # ── 4. Binary metadata ────────────────────────────────────────────────────
    df["is_signed"]       = (
        df["Signed"].fillna("false").astype(str).str.lower().eq("true").astype(int)
    )
    df["missing_company"] = df["Company"].isna().astype(int)

    # ── 5. Network behaviour (EventID 3) ──────────────────────────────────────
    df["dest_port"] = pd.to_numeric(
        df["DestinationPort"].fillna(0), errors="coerce"
    ).fillna(0)
    df["dest_external"] = (
        ~df["DestinationIp"].fillna("").str.startswith(("10.", "192.168.", "172."))
    ).astype(int)

    # ── 6. Temporal behaviour ─────────────────────────────────────────────────
    df["SystemTime"]     = pd.to_datetime(df["SystemTime"], errors="coerce")
    df["hour"]           = df["SystemTime"].dt.hour.fillna(0).astype(int)
    df["is_after_hours"] = ((df["hour"] < 7) | (df["hour"] > 19)).astype(int)

    # ── 7. Event type ─────────────────────────────────────────────────────────
    df["eventid"] = pd.to_numeric(
        df["EventID"].fillna(0), errors="coerce"
    ).fillna(0).astype(int)

    # ── 8. Categorical encoding ───────────────────────────────────────────────
    for col in ["process_name", "parent_process", "parent_child"]:
        df[col] = df[col].fillna("unknown")
        df[col] = LabelEncoder().fit_transform(df[col])

    # ── Final feature set (~23 features) ─────────────────────────────────────
    feature_cols = [
        # process identity
        "process_name",
        "parent_process",
        "parent_child",
        "rare_process_score",
        # command-line behaviour
        "cmd_length",
        "cmd_token_count",
        "cmd_entropy",
        "has_base64",
        "has_http",
        "has_ip",
        "has_download",
        "has_encodedcommand",
        # binary execution path
        "path_depth",
        "is_system32",
        "is_users_dir",
        "is_temp_exec",
        # binary metadata
        "is_signed",
        "missing_company",
        # network
        "dest_port",
        "dest_external",
        # temporal
        "hour",
        "is_after_hours",
        # event type
        "eventid",
    ]

    return df, feature_cols
