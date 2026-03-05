import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder

def extract_process_name(path):
    if pd.isna(path):
        return "unknown"
    return path.split("\\")[-1].lower()

def has_base64(cmd):
    if pd.isna(cmd):
        return 0
    return int(bool(re.search(r'[A-Za-z0-9+/]{20,}={0,2}', cmd)))

def has_http(cmd):
    if pd.isna(cmd):
        return 0
    return int("http" in cmd.lower())

def entropy(s):
    if pd.isna(s):
        return 0
    prob = [ float(s.count(c)) / len(s) for c in dict.fromkeys(list(s)) ]
    entropy = - sum([ p * np.log2(p) for p in prob ])
    return entropy

def feature_engineering(df):

    df = df.copy()

    # ---------------------------
    # process names
    # ---------------------------

    df["process_name"] = df["Image"].apply(extract_process_name)
    df["parent_process"] = df["ParentImage"].apply(extract_process_name)

    # parent child pair
    df["parent_child"] = df["parent_process"] + "->" + df["process_name"]

    # ---------------------------
    # command line features
    # ---------------------------

    df["cmd_length"] = df["CommandLine"].fillna("").apply(len)
    df["cmd_token_count"] = df["CommandLine"].fillna("").apply(lambda x: len(x.split()))

    df["has_base64"] = df["CommandLine"].apply(has_base64)
    df["has_http"] = df["CommandLine"].apply(has_http)

    df["cmd_entropy"] = df["CommandLine"].fillna("").apply(entropy)

    # ---------------------------
    # path features
    # ---------------------------

    df["path_depth"] = df["Image"].fillna("").apply(lambda x: x.count("\\"))

    df["is_system32"] = df["Image"].fillna("").str.contains("system32", case=False).astype(int)

    df["is_users_dir"] = df["Image"].fillna("").str.contains("users", case=False).astype(int)

    df["is_temp_exec"] = df["Image"].fillna("").str.contains("temp", case=False).astype(int)

    # ---------------------------
    # binary metadata
    # ---------------------------

    df["is_signed"] = df["Signed"].fillna("false").astype(str).str.lower().eq("true").astype(int)

    df["missing_company"] = df["Company"].isna().astype(int)

    # ---------------------------
    # network features
    # ---------------------------

    df["dest_port"] = df["DestinationPort"].fillna(0)

    df["dest_external"] = ~df["DestinationIp"].fillna("").str.startswith(("10.","192.168","172.")).astype(int)

    # ---------------------------
    # temporal features
    # ---------------------------

    df["SystemTime"] = pd.to_datetime(df["SystemTime"])

    df["hour"] = df["SystemTime"].dt.hour

    df["is_after_hours"] = ((df["hour"] < 7) | (df["hour"] > 19)).astype(int)

    # ---------------------------
    # encoding categorical features
    # ---------------------------

    encoders = {}

    for col in ["process_name","parent_process","parent_child","User","IntegrityLevel"]:

        le = LabelEncoder()

        df[col] = df[col].fillna("unknown")

        df[col] = le.fit_transform(df[col])

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
        "is_after_hours"
    ]

    return df, feature_cols