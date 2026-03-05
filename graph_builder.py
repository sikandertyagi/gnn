"""
graph_builder.py
────────────────
Builds a heterogeneous event graph from a Sysmon DataFrame.

Node types
──────────
  process  – unique Image path seen in the dataset
  user     – unique User value
  host     – unique Computer value
  ip       – unique DestinationIp value

Edge types
──────────
  (process, parent_of,  process)  – ParentImage → Image
  (process, connects_to, ip)      – Image → DestinationIp  (network events)
  (process, runs_as,    user)     – Image → User
  (process, runs_on,    host)     – Image → Computer
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder

from feature_engineering import extract_process_name


# ─────────────────────────────────────────────────────────────────────────────
# public API
# ─────────────────────────────────────────────────────────────────────────────

def build_event_graph(df: pd.DataFrame):
    """
    Parameters
    ----------
    df : DataFrame  (full or benign-only; must contain Sysmon columns)

    Returns
    -------
    data     : torch_geometric.data.HeteroData
    encoders : dict[str, LabelEncoder]  – one per node type
    """
    data     = HeteroData()
    encoders = _fit_encoders(df)

    data["process"].x = _process_features(df, encoders["process"])
    data["user"].x    = _user_features(encoders["user"])
    data["host"].x    = _host_features(encoders["host"])
    data["ip"].x      = _ip_features(df, encoders["ip"])

    data["process", "parent_of",   "process"].edge_index = _pp_edges(df, encoders["process"])
    data["process", "connects_to", "ip"      ].edge_index = _pi_edges(df, encoders["process"], encoders["ip"])
    data["process", "runs_as",     "user"    ].edge_index = _pu_edges(df, encoders["process"], encoders["user"])
    data["process", "runs_on",     "host"    ].edge_index = _ph_edges(df, encoders["process"], encoders["host"])

    return data, encoders


def node_feature_dims(data: HeteroData) -> dict:
    """Return {node_type: feature_dim} for all node types in *data*."""
    return {ntype: data[ntype].x.shape[1] for ntype in data.node_types}


# ─────────────────────────────────────────────────────────────────────────────
# internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_encoders(df: pd.DataFrame) -> dict:
    all_images = (
        pd.concat([df["Image"].fillna("unknown"), df["ParentImage"].fillna("unknown")])
        .unique()
    )
    proc_enc = LabelEncoder().fit(all_images)
    user_enc = LabelEncoder().fit(df["User"].fillna("unknown").unique())
    host_enc = LabelEncoder().fit(df["Computer"].fillna("unknown").unique())

    net_ips = df["DestinationIp"].dropna().unique()
    ip_enc  = LabelEncoder().fit(net_ips if len(net_ips) else ["0.0.0.0"])

    return {"process": proc_enc, "user": user_enc, "host": host_enc, "ip": ip_enc}


def _process_features(df: pd.DataFrame, enc: LabelEncoder) -> torch.Tensor:
    """
    Features per process node (indexed by enc.classes_):
      name_hash, path_depth, is_system32, is_users_dir, is_temp_exec,
      cmd_length_mean, cmd_entropy_mean, has_base64_rate, has_http_rate,
      is_signed_rate
    """
    import re

    # aggregate per-image statistics from events
    agg = (
        df.assign(
            _image=df["Image"].fillna("unknown"),
            _cmd_len=df["CommandLine"].fillna("").apply(len),
            _cmd_tok=df["CommandLine"].fillna("").apply(lambda x: len(x.split())),
            _b64=df["CommandLine"].fillna("").apply(
                lambda x: int(bool(re.search(r"[A-Za-z0-9+/]{20,}={0,2}", x)))
            ),
            _http=df["CommandLine"].fillna("").apply(lambda x: int("http" in x.lower())),
            _signed=df["Signed"].fillna("false").astype(str).str.lower().eq("true").astype(int),
        )
        .groupby("_image")
        .agg(
            cmd_length_mean=("_cmd_len", "mean"),
            cmd_entropy_mean=("_cmd_tok", "mean"),
            has_base64_rate=("_b64", "mean"),
            has_http_rate=("_http", "mean"),
            is_signed_rate=("_signed", "mean"),
        )
        .reindex(enc.classes_, fill_value=0.0)
    )

    images = pd.Series(enc.classes_)
    path_depth  = images.apply(lambda x: float(str(x).count("\\")))
    is_sys32    = images.str.contains("system32", case=False, na=False).astype(float)
    is_users    = images.str.contains("users",    case=False, na=False).astype(float)
    is_temp     = images.str.contains("temp",     case=False, na=False).astype(float)
    name_hash   = images.apply(lambda x: float(hash(extract_process_name(x)) % 10_000))

    feat = np.column_stack([
        name_hash.values,
        path_depth.values,
        is_sys32.values,
        is_users.values,
        is_temp.values,
        agg["cmd_length_mean"].values,
        agg["cmd_entropy_mean"].values,
        agg["has_base64_rate"].values,
        agg["has_http_rate"].values,
        agg["is_signed_rate"].values,
    ]).astype(np.float32)

    return torch.from_numpy(feat)


def _user_features(enc: LabelEncoder) -> torch.Tensor:
    users = pd.Series(enc.classes_)
    uid   = np.arange(len(users), dtype=np.float32)
    is_admin = users.apply(
        lambda u: float("system" in u.lower() or "admin" in u.lower() or "root" in u.lower())
    ).values.astype(np.float32)
    return torch.from_numpy(np.column_stack([uid, is_admin]))


def _host_features(enc: LabelEncoder) -> torch.Tensor:
    n = len(enc.classes_)
    # one-hot: each host is uniquely identified
    return torch.eye(n, dtype=torch.float32)


def _ip_features(df: pd.DataFrame, enc: LabelEncoder) -> torch.Tensor:
    net = df.dropna(subset=["DestinationIp"])
    # per-IP aggregates
    agg = (
        net.groupby("DestinationIp")
        .agg(
            port_mean=("DestinationPort", "mean"),
            port_max=("DestinationPort", "max"),
            event_count=("DestinationIp", "count"),
        )
        .reindex(enc.classes_, fill_value=0.0)
    )
    is_external = pd.Series(enc.classes_).apply(
        lambda ip: float(not str(ip).startswith(("10.", "192.168.", "172.")))
    ).values.astype(np.float32)

    feat = np.column_stack([
        agg["port_mean"].values.astype(np.float32),
        agg["port_max"].values.astype(np.float32),
        agg["event_count"].values.astype(np.float32),
        is_external,
    ])
    return torch.from_numpy(feat.astype(np.float32))


# ── edge builders ─────────────────────────────────────────────────────────────

def _make_edges(src_ids, dst_ids) -> torch.Tensor:
    if len(src_ids) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(np.stack([src_ids, dst_ids]), dtype=torch.long)


def _pp_edges(df, proc_enc) -> torch.Tensor:
    mask = df["Image"].notna() & df["ParentImage"].notna()
    if not mask.any():
        return torch.zeros((2, 0), dtype=torch.long)
    sub  = df[mask]
    src  = proc_enc.transform(sub["ParentImage"].fillna("unknown").values)
    dst  = proc_enc.transform(sub["Image"].fillna("unknown").values)
    return _make_edges(src, dst)


def _pi_edges(df, proc_enc, ip_enc) -> torch.Tensor:
    mask = df["Image"].notna() & df["DestinationIp"].notna()
    if not mask.any():
        return torch.zeros((2, 0), dtype=torch.long)
    sub = df[mask]
    src = proc_enc.transform(sub["Image"].fillna("unknown").values)
    dst = ip_enc.transform(sub["DestinationIp"].values)
    return _make_edges(src, dst)


def _pu_edges(df, proc_enc, user_enc) -> torch.Tensor:
    mask = df["Image"].notna() & df["User"].notna()
    if not mask.any():
        return torch.zeros((2, 0), dtype=torch.long)
    sub = df[mask]
    src = proc_enc.transform(sub["Image"].fillna("unknown").values)
    dst = user_enc.transform(sub["User"].fillna("unknown").values)
    return _make_edges(src, dst)


def _ph_edges(df, proc_enc, host_enc) -> torch.Tensor:
    mask = df["Image"].notna() & df["Computer"].notna()
    if not mask.any():
        return torch.zeros((2, 0), dtype=torch.long)
    sub = df[mask]
    src = proc_enc.transform(sub["Image"].fillna("unknown").values)
    dst = host_enc.transform(sub["Computer"].fillna("unknown").values)
    return _make_edges(src, dst)
