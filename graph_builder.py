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

Edge types (forward + reverse for bidirectional message passing)
──────────
  (process, parent_of,      process)  – ParentImage → Image
  (process, rev_parent_of,  process)  – Image → ParentImage  (reverse)
  (process, connects_to,    ip)       – Image → DestinationIp  (network events)
  (ip,      rev_connects_to,process)  – reverse of above
  (process, runs_as,        user)     – Image → User
  (user,    rev_runs_as,    process)  – reverse of above
  (process, runs_on,        host)     – Image → Computer
  (host,    rev_runs_on,    process)  – reverse of above
"""

import re
import zlib

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder

from feature_engineering import extract_process_name

# RFC 1918 + loopback pattern (same as feature_engineering.py)
_RFC1918 = re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|127\.)")


# ─────────────────────────────────────────────────────────────────────────────
# public API
# ─────────────────────────────────────────────────────────────────────────────

def build_event_graph(df: pd.DataFrame, encoders: dict = None):
    """
    Parameters
    ----------
    df       : DataFrame  (full or benign-only; must contain Sysmon columns)
    encoders : optional dict of pre-fitted LabelEncoders (from a full-graph
               call).  When provided the same node vocabulary is reused so
               that benign_graph and full_graph share identical feature dims.

    Returns
    -------
    data     : torch_geometric.data.HeteroData
    encoders : dict[str, LabelEncoder]  – one per node type
    """
    data = HeteroData()
    if encoders is None:
        encoders = _fit_encoders(df)

    # guard: every node type must have ≥1 node
    for ntype, col in [("process", "Image"), ("user", "User"),
                       ("host", "Computer"), ("ip", "DestinationIp")]:
        n = len(encoders[ntype].classes_)
        if n == 0:
            raise ValueError(
                f"Empty node type '{ntype}' – column '{col}' has no valid values."
            )

    data["process"].x = _process_features(df, encoders["process"])
    data["user"].x    = _user_features(encoders["user"])
    data["host"].x    = _host_features(encoders["host"])
    data["ip"].x      = _ip_features(df, encoders["ip"])

    # forward edges
    data["process", "parent_of",   "process"].edge_index = _pp_edges(df, encoders["process"])
    data["process", "connects_to", "ip"      ].edge_index = _pi_edges(df, encoders["process"], encoders["ip"])
    data["process", "runs_as",     "user"    ].edge_index = _pu_edges(df, encoders["process"], encoders["user"])
    data["process", "runs_on",     "host"    ].edge_index = _ph_edges(df, encoders["process"], encoders["host"])

    # reverse edges – enable bidirectional message passing
    data["process", "rev_parent_of",  "process"].edge_index = \
        data["process", "parent_of",   "process"].edge_index.flip(0)
    data["ip",   "rev_connects_to", "process"].edge_index = \
        data["process", "connects_to", "ip"    ].edge_index.flip(0)
    data["user", "rev_runs_as",     "process"].edge_index = \
        data["process", "runs_as",     "user"  ].edge_index.flip(0)
    data["host", "rev_runs_on",     "process"].edge_index = \
        data["process", "runs_on",     "host"  ].edge_index.flip(0)

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
        .astype(str)
        .unique()
    )
    proc_enc = LabelEncoder().fit(all_images)
    user_enc = LabelEncoder().fit(df["User"].fillna("unknown").astype(str).unique())
    host_enc = LabelEncoder().fit(df["Computer"].fillna("unknown").astype(str).unique())

    net_ips = df["DestinationIp"].dropna().astype(str).unique()
    ip_enc  = LabelEncoder().fit(net_ips if len(net_ips) else ["0.0.0.0"])

    return {"process": proc_enc, "user": user_enc, "host": host_enc, "ip": ip_enc}


def _process_features(df: pd.DataFrame, enc: LabelEncoder) -> torch.Tensor:
    """
    Features per process node (indexed by enc.classes_):
      name_hash (normalised), path_depth (normalised), is_system32, is_users_dir,
      is_temp_exec, cmd_length_mean (log-normalised), cmd_entropy_mean (normalised),
      has_base64_rate, has_http_rate, is_signed_rate
    All features are scaled to roughly [0, 1].
    """
    cmd = df["CommandLine"].fillna("").astype(str)
    agg = (
        df.assign(
            _image=df["Image"].fillna("unknown").astype(str),
            _cmd_len=cmd.str.len(),
            _cmd_tok=cmd.str.split().str.len().fillna(0),
            _b64=cmd.str.contains(r"[A-Za-z0-9+/]{20,}={0,2}", regex=True, na=False).astype(int),
            _http=cmd.str.contains("http", case=False, na=False).astype(int),
            _signed=df["Signed"].fillna("false").astype(str).str.lower().eq("true").astype(int),
        )
        .groupby("_image")
        .agg(
            cmd_length_mean=("_cmd_len", "mean"),
            cmd_token_mean=("_cmd_tok", "mean"),
            has_base64_rate=("_b64", "mean"),
            has_http_rate=("_http", "mean"),
            is_signed_rate=("_signed", "mean"),
        )
        .reindex(enc.classes_, fill_value=0.0)
    )

    images = pd.Series(enc.classes_)

    # CRC32-based name_hash: deterministic, stable across runs, always in [0, 1]
    name_hash = images.apply(
        lambda x: (zlib.crc32(extract_process_name(x).encode("utf-8")) & 0xFFFFFFFF) / 0xFFFFFFFF
    )

    # Fix #6: normalize continuous features to [0, 1]
    path_depth = images.apply(lambda x: float(str(x).count("\\")))
    path_depth = (path_depth / path_depth.clip(lower=1).max()).fillna(0.0)

    is_sys32 = images.str.contains("system32", case=False, na=False).astype(float)
    is_users = images.str.contains("users",    case=False, na=False).astype(float)
    is_temp  = images.str.contains("temp",     case=False, na=False).astype(float)

    # log-normalise cmd_length_mean (can be 0–thousands)
    cmd_len_norm = np.log1p(agg["cmd_length_mean"].values).astype(np.float32)
    max_len = cmd_len_norm.max()
    if max_len > 0:
        cmd_len_norm /= max_len

    # normalise cmd_token_mean (token count, typically 0–20)
    cmd_ent = agg["cmd_token_mean"].values.astype(np.float32)
    max_ent = cmd_ent.max()
    if max_ent > 0:
        cmd_ent /= max_ent

    feat = np.column_stack([
        name_hash.values,
        path_depth.values,
        is_sys32.values,
        is_users.values,
        is_temp.values,
        cmd_len_norm,
        cmd_ent,
        agg["has_base64_rate"].values,
        agg["has_http_rate"].values,
        agg["is_signed_rate"].values,
    ]).astype(np.float32)

    return torch.from_numpy(feat)


def _user_features(enc: LabelEncoder) -> torch.Tensor:
    users = pd.Series(enc.classes_)
    n     = len(users)
    # Fix #6: normalise uid to [0, 1]
    uid = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    is_admin = users.apply(
        lambda u: float("system" in u.lower() or "admin" in u.lower() or "root" in u.lower())
    ).values.astype(np.float32)
    return torch.from_numpy(np.column_stack([uid, is_admin]))


def _host_features(enc: LabelEncoder) -> torch.Tensor:
    # Fix #1: fixed dim=1 instead of one-hot (prevents model mismatch when
    # benign_graph and full_graph have different host counts).
    n = len(enc.classes_)
    host_ids = np.arange(n, dtype=np.float32) / max(n - 1, 1)
    return torch.from_numpy(host_ids.reshape(-1, 1))


def _ip_features(df: pd.DataFrame, enc: LabelEncoder) -> torch.Tensor:
    net = df.dropna(subset=["DestinationIp"])
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
        lambda ip: float(not (_RFC1918.match(str(ip)) or str(ip) == ""))
    ).values.astype(np.float32)

    # Fix #6: normalise port and count features to [0, 1]
    port_mean = agg["port_mean"].values.astype(np.float32) / 65535.0
    port_max  = agg["port_max"].values.astype(np.float32)  / 65535.0
    evt_count = np.log1p(agg["event_count"].values.astype(np.float32))
    max_cnt   = evt_count.max()
    if max_cnt > 0:
        evt_count /= max_cnt

    feat = np.column_stack([port_mean, port_max, evt_count, is_external])
    return torch.from_numpy(feat.astype(np.float32))


# ── edge builders ─────────────────────────────────────────────────────────────

def _make_edges(src_ids, dst_ids) -> torch.Tensor:
    if len(src_ids) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    edge = torch.tensor(np.stack([src_ids, dst_ids]), dtype=torch.long)
    # deduplicate parallel edges (same src → same dst seen multiple times)
    unique = torch.unique(edge.t(), dim=0)
    return unique.t().contiguous()


def _pp_edges(df, proc_enc) -> torch.Tensor:
    mask = df["Image"].notna() & df["ParentImage"].notna()
    if not mask.any():
        return torch.zeros((2, 0), dtype=torch.long)
    sub = df[mask]
    src = proc_enc.transform(sub["ParentImage"].fillna("unknown").astype(str).values)
    dst = proc_enc.transform(sub["Image"].fillna("unknown").astype(str).values)
    return _make_edges(src, dst)


def _pi_edges(df, proc_enc, ip_enc) -> torch.Tensor:
    mask = df["Image"].notna() & df["DestinationIp"].notna()
    if not mask.any():
        return torch.zeros((2, 0), dtype=torch.long)
    sub = df[mask]
    src = proc_enc.transform(sub["Image"].fillna("unknown").astype(str).values)
    dst = ip_enc.transform(sub["DestinationIp"].astype(str).values)
    return _make_edges(src, dst)


def _pu_edges(df, proc_enc, user_enc) -> torch.Tensor:
    mask = df["Image"].notna() & df["User"].notna()
    if not mask.any():
        return torch.zeros((2, 0), dtype=torch.long)
    sub = df[mask]
    src = proc_enc.transform(sub["Image"].fillna("unknown").astype(str).values)
    dst = user_enc.transform(sub["User"].fillna("unknown").astype(str).values)
    return _make_edges(src, dst)


def _ph_edges(df, proc_enc, host_enc) -> torch.Tensor:
    mask = df["Image"].notna() & df["Computer"].notna()
    if not mask.any():
        return torch.zeros((2, 0), dtype=torch.long)
    sub = df[mask]
    src = proc_enc.transform(sub["Image"].fillna("unknown").astype(str).values)
    dst = host_enc.transform(sub["Computer"].fillna("unknown").astype(str).values)
    return _make_edges(src, dst)
