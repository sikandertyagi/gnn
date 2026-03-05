"""
machine_graph_builder.py
────────────────────────
Builds one small HeteroData graph per machine (host) rather than a single
global graph.  This solves three scalability problems from the original design:

  1. Edge explosion     – each graph only contains edges from ONE machine;
                          edges are deduplicated (same parent→child pair seen
                          10 000 times → 1 edge, not 10 000).

  2. GNN OOM            – training uses PyG graph-level mini-batching
                          (N machines per step), so peak RAM is proportional
                          to MACHINE_GNN_BATCH_SIZE, not to total event count.

  3. Signal dilution    – a process that behaves anomalously on 1 machine is
                          no longer averaged into the embeddings of 999 normal
                          machines.  Anomaly is measured relative to the
                          behaviour of that specific machine's graph.

Node types per machine graph
────────────────────────────
  process  – unique Image / ParentImage values seen on this machine
  ip       – unique DestinationIp values from this machine
  user     – unique User values on this machine
  host     – the machine itself (single node; 1-dim feature = 0)

Edge types  (forward + bidirectional reverse, all deduplicated)
──────────────────────────────────────────────────────────────
  (process, parent_of,      process)   ParentImage → Image
  (process, connects_to,    ip)        Image → DestinationIp
  (process, runs_as,        user)      Image → User
  (process, runs_on,        host)      Image → host node 0

Metadata stored on each HeteroData object
──────────────────────────────────────────
  data.machine_name   : str       – value of the Computer column
  data.process_names  : list[str] – maps local process index → Image string
  data.machine_label  : int       – 0 = all events benign, 1 = contains attack
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from feature_engineering import extract_process_name


_MISSING = "__MISSING__"


# ── public API ────────────────────────────────────────────────────────────────

def build_machine_graphs(df: pd.DataFrame, min_events: int = 10) -> list:
    """
    Build one HeteroData per machine.

    Parameters
    ----------
    df          : full Sysmon DataFrame (must contain Computer and Label cols)
    min_events  : machines with fewer raw events are skipped

    Returns
    -------
    graphs : list[HeteroData]
        One graph per machine.  Each graph carries .machine_name,
        .process_names, and .machine_label as Python attributes.
    """
    df = df.copy()

    # guarantee required columns exist
    for col in ("Image", "ParentImage", "DestinationIp", "User", "Computer"):
        if col not in df.columns:
            df[col] = np.nan

    df["_image"]  = df["Image"].fillna(_MISSING).astype(str)
    df["_parent"] = df["ParentImage"].fillna(_MISSING).astype(str)
    df["_ip"]     = df["DestinationIp"].fillna(_MISSING).astype(str)
    df["_user"]   = df["User"].fillna(_MISSING).astype(str)
    df["_host"]   = df["Computer"].fillna(_MISSING).astype(str)

    graphs = []
    for host, host_df in df.groupby("_host", sort=False):
        if len(host_df) < min_events:
            continue
        g = _build_single_machine_graph(host, host_df)
        graphs.append(g)

    return graphs


def get_node_feature_dims(graphs: list) -> dict:
    """
    Return {node_type: feature_dim} derived from the first graph.
    All graphs share the same per-node-type feature schema, so any graph
    can be used as a reference.
    """
    if not graphs:
        raise ValueError("Graph list is empty — no machines passed min_events filter.")
    g = graphs[0]
    return {ntype: g[ntype].x.shape[1] for ntype in g.node_types}


# ── single-machine graph builder ──────────────────────────────────────────────

def _build_single_machine_graph(machine_name: str, df: pd.DataFrame) -> HeteroData:
    data = HeteroData()

    # ── local node vocabularies (unique to this machine) ──────────────────────
    all_procs  = pd.concat([df["_image"], df["_parent"]]).unique().tolist()
    proc_vocab = {name: idx for idx, name in enumerate(all_procs)}

    ip_vals    = [v for v in df["_ip"].unique() if v != _MISSING]
    ip_vocab   = {ip: idx for idx, ip in enumerate(ip_vals)}

    user_vals  = [v for v in df["_user"].unique() if v != _MISSING]
    user_vocab = {u: idx for idx, u in enumerate(user_vals)}

    # ── node feature matrices ─────────────────────────────────────────────────
    data["process"].x = _process_features(df, proc_vocab)
    data["ip"].x      = _ip_features(df, ip_vocab)
    data["user"].x    = _user_features(user_vocab)
    data["host"].x    = torch.zeros(1, 1, dtype=torch.float32)

    # ── edges: forward, deduplicated ──────────────────────────────────────────
    pp = _dedup(_pp_edges(df, proc_vocab))
    pi = _dedup(_pi_edges(df, proc_vocab, ip_vocab))
    pu = _dedup(_pu_edges(df, proc_vocab, user_vocab))
    ph = _dedup(_ph_edges(df, proc_vocab))

    data["process", "parent_of",      "process"].edge_index = pp
    data["process", "connects_to",    "ip"      ].edge_index = pi
    data["process", "runs_as",        "user"    ].edge_index = pu
    data["process", "runs_on",        "host"    ].edge_index = ph

    # reverse edges for bidirectional message passing
    data["process", "rev_parent_of",  "process"].edge_index = pp.flip(0)
    data["ip",      "rev_connects_to","process" ].edge_index = pi.flip(0)
    data["user",    "rev_runs_as",    "process" ].edge_index = pu.flip(0)
    data["host",    "rev_runs_on",    "process" ].edge_index = ph.flip(0)

    # ── metadata (Python attrs; not moved by .to(device)) ─────────────────────
    data.machine_name  = machine_name
    data.process_names = all_procs                       # local_idx → full path
    data.machine_label = int((df["Label"] != 0).any())  # 0 = fully benign

    return data


# ── node feature builders ─────────────────────────────────────────────────────

def _process_features(df: pd.DataFrame, vocab: dict) -> torch.Tensor:
    """
    10 features per process node, computed from this machine's events only.
    Features are normalised to [0, 1] per machine so the GNN sees consistent
    scales regardless of machine size.
    """
    n   = len(vocab)
    cmd = (df["CommandLine"].fillna("").astype(str)
           if "CommandLine" in df.columns
           else pd.Series([""] * len(df), index=df.index))
    signed = (df["Signed"].fillna("false").astype(str).str.lower().eq("true")
              if "Signed" in df.columns
              else pd.Series([False] * len(df), index=df.index))

    # aggregate per process image (vectorised)
    agg = (
        df.assign(
            _cmd_len = cmd.str.len(),
            _cmd_tok = cmd.str.split().str.len().fillna(0),
            _b64     = cmd.str.contains(r"[A-Za-z0-9+/]{20,}={0,2}",
                                         regex=True, na=False).astype(int),
            _http    = cmd.str.contains("http", case=False, na=False).astype(int),
            _signed  = signed.astype(int),
        )
        .groupby("_image")
        .agg(
            cmd_length_mean = ("_cmd_len", "mean"),
            cmd_token_mean  = ("_cmd_tok", "mean"),
            has_base64_rate = ("_b64",     "mean"),
            has_http_rate   = ("_http",    "mean"),
            is_signed_rate  = ("_signed",  "mean"),
        )
    )

    feat = np.zeros((n, 10), dtype=np.float32)
    for name, idx in vocab.items():
        # cols 0-4: path-derived (independent of event count)
        feat[idx, 0] = float(hash(extract_process_name(name)) % 10_000) / 10_000.0
        feat[idx, 1] = float(str(name).count("\\"))
        feat[idx, 2] = float("system32" in name.lower())
        feat[idx, 3] = float("users"    in name.lower())
        feat[idx, 4] = float("temp"     in name.lower())
        # cols 5-9: event-aggregate features
        if name in agg.index:
            row = agg.loc[name]
            feat[idx, 5] = float(np.log1p(row["cmd_length_mean"]))
            feat[idx, 6] = float(row["cmd_token_mean"])
            feat[idx, 7] = float(row["has_base64_rate"])
            feat[idx, 8] = float(row["has_http_rate"])
            feat[idx, 9] = float(row["is_signed_rate"])

    # normalise per-machine continuous cols to [0, 1]
    for col in (1, 5, 6):
        mx = feat[:, col].max()
        if mx > 0:
            feat[:, col] /= mx

    return torch.from_numpy(feat)


def _ip_features(df: pd.DataFrame, vocab: dict) -> torch.Tensor:
    """4 features per IP node: port_mean, port_max (both /65535), log-event-count, is_external."""
    n = len(vocab)
    if n == 0:
        return torch.zeros(1, 4, dtype=torch.float32)  # dummy when no network events

    net      = df[df["_ip"] != _MISSING]
    port_col = "DestinationPort" if "DestinationPort" in df.columns else None

    if port_col and len(net):
        agg = net.groupby("_ip").agg(
            port_mean  = (port_col, "mean"),
            port_max   = (port_col, "max"),
            evt_count  = ("_ip",    "count"),
        )
    else:
        agg = net.groupby("_ip").agg(evt_count=("_ip", "count"))
        agg["port_mean"] = 0.0
        agg["port_max"]  = 0.0

    feat = np.zeros((n, 4), dtype=np.float32)
    for ip, idx in vocab.items():
        feat[idx, 3] = float(not str(ip).startswith(("10.", "192.168.", "172.")))
        if ip in agg.index:
            row = agg.loc[ip]
            feat[idx, 0] = float(row["port_mean"]) / 65535.0
            feat[idx, 1] = float(row["port_max"])  / 65535.0
            feat[idx, 2] = float(np.log1p(row["evt_count"]))

    mx = feat[:, 2].max()
    if mx > 0:
        feat[:, 2] /= mx

    return torch.from_numpy(feat)


def _user_features(vocab: dict) -> torch.Tensor:
    """2 features: normalised index, is_privileged (system/admin/root in name)."""
    n = max(len(vocab), 1)
    feat = np.zeros((n, 2), dtype=np.float32)
    for name, idx in vocab.items():
        feat[idx, 0] = idx / max(n - 1, 1)
        feat[idx, 1] = float(
            "system" in name.lower() or
            "admin"  in name.lower() or
            "root"   in name.lower()
        )
    return torch.from_numpy(feat)


# ── edge builders ─────────────────────────────────────────────────────────────

def _dedup(edge: torch.Tensor) -> torch.Tensor:
    """
    Remove duplicate edges.
    Input / output shape: (2, E).
    Uses torch.unique on transposed (E, 2) tensor — no integer overflow risk.
    """
    if edge.shape[1] == 0:
        return edge
    unique = torch.unique(edge.t(), dim=0)
    return unique.t().contiguous()


def _make_edges(src: list, dst: list) -> torch.Tensor:
    if not src:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(
        np.stack([np.array(src, dtype=np.int64),
                  np.array(dst, dtype=np.int64)]),
        dtype=torch.long,
    )


def _pp_edges(df: pd.DataFrame, proc_vocab: dict) -> torch.Tensor:
    mask = (df["_image"] != _MISSING) & (df["_parent"] != _MISSING)
    sub  = df[mask]
    src  = [proc_vocab[p] for p in sub["_parent"]]
    dst  = [proc_vocab[p] for p in sub["_image"]]
    return _make_edges(src, dst)


def _pi_edges(df: pd.DataFrame, proc_vocab: dict, ip_vocab: dict) -> torch.Tensor:
    if not ip_vocab:
        return torch.zeros((2, 0), dtype=torch.long)
    mask = (df["_image"] != _MISSING) & (df["_ip"] != _MISSING)
    sub  = df[mask]
    src  = [proc_vocab[p]  for p in sub["_image"]]
    dst  = [ip_vocab[ip]   for ip in sub["_ip"]]
    return _make_edges(src, dst)


def _pu_edges(df: pd.DataFrame, proc_vocab: dict, user_vocab: dict) -> torch.Tensor:
    if not user_vocab:
        return torch.zeros((2, 0), dtype=torch.long)
    mask = (df["_image"] != _MISSING) & (df["_user"] != _MISSING)
    sub  = df[mask]
    src  = [proc_vocab[p] for p in sub["_image"]]
    dst  = [user_vocab[u] for u in sub["_user"]]
    return _make_edges(src, dst)


def _ph_edges(df: pd.DataFrame, proc_vocab: dict) -> torch.Tensor:
    """All non-missing process nodes on this machine → single host node 0."""
    procs = [p for p in df["_image"].unique() if p != _MISSING]
    src   = [proc_vocab[p] for p in procs]
    dst   = [0] * len(src)
    return _make_edges(src, dst)
