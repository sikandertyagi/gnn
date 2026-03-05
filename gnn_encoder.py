"""
gnn_encoder.py
──────────────
Heterogeneous Graph Neural Network encoder.

Architecture
────────────
  Input  : HeteroData with node feature matrices x_dict and edge_index_dict
  Layer 1: HeteroConv( SAGEConv per edge type ) + ReLU
  Layer 2: HeteroConv( SAGEConv per edge type ) + LayerNorm
  Output : node embedding dict  {node_type: Tensor(N, embed_dim)}

  The model architecture is identical to the original pipeline.
  What changed is HOW it is trained and scored (see below).

Edge types handled (forward + reverse)
───────────────────────────────────────
  (process, parent_of,      process)
  (process, rev_parent_of,  process)
  (process, connects_to,    ip)
  (ip,      rev_connects_to,process)
  (process, runs_as,        user)
  (user,    rev_runs_as,    process)
  (process, runs_on,        host)
  (host,    rev_runs_on,    process)

Fleet-scale training  (replaces original full-batch train_gnn)
──────────────────────────────────────────────────────────────
  train_gnn_fleet()
    · Takes a list of per-machine HeteroData graphs (from machine_graph_builder)
    · Uses PyG DataLoader for graph-level mini-batching
      → peak RAM = MACHINE_GNN_BATCH_SIZE machine graphs at a time
      → scales to thousands of machines without OOM
    · Each step: batch of N graphs → single batched HeteroData → forward/backward
    · Early stopping on average batch loss across the epoch

  compute_fleet_centroids()
    · Encodes all benign machine graphs (batched)
    · Averages process embeddings → one centroid vector (the "normal" centre)

  score_machine_graphs()
    · Encodes every machine graph individually (preserves process_names mapping)
    · Scores each process node as L2 distance from the benign centroid
    · Returns dict: machine_name → {process_name → score}

  map_graph_scores_to_events()
    · Maps per-machine, per-process scores back to individual event rows
    · Missing entries (no graph score for a process) default to 0
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader

from config import (
    GNN_EMBED_DIM, GNN_EPOCHS, GNN_LR, GNN_EARLY_STOPPING_PAT,
    MACHINE_GNN_BATCH_SIZE,
)


# ─────────────────────────────────────────────────────────────────────────────
# Model  (unchanged from original pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class HeteroGNNEncoder(nn.Module):
    """
    Two-layer heterogeneous GraphSAGE encoder with bidirectional edges.

    Parameters
    ----------
    node_feature_dims : dict[str, int]   e.g. {"process": 10, "user": 2, ...}
    embed_dim         : int              output embedding dimension per node
    """

    _EDGE_TYPES = [
        ("process", "parent_of",      "process"),
        ("process", "rev_parent_of",  "process"),
        ("process", "connects_to",    "ip"      ),
        ("ip",      "rev_connects_to","process" ),
        ("process", "runs_as",        "user"    ),
        ("user",    "rev_runs_as",    "process" ),
        ("process", "runs_on",        "host"    ),
        ("host",    "rev_runs_on",    "process" ),
    ]

    def __init__(self, node_feature_dims: dict, embed_dim: int = GNN_EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim

        self.input_projs = nn.ModuleDict({
            ntype: nn.Linear(fdim, embed_dim)
            for ntype, fdim in node_feature_dims.items()
        })

        self.conv1 = self._build_conv(embed_dim)
        self.conv2 = self._build_conv(embed_dim)
        self.norm  = nn.LayerNorm(embed_dim)

        self.decoders = nn.ModuleDict({
            ntype: nn.Linear(embed_dim, fdim)
            for ntype, fdim in node_feature_dims.items()
        })

    def encode(self, x_dict: dict, edge_index_dict: dict) -> dict:
        h = {ntype: F.relu(self.input_projs[ntype](x)) for ntype, x in x_dict.items()}

        # filter to edge types with >0 edges (handles sparse machine graphs)
        present = {k: v for k, v in edge_index_dict.items() if v.shape[1] > 0}

        h_new = self.conv1(h, present)
        h = {ntype: F.relu(h_new.get(ntype, feat)) for ntype, feat in h.items()}

        h_new = self.conv2(h, present)
        h = {ntype: self.norm(h_new.get(ntype, feat)) for ntype, feat in h.items()}
        return h

    def decode(self, h_dict: dict) -> dict:
        return {ntype: self.decoders[ntype](emb) for ntype, emb in h_dict.items()}

    def forward(self, x_dict: dict, edge_index_dict: dict):
        h = self.encode(x_dict, edge_index_dict)
        return self.decode(h)

    def _build_conv(self, dim: int) -> HeteroConv:
        return HeteroConv(
            {
                ("process", "parent_of",      "process"): SAGEConv(dim, dim),
                ("process", "connects_to",    "ip"      ): SAGEConv((dim, dim), dim),
                ("process", "runs_as",        "user"    ): SAGEConv((dim, dim), dim),
                ("process", "runs_on",        "host"    ): SAGEConv((dim, dim), dim),
                ("process", "rev_parent_of",  "process"): SAGEConv(dim, dim),
                ("ip",      "rev_connects_to","process" ): SAGEConv((dim, dim), dim),
                ("user",    "rev_runs_as",    "process" ): SAGEConv((dim, dim), dim),
                ("host",    "rev_runs_on",    "process" ): SAGEConv((dim, dim), dim),
            },
            aggr="mean",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Fleet-scale training
# ─────────────────────────────────────────────────────────────────────────────

def train_gnn_fleet(
    model:         HeteroGNNEncoder,
    benign_graphs: list,
    batch_size:    int   = MACHINE_GNN_BATCH_SIZE,
    epochs:        int   = GNN_EPOCHS,
    lr:            float = GNN_LR,
    patience:      int   = GNN_EARLY_STOPPING_PAT,
) -> HeteroGNNEncoder:
    """
    Train the GNN on per-machine benign HeteroData graphs using graph-level
    mini-batching via PyG DataLoader.

    Key difference from original train_gnn():
      · Instead of one massive full-batch graph → OOM at scale, we batch
        `batch_size` machine graphs per step.
      · PyG's DataLoader calls Batch.from_data_list() to concatenate node
        feature matrices and offset edge indices automatically.
      · Peak RAM = batch_size × (avg nodes per machine) × embed_dim — constant
        regardless of total fleet size.

    Parameters
    ----------
    benign_graphs : list of HeteroData from machine_graph_builder (label == 0)
    batch_size    : number of machine graphs per training step
    """
    if not benign_graphs:
        raise ValueError("No benign machine graphs available for GNN training.")

    device    = torch.device("cpu")
    model     = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = PyGDataLoader(benign_graphs, batch_size=batch_size, shuffle=True)

    best_loss  = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait       = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            batch = batch.to(device)
            x_dict = {
                ntype: batch[ntype].x
                for ntype in batch.node_types
            }
            edge_index_dict = {
                etype: batch[etype].edge_index
                for etype in batch.edge_types
            }

            optimizer.zero_grad()
            recon_dict = model(x_dict, edge_index_dict)

            loss = sum(
                criterion(recon_dict[ntype], x_dict[ntype])
                for ntype in recon_dict
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(loader), 1)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  GNN epoch {epoch:>3}/{epochs}  loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = copy.deepcopy(model.state_dict())
            wait       = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                print(f"  GNN early stopping at epoch {epoch}  "
                      f"(best={best_loss:.4f})")
                break

    print(f"  GNN training complete  best_loss={best_loss:.4f}")
    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Fleet-scale inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_fleet_centroids(
    model:         HeteroGNNEncoder,
    benign_graphs: list,
    batch_size:    int = MACHINE_GNN_BATCH_SIZE,
) -> dict:
    """
    Compute the mean process embedding across ALL benign machine graphs.

    This centroid represents "what a normal process looks like in a normal
    machine's graph neighbourhood."  Anomaly scores are measured as L2
    distance from this centroid.

    Returns
    -------
    centroids : dict  {"process": Tensor(embed_dim,)}
    """
    model.eval()
    device = next(model.parameters()).device

    loader = PyGDataLoader(benign_graphs, batch_size=batch_size, shuffle=False)

    all_proc_embs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            h = model.encode(
                {ntype: batch[ntype].x for ntype in batch.node_types},
                {etype: batch[etype].edge_index for etype in batch.edge_types},
            )
            all_proc_embs.append(h["process"].cpu())

    proc_cat = torch.cat(all_proc_embs, dim=0)          # (total_proc_nodes, D)
    return {"process": proc_cat.mean(dim=0)}


def score_machine_graphs(
    model:            HeteroGNNEncoder,
    all_graphs:       list,
    fleet_centroids:  dict,
) -> dict:
    """
    Score every process node in every machine graph as L2 distance from the
    benign centroid, then globally normalise to [0, 1].

    Why one graph at a time (not batched)?
      Batching merges node indices across machines.  Processing individually
      preserves the process_names list → local_index mapping so we can return
      human-readable {machine_name → {process_name → score}} results.

    Returns
    -------
    machine_scores : dict[str, dict[str, float]]
        machine_scores[machine][process_image_path] = normalised_score ∈ [0, 1]
    """
    model.eval()
    device   = next(model.parameters()).device
    centroid = fleet_centroids["process"].to(device)

    raw_scores: dict = {}   # machine_name → {proc_name → raw_distance}

    with torch.no_grad():
        for g in all_graphs:
            g_dev = g.to(device)
            h = model.encode(
                {ntype: g_dev[ntype].x for ntype in g_dev.node_types},
                {etype: g_dev[etype].edge_index for etype in g_dev.edge_types},
            )
            proc_emb = h["process"]                      # (N_local_proc, D)
            dists    = torch.norm(
                proc_emb - centroid.unsqueeze(0), dim=1
            ).cpu().numpy()                              # (N_local_proc,)

            # process_names is a plain Python list → survives .to(device)
            machine_scores: dict = {}
            for local_idx, proc_name in enumerate(g.process_names):
                machine_scores[proc_name] = float(dists[local_idx])

            raw_scores[g.machine_name] = machine_scores

    # ── global min-max normalisation across all machines ──────────────────────
    all_vals = [v for m in raw_scores.values() for v in m.values()]
    if not all_vals:
        return raw_scores

    mn  = min(all_vals)
    mx  = max(all_vals)
    rng = mx - mn + 1e-8

    normalised: dict = {}
    for machine, scores in raw_scores.items():
        normalised[machine] = {
            proc: (score - mn) / rng
            for proc, score in scores.items()
        }
    return normalised


def map_graph_scores_to_events(
    df:            "pd.DataFrame",
    machine_scores: dict,
) -> np.ndarray:
    """
    Map per-machine, per-process scores back to individual event rows.

    Look-up key: (Computer column, Image column).
    Events whose machine or process has no score (e.g. skipped due to
    min_events filter) receive score 0, treating them as baseline-normal.

    Returns
    -------
    scores : np.ndarray of shape (N_events,) dtype float32
    """
    import pandas as pd

    scores    = np.zeros(len(df), dtype=np.float32)
    image_col = df["Image"].fillna(_MISSING).astype(str).values
    host_col  = df["Computer"].fillna(_MISSING).astype(str).values

    for i, (img, host) in enumerate(zip(image_col, host_col)):
        m = machine_scores.get(host)
        if m is not None:
            scores[i] = m.get(img, 0.0)

    return scores


# keep _MISSING accessible for map_graph_scores_to_events
_MISSING = "__MISSING__"
