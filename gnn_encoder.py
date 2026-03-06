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

Training objective (benign-only)
──────────────────────────────────
  Self-supervised node-feature reconstruction:
    encode → decode (linear) → MSE vs original features
  Trained only on the graph built from benign events so the model learns
  "normal" graph topology.  At inference, high reconstruction error on a
  process node signals structural anomaly.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_geometric.data import HeteroData

from config import GNN_EMBED_DIM, GNN_EPOCHS, GNN_LR, GNN_EARLY_STOPPING_PAT


# ─────────────────────────────────────────────────────────────────────────────
# Model
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

        # project every node type to a common embedding space
        self.input_projs = nn.ModuleDict({
            ntype: nn.Linear(fdim, embed_dim)
            for ntype, fdim in node_feature_dims.items()
        })

        # two-layer HeteroConv
        self.conv1 = self._build_conv(embed_dim)
        self.conv2 = self._build_conv(embed_dim)
        self.norm  = nn.LayerNorm(embed_dim)

        # per-node-type decoder: reconstruct original features
        self.decoders = nn.ModuleDict({
            ntype: nn.Linear(embed_dim, fdim)
            for ntype, fdim in node_feature_dims.items()
        })

    # ── forward ───────────────────────────────────────────────────────────────

    def encode(self, x_dict: dict, edge_index_dict: dict) -> dict:
        h = {ntype: F.relu(self.input_projs[ntype](x)) for ntype, x in x_dict.items()}

        # filter to only edge types present in this graph (with >0 edges)
        present = {k: v for k, v in edge_index_dict.items() if v.shape[1] > 0}

        h_new = self.conv1(h, present)
        # Fix #3: fall back to projected features for node types with no
        # incoming edges so every type always has an embedding
        h = {ntype: F.relu(h_new.get(ntype, feat)) for ntype, feat in h.items()}

        h_new = self.conv2(h, present)
        h = {ntype: self.norm(h_new.get(ntype, feat)) for ntype, feat in h.items()}
        return h

    def decode(self, h_dict: dict) -> dict:
        return {ntype: self.decoders[ntype](emb) for ntype, emb in h_dict.items()}

    def forward(self, x_dict: dict, edge_index_dict: dict):
        h = self.encode(x_dict, edge_index_dict)
        return self.decode(h)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _build_conv(self, dim: int) -> HeteroConv:
        return HeteroConv(
            {
                # forward edges
                ("process", "parent_of",      "process"): SAGEConv(dim, dim),
                ("process", "connects_to",    "ip"      ): SAGEConv((dim, dim), dim),
                ("process", "runs_as",        "user"    ): SAGEConv((dim, dim), dim),
                ("process", "runs_on",        "host"    ): SAGEConv((dim, dim), dim),
                # Fix #3: reverse edges for bidirectional message passing
                ("process", "rev_parent_of",  "process"): SAGEConv(dim, dim),
                ("ip",      "rev_connects_to","process" ): SAGEConv((dim, dim), dim),
                ("user",    "rev_runs_as",    "process" ): SAGEConv((dim, dim), dim),
                ("host",    "rev_runs_on",    "process" ): SAGEConv((dim, dim), dim),
            },
            aggr="mean",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_gnn(
    model:    HeteroGNNEncoder,
    data:     HeteroData,
    epochs:   int   = GNN_EPOCHS,
    lr:       float = GNN_LR,
    patience: int   = GNN_EARLY_STOPPING_PAT,
) -> HeteroGNNEncoder:
    """
    Train the GNN encoder on *data* (should be built from benign events only)
    using self-supervised node-feature reconstruction.

    Early stopping monitors training loss (GNN training is full-graph, so there
    is no separate val set; the loss itself is a reliable convergence signal).
    """
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    print(f"  GNN training on {device}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_dict = {ntype: data[ntype].x.to(device) for ntype in data.node_types}
    edge_index_dict = {
        etype: data[etype].edge_index.to(device)
        for etype in data.edge_types
    }

    best_loss  = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    wait       = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        recon_dict = model(x_dict, edge_index_dict)
        loss = sum(
            criterion(recon_dict[ntype], x_dict[ntype])
            for ntype in recon_dict
        )
        loss.backward()

        # Fix #10: gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        loss_val = loss.item()
        if epoch % 5 == 0 or epoch == 1:
            print(f"  GNN epoch {epoch:>3}/{epochs}  loss={loss_val:.4f}")

        if loss_val < best_loss:
            best_loss  = loss_val
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
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def graph_anomaly_scores(model: HeteroGNNEncoder,
                          full_data: HeteroData,
                          benign_data: HeteroData,
                          process_enc) -> torch.Tensor:
    """
    Per-process-node anomaly score = reconstruction MSE, normalised to [0, 1].

    Uses benign node features (to avoid contamination from attack event
    statistics) but the full graph's edge topology (to detect structural
    anomalies such as new parent-child chains or novel IP connections).

    Attack-only processes that have no benign events will have zero feature
    vectors in benign_data (reindex fill_value=0.0) and will stand out as
    anomalous in the reconstruction.

    Returns
    -------
    scores : Tensor(N_process,)  – one score per process node
    """
    model.eval()
    device = next(model.parameters()).device
    # node features from benign graph — not contaminated by attack statistics
    x_dict = {ntype: benign_data[ntype].x.to(device) for ntype in full_data.node_types}
    # edge topology from full graph — captures attack-introduced relationships
    edge_index_dict = {
        etype: full_data[etype].edge_index.to(device)
        for etype in full_data.edge_types
    }
    with torch.no_grad():
        recon_dict = model(x_dict, edge_index_dict)   # encode + decode

    # per-node MSE for process nodes: shape (N_process,)
    recon_err = F.mse_loss(
        recon_dict["process"], x_dict["process"], reduction="none"
    ).mean(dim=1)

    # normalise to [0, 1] using benign range so attack scores can exceed 1
    mn, mx = recon_err.min(), recon_err.max()
    scores = (recon_err - mn) / (mx - mn + 1e-8)
    return scores.cpu()
