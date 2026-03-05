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
    Two-layer heterogeneous GraphSAGE encoder.

    Parameters
    ----------
    node_feature_dims : dict[str, int]   e.g. {"process": 10, "user": 2, ...}
    embed_dim         : int              output embedding dimension per node
    """

    _EDGE_TYPES = [
        ("process", "parent_of",   "process"),
        ("process", "connects_to", "ip"      ),
        ("process", "runs_as",     "user"    ),
        ("process", "runs_on",     "host"    ),
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

        # filter to only edge types present in this graph
        present = {k: v for k, v in edge_index_dict.items() if v.shape[1] > 0}

        h = self.conv1(h, present)
        h = {ntype: F.relu(feat) for ntype, feat in h.items()}

        h = self.conv2(h, present)
        h = {ntype: self.norm(feat) for ntype, feat in h.items()}
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
                ("process", "parent_of",   "process"): SAGEConv(dim, dim),
                ("process", "connects_to", "ip"      ): SAGEConv((dim, dim), dim),
                ("process", "runs_as",     "user"    ): SAGEConv((dim, dim), dim),
                ("process", "runs_on",     "host"    ): SAGEConv((dim, dim), dim),
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
    device    = torch.device("cpu")
    model     = model.to(device)
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

    model.load_state_dict(best_state)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_benign_centroids(model: HeteroGNNEncoder,
                              data: HeteroData) -> dict:
    """
    Compute mean embedding per node type over the (benign) training graph.
    Used as a reference point for anomaly scoring at inference time.
    """
    model.eval()
    device = next(model.parameters()).device
    x_dict = {ntype: data[ntype].x.to(device) for ntype in data.node_types}
    edge_index_dict = {
        etype: data[etype].edge_index.to(device)
        for etype in data.edge_types
    }
    with torch.no_grad():
        h = model.encode(x_dict, edge_index_dict)
    return {ntype: emb.mean(dim=0) for ntype, emb in h.items()}


def graph_anomaly_scores(model: HeteroGNNEncoder,
                          data: HeteroData,
                          benign_centroids: dict,
                          process_enc) -> torch.Tensor:
    """
    Per-process-node anomaly score = L2 distance from benign centroid,
    normalised to [0, 1].

    Returns
    -------
    scores : Tensor(N_process,)  – one score per process node
    """
    model.eval()
    device = next(model.parameters()).device
    x_dict = {ntype: data[ntype].x.to(device) for ntype in data.node_types}
    edge_index_dict = {
        etype: data[etype].edge_index.to(device)
        for etype in data.edge_types
    }
    with torch.no_grad():
        h = model.encode(x_dict, edge_index_dict)

    proc_emb  = h["process"]
    centroid  = benign_centroids["process"].to(device)
    distances = torch.norm(proc_emb - centroid.unsqueeze(0), dim=1)

    # normalise
    mn, mx = distances.min(), distances.max()
    scores = (distances - mn) / (mx - mn + 1e-8)
    return scores.cpu()
