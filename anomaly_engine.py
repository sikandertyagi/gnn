"""
anomaly_engine.py
─────────────────
Combines three anomaly signals into a composite event score.

    event_score = RECON_WEIGHT  * reconstruction_error   (transformer)
                + GRAPH_WEIGHT  * graph_anomaly_score     (GNN)
                + RARITY_WEIGHT * rarity_score            (rare behaviour)

All three input arrays must be aligned to the same event index.
Each component is min-max normalised before weighting.
"""

import numpy as np

from config import RECON_WEIGHT, GRAPH_WEIGHT, RARITY_WEIGHT


def compute_anomaly_scores(
    recon_errors:  np.ndarray,
    graph_scores:  np.ndarray,
    rarity_scores: np.ndarray,
) -> np.ndarray:
    """
    Parameters
    ----------
    recon_errors  : (N,)  MSE reconstruction error per sequence (transformer)
    graph_scores  : (N,)  GNN distance-from-centroid per event
    rarity_scores : (N,)  rare-behaviour score per event

    Returns
    -------
    scores : (N,)  composite anomaly score in [0, 1]
    """
    r = _normalise(recon_errors)
    g = _normalise(graph_scores)
    s = _normalise(rarity_scores)

    return RECON_WEIGHT * r + GRAPH_WEIGHT * g + RARITY_WEIGHT * s


# ─────────────────────────────────────────────────────────────────────────────

def _normalise(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)
