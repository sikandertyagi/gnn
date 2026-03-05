"""
anomaly_engine.py
─────────────────
Combines three anomaly signals into a composite event score.

    event_score = RECON_WEIGHT  * reconstruction_error   (transformer)
                + GRAPH_WEIGHT  * graph_anomaly_score     (GNN)
                + RARITY_WEIGHT * rarity_score            (rare behaviour)

All three input arrays must be aligned to the same event index.
Each component is min-max normalised before weighting.
NaN values (e.g. leading events without a full sequence window) are treated
as 0 after normalisation so they do not inflate the composite score.
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
                          May contain NaN for leading events (Fix #4).
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
    """
    Min-max normalise to [0, 1].

    Fix #7: if all values are identical (max == min) return zeros instead of
            dividing by ~0.
    Fix #4: NaN entries (leading events without a sequence window) are filled
            with 0 after normalisation so they do not appear anomalous.
    """
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    result = (arr - mn) / (mx - mn)
    # fill NaN (Fix #4) with 0 – treat unscored events as baseline-normal
    return np.where(np.isnan(result), 0.0, result).astype(np.float32)
