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

from config import RECON_WEIGHT, GRAPH_WEIGHT, RARITY_WEIGHT, TRAIN_LABEL


def compute_anomaly_scores(
    recon_errors:  np.ndarray,
    graph_scores:  np.ndarray,
    rarity_scores: np.ndarray,
    labels:        np.ndarray | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    recon_errors  : (N,)  MSE reconstruction error per sequence (transformer)
                          May contain NaN for leading events (Fix #4).
    graph_scores  : (N,)  GNN reconstruction error per event
    rarity_scores : (N,)  rare-behaviour score per event
    labels        : (N,)  event labels (0 = benign).  When provided, min-max
                          normalisation is fitted on benign rows only to prevent
                          attack score ranges from distorting the normalisation
                          baseline (test-set leakage).

    Returns
    -------
    scores : (N,)  composite anomaly score
    """
    benign_mask = (labels == TRAIN_LABEL) if labels is not None else None
    r = _normalise(recon_errors,  benign_mask)
    g = _normalise(graph_scores,  benign_mask)
    s = _normalise(rarity_scores, benign_mask)

    return RECON_WEIGHT * r + GRAPH_WEIGHT * g + RARITY_WEIGHT * s


# ─────────────────────────────────────────────────────────────────────────────

def _normalise(arr: np.ndarray,
               benign_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Min-max normalise, fitting the range on benign rows only when a mask is
    supplied.  This prevents attack event magnitudes from compressing the
    score range and making attacks appear less anomalous.

    Fix #7: if all values are identical (max == min) return zeros instead of
            dividing by ~0.
    Fix #4: NaN entries (leading events without a sequence window) are filled
            with 0 after normalisation so they do not appear anomalous.
    """
    if benign_mask is not None and benign_mask.any():
        mn = np.nanmin(arr[benign_mask])
        mx = np.nanmax(arr[benign_mask])
    else:
        mn = np.nanmin(arr)
        mx = np.nanmax(arr)

    if mx - mn < 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    result = (arr - mn) / (mx - mn)
    # fill NaN (Fix #4) with 0 – treat unscored events as baseline-normal
    return np.where(np.isnan(result), 0.0, result).astype(np.float32)
