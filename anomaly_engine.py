"""
anomaly_engine.py
─────────────────
Combines three anomaly signals into a composite event score.

    event_score = RECON_WEIGHT  * reconstruction_error   (transformer)
                + GRAPH_WEIGHT  * graph_anomaly_score     (GNN)
                + RARITY_WEIGHT * rarity_score            (rare behaviour)

All three input arrays must be aligned to the same event index.
Each component is min-max normalised before weighting.

Weight redistribution for unscored events
─────────────────────────────────────────
The transformer only scores EventID-1/3 events.  For all other events
recon_error is NaN.  Filling NaN with 0 and using the same weights caps
unscored events at (GRAPH_WEIGHT + RARITY_WEIGHT) = 0.65, creating a
systematic ceiling gap versus scored events (max = 1.0).

Fix: detect unscored events BEFORE normalisation and redistribute
RECON_WEIGHT proportionally across the two available components so both
groups use a composite that sums to 1.0:

    scored   → RECON_WEIGHT·r + GRAPH_WEIGHT·g + RARITY_WEIGHT·s
    unscored → (GRAPH_WEIGHT / remain)·g + (RARITY_WEIGHT / remain)·s
               where remain = GRAPH_WEIGHT + RARITY_WEIGHT
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
                          May contain NaN for events not scored by the transformer
                          (non-EventID-1/3 events and leading warmup events).
    graph_scores  : (N,)  GNN reconstruction error per event
    rarity_scores : (N,)  rare-behaviour score per event
    labels        : (N,)  event labels (0 = benign).  When provided, min-max
                          normalisation is fitted on benign rows only to prevent
                          attack score ranges from distorting the normalisation
                          baseline (test-set leakage).

    Returns
    -------
    scores : (N,)  composite anomaly score in [0, 1] for all events
    """
    benign_mask = (labels == TRAIN_LABEL) if labels is not None else None

    # Capture which events have a valid transformer score BEFORE normalisation
    # so we can apply the correct weight formula per event.
    has_recon = ~np.isnan(recon_errors)

    r = _normalise(recon_errors,  benign_mask)   # NaN → 0 after norm
    g = _normalise(graph_scores,  benign_mask)
    s = _normalise(rarity_scores, benign_mask)

    # Scored events: full three-component weighted sum.
    scored_composite = RECON_WEIGHT * r + GRAPH_WEIGHT * g + RARITY_WEIGHT * s

    # Unscored events: redistribute RECON_WEIGHT to the two available signals
    # so the composite still spans [0, 1] — same ceiling as scored events.
    remain = GRAPH_WEIGHT + RARITY_WEIGHT
    unscored_composite = (GRAPH_WEIGHT / remain) * g + (RARITY_WEIGHT / remain) * s

    return np.where(has_recon, scored_composite, unscored_composite).astype(np.float32)


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
    # fill NaN with 0 – treat unscored events as baseline-normal within their
    # component; weight redistribution in compute_anomaly_scores handles the rest.
    return np.where(np.isnan(result), 0.0, result).astype(np.float32)
