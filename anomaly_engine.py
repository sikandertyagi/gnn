"""
anomaly_engine.py
─────────────────
Combines three anomaly signals into a composite event score.

    event_score = RECON_WEIGHT  * reconstruction_error   (transformer)
                + GRAPH_WEIGHT  * graph_anomaly_score     (GNN)
                + RARITY_WEIGHT * rarity_score            (rare behaviour)

All three input arrays must be aligned to the same event index.
Each component is min-max normalised before weighting.

Weight redistribution for warmup events
────────────────────────────────────────
The pipeline now filters to EventID 1 & 3 globally, so every event is
eligible for transformer scoring.  However, the first (SEQUENCE_LENGTH - 1)
events per host have no full sliding window and therefore receive NaN
recon_error (warmup period).

Filling NaN with 0 and applying the standard weights would cap warmup
events at (GRAPH_WEIGHT + RARITY_WEIGHT) = 0.65, while fully-scored events
can reach 1.0.  To keep both groups on equal footing, RECON_WEIGHT is
redistributed proportionally to the two available signals for warmup events:

    scored   → RECON_WEIGHT·r + GRAPH_WEIGHT·g + RARITY_WEIGHT·s
    warmup   → (GRAPH_WEIGHT / remain)·g + (RARITY_WEIGHT / remain)·s
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
                          May contain NaN for the first (SEQUENCE_LENGTH - 1)
                          events per host (warmup — no full window available).
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
    # fill NaN with 0 – warmup events are baseline-normal within this component;
    # weight redistribution in compute_anomaly_scores handles the ceiling parity.
    return np.where(np.isnan(result), 0.0, result).astype(np.float32)
