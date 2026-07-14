"""
anomaly_engine.py
─────────────────
Combines four anomaly signals into a composite event score.

    event_score = DENSE_WEIGHT  * dense_error          (single-event dense AE)
               + RECON_WEIGHT  * reconstruction_error   (sequence transformer)
               + GRAPH_WEIGHT  * graph_score             (GNN, disabled)
               + RARITY_WEIGHT * rarity_score            (rare behaviour)

Current weights: DENSE=0.50, RECON=0.20, GRAPH=0.00, RARITY=0.30 (sum=1.0).

All input arrays must be aligned to the same event index.
Each component is min-max normalised before weighting.

Weight redistribution for warmup events
────────────────────────────────────────
The first (SEQUENCE_LENGTH - 1) events per host have no full sliding window
and therefore receive NaN recon_error (warmup period).

For warmup events, RECON_WEIGHT is redistributed proportionally among the
remaining active signals so the composite still spans [0, 1]:

    scored   → DENSE·d + RECON·r + GRAPH·g + RARITY·s
    warmup   → (DENSE / remain)·d + (GRAPH / remain)·g + (RARITY / remain)·s
               where remain = DENSE + GRAPH + RARITY
"""

import numpy as np

from config import DENSE_WEIGHT, RECON_WEIGHT, GRAPH_WEIGHT, RARITY_WEIGHT, TRAIN_LABEL


def compute_anomaly_scores(
    recon_errors:  np.ndarray,
    graph_scores:  np.ndarray,
    rarity_scores: np.ndarray,
    labels:        np.ndarray | None = None,
    dense_errors:  np.ndarray | None = None,
) -> np.ndarray:
    """
    Parameters
    ----------
    recon_errors  : (N,)  MSE reconstruction error per sequence (transformer)
                          May contain NaN for warmup events.
    graph_scores  : (N,)  GNN reconstruction error per event
    rarity_scores : (N,)  rare-behaviour score per event
    labels        : (N,)  event labels (0 = benign).  When provided, min-max
                          normalisation is fitted on benign rows only.
    dense_errors  : (N,)  dense AE reconstruction error per event (optional —
                          None falls back to the three-signal composite)

    Returns
    -------
    scores : (N,)  composite anomaly score in [0, 1] for all events
    """
    benign_mask = (labels == TRAIN_LABEL) if labels is not None else None

    has_recon = ~np.isnan(recon_errors)

    r = _normalise(recon_errors,  benign_mask)
    g = _normalise(graph_scores,  benign_mask)
    s = _normalise(rarity_scores, benign_mask)

    if dense_errors is not None:
        d = _normalise(dense_errors, benign_mask)

        scored_composite = (
            DENSE_WEIGHT * d + RECON_WEIGHT * r +
            GRAPH_WEIGHT * g + RARITY_WEIGHT * s
        )

        remain = DENSE_WEIGHT + GRAPH_WEIGHT + RARITY_WEIGHT
        if remain > 0:
            unscored_composite = (
                (DENSE_WEIGHT / remain) * d +
                (GRAPH_WEIGHT / remain) * g +
                (RARITY_WEIGHT / remain) * s
            )
        else:
            unscored_composite = np.zeros_like(r)
    else:
        # legacy three-signal path (no dense AE)
        w_recon  = RECON_WEIGHT + DENSE_WEIGHT   # absorb dense weight into recon
        w_rarity = RARITY_WEIGHT
        scored_composite = w_recon * r + GRAPH_WEIGHT * g + w_rarity * s

        remain = GRAPH_WEIGHT + w_rarity
        if remain > 0:
            unscored_composite = (GRAPH_WEIGHT / remain) * g + (w_rarity / remain) * s
        else:
            unscored_composite = np.zeros_like(r)

    return np.where(has_recon, scored_composite, unscored_composite).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────

def _normalise(arr: np.ndarray,
               benign_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Min-max normalise, fitting the range on benign rows only when a mask is
    supplied.  NaN entries are filled with 0 after normalisation.
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
    return np.where(np.isnan(result), 0.0, result).astype(np.float32)
