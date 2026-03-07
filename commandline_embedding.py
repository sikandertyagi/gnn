"""
commandline_embedding.py
────────────────────────
Semantic command-line embeddings using SentenceTransformers.

Pipeline
────────
  1. Encode each CommandLine string with all-MiniLM-L6-v2 → 384-d float32
  2. Reduce to 32 dimensions with PCA (fitted once, saved to disk)

Caching strategy
────────────────
  Raw 384-d embeddings are cached keyed by a SHA-256 hash of the input
  command strings (order-sensitive).  On repeated runs (e.g. hyperparameter
  sweeps on the same dataset) both the SentenceTransformer encoding and the
  PCA reduction are skipped entirely.

  PCA model is stored separately at CMD_EMBED_PCA_PATH (default cmd_pca.pkl).
  If it does not exist when embed_commandlines() is called it is fitted on the
  current batch and saved.  Subsequent calls reuse the fitted model so the
  projection is stable across train / inference runs.

Determinism
───────────
  · SentenceTransformer encoding is deterministic for fixed model weights
    and fixed input order.
  · PCA uses random_state=42 (sklearn default reproducibility).
  · SHA-256 cache key guarantees cache hits only when the exact same
    command strings (in the same order) are presented.
"""

import hashlib
import os

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from config import CMD_EMBED_MODEL, CMD_EMBED_N_COMPONENTS, CMD_EMBED_BATCH_SIZE, \
                   CMD_EMBED_CACHE_DIR, CMD_EMBED_PCA_PATH


# ── public API ──────────────────────────────────────────────────────────────────

def embed_commandlines(
    cmd_series: pd.Series,
    pca_path:   str = CMD_EMBED_PCA_PATH,
    cache_dir:  str = CMD_EMBED_CACHE_DIR,
) -> np.ndarray:
    """
    Parameters
    ----------
    cmd_series : pd.Series of raw CommandLine strings (length N)
    pca_path   : path to save / load the fitted PCA model
    cache_dir  : directory for SHA-256 keyed raw-embedding cache files

    Returns
    -------
    embeddings : ndarray (N, CMD_EMBED_N_COMPONENTS)  float32
                 PCA-projected semantic embeddings; one row per command line
    """
    os.makedirs(cache_dir, exist_ok=True)

    key        = _cache_key(cmd_series)
    cache_file = os.path.join(cache_dir, f"{key}.npy")

    # ── load raw embeddings from cache if available ─────────────────────────
    if os.path.exists(cache_file):
        raw = np.load(cache_file)                    # (N, 384)
    else:
        raw = _encode(cmd_series)
        np.save(cache_file, raw)

    # ── fit PCA once; reuse on subsequent calls ─────────────────────────────
    if os.path.exists(pca_path):
        pca = joblib.load(pca_path)
    else:
        pca = PCA(n_components=CMD_EMBED_N_COMPONENTS, random_state=42)
        pca.fit(raw)
        joblib.dump(pca, pca_path)

    return pca.transform(raw).astype(np.float32)     # (N, 32)


# ── internal helpers ────────────────────────────────────────────────────────────

def _cache_key(cmd_series: pd.Series) -> str:
    """SHA-256 of all command strings concatenated in order."""
    h = hashlib.sha256()
    for s in cmd_series.fillna("").astype(str):
        h.update(s.encode("utf-8", errors="replace"))
    return h.hexdigest()


def _encode(cmd_series: pd.Series) -> np.ndarray:
    """Encode command strings to 384-d float32 vectors via SentenceTransformer."""
    model = SentenceTransformer(CMD_EMBED_MODEL)
    texts = cmd_series.fillna("").astype(str).tolist()
    return model.encode(
        texts,
        batch_size=CMD_EMBED_BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype(np.float32)                             # (N, 384)
