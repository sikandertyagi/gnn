"""
process_chain_builder.py
────────────────────────
Builds process ancestry chains from Sysmon EventID 1 (process creation)
events and trains Word2Vec embeddings on the resulting chains.

Process chain extraction
────────────────────────
  For each process-create event, walk backwards from the current process to
  its root ancestor using ProcessGuid → ParentProcessGuid lookups:

      explorer.exe → cmd.exe → powershell.exe → rundll32.exe

  Chains are capped at max_depth=4 (configurable) and always ordered
  root-first.  Missing parents are handled gracefully: the walk stops at
  the first unknown GUID.

Word2Vec embeddings
───────────────────
  Process chains are treated as "sentences" for gensim Word2Vec.  Each
  executable name is a "word".  The trained model maps every process name
  to a dense vector (default 32-d) that captures co-occurrence context
  in parent-child trees.

  For each event, the chain embedding is the mean of its constituent
  process-name vectors.  Unknown process names (below min_count) fall
  back to a zero vector.

Performance
───────────
  · Chain extraction is O(n) — single pass to build the GUID→parent dict,
    then O(max_depth) per event.
  · Word2Vec training is handled by gensim's C-optimised backend.
  · Embeddings are cached in a dict for O(1) per-name lookup.
"""

import os
import re
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from config import (
    CHAIN_EMBED_DIM,
    CHAIN_MAX_DEPTH,
    CHAIN_W2V_WINDOW,
    CHAIN_W2V_MIN_COUNT,
    CHAIN_W2V_MODEL_PATH,
)


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Build process chains
# ─────────────────────────────────────────────────────────────────────────────

def _extract_basename(path) -> str:
    """Lowercase executable name from a Windows or Linux image path."""
    if pd.isna(path) or not str(path).strip():
        return "unknown"
    return re.split(r"[/\\]", str(path).strip())[-1].lower()


def build_process_chains(
    df: pd.DataFrame,
    max_depth: int = CHAIN_MAX_DEPTH,
) -> list[list[str]]:
    """
    Build process ancestry chains from Sysmon process-create events.

    Parameters
    ----------
    df        : DataFrame with ProcessGuid, ParentProcessGuid, Image,
                ParentImage columns (and EventID to filter on type 1).
    max_depth : maximum number of ancestors to walk (including self).

    Returns
    -------
    chains : list[list[str]] of length len(df).  Each entry is a list of
             executable basenames ordered root-first → current-process-last.
             Non-EventID-1 rows get a single-element chain from their Image.
    """
    # ── build GUID → (parent_guid, process_name) lookup from EventID 1 rows ──
    guid_col   = "ProcessGuid"  if "ProcessGuid"  in df.columns else None
    parent_col = "ParentProcessGuid" if "ParentProcessGuid" in df.columns else None

    if guid_col is None or parent_col is None:
        # No GUID columns → fallback: single-element chains
        return [[_extract_basename(img)] for img in df["Image"]]

    # Dict: guid → (parent_guid, exe_name)  — O(n) build
    guid_to_parent: dict[str, str] = {}
    guid_to_name:   dict[str, str] = {}

    eid = pd.to_numeric(df["EventID"], errors="coerce").fillna(0).astype(int)
    mask_eid1 = eid == 1

    for guid, pguid, img in zip(
        df.loc[mask_eid1, guid_col],
        df.loc[mask_eid1, parent_col],
        df.loc[mask_eid1, "Image"],
    ):
        g = str(guid).strip()
        guid_to_parent[g] = str(pguid).strip() if pd.notna(pguid) else ""
        guid_to_name[g]   = _extract_basename(img)

    # Also populate parent names from ParentImage where available
    for guid, pguid, pimg in zip(
        df.loc[mask_eid1, guid_col],
        df.loc[mask_eid1, parent_col],
        df.loc[mask_eid1, "ParentImage"],
    ):
        pg = str(pguid).strip()
        if pg and pg not in guid_to_name and pd.notna(pimg):
            guid_to_name[pg] = _extract_basename(pimg)

    # ── walk chains for every row — O(max_depth) per row ─────────────────────
    chains: list[list[str]] = []

    all_guids   = df[guid_col].values
    all_pguids  = df[parent_col].values
    all_images  = df["Image"].values
    all_pimages = df["ParentImage"].values

    for i in range(len(df)):
        current_name = _extract_basename(all_images[i])
        current_guid = str(all_guids[i]).strip() if pd.notna(all_guids[i]) else ""

        # Walk ancestors
        ancestors: list[str] = []
        visited: set[str] = set()
        g = current_guid

        # First, collect the current process
        # Then walk up the parent chain
        walk_guid = guid_to_parent.get(g, "")
        depth = 1  # current process counts as depth 1

        while walk_guid and walk_guid not in visited and depth < max_depth:
            visited.add(walk_guid)
            name = guid_to_name.get(walk_guid, "")
            if not name:
                break
            ancestors.append(name)
            walk_guid = guid_to_parent.get(walk_guid, "")
            depth += 1

        # ancestors is child→root order; reverse to root→child, then append current
        ancestors.reverse()
        ancestors.append(current_name)
        chains.append(ancestors)

    return chains


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Train Word2Vec on process chains
# ─────────────────────────────────────────────────────────────────────────────

def train_chain_w2v(
    chains: list[list[str]],
    model_path: str = CHAIN_W2V_MODEL_PATH,
) -> Word2Vec:
    """
    Train Word2Vec on process chains and save the model.

    If a saved model already exists at *model_path*, it is loaded instead
    of retraining (same as PCA caching in commandline_embedding.py).

    Parameters
    ----------
    chains     : list of process-name chains (each chain is a "sentence")
    model_path : path to save / load the trained Word2Vec model

    Returns
    -------
    model : trained gensim Word2Vec model
    """
    if os.path.exists(model_path):
        return Word2Vec.load(model_path)

    model = Word2Vec(
        sentences=chains,
        vector_size=CHAIN_EMBED_DIM,
        window=CHAIN_W2V_WINDOW,
        min_count=CHAIN_W2V_MIN_COUNT,
        workers=max(1, cpu_count()),
        seed=42,
        epochs=10,
    )
    model.save(model_path)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Convert chains to event-level feature vectors
# ─────────────────────────────────────────────────────────────────────────────

def chains_to_embeddings(
    chains: list[list[str]],
    w2v_model: Word2Vec,
) -> np.ndarray:
    """
    Convert process chains to fixed-size embeddings via mean-pooling.

    For each chain, look up the Word2Vec vector for every process name in
    the chain and return their element-wise mean.  Process names not in the
    vocabulary (below min_count during training) contribute a zero vector.

    Parameters
    ----------
    chains    : list of process-name chains (length N)
    w2v_model : trained Word2Vec model

    Returns
    -------
    embeddings : ndarray (N, CHAIN_EMBED_DIM) float32
    """
    dim = w2v_model.wv.vector_size
    wv  = w2v_model.wv

    # Pre-cache all known vectors for O(1) lookup
    vocab_vecs: dict[str, np.ndarray] = {}
    for word in wv.key_to_index:
        vocab_vecs[word] = wv[word]

    zero = np.zeros(dim, dtype=np.float32)
    embeddings = np.empty((len(chains), dim), dtype=np.float32)

    for i, chain in enumerate(chains):
        vecs = [vocab_vecs.get(name, zero) for name in chain]
        if vecs:
            embeddings[i] = np.mean(vecs, axis=0)
        else:
            embeddings[i] = zero

    return embeddings
