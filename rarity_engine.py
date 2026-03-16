"""
rarity_engine.py
────────────────
Tracks baseline frequency of behavioural patterns from benign events and
computes per-event rarity scores in [0, 1].

Tracked patterns
────────────────
  parent_child : (parent_proc_name, child_proc_name)
  proc_ip      : (process_name, destination_ip)
  network_dest : destination_ip

Scalability
───────────
  fit()            – pandas groupby, O(N) vectorised (no iterrows)
  score_dataframe() – pandas merge, O(N) vectorised (no iterrows)
  score_row()      – kept for single-event use (e.g. streaming)
"""

import re

import numpy as np
import pandas as pd


_MISSING = "__MISSING__"          # sentinel for NaN join keys


class RarityEngine:

    def __init__(self):
        # DataFrames used for vectorised merge-based scoring
        self._pc_counts: pd.DataFrame = pd.DataFrame()   # _parent, _child, count
        self._pi_counts: pd.DataFrame = pd.DataFrame()   # _proc,  DestinationIp, count
        self._nd_counts: pd.DataFrame = pd.DataFrame()   # DestinationIp, count

        self._total_pc: int = 1
        self._total_pi: int = 1
        self._total_nd: int = 1
        self._vocab_pc: int = 0
        self._vocab_pi: int = 0
        self._vocab_nd: int = 0

    # ── fitting (vectorised) ──────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "RarityEngine":
        """Fit on benign (Label == 0) rows using vectorised pandas ops."""
        benign = df[df["Label"] == 0]

        # parent → child process pairs
        pc = benign[["ParentImage", "Image"]].dropna().copy()
        pc["_parent"] = _extract_name(pc["ParentImage"])
        pc["_child"]  = _extract_name(pc["Image"])
        self._pc_counts = (
            pc.groupby(["_parent", "_child"], sort=False)
            .size()
            .reset_index(name="count")
        )
        self._total_pc = max(int(self._pc_counts["count"].sum()), 1)
        self._vocab_pc = len(self._pc_counts)

        # process → destination IP
        pi = benign[["Image", "DestinationIp"]].dropna().copy()
        pi["_proc"] = _extract_name(pi["Image"])
        pi["DestinationIp"] = pi["DestinationIp"].astype(str)
        self._pi_counts = (
            pi.groupby(["_proc", "DestinationIp"], sort=False)
            .size()
            .reset_index(name="count")
        )
        self._total_pi = max(int(self._pi_counts["count"].sum()), 1)
        self._vocab_pi = len(self._pi_counts)

        # network destination
        nd = benign["DestinationIp"].dropna().astype(str)
        _vc = nd.value_counts().reset_index()
        # pandas <2.0 reset_index gives ["index", "DestinationIp"]; ≥2.0 gives ["DestinationIp", "count"]
        if "index" in _vc.columns:
            _vc = _vc.rename(columns={"index": "DestinationIp", "DestinationIp": "count"})
        self._nd_counts = _vc
        self._total_nd = max(int(self._nd_counts["count"].sum()), 1)
        self._vocab_nd = len(self._nd_counts)

        return self

    # ── scoring (vectorised) ──────────────────────────────────────────────────

    def score_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        Compute rarity score for every row in *df*.
        Returns ndarray of shape (N,) in [0, 1].
        Fully vectorised – safe for millions of rows.
        """
        N      = len(df)
        scores = np.full((N, 3), np.nan, dtype=np.float64)
        idx    = df.index  # preserve original index for alignment

        # ── parent-child rarity ───────────────────────────────────────────────
        pc_valid = df["ParentImage"].notna() & df["Image"].notna()
        if pc_valid.any() and not self._pc_counts.empty:
            tmp = pd.DataFrame({
                "_parent": _extract_name(df["ParentImage"].fillna(_MISSING)),
                "_child":  _extract_name(df["Image"].fillna(_MISSING)),
            }, index=idx)
            tmp = tmp.merge(self._pc_counts, on=["_parent", "_child"], how="left")
            freq = tmp["count"].fillna(0).values
            # Fix #9: Jeffreys (alpha=0.5) smoothing for better stability on small datasets
            p    = (freq + 0.5) / (self._total_pc + 0.5 * self._vocab_pc)
            scores[:, 0] = np.where(pc_valid.values, 1.0 - p, np.nan)

        # ── process-IP rarity ─────────────────────────────────────────────────
        pi_valid = df["Image"].notna() & df["DestinationIp"].notna()
        if pi_valid.any() and not self._pi_counts.empty:
            tmp = pd.DataFrame({
                "_proc":        _extract_name(df["Image"].fillna(_MISSING)),
                "DestinationIp": df["DestinationIp"].fillna(_MISSING).astype(str),
            }, index=idx)
            tmp = tmp.merge(self._pi_counts, on=["_proc", "DestinationIp"], how="left")
            freq = tmp["count"].fillna(0).values
            p    = (freq + 0.5) / (self._total_pi + 0.5 * self._vocab_pi)
            scores[:, 1] = np.where(pi_valid.values, 1.0 - p, np.nan)

        # ── network-destination rarity ────────────────────────────────────────
        nd_valid = df["DestinationIp"].notna()
        if nd_valid.any() and not self._nd_counts.empty:
            tmp = pd.DataFrame({
                "DestinationIp": df["DestinationIp"].fillna(_MISSING).astype(str),
            }, index=idx)
            tmp = tmp.merge(self._nd_counts, on="DestinationIp", how="left")
            freq = tmp["count"].fillna(0).values
            p    = (freq + 0.5) / (self._total_nd + 0.5 * self._vocab_nd)
            scores[:, 2] = np.where(nd_valid.values, 1.0 - p, np.nan)

        # mean across whichever signals are valid per row; default 0 if none
        with np.errstate(all="ignore"):
            result = np.nanmean(scores, axis=1)
        result = np.where(np.isnan(result), 0.0, result)
        return result.astype(np.float32)

    # ── single-row scoring (kept for streaming / debugging) ───────────────────

    def score_row(self, row: pd.Series) -> float:
        """Score a single event row. Prefer score_dataframe() for bulk use."""
        return float(self.score_dataframe(row.to_frame().T)[0])


# ─────────────────────────────────────────────────────────────────────────────

def _extract_name(series: pd.Series) -> pd.Series:
    """Vectorised basename extraction for Windows and Linux paths."""
    return series.astype(str).str.split(r"[/\\]").str[-1].str.lower()
