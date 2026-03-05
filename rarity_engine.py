"""
rarity_engine.py
────────────────
Tracks baseline frequency of behavioural patterns observed in benign events
and computes a rarity score in [0, 1] for new events.

Tracked frequencies
───────────────────
  parent_child_freq    : (parent_process_name, child_process_name)
  proc_ip_freq         : (process_name, destination_ip)
  network_dest_freq    : destination_ip

All counters are fitted on label=0 (benign) data only.
"""

from collections import Counter

import numpy as np
import pandas as pd


class RarityEngine:

    def __init__(self):
        self.parent_child_freq:  Counter = Counter()
        self.proc_ip_freq:       Counter = Counter()
        self.network_dest_freq:  Counter = Counter()

        self._total_pc: int = 1
        self._total_pi: int = 1
        self._total_nd: int = 1

    # ── fitting ───────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "RarityEngine":
        """Fit counters on benign (Label == 0) rows of *df*."""
        benign = df[df["Label"] == 0]

        # parent → child process pairs
        pc = benign[["ParentImage", "Image"]].dropna()
        for _, row in pc.iterrows():
            parent = _proc_name(row["ParentImage"])
            child  = _proc_name(row["Image"])
            self.parent_child_freq[(parent, child)] += 1
        self._total_pc = max(sum(self.parent_child_freq.values()), 1)

        # process → destination IP
        pi = benign[["Image", "DestinationIp"]].dropna()
        for _, row in pi.iterrows():
            proc = _proc_name(row["Image"])
            ip   = str(row["DestinationIp"])
            self.proc_ip_freq[(proc, ip)] += 1
        self._total_pi = max(sum(self.proc_ip_freq.values()), 1)

        # network destination frequency
        nd = benign["DestinationIp"].dropna()
        for ip in nd:
            self.network_dest_freq[str(ip)] += 1
        self._total_nd = max(sum(self.network_dest_freq.values()), 1)

        return self

    # ── scoring ───────────────────────────────────────────────────────────────

    def score_row(self, row: pd.Series) -> float:
        """Return rarity score in [0, 1] for a single event row."""
        signals: list = []

        if pd.notna(row.get("ParentImage")) and pd.notna(row.get("Image")):
            parent = _proc_name(row["ParentImage"])
            child  = _proc_name(row["Image"])
            freq   = self.parent_child_freq.get((parent, child), 0)
            vocab  = len(self.parent_child_freq)
            p      = (freq + 1) / (self._total_pc + vocab + 1)
            signals.append(1.0 - p)

        if pd.notna(row.get("Image")) and pd.notna(row.get("DestinationIp")):
            proc = _proc_name(row["Image"])
            ip   = str(row["DestinationIp"])
            freq = self.proc_ip_freq.get((proc, ip), 0)
            vocab = len(self.proc_ip_freq)
            p    = (freq + 1) / (self._total_pi + vocab + 1)
            signals.append(1.0 - p)

        if pd.notna(row.get("DestinationIp")):
            ip   = str(row["DestinationIp"])
            freq = self.network_dest_freq.get(ip, 0)
            vocab = len(self.network_dest_freq)
            p    = (freq + 1) / (self._total_nd + vocab + 1)
            signals.append(1.0 - p)

        return float(np.mean(signals)) if signals else 0.0

    def score_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorised scoring over a DataFrame; returns ndarray of shape (N,)."""
        return np.array([self.score_row(row) for _, row in df.iterrows()])


# ─────────────────────────────────────────────────────────────────────────────

def _proc_name(path) -> str:
    return str(path).split("\\")[-1].lower()
