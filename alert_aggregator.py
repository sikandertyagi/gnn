"""
alert_aggregator.py
───────────────────
Converts per-event anomaly scores into human-readable alerts.

Algorithm
─────────
1. Flag every event whose composite score >= ALERT_THRESHOLD.
2. Sort flagged events by SystemTime.
3. Group consecutive events whose gap is <= ALERT_WINDOW seconds into one
   "attack chain".
4. Summarise each chain (processes involved, destination IPs, score stats).
"""

import numpy as np
import pandas as pd

from config import ALERT_THRESHOLD, ALERT_WINDOW


def aggregate_alerts(
    df:         pd.DataFrame,
    scores:     np.ndarray,
    threshold:  float = ALERT_THRESHOLD,
    window_sec: int   = ALERT_WINDOW,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    df        : original event DataFrame (must include SystemTime, Image,
                DestinationIp columns)
    scores    : (N,) composite anomaly scores aligned to *df* rows
    threshold : minimum score to flag an event
    window_sec: max gap in seconds to group events into one chain

    Returns
    -------
    DataFrame of alert chains with columns:
        chain_id, start_time, end_time, duration_s, num_events,
        max_score, mean_score, processes, dest_ips, labels
    """
    result            = df.copy().reset_index(drop=True)
    result["_score"]  = scores
    result["_alert"]  = scores >= threshold

    flagged = result[result["_alert"]].copy()

    if flagged.empty:
        print(f"  No events exceeded threshold {threshold:.2f}")
        return pd.DataFrame(columns=[
            "chain_id", "start_time", "end_time", "duration_s",
            "num_events", "max_score", "mean_score",
            "processes", "dest_ips", "labels",
        ])

    # Fix #8: drop rows with unparseable/NaN SystemTime before sorting
    flagged = flagged[flagged["SystemTime"].notna()].copy()
    if flagged.empty:
        print(f"  No events with valid SystemTime exceeded threshold {threshold:.2f}")
        return pd.DataFrame(columns=[
            "chain_id", "start_time", "end_time", "duration_s",
            "num_events", "max_score", "mean_score",
            "processes", "dest_ips", "labels",
        ])

    flagged = flagged.sort_values("SystemTime").reset_index(drop=True)

    chains:     list = []
    chain_rows: list = [flagged.iloc[0]]

    for i in range(1, len(flagged)):
        row  = flagged.iloc[i]
        prev = chain_rows[-1]
        try:
            gap = (
                pd.Timestamp(row["SystemTime"]) -
                pd.Timestamp(prev["SystemTime"])
            ).total_seconds()
        except Exception:
            gap = 0.0

        if gap <= window_sec:
            chain_rows.append(row)
        else:
            chains.append(_summarise(len(chains), chain_rows))
            chain_rows = [row]

    chains.append(_summarise(len(chains), chain_rows))

    alerts_df = pd.DataFrame(chains)
    print(f"  {len(alerts_df)} alert chain(s) found "
          f"({len(flagged)} flagged events, threshold={threshold:.2f})")
    return alerts_df


# ─────────────────────────────────────────────────────────────────────────────

def _summarise(chain_id: int, rows: list) -> dict:
    scores = [float(r["_score"]) for r in rows]

    processes = list(dict.fromkeys(
        str(r.get("Image", "unknown")).split("\\")[-1].lower()
        for r in rows
    ))

    dest_ips = list(dict.fromkeys(
        str(r.get("DestinationIp", ""))
        for r in rows
        if pd.notna(r.get("DestinationIp")) and str(r.get("DestinationIp")) != ""
    ))

    labels = list(dict.fromkeys(
        str(int(r["Label"])) for r in rows if pd.notna(r.get("Label"))
    ))

    try:
        start = pd.Timestamp(rows[0]["SystemTime"])
        end   = pd.Timestamp(rows[-1]["SystemTime"])
        dur   = (end - start).total_seconds()
    except Exception:
        start, end, dur = rows[0].get("SystemTime"), rows[-1].get("SystemTime"), 0.0

    return {
        "chain_id":   chain_id,
        "start_time": start,
        "end_time":   end,
        "duration_s": dur,
        "num_events": len(rows),
        "max_score":  max(scores),
        "mean_score": float(np.mean(scores)),
        "processes":  " -> ".join(processes),
        "dest_ips":   ", ".join(dest_ips) if dest_ips else "none",
        "labels":     ", ".join(labels),
    }
