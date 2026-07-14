"""
rescore.py
──────────
Re-compute composite scores and evaluation from a saved anomaly_scores.csv
without retraining any model.  Reads raw per-event component scores, applies
the current weights + normalization from anomaly_engine, and runs the full
evaluation pipeline.

Usage:
    python rescore.py                          # uses SCORES_PATH from config
    python rescore.py /path/to/scores.csv      # custom path
"""

import sys

import numpy as np
import pandas as pd

from config import (
    SCORES_PATH, ALERTS_PATH, TRAIN_LABEL,
    ALERT_THRESHOLD, ALERT_PERCENTILE,
    HAS_GROUND_TRUTH,
    ANOMALY_REPORT_PATH, FLAGGED_EVENTS_PATH,
    DATA_PATH, HIGH_SIGNAL_EVENTIDS,
)
from anomaly_engine import compute_anomaly_scores
from alert_aggregator import aggregate_alerts
from metrics import evaluate
from anomaly_report import generate_investigation_report


def main():
    scores_path = sys.argv[1] if len(sys.argv) > 1 else SCORES_PATH

    print(f"[1/4] Loading scores from {scores_path} ...")
    df_scores = pd.read_csv(scores_path)
    print(f"      {len(df_scores):,} events")

    print(f"\n[2/4] Loading event data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    if "Label" not in df.columns:
        df["Label"] = 0
    df["EventID"] = pd.to_numeric(df["EventID"], errors="coerce")
    if HIGH_SIGNAL_EVENTIDS is not None:
        df = df[df["EventID"].isin(HIGH_SIGNAL_EVENTIDS)].reset_index(drop=True)
        print(f"      {len(df):,} events after EventID filter")
    else:
        print(f"      {len(df):,} events (no EventID filter)")

    print("\n[3/4] Re-computing composite scores with current weights...")
    dense_errors = df_scores["dense_error"].values if "dense_error" in df_scores.columns else None

    composite = compute_anomaly_scores(
        df_scores["recon_error"].values,
        df_scores["graph_score"].values,
        df_scores["rarity_score"].values,
        df_scores["label"].values,
        dense_errors=dense_errors,
    )

    df_scores["score"] = composite

    print("\n  Score summary by label:")
    cols = ["score"]
    if "dense_error" in df_scores.columns:
        cols.append("dense_error")
    cols += ["recon_error", "graph_score", "rarity_score"]
    print(df_scores.groupby("label")[cols].mean().to_string())

    # threshold
    if ALERT_THRESHOLD is not None:
        threshold = ALERT_THRESHOLD
        print(f"\n  Alert threshold (fixed)     : {threshold:.6f}")
    else:
        benign_scores = composite[df_scores["label"].values == TRAIN_LABEL]
        threshold = float(np.percentile(benign_scores, ALERT_PERCENTILE))
        print(f"\n  Alert threshold (p{ALERT_PERCENTILE} benign): {threshold:.6f}")

    print("\n[4/4] Alerts & evaluation...")
    alerts = aggregate_alerts(df, composite, threshold=threshold)
    alerts.to_csv(ALERTS_PATH, index=False)

    generate_investigation_report(
        df_scores=df_scores, df_events=df, df_alerts=alerts,
        report_path=ANOMALY_REPORT_PATH, csv_path=FLAGGED_EVENTS_PATH,
        threshold=threshold,
    )

    if not HAS_GROUND_TRUTH:
        print(f"\n  HAS_GROUND_TRUTH=False — AUC/F1 metrics skipped.")
        return

    if df_scores["label"].nunique() <= 1:
        print("\n  No anomalous labels — skipping evaluation.")
        return

    metrics = evaluate(
        df_scores=df_scores, df_alerts=alerts, df_events=df,
        report_path="threshold_sweep.csv",
    )
    print(f"\n  ROC-AUC : {metrics.get('roc_auc', float('nan')):.4f}")
    print(f"  PR-AUC  : {metrics.get('pr_auc',  float('nan')):.4f}")
    best = metrics.get("best_f1_threshold", {})
    print(f"  Best F1 : {best.get('f1', 0):.4f}  "
          f"@ threshold={best.get('threshold', '?')}"
          f"  P={best.get('precision', 0):.4f}  R={best.get('recall', 0):.4f}")


if __name__ == "__main__":
    main()
