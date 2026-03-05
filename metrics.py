"""
metrics.py
──────────
Comprehensive evaluation of anomaly detection performance on labelled
Sysmon data.

Label convention (sysmondataless.csv)
──────────────────────────────────────
  0 → benign
  1 → confirmed attack
  2 → suspicious / likely malicious

Binary evaluation treats labels {1, 2} as positive (anomalous).

Metrics computed
────────────────
  Event-level (score vs ground-truth label)
    · ROC-AUC
    · PR-AUC  (better for class imbalance)
    · At the F1-optimal threshold:
        Precision / Recall / F1 / Accuracy / MCC
        Confusion matrix  (TP  FP  FN  TN)
        False-Positive Rate, False-Negative Rate
    · At the Youden-J-optimal threshold (sensitivity + specificity)
    · Threshold sweep table  (every 0.05 step)

  Per-component comparison
    · ROC-AUC for recon_error, graph_score, rarity_score, composite

  Per-label score statistics
    · mean / std / median / p95 per class

  Alert-level  (from alert_aggregator output)
    · Alert Precision  – fraction of chains that contain ≥1 malicious event
    · Alert Recall     – fraction of malicious events covered by any chain
    · Mean events per chain
    · Mean max score per chain
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    matthews_corrcoef,
    accuracy_score,
    f1_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    df_scores:  pd.DataFrame,
    df_alerts:  Optional[pd.DataFrame] = None,
    df_events:  Optional[pd.DataFrame] = None,
    report_path: Optional[str] = "evaluation_report.csv",
) -> dict:
    """
    Parameters
    ----------
    df_scores  : DataFrame with columns
                   score, recon_error, graph_score, rarity_score, label
    df_alerts  : DataFrame returned by alert_aggregator.aggregate_alerts()
    df_events  : original event DataFrame (needed for alert-level recall)
    report_path: if set, write the threshold sweep table to this CSV

    Returns
    -------
    metrics : dict of all computed metrics
    """
    labels  = df_scores["label"].values.astype(int)
    scores  = df_scores["score"].values
    binary  = (labels > 0).astype(int)   # {1,2} → positive

    metrics: dict = {}

    print("\n" + "=" * 64)
    print("  ANOMALY DETECTION EVALUATION REPORT")
    print("=" * 64)

    # ── dataset summary ───────────────────────────────────────────────────────
    _section("Dataset")
    n_total  = len(labels)
    n_benign = int((labels == 0).sum())
    n_attack = int((labels == 1).sum())
    n_susp   = int((labels == 2).sum())
    n_pos    = n_attack + n_susp
    metrics.update(dict(n_total=n_total, n_benign=n_benign,
                        n_attack=n_attack, n_suspicious=n_susp,
                        n_positive=n_pos))
    _row("Total events",   n_total)
    _row("Benign  (0)",    f"{n_benign}  ({100*n_benign/n_total:.1f}%)")
    _row("Attack  (1)",    f"{n_attack}  ({100*n_attack/n_total:.1f}%)")
    _row("Suspicious (2)", f"{n_susp}  ({100*n_susp/n_total:.1f}%)")
    _row("Positive total", f"{n_pos}  ({100*n_pos/n_total:.1f}%)")

    # ── score distribution per label ──────────────────────────────────────────
    _section("Score Distribution by Label")
    dist_rows = []
    for lbl, name in [(0, "Benign"), (1, "Attack"), (2, "Suspicious")]:
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        s = scores[mask]
        row = dict(label=lbl, name=name, count=int(mask.sum()),
                   mean=s.mean(), std=s.std(),
                   median=np.median(s), p95=np.percentile(s, 95))
        dist_rows.append(row)
        _row(f"{name:12s}",
             f"mean={s.mean():.4f}  std={s.std():.4f}  "
             f"median={np.median(s):.4f}  p95={np.percentile(s,95):.4f}")
    metrics["score_distribution"] = dist_rows

    # ── ROC-AUC / PR-AUC ─────────────────────────────────────────────────────
    _section("Ranking Metrics (binary: label>0 is positive)")
    try:
        roc_auc = roc_auc_score(binary, scores)
    except Exception:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(binary, scores)
    except Exception:
        pr_auc = float("nan")

    metrics.update(roc_auc=roc_auc, pr_auc=pr_auc)
    _row("ROC-AUC", f"{roc_auc:.4f}")
    _row("PR-AUC (Avg Precision)", f"{pr_auc:.4f}")

    # random-chance baselines
    prev = n_pos / n_total
    _row("Random ROC-AUC (baseline)", "0.5000")
    _row("Random PR-AUC  (baseline)", f"{prev:.4f}")

    # ── per-component AUC ─────────────────────────────────────────────────────
    _section("Per-Component ROC-AUC")
    for col in ["recon_error", "graph_score", "rarity_score", "score"]:
        if col not in df_scores.columns:
            continue
        try:
            auc = roc_auc_score(binary, df_scores[col].values)
        except Exception:
            auc = float("nan")
        label_str = "composite" if col == "score" else col
        metrics[f"roc_auc_{col}"] = auc
        _row(label_str, f"{auc:.4f}")

    # ── threshold sweep ───────────────────────────────────────────────────────
    sweep = _threshold_sweep(binary, scores)
    if report_path:
        pd.DataFrame(sweep).to_csv(report_path, index=False)

    # find best F1 threshold
    best_f1_row = max(sweep, key=lambda r: r["f1"])
    # find best Youden-J threshold
    best_j_row  = max(sweep, key=lambda r: r["youden_j"])

    _section("Threshold Sweep (step 0.05)")
    _table(sweep, ["threshold","precision","recall","f1","accuracy",
                   "fpr","fnr","mcc","tp","fp","fn","tn"])

    # ── metrics at F1-optimal threshold ───────────────────────────────────────
    _section(f"Metrics at F1-Optimal Threshold  ({best_f1_row['threshold']:.2f})")
    for k, v in best_f1_row.items():
        _row(k, f"{v:.4f}" if isinstance(v, float) else v)
    metrics["best_f1_threshold"]  = best_f1_row
    metrics["threshold_f1_opt"]   = best_f1_row["threshold"]

    # ── metrics at Youden-J threshold ─────────────────────────────────────────
    _section(f"Metrics at Youden-J Threshold  ({best_j_row['threshold']:.2f})")
    for k, v in best_j_row.items():
        _row(k, f"{v:.4f}" if isinstance(v, float) else v)
    metrics["best_youden_j_threshold"] = best_j_row
    metrics["threshold_youden_opt"]    = best_j_row["threshold"]

    # ── alert-level metrics ───────────────────────────────────────────────────
    if df_alerts is not None and not df_alerts.empty and df_events is not None:
        _section("Alert-Level Metrics")
        alert_m = _alert_metrics(df_alerts, df_events, labels, scores,
                                  best_f1_row["threshold"])
        metrics["alert_metrics"] = alert_m
        for k, v in alert_m.items():
            _row(k, f"{v:.4f}" if isinstance(v, float) else v)

    # ── detection of confirmed attacks (label=1) ──────────────────────────────
    _section("Detection of Confirmed Attacks (label=1)")
    thr = best_f1_row["threshold"]
    if n_attack > 0:
        attack_mask   = labels == 1
        detected      = (scores[attack_mask] >= thr).sum()
        det_rate      = detected / n_attack
        metrics["attack_detection_rate"]    = det_rate
        metrics["attacks_detected"]         = int(detected)
        metrics["attacks_missed"]           = int(n_attack - detected)
        _row("Attacks detected", f"{detected}/{n_attack}  ({100*det_rate:.1f}%)")
        _row("Attacks missed",   f"{n_attack-detected}/{n_attack}")
        _row("Mean score (attacks)",    f"{scores[attack_mask].mean():.4f}")
        _row("Mean score (benign)",     f"{scores[labels==0].mean():.4f}")
    else:
        _row("No label=1 events in dataset", "")

    if n_susp > 0:
        susp_mask = labels == 2
        detected_s = (scores[susp_mask] >= thr).sum()
        det_s = detected_s / n_susp
        metrics["suspicious_detection_rate"] = det_s
        _row("Suspicious detected", f"{detected_s}/{n_susp}  ({100*det_s:.1f}%)")

    print("\n" + "=" * 64 + "\n")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _threshold_sweep(binary: np.ndarray, scores: np.ndarray) -> list[dict]:
    thresholds = np.arange(0.0, 1.05, 0.05)
    rows = []
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tn, fp, fn, tp = confusion_matrix(binary, pred, labels=[0,1]).ravel()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            acc  = accuracy_score(binary, pred)
            mcc  = matthews_corrcoef(binary, pred) if len(set(binary)) > 1 else 0.0
            yj   = rec + (1 - fpr) - 1       # Youden's J = sensitivity + specificity - 1
        rows.append(dict(
            threshold=round(float(thr), 2),
            precision=round(prec, 4), recall=round(rec, 4),
            f1=round(f1, 4), accuracy=round(acc, 4),
            fpr=round(fpr, 4), fnr=round(fnr, 4),
            mcc=round(float(mcc), 4), youden_j=round(float(yj), 4),
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        ))
    return rows


def _alert_metrics(df_alerts, df_events, labels, scores, threshold) -> dict:
    """
    Alert Precision : % of alert chains that contain ≥1 malicious event
    Alert Recall    : % of malicious events covered by at least one chain
    """
    n_chains = len(df_alerts)
    if "labels" not in df_alerts.columns:
        return dict(n_chains=n_chains)

    # chains that contain any malicious label
    def _has_malicious(label_str):
        parts = [p.strip() for p in str(label_str).split(",")]
        return any(p in {"1", "2"} for p in parts)

    malicious_chains = df_alerts["labels"].apply(_has_malicious).sum()
    alert_precision  = malicious_chains / n_chains if n_chains > 0 else 0.0

    # coverage: which malicious events are above threshold?
    malicious_mask   = labels > 0
    n_malicious      = malicious_mask.sum()
    covered          = (scores[malicious_mask] >= threshold).sum()
    alert_recall     = covered / n_malicious if n_malicious > 0 else 0.0

    return dict(
        n_chains            = n_chains,
        malicious_chains    = int(malicious_chains),
        alert_precision     = float(alert_precision),
        alert_recall        = float(alert_recall),
        mean_events_per_chain = float(df_alerts["num_events"].mean()),
        mean_max_score      = float(df_alerts["max_score"].mean()),
    )


def _section(title: str):
    print(f"\n  ── {title} " + "─" * max(0, 52 - len(title)))


def _row(key: str, value):
    print(f"    {key:<35} {value}")


def _table(rows: list[dict], cols: list[str]):
    header = "  ".join(f"{c:>11}" for c in cols)
    print(f"    {header}")
    print("    " + "-" * len(header))
    for r in rows:
        line = "  ".join(f"{str(r.get(c,''))[:11]:>11}" for c in cols)
        print(f"    {line}")
