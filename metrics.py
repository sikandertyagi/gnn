"""
metrics.py
──────────
Comprehensive evaluation of anomaly detection performance on labelled
Sysmon data.

Label convention
────────────────
  0 → benign
  1 → confirmed attack
  2 → suspicious / likely malicious

Binary evaluation treats labels {1, 2} as positive (anomalous).

Metrics computed
────────────────
  Event-level (score vs ground-truth label)
    · ROC-AUC
    · PR-AUC  (preferable for class-imbalanced datasets)
    · At the F1-optimal threshold:
        Precision / Recall / F1 / Accuracy / MCC / Cohen's Kappa / G-Mean
        Confusion matrix  (TP  FP  FN  TN)
        FPR / FNR
    · At the Youden-J-optimal threshold
    · TPR at fixed FPR: 0.1%, 0.5%, 1%, 5%, 10%
      (standard in security / IDS papers for operational trade-off tables)
    · Fine threshold sweep (step 0.01)

  Ablation / per-component comparison
    · Individual ROC-AUC for recon_error, graph_score, rarity_score, composite

  Per-label score statistics
    · mean / std / median / p95 per class

  Alert-level
    · Alert Precision  – fraction of chains containing >=1 malicious event
    · Alert Recall     – fraction of malicious events covered by any chain

Research-paper artefacts saved to disk
────────────────────────────────────────
  roc_curve.csv            fpr, tpr, threshold    -> Figure: ROC curve
  pr_curve.csv             precision, recall, threshold  -> Figure: PR curve
  score_distributions.csv  per-class score stats
  threshold_sweep.csv      full sweep table  -> supplementary table
  metrics.json             all scalar metrics -> paper Table 1
"""

from __future__ import annotations

import json
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
    cohen_kappa_score,
)

from config import (
    METRICS_JSON_PATH, ROC_CURVE_PATH,
    PR_CURVE_PATH, SCORE_DIST_PATH,
)


# ─────────────────────────────────────────────────────────────────────────────
# public API
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    df_scores:  pd.DataFrame,
    df_alerts:  Optional[pd.DataFrame] = None,
    df_events:  Optional[pd.DataFrame] = None,
    report_path: Optional[str] = "threshold_sweep.csv",
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
    metrics : dict of all computed metrics (JSON-serialisable)
    """
    labels = df_scores["label"].values.astype(int)
    scores = df_scores["score"].values.astype(np.float64)
    binary = (labels > 0).astype(int)

    metrics: dict = {}

    print("\n" + "=" * 68)
    print("  ANOMALY DETECTION EVALUATION REPORT")
    print("=" * 68)

    # ── dataset summary ───────────────────────────────────────────────────────
    _section("Dataset")
    n_total  = len(labels)
    n_benign = int((labels == 0).sum())
    n_attack = int((labels == 1).sum())
    n_susp   = int((labels == 2).sum())
    n_pos    = n_attack + n_susp
    metrics.update(n_total=n_total, n_benign=n_benign,
                   n_attack=n_attack, n_suspicious=n_susp, n_positive=n_pos)
    _row("Total events",    n_total)
    _row("Benign   (0)",    f"{n_benign}  ({100*n_benign/n_total:.1f}%)")
    _row("Attack   (1)",    f"{n_attack}  ({100*n_attack/n_total:.1f}%)")
    _row("Suspicious (2)",  f"{n_susp}  ({100*n_susp/n_total:.1f}%)")
    _row("Positive total",  f"{n_pos}  ({100*n_pos/n_total:.1f}%)")

    # ── score distribution per label ──────────────────────────────────────────
    _section("Score Distribution by Label")
    dist_rows = []
    for lbl, name in [(0, "Benign"), (1, "Attack"), (2, "Suspicious")]:
        mask = labels == lbl
        if not mask.any():
            continue
        s = scores[mask]
        row = dict(label=lbl, name=name, count=int(mask.sum()),
                   mean=round(float(s.mean()), 6),
                   std=round(float(s.std()), 6),
                   median=round(float(np.median(s)), 6),
                   p25=round(float(np.percentile(s, 25)), 6),
                   p75=round(float(np.percentile(s, 75)), 6),
                   p95=round(float(np.percentile(s, 95)), 6),
                   p99=round(float(np.percentile(s, 99)), 6))
        dist_rows.append(row)
        _row(f"{name:12s}",
             f"mean={s.mean():.4f}  std={s.std():.4f}  "
             f"median={np.median(s):.4f}  p95={np.percentile(s,95):.4f}")
    metrics["score_distribution"] = dist_rows
    pd.DataFrame(dist_rows).to_csv(SCORE_DIST_PATH, index=False)

    # ── ROC-AUC / PR-AUC ─────────────────────────────────────────────────────
    _section("Ranking Metrics  (binary: label > 0 = positive)")
    roc_auc = _safe_auc(roc_auc_score, binary, scores)
    pr_auc  = _safe_auc(average_precision_score, binary, scores)
    metrics.update(roc_auc=roc_auc, pr_auc=pr_auc)
    _row("ROC-AUC",                 f"{roc_auc:.4f}")
    _row("PR-AUC (Avg Precision)",  f"{pr_auc:.4f}")
    _row("Random ROC-AUC baseline", "0.5000")
    _row("Random PR-AUC  baseline", f"{n_pos/n_total:.4f}")

    # ── save ROC / PR curves for paper figures ────────────────────────────────
    _save_roc_curve(binary, scores, ROC_CURVE_PATH)
    _save_pr_curve(binary,  scores, PR_CURVE_PATH)

    # ── per-component AUC (ablation table) ────────────────────────────────────
    _section("Per-Component ROC-AUC  (ablation)")
    for col in ["recon_error", "graph_score", "rarity_score", "score"]:
        if col not in df_scores.columns:
            continue
        auc = _safe_auc(roc_auc_score, binary, df_scores[col].values)
        label_str = "composite" if col == "score" else col
        metrics[f"roc_auc_{col}"] = auc
        _row(label_str, f"{auc:.4f}")

    # ── TPR at fixed FPR operating points ─────────────────────────────────────
    _section("TPR at Fixed FPR  (operational trade-off table)")
    fpr_targets = [0.001, 0.005, 0.01, 0.05, 0.10]
    tpr_at_fpr  = _tpr_at_fpr_table(binary, scores, fpr_targets)
    metrics["tpr_at_fpr"] = tpr_at_fpr
    for fpr_t, tpr_v in tpr_at_fpr.items():
        _row(f"TPR @ FPR={fpr_t}", f"{tpr_v:.4f}  ({100*tpr_v:.1f}%)")

    # ── threshold sweep ───────────────────────────────────────────────────────
    sweep = _threshold_sweep(binary, scores)
    if report_path:
        pd.DataFrame(sweep).to_csv(report_path, index=False)

    best_f1_row = max(sweep, key=lambda r: r["f1"])
    best_j_row  = max(sweep, key=lambda r: r["youden_j"])

    _section("Threshold Sweep (step 0.01) — every 5th row shown")
    _table(sweep[::5], ["threshold", "precision", "recall", "f1",
                        "fpr", "fnr", "mcc", "g_mean", "tp", "fp", "fn", "tn"])
    _row("(full sweep saved to", str(report_path or "—") + ")")

    # ── metrics at F1-optimal threshold ───────────────────────────────────────
    _section(f"Metrics at F1-Optimal Threshold  ({best_f1_row['threshold']:.2f})")
    for k, v in best_f1_row.items():
        _row(k, f"{v:.4f}" if isinstance(v, float) else v)
    metrics["best_f1_threshold"] = best_f1_row
    metrics["threshold_f1_opt"]  = best_f1_row["threshold"]

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

    # ── detection rate per label at F1-optimal threshold ──────────────────────
    thr = best_f1_row["threshold"]
    _section("Detection Rate at F1-Optimal Threshold")
    for lbl, name in [(1, "Attack (1)"), (2, "Suspicious (2)")]:
        mask = labels == lbl
        if not mask.any():
            continue
        det  = int((scores[mask] >= thr).sum())
        tot  = int(mask.sum())
        rate = det / tot
        metrics[f"detection_rate_label_{lbl}"] = rate
        _row(f"{name} detected",   f"{det}/{tot}  ({100*rate:.1f}%)")
        _row(f"{name} mean score", f"{scores[mask].mean():.4f}")
    if n_benign > 0:
        _row("Benign mean score", f"{scores[labels==0].mean():.4f}")

    # ── save JSON summary ─────────────────────────────────────────────────────
    _save_metrics_json(metrics, METRICS_JSON_PATH)
    print(f"\n  All metrics   -> {METRICS_JSON_PATH}")
    print(f"  ROC curve     -> {ROC_CURVE_PATH}")
    print(f"  PR curve      -> {PR_CURVE_PATH}")
    print(f"  Score dist    -> {SCORE_DIST_PATH}")
    print("\n" + "=" * 68 + "\n")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# threshold sweep
# ─────────────────────────────────────────────────────────────────────────────

def _threshold_sweep(binary: np.ndarray, scores: np.ndarray) -> list:
    """Fine-grained sweep (step 0.01) including G-Mean and Cohen's Kappa."""
    thresholds = np.round(np.arange(0.0, 1.01, 0.01), 2)
    rows = []
    for thr in thresholds:
        pred = (scores >= thr).astype(int)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tn, fp, fn, tp = confusion_matrix(binary, pred, labels=[0, 1]).ravel()
            prec  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            f1    = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0
            fpr   = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr   = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            acc   = accuracy_score(binary, pred)
            mcc   = float(matthews_corrcoef(binary, pred)) if len(set(binary)) > 1 else 0.0
            kappa = float(cohen_kappa_score(binary, pred)) if len(set(pred)) > 1 else 0.0
            g_mean = float(np.sqrt(rec * spec))
            yj    = rec + spec - 1.0
        rows.append(dict(
            threshold   = round(float(thr), 2),
            precision   = round(prec,  4),
            recall      = round(rec,   4),
            specificity = round(spec,  4),
            f1          = round(f1,    4),
            accuracy    = round(acc,   4),
            fpr         = round(fpr,   4),
            fnr         = round(fnr,   4),
            mcc         = round(mcc,   4),
            kappa       = round(kappa, 4),
            g_mean      = round(g_mean,4),
            youden_j    = round(float(yj), 4),
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        ))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# TPR at fixed FPR
# ─────────────────────────────────────────────────────────────────────────────

def _tpr_at_fpr_table(binary, scores, fpr_targets) -> dict:
    """
    Interpolate TPR from the ROC curve at each requested FPR.
    Standard table in IDS / network security papers.
    """
    result = {}
    try:
        fpr_arr, tpr_arr, _ = roc_curve(binary, scores)
    except Exception:
        return {f"fpr={t:.3f}": float("nan") for t in fpr_targets}

    for target in fpr_targets:
        idx = np.searchsorted(fpr_arr, target)
        if idx == 0:
            tpr_val = float(tpr_arr[0])
        elif idx >= len(fpr_arr):
            tpr_val = float(tpr_arr[-1])
        else:
            slope   = (tpr_arr[idx] - tpr_arr[idx-1]) / (
                fpr_arr[idx] - fpr_arr[idx-1] + 1e-12)
            tpr_val = float(tpr_arr[idx-1] + slope * (target - fpr_arr[idx-1]))
        result[f"fpr={target:.3f}"] = round(tpr_val, 4)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# curve / JSON saving
# ─────────────────────────────────────────────────────────────────────────────

def _save_roc_curve(binary, scores, path):
    try:
        fpr, tpr, thr = roc_curve(binary, scores)
        pd.DataFrame({"fpr": fpr, "tpr": tpr,
                      "threshold": list(thr) + [float("nan")]}).to_csv(path, index=False)
    except Exception:
        pass


def _save_pr_curve(binary, scores, path):
    try:
        prec, rec, thr = precision_recall_curve(binary, scores)
        pd.DataFrame({"precision": prec, "recall": rec,
                      "threshold": list(thr) + [float("nan")]}).to_csv(path, index=False)
    except Exception:
        pass


def _save_metrics_json(metrics: dict, path: str):
    """Serialise metrics dict to JSON, converting numpy scalars."""
    def _cvt(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, dict):           return {k: _cvt(v) for k, v in obj.items()}
        if isinstance(obj, list):           return [_cvt(v) for v in obj]
        return obj
    try:
        with open(path, "w") as f:
            json.dump(_cvt(metrics), f, indent=2)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# alert-level
# ─────────────────────────────────────────────────────────────────────────────

def _alert_metrics(df_alerts, df_events, labels, scores, threshold) -> dict:
    n_chains = len(df_alerts)
    if "labels" not in df_alerts.columns:
        return dict(n_chains=n_chains)

    def _has_malicious(label_str):
        return any(p.strip() in {"1", "2"} for p in str(label_str).split(","))

    malicious_chains = df_alerts["labels"].apply(_has_malicious).sum()
    alert_precision  = malicious_chains / n_chains if n_chains > 0 else 0.0

    malicious_mask = labels > 0
    n_malicious    = malicious_mask.sum()
    covered        = (scores[malicious_mask] >= threshold).sum()
    alert_recall   = covered / n_malicious if n_malicious > 0 else 0.0

    return dict(
        n_chains              = n_chains,
        malicious_chains      = int(malicious_chains),
        alert_precision       = round(float(alert_precision), 4),
        alert_recall          = round(float(alert_recall), 4),
        mean_events_per_chain = round(float(df_alerts["num_events"].mean()), 4),
        mean_max_score        = round(float(df_alerts["max_score"].mean()), 4),
    )


# ─────────────────────────────────────────────────────────────────────────────
# formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_auc(fn, binary, scores):
    try:
        return round(float(fn(binary, scores)), 4)
    except Exception:
        return float("nan")


def _section(title: str):
    print(f"\n  -- {title} " + "-" * max(0, 56 - len(title)))


def _row(key: str, value):
    print(f"    {key:<40} {value}")


def _table(rows: list, cols: list):
    header = "  ".join(f"{c:>10}" for c in cols)
    print(f"    {header}")
    print("    " + "-" * len(header))
    for r in rows:
        line = "  ".join(f"{str(r.get(c,''))[:10]:>10}" for c in cols)
        print(f"    {line}")
