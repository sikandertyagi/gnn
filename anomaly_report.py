"""
anomaly_report.py
─────────────────
Generates a human-readable investigation report and a filtered CSV for manual
triage of anomalies detected in the evaluation period.

Outputs
───────
  anomaly_report.txt   Narrative report structured for analyst review:
                         · Executive summary (counts, hosts, date range)
                         · Alert chain summaries (grouped events)
                         · Per-host flagged event listings
                         · Full detail per event + "why suspicious" hints

  flagged_events.csv   All flagged events (score ≥ ALERT_THRESHOLD) in the
                       evaluation period (Label ≠ 0), enriched with all
                       original Sysmon/Defend fields plus score columns.
                       Sorted by anomaly score descending.
                       Open in Excel / LibreOffice for filtering.

Label conventions used internally by the pipeline
───────────────────────────────────────────────────
  Label = 0  Training period (Nov–Dec 2025).  Models trained on this.
  Label = 2  Evaluation period (Jan 2026).    Scored, never trained on.
             This is NOT a ground-truth annotation — it is a period marker.

Analyst verification workflow
──────────────────────────────
  1. Run: python elastic_ingest.py   → pulls data from Elastic SIEM
  2. Run: python main.py             → trains models, scores all events,
                                       writes anomaly_report.txt +
                                       flagged_events.csv
  3. Open anomaly_report.txt         → read narrative investigation guide
  4. Open flagged_events.csv         → sort/filter in spreadsheet
  5. For each suspicious entry:
       · Check host timeline in Kibana / SIEM around the flagged timestamp
       · Verify process lineage (ParentImage → Image)
       · Search CommandLine for known IOCs
       · Look up DestinationIp in threat-intel feeds
  6. Confirmed attacks: run relabel_anomalies.py → set HAS_GROUND_TRUTH=True
                        → re-run main.py for AUC/F1 metrics
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from config import ALERT_THRESHOLD


# ── "why suspicious" hint engine ─────────────────────────────────────────────

def _why_suspicious(row: pd.Series) -> list[str]:
    """
    Return a list of human-readable reason strings for why this event was
    flagged, derived from its feature values.  Each reason is one line.
    """
    reasons = []

    score = row.get("score", 0)
    rarity = row.get("rarity_score", 0)
    recon  = row.get("recon_error",  0)

    # Score composition
    if rarity > 0.7:
        reasons.append(
            f"High rarity score ({rarity:.2f}): behaviour pattern not seen "
            "during Nov–Dec 2025 baseline"
        )
    if not pd.isna(recon) and recon > 0.7:
        reasons.append(
            f"High reconstruction error ({recon:.2f}): event sequence deviates "
            "from normal host behaviour"
        )

    # Command-line indicators (feature columns added by feature_engineering)
    if row.get("has_base64", 0):
        reasons.append("Base64 string (≥20 chars) found in CommandLine")
    if row.get("has_encodedcommand", 0):
        reasons.append("PowerShell -EncodedCommand / -enc flag detected")
    if row.get("has_download", 0):
        reasons.append(
            "Download utility in command (wget / curl / Invoke-WebRequest / "
            "bitsadmin / certutil)"
        )
    if row.get("has_http", 0):
        reasons.append("HTTP/HTTPS URL found in CommandLine")
    if row.get("has_ip", 0):
        reasons.append("Raw IPv4 address embedded in CommandLine")

    # Process location
    if row.get("is_temp_exec", 0):
        reasons.append(
            "Executable launched from a temp / user-writable directory "
            "(AppData, Temp, Downloads, ProgramData)"
        )
    if not row.get("is_system32", 0) and not row.get("is_signed", 1):
        reasons.append(
            "Unsigned binary running outside System32 — uncommon for legitimate software"
        )
    if row.get("missing_company", 0):
        reasons.append("Binary has no Company metadata in PE header")

    # Network
    eid = int(row.get("EventID", 0))
    if eid == 3:
        if row.get("dest_external", 0):
            ip   = str(row.get("DestinationIp", ""))
            port = int(row.get("DestinationPort", 0))
            reasons.append(
                f"Outbound connection to external IP {ip}:{port} — "
                "verify against threat-intel feeds"
            )
        else:
            reasons.append("Lateral movement candidate: internal network connection")

    # Integrity
    integrity = str(row.get("IntegrityLevel", "")).lower()
    if integrity in ("high", "system"):
        reasons.append(
            f"Process running at elevated integrity level: {integrity.title()} — "
            "check if privilege escalation occurred"
        )

    # cmd entropy
    cmd_ent = row.get("cmd_entropy", 0)
    if cmd_ent > 4.5:
        reasons.append(
            f"High command entropy ({cmd_ent:.2f}) — may indicate obfuscated or "
            "encoded payload"
        )

    if not reasons:
        reasons.append(
            f"Composite anomaly score {score:.3f} exceeds threshold "
            f"{ALERT_THRESHOLD:.2f} — verify against baseline behaviour"
        )

    return reasons


# ── event detail formatter ────────────────────────────────────────────────────

def _format_event(idx: int, rank: int, ev: pd.Series) -> str:
    """Return a multi-line string with full details for one flagged event."""

    eid      = int(ev.get("EventID", 0))
    eid_name = {1: "Process Creation", 3: "Network Connection"}.get(eid, f"EventID {eid}")

    score  = ev.get("score",        float("nan"))
    rarity = ev.get("rarity_score", float("nan"))
    recon  = ev.get("recon_error",  float("nan"))

    cmd = str(ev.get("CommandLine", "")).strip() or "(none)"
    # wrap long command lines for readability
    cmd_wrapped = "\n              ".join(textwrap.wrap(cmd, width=90))

    lines = [
        f"  ┌─ [{rank:>4}]  Score: {score:.4f}  "
        f"│ rarity={rarity:.3f}  recon={'n/a' if pd.isna(recon) else f'{recon:.3f}'}",
        f"  │  Row index   : {idx}",
        f"  │  Timestamp   : {ev.get('SystemTime', '')}",
        f"  │  Host        : {ev.get('Computer', '')}",
        f"  │  User        : {ev.get('User', '')}",
        f"  │  EventID     : {eid}  ({eid_name})",
        f"  │  Image       : {ev.get('Image', '')}",
        f"  │  ParentImage : {ev.get('ParentImage', '')}",
        f"  │  CommandLine : {cmd_wrapped}",
    ]

    if eid == 3:
        lines.append(
            f"  │  Dest IP:Port: {ev.get('DestinationIp', '')}:"
            f"{int(ev.get('DestinationPort', 0))}"
        )

    if ev.get("IntegrityLevel"):
        lines.append(f"  │  Integrity   : {ev.get('IntegrityLevel', '')}")

    lines.append("  │")
    lines.append("  │  WHY FLAGGED:")
    for reason in _why_suspicious(ev):
        wrapped = textwrap.fill(
            reason, width=88,
            initial_indent="  │    · ",
            subsequent_indent="  │      ",
        )
        lines.append(wrapped)

    lines.append("  └" + "─" * 78)
    return "\n".join(lines)


# ── main report generator ─────────────────────────────────────────────────────

def generate_investigation_report(
    df_scores:   pd.DataFrame,
    df_events:   pd.DataFrame,
    df_alerts:   pd.DataFrame,
    report_path: str = "anomaly_report.txt",
    csv_path:    str = "flagged_events.csv",
    top_n:       int = 200,
) -> None:
    """
    Write anomaly_report.txt and flagged_events.csv.

    Parameters
    ──────────
    df_scores  : DataFrame with columns score, recon_error, graph_score,
                 rarity_score, label  (one row per event, same index as df_events)
    df_events  : Original enriched event DataFrame (post feature_engineering)
    df_alerts  : Alert chains DataFrame from alert_aggregator
    report_path: Path for the text investigation report
    csv_path   : Path for the flagged-events CSV
    top_n      : Maximum number of events to detail individually in the report
    """

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # ── join scores onto events ───────────────────────────────────────────────
    df = df_events.copy()
    df["score"]        = df_scores["score"].values
    df["rarity_score"] = df_scores["rarity_score"].values
    df["recon_error"]  = df_scores["recon_error"].values
    df["graph_score"]  = df_scores["graph_score"].values
    df["label"]        = df_scores["label"].values

    # ── evaluation-period events only (Label ≠ 0) ────────────────────────────
    eval_df = df[df["label"] != 0].copy()

    # ── flagged events: score ≥ threshold in evaluation period ───────────────
    flagged = eval_df[eval_df["score"] >= ALERT_THRESHOLD].copy()
    flagged = flagged.sort_values("score", ascending=False).reset_index()
    # 'index' column is the original row index in df_events (for reference)
    flagged = flagged.rename(columns={"index": "original_row_index"})

    n_eval    = len(eval_df)
    n_flagged = len(flagged)
    n_hosts_eval    = eval_df["Computer"].nunique()
    n_hosts_flagged = flagged["Computer"].nunique() if n_flagged else 0

    ts_min = eval_df["SystemTime"].min() if n_eval else "N/A"
    ts_max = eval_df["SystemTime"].max() if n_eval else "N/A"

    # ── score distribution for eval period ───────────────────────────────────
    if n_eval:
        pcts = np.nanpercentile(eval_df["score"].values, [50, 75, 90, 95, 99])
        dist_lines = (
            f"    p50={pcts[0]:.3f}  p75={pcts[1]:.3f}  p90={pcts[2]:.3f}  "
            f"p95={pcts[3]:.3f}  p99={pcts[4]:.3f}"
        )
    else:
        dist_lines = "    (no evaluation-period events)"

    # ── build text report ─────────────────────────────────────────────────────
    W = 82   # line width
    lines: list[str] = []
    hr  = "═" * W
    hr2 = "─" * W

    def section(title: str) -> None:
        lines.append("")
        lines.append(hr2)
        lines.append(f"  {title}")
        lines.append(hr2)

    # header
    lines += [
        hr,
        "  ANOMALY INVESTIGATION REPORT",
        f"  Generated : {now_str}",
        f"  Pipeline  : GNN Anomaly Detection — Elastic SIEM Integration",
        f"  Threshold : {ALERT_THRESHOLD}  (events above this score are flagged)",
        hr,
    ]

    # executive summary
    section("EXECUTIVE SUMMARY")
    lines += [
        f"  Evaluation period events : {n_eval:,}",
        f"  Flagged events (≥{ALERT_THRESHOLD:.2f})    : {n_flagged:,}"
        f"  ({100 * n_flagged / max(n_eval, 1):.2f}% of eval period)",
        f"  Hosts in evaluation      : {n_hosts_eval}",
        f"  Hosts with flagged events: {n_hosts_flagged}",
        f"  Eval date range          : {ts_min}  →  {ts_max}",
        "",
        "  Score distribution (evaluation period):",
        dist_lines,
    ]

    # per-host summary table
    section("FLAGGED EVENTS BY HOST")
    if n_flagged:
        host_summary = (
            flagged.groupby("Computer")
            .agg(
                flagged_events=("score", "count"),
                max_score=("score", "max"),
                mean_score=("score", "mean"),
            )
            .sort_values("max_score", ascending=False)
            .reset_index()
        )
        col_w = [30, 16, 12, 12]
        hdr = (
            f"  {'Host':<{col_w[0]}}  {'Flagged Events':>{col_w[1]}}"
            f"  {'Max Score':>{col_w[2]}}  {'Mean Score':>{col_w[3]}}"
        )
        lines.append(hdr)
        lines.append("  " + "─" * (sum(col_w) + 6))
        for _, row in host_summary.iterrows():
            lines.append(
                f"  {str(row['Computer']):<{col_w[0]}}"
                f"  {int(row['flagged_events']):>{col_w[1]}}"
                f"  {row['max_score']:>{col_w[2]}.4f}"
                f"  {row['mean_score']:>{col_w[3]}.4f}"
            )
    else:
        lines.append(
            f"  No events exceeded the threshold ({ALERT_THRESHOLD}).  "
            "The Jan 2026 activity appears consistent with the Nov–Dec 2025 baseline.\n"
            "  Consider lowering ALERT_THRESHOLD in config.py if you expect more findings."
        )

    # alert chain summaries
    section("ALERT CHAINS  (groups of temporally close flagged events per host)")
    if not df_alerts.empty:
        chains_to_show = df_alerts.nlargest(50, "max_score")
        for _, chain in chains_to_show.iterrows():
            lines += [
                f"",
                f"  Chain #{int(chain.get('chain_id', 0))}",
                f"    Host       : {chain.get('host', '')}",
                f"    Time span  : {chain.get('start_time', '')}  →  {chain.get('end_time', '')}",
                f"    Duration   : {chain.get('duration_s', 0):.0f}s",
                f"    Events     : {int(chain.get('num_events', 0))}   "
                f"Max score: {chain.get('max_score', 0):.4f}",
                f"    Processes  : {chain.get('processes', '')}",
                f"    Dest IPs   : {chain.get('dest_ips', '(none)')}",
                f"    ──────────────────────────────────────────────────────────",
            ]
    else:
        lines.append(
            "  No alert chains formed.  Either no events exceeded the threshold\n"
            "  or flagged events were too spread out in time to cluster."
        )

    # detailed event listings
    section(
        f"FLAGGED EVENT DETAILS  "
        f"(top {min(top_n, n_flagged)} of {n_flagged} — sorted by score)"
    )

    if n_flagged == 0:
        lines.append("  No flagged events to show.")
    else:
        lines.append(
            "  Each entry shows full event context plus reasons for the flag.\n"
            "  Use these details to investigate in Kibana / your SIEM:\n"
            "    1. Search by Timestamp + Host to find the event in Kibana\n"
            "    2. Check process lineage (ParentImage → Image)\n"
            "    3. Search CommandLine for IOCs (hashes, IPs, domains)\n"
            "    4. Look up DestinationIp in threat-intel feeds\n"
            "    5. Review adjacent events on the same host ± 5 minutes\n"
        )
        shown = flagged.head(top_n)
        for rank, (_, ev) in enumerate(shown.iterrows(), start=1):
            orig_idx = int(ev.get("original_row_index", rank - 1))
            lines.append(_format_event(orig_idx, rank, ev))
            lines.append("")

        if n_flagged > top_n:
            lines.append(
                f"  … {n_flagged - top_n} more flagged events not shown here.\n"
                f"  See {csv_path} for the complete list."
            )

    # footer
    lines += [
        "",
        hr,
        "  NEXT STEPS",
        hr2,
        "  · Open flagged_events.csv in a spreadsheet for filtering by host,",
        "    process, IP, or score.",
        "  · Use relabel_anomalies.py to mark confirmed attacks as Label=1:",
        "      python relabel_anomalies.py --host HOSTNAME",
        "      python relabel_anomalies.py --process mimikatz --ip 1.2.3.4",
        "  · Then set HAS_GROUND_TRUTH = True in config.py and re-run main.py",
        "    to compute AUC / F1 / ROC metrics.",
        hr,
        "",
    ]

    report_text = "\n".join(lines)

    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    print(f"\n  Investigation report → {report_path}")

    # ── flagged_events.csv ────────────────────────────────────────────────────
    # Select human-useful columns (skip internal feature vectors)
    keep_cols = [
        "original_row_index",
        "score", "rarity_score", "recon_error", "graph_score",
        "label",
        "Computer", "User", "SystemTime", "EventID",
        "Image", "ParentImage", "CommandLine",
        "DestinationIp", "DestinationPort",
        "IntegrityLevel", "Company", "Signed",
        # readable feature flags
        "has_base64", "has_encodedcommand", "has_download",
        "has_http", "has_ip",
        "is_system32", "is_temp_exec", "is_signed", "missing_company",
        "cmd_entropy",
        "ProcessGuid", "ParentProcessGuid",
    ]

    # only keep columns that actually exist in the flagged DataFrame
    available = [c for c in keep_cols if c in flagged.columns]
    flagged[available].to_csv(csv_path, index=False)
    print(f"  Flagged events CSV   → {csv_path}  ({n_flagged} rows)")
