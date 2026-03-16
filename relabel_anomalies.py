"""
relabel_anomalies.py
────────────────────
Utility to promote specific evaluation events from Label=2 (suspicious) to
Label=1 (confirmed attack) in the ingested CSV.

This step is needed to get meaningful AUC / F1 metrics from main.py.
Without ground-truth Label=1 events the metrics module still runs but
scores are unsorted — you only know which events look anomalous, not
whether those are the right ones.

Labelling strategies (choose whichever you have evidence for)
──────────────────────────────────────────────────────────────
  --host       Flag all events from a specific machine as confirmed attacks
  --process    Flag events where Image contains a process name substring
  --ip         Flag network events that connected to a suspicious IP
  --time       Flag events in a specific time window (combine with --host)
  --ids        Flag a CSV file of row indices (from anomaly_scores.csv review)
  --query      Arbitrary pandas query string for full flexibility

Multiple criteria can be combined: a row is labelled 1 if ANY criterion matches.

Usage examples
──────────────
  # Mark all Jan events on "WORKSTATION-04" as confirmed attacks
  python relabel_anomalies.py --host WORKSTATION-04

  # Mark events involving a specific suspicious process
  python relabel_anomalies.py --process "mimikatz"

  # Mark events connecting to a known C2 IP
  python relabel_anomalies.py --ip "185.220.101.45"

  # Mark events between 02:00 and 04:00 on Jan 15 on a specific host
  python relabel_anomalies.py \\
      --host WORKSTATION-04 \\
      --time "2026-01-15T02:00:00Z" "2026-01-15T04:00:00Z"

  # Arbitrary pandas query
  python relabel_anomalies.py \\
      --query "Image.str.contains('psexec', case=False) and Computer == 'SRV-DC'"

  # Provide row indices from manual review of anomaly_scores.csv
  python relabel_anomalies.py --ids indices.csv

  # Preview without saving
  python relabel_anomalies.py --host WORKSTATION-04 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _build_mask(df: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    """Return a boolean Series: True for rows that should become Label=1."""
    # start from all-False
    mask = pd.Series(False, index=df.index)

    if args.host:
        for h in args.host:
            mask |= df["Computer"].str.lower() == h.lower()
        log.info("  --host     : %d rows matched", mask.sum())

    if args.process:
        proc_mask = pd.Series(False, index=df.index)
        for p in args.process:
            proc_mask |= df["Image"].str.contains(p, case=False, na=False)
        before = mask.sum()
        mask |= proc_mask
        log.info("  --process  : %d rows matched", mask.sum() - before)

    if args.ip:
        ip_mask = pd.Series(False, index=df.index)
        for ip in args.ip:
            ip_mask |= df["DestinationIp"].str.contains(ip, case=False, na=False)
        before = mask.sum()
        mask |= ip_mask
        log.info("  --ip       : %d rows matched", mask.sum() - before)

    if args.time:
        if len(args.time) != 2:
            log.error("--time requires exactly two arguments: <start> <end>")
            sys.exit(1)
        ts_start, ts_end = args.time
        ts_col = pd.to_datetime(df["SystemTime"], utc=True, errors="coerce")
        t0 = pd.to_datetime(ts_start, utc=True)
        t1 = pd.to_datetime(ts_end,   utc=True)
        time_mask = (ts_col >= t0) & (ts_col <= t1)
        # if --host was also specified, narrow to host AND time
        if args.host:
            host_mask = pd.Series(False, index=df.index)
            for h in args.host:
                host_mask |= df["Computer"].str.lower() == h.lower()
            time_mask = time_mask & host_mask
        before = mask.sum()
        mask |= time_mask
        log.info("  --time     : %d rows matched", mask.sum() - before)

    if args.query:
        try:
            query_mask = df.eval(args.query)
        except Exception as exc:
            log.error("--query failed: %s", exc)
            sys.exit(1)
        before = mask.sum()
        mask |= query_mask.astype(bool)
        log.info("  --query    : %d rows matched", mask.sum() - before)

    if args.ids:
        ids_path = Path(args.ids)
        if not ids_path.exists():
            log.error("--ids file not found: %s", ids_path)
            sys.exit(1)
        idx_df = pd.read_csv(ids_path)
        # expect a column named 'index' or just use the first column
        col = "index" if "index" in idx_df.columns else idx_df.columns[0]
        target_ids = idx_df[col].astype(int).tolist()
        valid_ids  = [i for i in target_ids if i in df.index]
        before = mask.sum()
        mask.iloc[valid_ids] = True
        log.info(
            "  --ids      : %d indices requested, %d valid",
            len(target_ids), mask.sum() - before,
        )

    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Promote suspicious events (Label=2) to confirmed attacks (Label=1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        default="elastic_data.csv",
        help="Input CSV (output of elastic_ingest.py, default: elastic_data.csv)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output CSV path (default: overwrites --input in-place)",
    )
    parser.add_argument(
        "--host",
        nargs="+",
        metavar="HOSTNAME",
        help="Hostname(s) to mark as compromised (case-insensitive)",
    )
    parser.add_argument(
        "--process",
        nargs="+",
        metavar="SUBSTRING",
        help="Mark events where Image contains this substring",
    )
    parser.add_argument(
        "--ip",
        nargs="+",
        metavar="IP",
        help="Mark network events connecting to this IP (substring match)",
    )
    parser.add_argument(
        "--time",
        nargs=2,
        metavar=("START", "END"),
        help="ISO-8601 UTC time window, e.g. 2026-01-15T02:00:00Z 2026-01-15T04:00:00Z",
    )
    parser.add_argument(
        "--query",
        metavar="PANDAS_QUERY",
        help="Arbitrary pandas df.eval() expression",
    )
    parser.add_argument(
        "--ids",
        metavar="INDICES_CSV",
        help="CSV file with an 'index' column of row numbers to relabel",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing the file",
    )
    args = parser.parse_args()

    # require at least one criterion
    if not any([args.host, args.process, args.ip, args.time, args.query, args.ids]):
        parser.error(
            "Provide at least one labelling criterion: "
            "--host, --process, --ip, --time, --query, or --ids"
        )

    # ── load ──────────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input file not found: %s", input_path)
        sys.exit(1)

    log.info("Loading %s …", input_path)
    df = pd.read_csv(input_path, low_memory=False)

    if "Label" not in df.columns:
        log.warning("No 'Label' column found; initialising all rows to Label=0.")
        df["Label"] = 0

    before_counts = df["Label"].value_counts().sort_index()
    log.info("Label distribution BEFORE relabelling:")
    for lbl, cnt in before_counts.items():
        log.info("  Label=%-2d : %d", lbl, cnt)

    # ── build mask ────────────────────────────────────────────────────────────
    log.info("Applying labelling criteria …")
    mask = _build_mask(df, args)

    # only relabel rows currently at Label=2 (don't overwrite manual Label=1)
    target_mask = mask & (df["Label"] == 2)
    n_to_relabel = target_mask.sum()

    log.info("")
    log.info("Total rows matching criteria : %d", mask.sum())
    log.info("Of which currently Label=2   : %d  (will be set to Label=1)", n_to_relabel)

    if n_to_relabel == 0:
        log.warning("No Label=2 rows matched the criteria — nothing to change.")
        return

    if args.dry_run:
        log.info("[DRY-RUN] No file written.")
        sample = df[target_mask][
            ["Computer", "SystemTime", "Image", "CommandLine", "DestinationIp", "Label"]
        ].head(10)
        log.info("Sample of rows that WOULD be relabelled:\n%s", sample.to_string(index=True))
        return

    # ── apply ──────────────────────────────────────────────────────────────────
    df.loc[target_mask, "Label"] = 1

    after_counts = df["Label"].value_counts().sort_index()
    log.info("Label distribution AFTER relabelling:")
    for lbl, cnt in after_counts.items():
        log.info("  Label=%-2d : %d", lbl, cnt)

    # ── save ──────────────────────────────────────────────────────────────────
    output_path = args.output or str(input_path)
    df.to_csv(output_path, index=False)
    log.info("Saved → %s  (%d events, %d confirmed attacks)", output_path, len(df), (df["Label"] == 1).sum())
    log.info("")
    log.info("Run the pipeline:  python main.py")


if __name__ == "__main__":
    main()
