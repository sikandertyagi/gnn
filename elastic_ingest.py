"""
elastic_ingest.py
─────────────────
One-shot data preparation script: pulls endpoint telemetry from Elasticsearch
and writes the combined CSV that main.py consumes.

Workflow
────────
  1. Connect to Elasticsearch (credentials from config file or env vars)
  2. Fetch TRAINING data  (Nov 2025 – Dec 2025) → Label = 0  (benign)
  3. Fetch EVALUATION data (Jan 2026)            → Label = 2  (suspicious)
  4. Merge, deduplicate, sort by SystemTime
  5. Write combined CSV to output_path
  6. Print summary table

Then run:
  python main.py

Label semantics
───────────────
  0 → benign   — training period, all three models trained on this
  1 → confirmed attack — set via relabel_anomalies.py when ground truth known
  2 → suspicious / unknown — evaluation period; pipeline scores but never
                              trains on; treated as positive class in AUC/F1

Usage
─────
  python elastic_ingest.py                         # uses elastic_config.yml
  python elastic_ingest.py --config /path/to/cfg.yml
  python elastic_ingest.py --dry-run               # print counts, skip CSV save

Environment variable overrides (highest priority)
──────────────────────────────────────────────────
  ES_HOST          Full URL, e.g. https://elastic.corp:9200
  ES_CLOUD_ID      Elastic Cloud deployment ID
  ES_API_KEY       API key (base64 id:key)
  ES_USERNAME      Basic auth username
  ES_PASSWORD      Basic auth password
  ES_CA_CERTS      Path to CA certificate bundle
  ES_SYSMON_INDEX  Index pattern for Sysmon
  ES_DEFEND_INDEX  Index pattern for Elastic Defend
  ES_WAZUH_INDEX   Index pattern for Wazuh
  ES_OUTPUT_PATH   Output CSV path
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ── allow running from the project root ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from elastic_connector import ElasticConnector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── config loading ─────────────────────────────────────────────────────────────
def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        log.warning("PyYAML not installed; ignoring config file. Install: pip install pyyaml")
        return {}
    p = Path(path)
    if not p.exists():
        log.warning("Config file not found: %s — relying on env vars / defaults.", path)
        return {}
    with open(p) as fh:
        data = yaml.safe_load(fh) or {}
    log.info("Config loaded from %s", path)
    return data


def _load_config(path: str) -> Dict[str, Any]:
    cfg = _load_yaml(path)

    # environment variable overrides
    _ENV = {
        "ES_HOST":         "host",
        "ES_CLOUD_ID":     "cloud_id",
        "ES_API_KEY":      "api_key",
        "ES_USERNAME":     "username",
        "ES_PASSWORD":     "password",
        "ES_CA_CERTS":     "ca_certs",
        "ES_SYSMON_INDEX": "sysmon_index",
        "ES_DEFEND_INDEX": "defend_index",
        "ES_WAZUH_INDEX":  "wazuh_index",
        "ES_OUTPUT_PATH":  "output_path",
    }
    for env_var, key in _ENV.items():
        val = os.environ.get(env_var)
        if val is not None:
            cfg[key] = val
            log.debug("Config override from env: %s = %s", key, val)

    return cfg


def _build_connector(cfg: Dict[str, Any]) -> ElasticConnector:
    return ElasticConnector(
        host=cfg.get("host"),
        cloud_id=cfg.get("cloud_id"),
        api_key=cfg.get("api_key"),
        username=cfg.get("username"),
        password=cfg.get("password"),
        ca_certs=cfg.get("ca_certs"),
        verify_certs=cfg.get("verify_certs", True),
        page_size=cfg.get("page_size", 5_000),
    )


# ── data fetching ──────────────────────────────────────────────────────────────
def _fetch_window(
    conn: ElasticConnector,
    start: str,
    end: str,
    label: int,
    period_name: str,
    sources: List[str],
    sysmon_index: str,
    defend_index: str,
    wazuh_index: str,
) -> pd.DataFrame:
    log.info("")
    log.info("── %s  [%s → %s]  Label=%d ──", period_name, start, end, label)
    df = conn.fetch_all(
        start=start,
        end=end,
        sources=sources,
        sysmon_index=sysmon_index,
        defend_index=defend_index,
        wazuh_index=wazuh_index,
    )
    df["Label"] = label

    by_eid = df.groupby("EventID").size()
    log.info("  EventID breakdown: %s", dict(by_eid))
    log.info("  Hosts found      : %d", df["Computer"].nunique())
    return df


# ── summary printer ────────────────────────────────────────────────────────────
def _print_summary(df: pd.DataFrame, output_path: str) -> None:
    sep = "=" * 62
    log.info("")
    log.info(sep)
    log.info("  DATASET SUMMARY")
    log.info(sep)

    label_names = {0: "benign/training", 1: "confirmed attack", 2: "suspicious/eval"}
    for lbl, count in df["Label"].value_counts().sort_index().items():
        log.info("  Label=%-1d %-20s : %6d events", lbl, f"({label_names.get(lbl, '?')})", count)

    log.info("")
    for eid, count in df["EventID"].value_counts().sort_index().items():
        eid_name = {1: "process creation", 3: "network connection"}.get(int(eid), "?")
        log.info("  EventID=%-2d %-18s : %6d events", eid, f"({eid_name})", count)

    log.info("")
    log.info("  Total events  : %d", len(df))
    log.info("  Unique hosts  : %d", df["Computer"].nunique())
    log.info("  Unique procs  : %d", df["Image"].nunique())
    log.info("  Date range    : %s  →  %s",
             df["SystemTime"].min(), df["SystemTime"].max())
    log.info(sep)
    log.info("")
    log.info("  Saved → %s", output_path)
    log.info("")
    log.info("  Next step: ensure config.py has")
    log.info("    DATA_PATH = \"%s\"", output_path)
    log.info("  then run:  python main.py")
    log.info(sep)


# ── main ───────────────────────────────────────────────────────────────────────
def main(config_path: str = "elastic_config.yml", dry_run: bool = False) -> None:
    cfg = _load_config(config_path)

    # ── date windows ──────────────────────────────────────────────────────────
    train_start = cfg.get("train_start", "2025-11-01T00:00:00Z")
    train_end   = cfg.get("train_end",   "2025-12-31T23:59:59Z")
    eval_start  = cfg.get("eval_start",  "2026-01-01T00:00:00Z")
    eval_end    = cfg.get("eval_end",    "2026-01-31T23:59:59Z")

    # ── sources & index patterns ──────────────────────────────────────────────
    sources      = cfg.get("sources", ["sysmon", "defend"])
    sysmon_index = cfg.get(
        "sysmon_index",
        "winlogbeat-*,logs-windows.sysmon_operational-*",
    )
    defend_index = cfg.get(
        "defend_index",
        "logs-endpoint.events.process-*,logs-endpoint.events.network-*",
    )
    wazuh_index  = cfg.get(
        "wazuh_index",
        "wazuh-alerts-4.x-*,wazuh-archives-4.x-*",
    )
    output_path  = cfg.get("output_path", "elastic_data.csv")

    log.info("Sources           : %s", sources)
    log.info("Training window   : %s → %s", train_start, train_end)
    log.info("Evaluation window : %s → %s", eval_start,  eval_end)
    log.info("Output path       : %s", output_path)

    if dry_run:
        log.info("[DRY-RUN] Skipping Elasticsearch connection.")
        return

    # ── connect ───────────────────────────────────────────────────────────────
    conn = _build_connector(cfg)

    # ── fetch training data: Nov–Dec 2025 → Label=0 (benign) ─────────────────
    df_train = _fetch_window(
        conn,
        start=train_start,
        end=train_end,
        label=0,
        period_name="TRAINING  (Nov–Dec 2025)",
        sources=sources,
        sysmon_index=sysmon_index,
        defend_index=defend_index,
        wazuh_index=wazuh_index,
    )

    if df_train.empty:
        log.error(
            "No training data fetched for %s → %s.\n"
            "  Check your index patterns, date range, and ES credentials.",
            train_start, train_end,
        )
        sys.exit(1)

    # ── fetch evaluation data: Jan 2026 → Label=2 (suspicious/unknown) ───────
    df_eval = _fetch_window(
        conn,
        start=eval_start,
        end=eval_end,
        label=2,
        period_name="EVALUATION (Jan  2026)",
        sources=sources,
        sysmon_index=sysmon_index,
        defend_index=defend_index,
        wazuh_index=wazuh_index,
    )

    if df_eval.empty:
        log.warning(
            "No evaluation data fetched for %s → %s. "
            "Pipeline will still run but evaluation metrics will be skipped.",
            eval_start, eval_end,
        )

    # ── merge & sort ──────────────────────────────────────────────────────────
    df = pd.concat([df_train, df_eval], ignore_index=True)
    df["_sort_ts"] = pd.to_datetime(df["SystemTime"], utc=True, errors="coerce")
    df = df.sort_values("_sort_ts").drop(columns=["_sort_ts"]).reset_index(drop=True)

    # ── save ──────────────────────────────────────────────────────────────────
    df.to_csv(output_path, index=False)
    _print_summary(df, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch Elastic SIEM data → prepare anomaly pipeline input CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c",
        default="elastic_config.yml",
        help="Path to YAML config file (default: elastic_config.yml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config and exit without connecting to Elasticsearch",
    )
    args = parser.parse_args()
    main(config_path=args.config, dry_run=args.dry_run)
