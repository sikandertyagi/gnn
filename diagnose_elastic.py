"""
diagnose_elastic.py
────────────────────
Diagnoses why elastic_ingest.py returns 0 events.

Runs progressively more targeted queries and prints exactly what ES returns,
including timed_out flags and query structure.

Usage:
  python diagnose_elastic.py --config elastic_config.yml
"""
import argparse, sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from elasticsearch import Elasticsearch
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def run(es, label, index, body):
    print(f"\n{'─'*60}")
    print(f"TEST: {label}")
    try:
        resp = es.search(index=index, expand_wildcards="all", body=body)
        total  = resp["hits"]["total"]
        timed  = resp.get("timed_out", False)
        hits   = resp["hits"]["hits"]
        print(f"  total={total}  timed_out={timed}  returned={len(hits)}")
        for h in hits[:2]:
            src = h.get("_source", {})
            print("  doc:", json.dumps({k: src.get(k) for k in
                ["@timestamp","event.category","event.type","event.action",
                 "process.name","host.hostname"]}, indent=4))
    except Exception as e:
        print(f"  ERROR: {e}")


def main(config_path):
    cfg = load_cfg(config_path)

    es = Elasticsearch(
        hosts=[cfg["host"]],
        basic_auth=(cfg.get("username"), cfg.get("password")),
        verify_certs=False,
        ssl_show_warn=False,
    )

    index       = cfg.get("defend_index",
                          ".ds-logs-endpoint.events.process-*,.ds-logs-endpoint.events.network-*")
    train_start = cfg.get("train_start", "2025-11-01T00:00:00Z")
    train_end   = cfg.get("train_end",   "2025-12-31T23:59:59Z")
    date_filter = {"range": {"@timestamp": {"gte": train_start, "lte": train_end}}}

    src_fields = ["@timestamp", "event.category", "event.type", "event.action",
                  "process.name", "host.hostname"]

    # ── 1. Baseline: date range only, no sort ─────────────────────────────────
    run(es, "date range only (no sort, size=3)", index, {
        "query": date_filter,
        "size": 3,
        "_source": src_fields,
    })

    # ── 2. Exact _paginate body: date range + _id sort ────────────────────────
    run(es, "date range + sort by @timestamp,_id (mirrors _paginate)", index, {
        "query": date_filter,
        "size": 3,
        "_source": src_fields,
        "sort": [{"@timestamp": {"order": "asc"}}, {"_id": {"order": "asc"}}],
    })

    # ── 3. Full defend query (same as production), NO sort ────────────────────
    defend_query = {
        "bool": {
            "must": [
                date_filter,
                {
                    "bool": {
                        "should": [
                            {"bool": {"must": [
                                {"term": {"event.category": "process"}},
                                {"term": {"event.type":     "start"}},
                            ]}},
                            {"bool": {"must": [
                                {"term":  {"event.category": "network"}},
                                {"terms": {"event.type": ["connection", "start", "protocol"]}},
                            ]}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
            ]
        }
    }
    run(es, "full defend_query (no sort)", index, {
        "query": defend_query,
        "size": 3,
        "_source": src_fields,
    })

    # ── 4. Network only, ANY type ─────────────────────────────────────────────
    run(es, "network category only (any type)", index, {
        "query": {"bool": {"must": [
            date_filter,
            {"term": {"event.category": "network"}},
        ]}},
        "size": 3,
        "_source": src_fields,
    })

    # ── 5. Process only, ANY type ─────────────────────────────────────────────
    run(es, "process category only (any type)", index, {
        "query": {"bool": {"must": [
            date_filter,
            {"term": {"event.category": "process"}},
        ]}},
        "size": 3,
        "_source": src_fields,
    })

    # ── 6. Agg: unique event.category + event.type values ────────────────────
    # Run on a single small index first to avoid cluster-wide timeout
    small_index = ".ds-logs-endpoint.events.network-default-2025.11.07-000007"
    print(f"\n{'─'*60}")
    print(f"TEST: agg on single Nov index ({small_index})")
    try:
        resp = es.search(
            index=small_index,
            expand_wildcards="all",
            body={
                "query": date_filter,
                "size": 0,
                "aggs": {
                    "categories": {"terms": {"field": "event.category", "size": 20}},
                    "types":      {"terms": {"field": "event.type",     "size": 20}},
                },
            },
            request_timeout=30,
        )
        cats  = resp["aggregations"]["categories"]["buckets"]
        types = resp["aggregations"]["types"]["buckets"]
        print("  event.category:", [(b["key"], b["doc_count"]) for b in cats])
        print("  event.type    :", [(b["key"], b["doc_count"]) for b in types])
    except Exception as e:
        print(f"  ERROR: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="elastic_config.yml")
    args = p.parse_args()
    main(args.config)
