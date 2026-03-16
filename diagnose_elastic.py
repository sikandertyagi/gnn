"""
diagnose_elastic.py
────────────────────
Samples raw documents from the defend indices and shows exactly what
event.category / event.type values are present, without any filter.

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


def main(config_path):
    cfg = load_cfg(config_path)

    es = Elasticsearch(
        hosts=[cfg["host"]],
        basic_auth=(cfg.get("username"), cfg.get("password")),
        verify_certs=False,
        ssl_show_warn=False,
    )

    index = cfg.get(
        "defend_index",
        "logs-endpoint.events.process-*,.ds-logs-endpoint.events.process-*",
    )
    train_start = cfg.get("train_start", "2025-11-01T00:00:00Z")
    train_end   = cfg.get("train_end",   "2025-12-31T23:59:59Z")

    print(f"\n=== Indices matching pattern ===")
    try:
        cat = es.cat.indices(index=index, expand_wildcards="all", h="index,docs.count,store.size", s="index")
        print(cat.body if hasattr(cat, "body") else cat)
    except Exception as e:
        print(f"  (cat indices error: {e})")

    print(f"\n=== Sample docs (match_all, no category filter, date range {train_start} → {train_end}) ===")
    resp = es.search(
        index=index,
        expand_wildcards="all",
        body={
            "query": {"range": {"@timestamp": {"gte": train_start, "lte": train_end}}},
            "size": 3,
            "_source": ["@timestamp", "event.category", "event.type", "event.action",
                        "host.hostname", "process.name", "process.executable"],
        },
    )
    hits = resp["hits"]["hits"]
    total = resp["hits"]["total"]
    print(f"Total hits (date range only, no category filter): {total}")
    for h in hits:
        print(json.dumps(h["_source"], indent=2))

    print(f"\n=== Unique event.category values (agg) ===")
    resp2 = es.search(
        index=index,
        expand_wildcards="all",
        body={
            "query": {"range": {"@timestamp": {"gte": train_start, "lte": train_end}}},
            "size": 0,
            "aggs": {
                "categories": {"terms": {"field": "event.category", "size": 20}},
                "types":      {"terms": {"field": "event.type",     "size": 20}},
            },
        },
    )
    cats = resp2["aggregations"]["categories"]["buckets"]
    types = resp2["aggregations"]["types"]["buckets"]
    print("event.category:", [(b["key"], b["doc_count"]) for b in cats])
    print("event.type    :", [(b["key"], b["doc_count"]) for b in types])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", "-c", default="elastic_config.yml")
    args = p.parse_args()
    main(args.config)
