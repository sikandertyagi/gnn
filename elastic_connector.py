"""
elastic_connector.py
────────────────────
Fetches endpoint telemetry from Elasticsearch and normalises it to the
pipeline's CSV schema (EventID, Image, ParentImage, CommandLine, …).

Supported data sources
──────────────────────
  sysmon   → Sysmon events forwarded via Winlogbeat or Elastic Agent
             Indices: winlogbeat-*, logs-windows.sysmon_operational-*
             Fields:  winlog.event_data.* (ECS-enriched)

  defend   → Elastic Defend endpoint events
             Indices: logs-endpoint.events.process-*,
                      logs-endpoint.events.network-*
             Fields:  ECS (process.*, destination.*, user.*, host.*)

  wazuh    → Wazuh alerts/archives forwarded to Elasticsearch
             Indices: wazuh-alerts-4.x-*, wazuh-archives-4.x-*
             Fields:  data.win.eventdata.* (Wazuh 3.x–4.x) or
                      winlog.event_data.* when Wazuh uses the Elastic
                      integration module

Output schema (pipeline columns)
──────────────────────────────────
  EventID, Image, ParentImage, CommandLine, User, Computer,
  SystemTime, DestinationIp, DestinationPort,
  ProcessGuid, ParentProcessGuid, IntegrityLevel, Company, Signed
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── pipeline output schema with safe fill defaults ──────────────────────────
_SCHEMA_DEFAULTS: Dict[str, Any] = {
    "EventID":            0,
    "Image":              "",
    "ParentImage":        "",
    "CommandLine":        "",
    "User":               "",
    "Computer":           "",
    "SystemTime":         "",
    "DestinationIp":      "",
    "DestinationPort":    0,
    "ProcessGuid":        "",
    "ParentProcessGuid":  "",
    "IntegrityLevel":     "",
    "Company":            "",
    "Signed":             False,
}

# ── Sysmon / Winlogbeat field map ────────────────────────────────────────────
# EventID read separately from winlog.event_id / event.code
_SYSMON_FIELD_MAP: Dict[str, str] = {
    "Image":             "winlog.event_data.Image",
    "ParentImage":       "winlog.event_data.ParentImage",
    "CommandLine":       "winlog.event_data.CommandLine",
    "User":              "winlog.event_data.User",
    "Computer":          "host.name",
    "SystemTime":        "@timestamp",
    "DestinationIp":     "winlog.event_data.DestinationIp",
    "DestinationPort":   "winlog.event_data.DestinationPort",
    "ProcessGuid":       "winlog.event_data.ProcessGuid",
    "ParentProcessGuid": "winlog.event_data.ParentProcessGuid",
    "IntegrityLevel":    "winlog.event_data.IntegrityLevel",
    "Company":           "winlog.event_data.Company",
    "Signed":            "winlog.event_data.Signed",
}

# ── Elastic Defend field map (ECS) ───────────────────────────────────────────
_DEFEND_FIELD_MAP: Dict[str, str] = {
    "Image":             "process.executable",
    "ParentImage":       "process.parent.executable",
    "CommandLine":       "process.command_line",
    "User":              "user.name",
    "Computer":          "host.name",
    "SystemTime":        "@timestamp",
    "DestinationIp":     "destination.ip",
    "DestinationPort":   "destination.port",
    "ProcessGuid":       "process.entity_id",
    "ParentProcessGuid": "process.parent.entity_id",
    "IntegrityLevel":    "process.token.integrity_level_name",
    "Company":           "process.code_signature.subject_name",
    "Signed":            "process.code_signature.trusted",
}

# ── Wazuh field map ───────────────────────────────────────────────────────────
# Wazuh 3.x / 4.x stores Sysmon fields under data.win.eventdata.*
# Newer Wazuh (4.9+ with ECS output) mirrors Winlogbeat; use sysmon source for that.
_WAZUH_FIELD_MAP: Dict[str, str] = {
    "Image":             "data.win.eventdata.image",
    "ParentImage":       "data.win.eventdata.parentImage",
    "CommandLine":       "data.win.eventdata.commandLine",
    "User":              "data.win.eventdata.user",
    "Computer":          "agent.name",
    "SystemTime":        "@timestamp",
    "DestinationIp":     "data.win.eventdata.destinationIp",
    "DestinationPort":   "data.win.eventdata.destinationPort",
    "ProcessGuid":       "data.win.eventdata.processGuid",
    "ParentProcessGuid": "data.win.eventdata.parentProcessGuid",
    "IntegrityLevel":    "data.win.eventdata.integrityLevel",
    "Company":           "data.win.eventdata.company",
    "Signed":            "data.win.eventdata.signed",
}

# Wazuh rule IDs that correspond to Sysmon EventID 1 and 3
_WAZUH_SYSMON_EID1_RULES = {"92000", "92001", "92002", "92003"}   # process creation
_WAZUH_SYSMON_EID3_RULES = {"92022", "92023"}                      # network connection
# Also check sysmon_id field if rule.id matching is insufficient
_WAZUH_SYSMON_EID_FIELD = "data.win.eventdata.sysmonId"


# ── utility: walk dot-separated key into nested dict ────────────────────────
def _get_nested(doc: Dict, dotted_key: str) -> Any:
    """Return value at dot-separated path in nested dict, or None if missing."""
    node = doc
    for part in dotted_key.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


# ── per-source row extractors ─────────────────────────────────────────────────
def _extract_sysmon_row(hit: Dict) -> Dict[str, Any]:
    src = hit.get("_source", {})
    raw_eid = (
        _get_nested(src, "winlog.event_id")
        or _get_nested(src, "event.code")
        or 0
    )
    try:
        event_id = int(raw_eid)
    except (ValueError, TypeError):
        event_id = 0

    row: Dict[str, Any] = {"EventID": event_id}
    for col, es_field in _SYSMON_FIELD_MAP.items():
        row[col] = _get_nested(src, es_field)
    return row


def _extract_defend_row(hit: Dict) -> Dict[str, Any]:
    src = hit.get("_source", {})
    event_cats = src.get("event", {}).get("category", [])
    if isinstance(event_cats, str):
        event_cats = [event_cats]

    # Synthesise Sysmon-equivalent EventID
    if "process" in event_cats:
        event_id = 1
    elif "network" in event_cats:
        event_id = 3
    else:
        event_id = 0

    row: Dict[str, Any] = {"EventID": event_id}
    for col, es_field in _DEFEND_FIELD_MAP.items():
        row[col] = _get_nested(src, es_field)
    return row


def _extract_wazuh_row(hit: Dict) -> Dict[str, Any]:
    src = hit.get("_source", {})

    # Determine EventID from Wazuh rule ID or sysmonId field
    rule_id = str(_get_nested(src, "rule.id") or "")
    sysmon_id = str(_get_nested(src, _WAZUH_SYSMON_EID_FIELD) or "")

    if rule_id in _WAZUH_SYSMON_EID1_RULES or sysmon_id == "1":
        event_id = 1
    elif rule_id in _WAZUH_SYSMON_EID3_RULES or sysmon_id == "3":
        event_id = 3
    else:
        # Fall back: try to parse from data.win.system.eventID
        raw_eid = _get_nested(src, "data.win.system.eventID") or 0
        try:
            event_id = int(raw_eid)
        except (ValueError, TypeError):
            event_id = 0

    row: Dict[str, Any] = {"EventID": event_id}
    for col, es_field in _WAZUH_FIELD_MAP.items():
        row[col] = _get_nested(src, es_field)
    return row


# ── row normalisation ─────────────────────────────────────────────────────────
def _normalise_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce types and fill missing values with schema defaults."""
    result: Dict[str, Any] = {}
    for col, default in _SCHEMA_DEFAULTS.items():
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            val = default
        result[col] = val

    # EventID: int
    try:
        result["EventID"] = int(result["EventID"])
    except (ValueError, TypeError):
        result["EventID"] = 0

    # DestinationPort: int
    try:
        result["DestinationPort"] = int(result["DestinationPort"])
    except (ValueError, TypeError):
        result["DestinationPort"] = 0

    # Signed: bool
    if isinstance(result["Signed"], str):
        result["Signed"] = result["Signed"].strip().lower() in ("true", "1", "yes")

    return result


# ── main connector class ──────────────────────────────────────────────────────
class ElasticConnector:
    """
    Connects to an Elasticsearch cluster and fetches endpoint telemetry as a
    normalised DataFrame matching the GNN anomaly pipeline's CSV schema.

    Authentication (choose one)
    ────────────────────────────
      api_key              Base64 "id:api_key" string (recommended)
      username + password  Basic auth (use with HTTPS only)

    TLS
    ────
      ca_certs    Path to CA certificate bundle (for self-signed / private CA)
      verify_certs  False disables cert verification (dev/lab only)

    Parameters
    ──────────
    host : str
        Full URL including scheme and port, e.g. "https://elastic.corp:9200"
        For Elastic Cloud use the Cloud ID instead: set cloud_id= and omit host.
    cloud_id : str, optional
        Elastic Cloud deployment ID (alternative to host).
    api_key : str, optional
        API key in base64 "id:key" format.
    username / password : str, optional
        Basic auth credentials.
    ca_certs : str, optional
        Path to CA bundle file.
    verify_certs : bool
        Whether to verify TLS certificates (default True).
    page_size : int
        Hits per search_after page (default 5 000). Reduce for slow clusters.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        cloud_id: Optional[str] = None,
        api_key: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        ca_certs: Optional[str] = None,
        verify_certs: bool = True,
        page_size: int = 5_000,
    ):
        try:
            from elasticsearch import Elasticsearch
        except ImportError as exc:
            raise ImportError(
                "Install the Elasticsearch client: "
                "pip install 'elasticsearch>=8.0.0,<9.0.0'"
            ) from exc

        # Suppress urllib3 InsecureRequestWarning when cert verification is
        # disabled — without this every paginated request prints a warning.
        if not verify_certs:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        kwargs: Dict[str, Any] = {
            "verify_certs":  verify_certs,
            "ssl_show_warn": verify_certs,   # mirror: False hides ES-client SSL logs
        }

        if cloud_id:
            kwargs["cloud_id"] = cloud_id
        elif host:
            kwargs["hosts"] = [host]
        else:
            raise ValueError("Provide either host= or cloud_id=")

        if ca_certs:
            kwargs["ca_certs"] = ca_certs

        if api_key:
            kwargs["api_key"] = api_key
        elif username and password:
            kwargs["basic_auth"] = (username, password)

        self._es = Elasticsearch(**kwargs)
        self._page_size = page_size
        logger.info(
            "ElasticConnector: connected to %s",
            cloud_id if cloud_id else host,
        )

    # ── low-level pagination ──────────────────────────────────────────────────
    def _paginate(
        self,
        index: str,
        query: Dict,
    ) -> Generator[Dict, None, None]:
        """
        Yield raw hit dicts using search_after + (_id, @timestamp) sort.
        This is more reliable than scroll for time-windowed queries.
        """
        body: Dict[str, Any] = {
            "query": query,
            "size": self._page_size,
            "sort": [
                {"@timestamp": {"order": "asc"}},
                {"_id":        {"order": "asc"}},
            ],
        }
        search_after = None
        total = 0

        while True:
            if search_after:
                body["search_after"] = search_after

            resp = self._es.search(index=index, body=body, expand_wildcards="all")
            hits = resp["hits"]["hits"]
            if not hits:
                break

            for hit in hits:
                yield hit
            total += len(hits)
            search_after = hits[-1]["sort"]
            logger.debug("  …%d events fetched from %s", total, index)

    # ── query builders ────────────────────────────────────────────────────────
    @staticmethod
    def _date_range_filter(start: str, end: str) -> Dict:
        return {"range": {"@timestamp": {"gte": start, "lte": end}}}

    def _sysmon_query(self, start: str, end: str, event_ids: List[int]) -> Dict:
        return {
            "bool": {
                "must": [
                    self._date_range_filter(start, end),
                    # winlog.event_id is a keyword field in Winlogbeat
                    {"terms": {"winlog.event_id": [str(e) for e in event_ids]}},
                ]
            }
        }

    def _defend_query(self, start: str, end: str) -> Dict:
        return {
            "bool": {
                "must": [
                    self._date_range_filter(start, end),
                    {
                        "bool": {
                            "should": [
                                # Process start event
                                {
                                    "bool": {
                                        "must": [
                                            {"term":  {"event.category": "process"}},
                                            {"term":  {"event.type":     "start"}},
                                        ]
                                    }
                                },
                                # Network connection event
                                {
                                    "bool": {
                                        "must": [
                                            {"term": {"event.category": "network"}},
                                            {
                                                "terms": {
                                                    "event.type": [
                                                        "connection",
                                                        "start",
                                                        "protocol",
                                                    ]
                                                }
                                            },
                                        ]
                                    }
                                },
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                ]
            }
        }

    def _wazuh_query(self, start: str, end: str, event_ids: List[int]) -> Dict:
        """
        Wazuh stores Sysmon events as alerts; filter by rule.id ranges that
        correspond to Sysmon EventID 1 and 3, or by the raw sysmonId field.
        """
        sysmon_id_strs = [str(e) for e in event_ids]
        eid1_rules = sorted(_WAZUH_SYSMON_EID1_RULES) if 1 in event_ids else []
        eid3_rules = sorted(_WAZUH_SYSMON_EID3_RULES) if 3 in event_ids else []
        all_rules  = eid1_rules + eid3_rules

        should_clauses: List[Dict] = []
        if all_rules:
            should_clauses.append({"terms": {"rule.id": all_rules}})
        should_clauses.append(
            {"terms": {_WAZUH_SYSMON_EID_FIELD: sysmon_id_strs}}
        )
        should_clauses.append(
            {"terms": {"data.win.system.eventID": sysmon_id_strs}}
        )

        return {
            "bool": {
                "must": [
                    self._date_range_filter(start, end),
                    {"bool": {"should": should_clauses, "minimum_should_match": 1}},
                ]
            }
        }

    # ── public fetch methods ──────────────────────────────────────────────────
    def fetch_sysmon(
        self,
        start: str,
        end: str,
        index: str = "winlogbeat-*,logs-windows.sysmon_operational-*",
        event_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Fetch Sysmon process-creation (EventID 1) and network-connection
        (EventID 3) events from Winlogbeat / Elastic Agent indices.

        Parameters
        ──────────
        start, end : ISO-8601 UTC strings, e.g. "2025-11-01T00:00:00Z"
        index      : comma-separated Elasticsearch index pattern
        event_ids  : EventIDs to fetch (default: [1, 3])
        """
        if event_ids is None:
            event_ids = [1, 3]

        query = self._sysmon_query(start, end, event_ids)
        rows  = [
            _normalise_row(_extract_sysmon_row(hit))
            for hit in self._paginate(index, query)
        ]
        df = _rows_to_df(rows)
        logger.info("fetch_sysmon : %6d events  [%s → %s]", len(df), start, end)
        return df

    def fetch_defend(
        self,
        start: str,
        end: str,
        index: str = (
            "logs-endpoint.events.process-*,"
            "logs-endpoint.events.network-*"
        ),
    ) -> pd.DataFrame:
        """
        Fetch Elastic Defend process-start and network-connection events.

        Parameters
        ──────────
        start, end : ISO-8601 UTC strings
        index      : comma-separated index pattern
        """
        query = self._defend_query(start, end)
        rows  = [
            _normalise_row(_extract_defend_row(hit))
            for hit in self._paginate(index, query)
        ]
        df = _rows_to_df(rows)
        logger.info("fetch_defend : %6d events  [%s → %s]", len(df), start, end)
        return df

    def fetch_wazuh(
        self,
        start: str,
        end: str,
        index: str = "wazuh-alerts-4.x-*,wazuh-archives-4.x-*",
        event_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Fetch Sysmon EventID 1 and 3 events from Wazuh alert/archive indices.

        Parameters
        ──────────
        start, end : ISO-8601 UTC strings
        index      : Wazuh index pattern
        event_ids  : Sysmon EventIDs to extract (default: [1, 3])
        """
        if event_ids is None:
            event_ids = [1, 3]

        query = self._wazuh_query(start, end, event_ids)
        rows  = [
            _normalise_row(_extract_wazuh_row(hit))
            for hit in self._paginate(index, query)
        ]
        df = _rows_to_df(rows)
        logger.info("fetch_wazuh  : %6d events  [%s → %s]", len(df), start, end)
        return df

    def fetch_all(
        self,
        start: str,
        end: str,
        sources: Optional[List[str]] = None,
        sysmon_index: str = "winlogbeat-*,logs-windows.sysmon_operational-*",
        defend_index: str = (
            "logs-endpoint.events.process-*,"
            "logs-endpoint.events.network-*"
        ),
        wazuh_index: str = "wazuh-alerts-4.x-*,wazuh-archives-4.x-*",
    ) -> pd.DataFrame:
        """
        Fetch from all requested sources, merge, deduplicate, and sort.

        Parameters
        ──────────
        sources : list containing any of "sysmon", "defend", "wazuh"
                  (default: ["sysmon", "defend"])
        """
        if sources is None:
            sources = ["sysmon", "defend"]

        frames: List[pd.DataFrame] = []

        if "sysmon" in sources:
            frames.append(self.fetch_sysmon(start, end, index=sysmon_index))
        if "defend" in sources:
            frames.append(self.fetch_defend(start, end, index=defend_index))
        if "wazuh" in sources:
            frames.append(self.fetch_wazuh(start, end, index=wazuh_index))

        if not frames:
            return pd.DataFrame(columns=list(_SCHEMA_DEFAULTS.keys()))

        df = pd.concat(frames, ignore_index=True)

        # keep only EventID 1 (process creation) and 3 (network connection)
        df = df[df["EventID"].isin([1, 3])].copy()

        # deduplicate by (Computer, SystemTime, Image, EventID)
        before = len(df)
        df["_ts_parsed"] = pd.to_datetime(df["SystemTime"], utc=True, errors="coerce")
        df = df.drop_duplicates(
            subset=["Computer", "_ts_parsed", "Image", "EventID"]
        ).sort_values("_ts_parsed").reset_index(drop=True)
        after = len(df)

        if before - after > 0:
            logger.info(
                "Deduplication: %d → %d events (%d duplicates removed)",
                before, after, before - after,
            )

        # normalise SystemTime to pipeline-expected string format
        df["SystemTime"] = df["_ts_parsed"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
        df = df.drop(columns=["_ts_parsed"])

        logger.info(
            "fetch_all    : %6d events  [%s → %s]  sources=%s",
            len(df), start, end, sources,
        )
        return df


# ── helpers ────────────────────────────────────────────────────────────────────
def _rows_to_df(rows: List[Dict]) -> pd.DataFrame:
    """Convert list of normalised row dicts to a DataFrame with correct columns."""
    if rows:
        return pd.DataFrame(rows, columns=list(_SCHEMA_DEFAULTS.keys()))
    return pd.DataFrame(columns=list(_SCHEMA_DEFAULTS.keys()))
