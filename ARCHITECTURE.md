# Architecture Reference — Endpoint Anomaly Detection Pipeline

This document is the authoritative description of the pipeline. It is written
so that any reader — human or AI tool — with no prior context can understand
**what every file does, what it consumes, what it produces, and how all pieces
connect**. Every section reflects the current code exactly.

---

## Table of Contents

1. [System Goal](#1-system-goal)
2. [End-to-End Data Flow](#2-end-to-end-data-flow)
3. [Data Ingestion](#3-data-ingestion)
4. [Feature Vector Specification](#4-feature-vector-specification)
5. [File Reference](#5-file-reference)
6. [Composite Scoring Formula](#6-composite-scoring-formula)
7. [Key Design Decisions](#7-key-design-decisions)
8. [Artifacts on Disk](#8-artifacts-on-disk)
9. [Configuration Quick Reference](#9-configuration-quick-reference)
10. [How to Run](#10-how-to-run)

---

## 1. System Goal

Detect cyber-attack activity (malware execution, C2 communication, privilege
escalation, lateral movement) in Windows endpoint telemetry — **without relying
on signatures and without requiring labelled attack data at training time**.

The pipeline ingests data from Elasticsearch (Elastic Defend, Sysmon via
Winlogbeat, or Wazuh) and processes it through multiple anomaly detection models.

The fundamental principle:

> *Train every model exclusively on normal (benign) behaviour. Anything the
> model cannot reconstruct or has never seen before is, by definition, anomalous.*

Four complementary anomaly signals are computed independently and fused into a
single `composite_score` in [0, 1] per event.

| Signal | Module | What it measures | Weight |
|---|---|---|---|
| `dense_error` | `dense_autoencoder.py` | How poorly a single-event dense AE reconstructs the event | 0.50 |
| `rarity_score` | `rarity_engine.py` | How rare the process-lineage / destination-IP patterns are vs benign baseline | 0.30 |
| `recon_error` | `transformer_autoencoder.py` | How poorly the Transformer reconstructs a 20-event sequence | 0.20 |
| `graph_score` | `gnn_encoder.py` | How anomalous a process node is in the system-call graph | 0.00 (disabled) |

---

## 2. End-to-End Data Flow

```
Elasticsearch (Elastic Defend / Sysmon / Wazuh)
        │
        ▼  elastic_ingest.py — DATA INGESTION
        │
        │  Connects to Elasticsearch via elastic_connector.py.
        │  Fetches training data (Nov 2025) → Label=0 (benign baseline)
        │  Fetches evaluation data (Jan 17–25 2026) → Label=2 (unlabelled, to-be-predicted)
        │  Streams to CSV in chunks (memory-efficient).
        │
        ▼  elastic_data.csv  (combined CSV, one row per event)
        │
        ▼  main.py — step 1: LOAD
        │
        │  Optional EventID filter (configurable via HIGH_SIGNAL_EVENTIDS).
        │  DataFrame index reset to contiguous 0..N-1 RangeIndex.
        │
        ▼  main.py — step 2: FEATURE ENGINEERING  (feature_engineering.py)
        │
        │  Raw event columns → fixed-width numeric feature matrix.
        │  Encoding strategy:
        │    · High-cardinality cols (Computer, DestinationPortName):
        │        frequency encoding + CRC32 hash-bucket encoding + flags
        │    · Low-cardinality cols (EventID, Initiated, SourceIsIpv6, time parts):
        │        OneHotEncoding (bounded unique values)
        │    · Numerical cols: passed through as-is
        │
        │  Feature dimension is FIXED regardless of fleet size.
        │
        ▼  main.py — step 3: NORMALISATION  (normaliser.py)
        │
        │  MinMaxScaler fitted on benign rows (Label==0) only.
        │  Binary columns (OHE + hash buckets + flags) are skipped —
        │  already in {0, 1}.
        │  Float columns (frequencies + numericals) are scaled to [0, 1].
        │
        ├──────────────────────────────────────────────────────┐
        │                                                      │
        ▼  steps 4–6: GNN PATH                                ▼  step 7: RARITY PATH
        │                                                      │
        │  graph_builder.py                                    │  rarity_engine.py
        │  Heterogeneous graph:                                │
        │    process → process  (parent_of)                   │  Fit on benign rows.
        │    process → ip       (connects_to)                 │  Score all events via
        │    process → user     (runs_as)                     │  vectorised pandas merge.
        │    process → host     (runs_on)                     │  Three sub-signals:
        │    + all reverse edges                               │    parent→child rarity
        │                                                      │    process→IP rarity
        │  gnn_encoder.py                                      │    destination-IP rarity
        │  Two-layer HeteroGraphSAGE encoder.                 │
        │  Trained on benign-only graph (self-supervised       │  → event_rarity_scores
        │  node-feature MSE reconstruction).                   │    (N_events,) ∈ [0,1]
        │                                                      │
        │  → event_graph_scores (N_events,) ∈ [0,1]          │
        │                                                      │
        ├──────────────────────┬───────────────────────────────┘
        │                      │
        ▼  step 8: DENSE AE    ▼  step 9: TRANSFORMER PATH
        │                      │
        │  dense_autoencoder   │  sequence_builder.py
        │  Single-event dense  │  1. _add_temporal_features():
        │  autoencoder.        │       · time_delta_seconds
        │  Trained on benign   │       · log_time_delta
        │  events only.        │       · event_burst_count
        │  Architecture:       │     All three z-scored on benign rows.
        │  F→128→64→32→16      │
        │  16→32→64→128→F      │  2. Sliding windows (seq_len=20) per host:
        │  (LayerNorm + ReLU,  │       Small path (<200k): in-memory ndarray
        │   final sigmoid)     │       Large path (≥200k): np.memmap on disk
        │                      │
        │  → event_dense_errors│  transformer_autoencoder.py
        │    (N_events,)       │  Trained on benign sequences only.  MSE loss.
        │                      │
        │                      │  → event_recon_errors (N_events,)
        │                      │    NaN for warmup events
        │                      │
        └──────────┬───────────┘
                   │
                   ▼  main.py — step 10: COMPOSITE SCORING
                   │
                   │  anomaly_engine.py
                   │  Each signal min-max normalised on benign rows.
                   │  composite = 0.50·dense + 0.20·recon + 0.00·graph + 0.30·rarity
                   │  Warmup events: recon weight redistributed to remaining signals.
                   │
                   ▼  alert_aggregator.py
                   │
                   │  Flag events with score ≥ threshold (p97.5 of benign scores).
                   │  Group consecutive flagged events (gap ≤ 300 s) per host
                   │  → "attack chains"
                   │  → alerts.csv
                   │
                   ▼  anomaly_report.py
                   │
                   │  anomaly_report.txt  → narrative triage report
                   │  flagged_events.csv  → all flagged rows with context
                   │
                   ▼  metrics.py  (only when HAS_GROUND_TRUTH = True)
                   │
                   │  ROC-AUC, PR-AUC, F1, threshold sweep,
                   │  per-component ablation AUCs, alert-level metrics.
                   │  → metrics.json, roc_curve.csv, pr_curve.csv, …
```

---

## 3. Data Ingestion

### `elastic_ingest.py`

One-shot data preparation script that pulls endpoint telemetry from
Elasticsearch and writes the combined CSV consumed by `main.py`.

**Usage:**
```bash
python elastic_ingest.py                         # uses elastic_config.yml
python elastic_ingest.py --config /path/to/cfg.yml
python elastic_ingest.py --dry-run               # print config only
python elastic_ingest.py --resume                # resume interrupted run
python elastic_ingest.py --eval-only             # append eval window to existing CSV
```

**Streaming architecture:** Results are written to CSV in chunks as they arrive
from Elasticsearch (`_stream_window_to_csv`), never holding more than one page
(~5,000 rows) in memory. Each chunk filters to EventID 1 (process creation)
and EventID 3 (network connection), drops rows with empty `Image` fields.

**Label semantics:**

| Label | Meaning | Used for |
|---|---|---|
| 0 | Benign / training period | All models trained exclusively on this |
| 1 | Confirmed attack | Set manually via `relabel_anomalies.py`; enables AUC/F1 metrics |
| 2 | Unlabelled / to-be-predicted | Scored against Label=0 baseline; never trained on |

**`--resume` mode:** Reads the existing output CSV to determine where the
previous run stopped (max timestamp + 1ms). If `last_label == 0`, resumes
training window. If `last_label == 2`, skips training and resumes eval window.
No separate checkpoint file — the CSV itself is the state record.

**`--eval-only` mode:** Skips the training window entirely and appends only
the evaluation window to the existing CSV. Use when training data is complete
and you want to add a new eval period without re-fetching training data.

### `elastic_connector.py`

Low-level Elasticsearch client supporting three data source types:

| Source | Index patterns | Field namespace |
|---|---|---|
| Sysmon (Winlogbeat) | `winlogbeat-*`, `logs-windows.sysmon_operational-*` | `winlog.event_data.*` |
| Elastic Defend | `logs-endpoint.events.process-*`, `logs-endpoint.events.network-*` | `process.*`, `destination.*` (ECS) |
| Wazuh | `wazuh-alerts-4.x-*`, `wazuh-archives-4.x-*` | `data.win.eventdata.*` |

**Output schema (14 columns):**
`EventID, Image, ParentImage, CommandLine, User, Computer, SystemTime,
DestinationIp, DestinationPort, ProcessGuid, ParentProcessGuid,
IntegrityLevel, Company, Signed`

**Features:** `search_after` pagination, retry with exponential backoff (up to
6 retries), row normalisation with schema defaults, deduplication by
`(Computer, SystemTime, Image, EventID)`.

### `elastic_config.yml`

YAML configuration for ingestion. All values can be overridden by environment
variables (`ES_HOST`, `ES_USERNAME`, `ES_PASSWORD`, etc.).

Key sections:
- Connection (host, credentials, TLS, page_size)
- Data sources (`defend`, `sysmon`, `wazuh` — mix and match)
- Date windows (`train_start/end`, `eval_start/end`)
- Output path

### `diagnose_elastic.py`

Diagnostic tool for debugging zero-result queries. Runs progressively more
targeted queries against Elasticsearch and prints exactly what ES returns.

---

## 4. Feature Vector Specification

Total feature count is **fixed regardless of fleet size**. The exact number
depends on the low-cardinality column value counts, but the high-cardinality
columns produce a constant number of features.

### Column ordering: binary columns first, then float columns

The first `n_binary_cols` columns in `feature_cols` are binary ({0, 1}) and
skipped by the MinMaxScaler. The remaining columns are floats that get scaled.

### Section A — Low-cardinality OHE (binary, not scaled)

OneHotEncoded columns for bounded-cardinality categoricals:

| Source column | Typical unique values |
|---|---|
| `EventID` | 2 (EID 1 and 3) |
| `Initiated` | 2–3 |
| `SourceIsIpv6` | 2–3 |
| `SystemTime_year` | 1–2 |
| `SystemTime_month` | 2–4 |
| `SystemTime_week` | 4–8 |
| `SystemTime_day_of_week` | 7 |

### Section B — Hash-bucket encoding (binary, not scaled)

CRC32 hash of high-cardinality string values mapped to fixed-width buckets.
Deterministic across runs (uses `zlib.crc32`, not `hash()`).

| Source column | Buckets | Features produced |
|---|---|---|
| `Computer` | 16 | `Computer_hash_0` … `Computer_hash_15` |
| `DestinationPortName` | 8 | `DestinationPortName_hash_0` … `DestinationPortName_hash_7` |

### Section C — Flags (binary, not scaled)

| Name | Description |
|---|---|
| `port_is_wellknown` | 1 if `DestinationPortName` is in the well-known set (HTTP, HTTPS, DNS, SSH, RDP, SMB, etc.) |

### Section D — Frequency encoding (float, scaled)

Proportion of events in the dataset with that value. Unseen values at
inference time get frequency 0.0 (a useful anomaly signal — rare = suspicious).

| Name | Source |
|---|---|
| `Computer_freq` | `Computer` |
| `DestinationPortName_freq` | `DestinationPortName` |

### Section E — Numerical features (float, scaled)

| Name | Source | Description |
|---|---|---|
| `EventRecordID` | Raw column | ES internal record ID |
| `Execution_ProcessID` | Raw column | PID at execution time |
| `ProcessId` | Raw column | Process ID |
| `SystemTime_day` | Parsed from `SystemTime` | Day of month (1–31) |
| `SystemTime_hour` | Parsed from `SystemTime` | Hour of day (0–23) |
| `SystemTime_minute` | Parsed from `SystemTime` | Minute (0–59) |

### Section F — Temporal behaviour features (added by `sequence_builder`)

Added **after** the scaler runs, z-scored on benign rows inside
`_add_temporal_features`.

| Name | Formula | Notes |
|---|---|---|
| `time_delta_seconds` | `diff(SystemTime)` per host, clipped ≥ 0 | 0 for first event per host |
| `log_time_delta` | `log1p(time_delta_seconds)` | Compresses multi-hour idle gaps |
| `event_burst_count` | Events on same host in [t−60s, t) | O(n log n) via `searchsorted` |

### Derived columns (in DataFrame, not in feature matrix)

These columns are written to the DataFrame for use by downstream modules
(RarityEngine, graph builder) but are **not** part of the model feature vector:

| Name | Used by |
|---|---|
| `process_name` | RarityEngine, graph_builder |
| `parent_process` | RarityEngine |
| `parent_child` | RarityEngine |

---

## 5. File Reference

---

### `config.py`

**Role:** Single source of truth for all hyperparameters, weights, and file paths.
**Imported by:** Every other module.
**Never modify any other file to change a number** — edit `config.py` instead.

Key parameters:

```
DATA_DIR                 = "/gulshan_data/anomaly_detection"
ARTIFACTS_DIR            = "/home/rohit/elastic"
DATA_PATH                = DATA_DIR + "/elastic_data.csv"
RANDOM_SEED              = 42
SEQUENCE_LENGTH          = 20         sliding-window length
TRAIN_LABEL              = 0          label meaning "benign"
BATCH_SIZE               = 512
EPOCHS                   = 20         max transformer training epochs
LEARNING_RATE            = 1e-3
VAL_RATIO                = 0.15
EARLY_STOPPING_PATIENCE  = 5
EMBED_DIM                = 96         transformer internal width
NUM_HEADS                = 4
NUM_LAYERS               = 2
FF_DIM                   = 256
HIGH_SIGNAL_EVENTIDS     = None       None = use all events
GNN_EMBED_DIM            = 64
GNN_EPOCHS               = 15
GNN_LR                   = 1e-3
DENSE_HIDDEN_DIMS        = (128, 64, 32, 16)
DENSE_EPOCHS             = 100
DENSE_WEIGHT             = 0.50
RECON_WEIGHT             = 0.20
GRAPH_WEIGHT             = 0.00       disabled (GNN ablation AUC < random)
RARITY_WEIGHT            = 0.30
ALERT_PERCENTILE         = 97.5       percentile of benign scores → threshold
ALERT_WINDOW             = 300        seconds
LARGE_DATASET_THRESHOLD  = 200_000
INFER_BATCH_SIZE         = 2048
HAS_GROUND_TRUTH         = False
```

---

### `main.py`

**Role:** Orchestrates all ten pipeline steps from CSV load to report output.
**Entry point:** `python main.py`

| Step | Action | Module called |
|---|---|---|
| 1 | Load CSV; optional EventID filter; reset index | — |
| 2 | Feature engineering (OHE + freq/hash encoding) | `feature_engineering.feature_engineering` |
| 3 | Fit + apply MinMaxScaler (benign-only fit) | `normaliser.fit_scaler`, `apply_scaler` |
| 4 | Build full + benign-only heterogeneous graphs | `graph_builder.build_event_graph` |
| 5 | Train GNN on benign graph | `gnn_encoder.train_gnn` |
| 6 | Compute per-event graph anomaly scores | `gnn_encoder.graph_anomaly_scores` |
| 7 | Fit rarity engine; score all events | `rarity_engine.RarityEngine` |
| 8 | Train dense autoencoder; score all events | `dense_autoencoder.train_dense_ae` |
| 9 | Build sequences; train Transformer; compute recon errors | `sequence_builder`, `transformer_autoencoder`, `train`, `evaluate` |
| 10 | Composite score; alerts; report; evaluation | `anomaly_engine`, `alert_aggregator`, `anomaly_report`, `metrics` |

**Score alignment (step 9):**
`build_sequences` iterates hosts in DataFrame order. Window `j` on a host ends
at host event index `j + SEQUENCE_LENGTH − 1`. Each window score maps to that
last event's position. The first `SEQUENCE_LENGTH − 1 = 19` events per host
are warmup and receive `NaN`.

---

### `feature_engineering.py`

**Role:** Converts raw event rows into a fixed-width numeric feature matrix.

**Encoding strategy:**

| Column type | Encoding | Dimension |
|---|---|---|
| `Computer` (high-cardinality) | 1 frequency + 16 hash buckets | 17 (fixed) |
| `DestinationPortName` (high-cardinality) | 1 frequency + 8 hash buckets + 1 well-known flag | 10 (fixed) |
| Low-cardinality categoricals (7 cols) | OneHotEncoding | ~15–20 (bounded) |
| Numerical (6 cols) | Pass-through | 6 |
| **Total** | | **~48–53 (fixed)** |

**Why not OHE for everything:** With hundreds of unique hostnames or ports, OHE
produces thousands of sparse binary columns, causing memory explosion, curse of
dimensionality, and broken generalisation for unseen values. Frequency + hash
encoding keeps the dimension fixed and handles unseen values gracefully
(frequency=0 is itself an anomaly signal).

**Frequency maps** are saved to `freq_maps.pkl` for use during inference.
**OHE encoder** is saved to `ohe_encoder.pkl`.

**Missing-column safety:** `_OPTIONAL_COLS` dict defines every optional column
with a safe default. Any column absent from the CSV is filled before feature
computation, so the pipeline runs on any endpoint data config.

---

### `normaliser.py`

**Role:** Fits and applies a `MinMaxScaler` to the float portion of the feature
matrix, leaving binary columns (OHE + hash buckets + flags) untouched.

**Public API:**
```python
scaler = fit_scaler(df, feature_cols, n_ohe_cols)  # returns + saves scaler.pkl
df     = apply_scaler(df, feature_cols, scaler, n_ohe_cols)
scaler = load_scaler()                              # reload for inference
```

**What is scaled:** `feature_cols[n_ohe_cols:]` — frequency and numerical columns.
**What is NOT scaled:** Binary columns at positions `0` to `n_ohe_cols-1`.

**Benign-only fitting:** `scaler.fit(df[df["Label"] == TRAIN_LABEL][numeric_cols])`.
Prevents attack feature distributions from widening the scale and making attack
values appear closer to normal.

---

### `sequence_builder.py`

**Role:** Injects temporal features and builds overlapping sliding-window
sequences of events per host.

**Temporal feature injection — `_add_temporal_features(df, feature_cols)`:**
Called at the top of both build functions. Sorts df by `[Computer, SystemTime]`,
computes three features, z-scores them on benign rows, then appends the names
to `feature_cols`.

**Sliding windows** use `numpy.lib.stride_tricks.sliding_window_view` (zero-copy).
Label: label of the **last** event in the window.

**Memmap algorithm (large datasets):**
1. Pass 1 — count total windows per host.
2. Allocate two memmap files with exact required shape.
3. Pass 2 — write host-by-host. Peak RAM = one host's event matrix.

---

### `transformer_autoencoder.py`

**Role:** Defines the `TransformerAutoencoder` nn.Module.

**Architecture:**
```
Input  (B, T=20, F)
  │
  ├─ input_proj: Linear(F, 96) + LayerNorm(96)
  ├─ pos_enc: SinusoidalPositionalEncoding(96)
  ├─ encoder: TransformerEncoder(d=96, heads=4, layers=2, ff=256)
  ├─ mean-pool over T → (B, 96)
  └─ bottleneck: Linear(96, 24)

Bottleneck z  (B, 24)
  │
  ├─ bottleneck_expand: Linear(24, 96)
  ├─ broadcast to (B, T, 96) → memory
  ├─ query tokens → query_proj: Linear(96, 96)
  ├─ decoder: TransformerDecoder(d=96, heads=4, layers=2, ff=256)
  └─ output_proj: Linear(96, F)

Output  (B, T=20, F)
```

**Training objective:** `MSELoss(output, input)` on benign sequences only.
High MSE at inference = model cannot explain the sequence = anomaly.

---

### `dense_autoencoder.py`

**Role:** Single-event dense (feedforward) autoencoder for sharp per-event
anomaly detection without sequence dilution.

**Architecture (validated at ROC-AUC 0.9950 standalone):**
```
Encoder: F → 128 → 64 → 32 → 16  (ReLU + LayerNorm)
Decoder: 16 → 32 → 64 → 128 → F  (ReLU + LayerNorm, final sigmoid)
```

**Training:** MSE on benign events only. Adam with ReduceLROnPlateau scheduler.
**Scoring:** Per-event MSE reconstruction error.

---

### `train.py`

**Role:** Two training entry points for the Transformer autoencoder.

- **`train_model`** — in-memory path. AdamW + linear warmup + cosine annealing.
  AMP, gradient clipping, early stopping.
- **`train_model_large`** — disk-backed path. `MemmapDataset` reads from
  `np.memmap`. Peak RAM = one batch at a time.

---

### `evaluate.py`

**Role:** Batched inference — computes per-sequence MSE reconstruction error.
Handles both in-memory `ndarray` and disk-backed `memmap` inputs.

---

### `rarity_engine.py`

**Role:** Scores how rarely each event's behavioural patterns appeared during
benign training. Three sub-signals averaged per event:

| Signal | Pair tracked |
|---|---|
| `parent_child` | `(parent_process_name, child_process_name)` |
| `proc_ip` | `(process_name, DestinationIp)` |
| `network_dest` | `DestinationIp` |

**Scoring formula (Jeffreys / α=0.5 smoothing):**
```
p(pattern) = (count + 0.5) / (total_count + 0.5 × vocab_size)
rarity     = 1 − p(pattern)
```

All operations vectorised via pandas `groupby` + `merge`. Safe for millions
of events.

---

### `graph_builder.py`

**Role:** Converts events into `torch_geometric.data.HeteroData` graph.

**Node types:** `process`, `ip`, `user`, `host`.
**Edge types (bidirectional):** `parent_of`, `connects_to`, `runs_as`, `runs_on`
(+ reverses for bidirectional message-passing).

**Shared encoders:** Both full and benign graphs use the same encoders so node
orderings match and GNN embeddings map correctly back to events.

---

### `gnn_encoder.py`

**Role:** Two-layer heterogeneous GraphSAGE encoder.

**Training:** Benign-only graph, MSE reconstruction loss.
**Inference:** Benign node features + full graph topology. Per-process MSE
normalised to [0, 1].

**Current status:** `GRAPH_WEIGHT = 0.00`. Ablation ROC-AUC = 0.31 (worse than
random). Code retained for future improvement.

---

### `anomaly_engine.py`

**Role:** Fuses four per-event score arrays into one composite score.

**Normalisation:** Min-max, fitted on benign rows only.

**Composite formula:**
```
scored events  →  0.50·dense + 0.20·recon + 0.00·graph + 0.30·rarity
warmup events  →  recon weight redistributed to remaining active signals
```

---

### `alert_aggregator.py`

**Role:** Groups high-scoring events into attack-chain records per host.
Consecutive flagged events within `ALERT_WINDOW` seconds are merged into
one chain.

---

### `anomaly_report.py`

**Role:** Generates human-readable investigation outputs for manual triage.

**Outputs:**
- `anomaly_report.txt` — narrative report: executive summary, alert chain
  summaries, per-host flagged event listings, "why suspicious" hints.
- `flagged_events.csv` — all flagged events with full context, sorted by score.

Always generated regardless of ground truth availability.

---

### `relabel_anomalies.py`

**Role:** Utility to promote evaluation events from Label=2 to Label=1
(confirmed attack) after manual review.

**Labelling strategies:**
- `--host` — flag all events from a specific machine
- `--process` — flag events matching a process name substring
- `--ip` — flag network events connecting to a suspicious IP
- `--time` — flag events in a specific time window
- `--ids` — flag specific row indices from anomaly_scores.csv
- `--query` — arbitrary pandas query string

After relabelling, set `HAS_GROUND_TRUTH = True` in `config.py` and re-run
`main.py` for AUC/F1 evaluation metrics.

---

### `metrics.py`

**Role:** Research-paper-grade evaluation (only when `HAS_GROUND_TRUTH = True`).

Binary evaluation: `label > 0` is positive.

**Metrics computed:** ROC-AUC, PR-AUC, per-component ablation AUCs,
F1-optimal threshold, Youden-J threshold, operational TPR table,
101-point threshold sweep, alert-level precision/recall, per-label statistics.

---

### `rescore.py`

**Role:** Quick re-evaluation without retraining. Reloads saved model weights
and recomputes anomaly scores with potentially different weights or thresholds.

---

## 6. Composite Scoring Formula

```
composite_score[i] = DENSE_WEIGHT  × norm(dense_error[i])
                   + RECON_WEIGHT  × norm(recon_error[i])
                   + GRAPH_WEIGHT  × norm(graph_score[i])
                   + RARITY_WEIGHT × norm(rarity_score[i])

                   = 0.50 × norm(dense_error[i])
                   + 0.20 × norm(recon_error[i])
                   + 0.00 × norm(graph_score[i])
                   + 0.30 × norm(rarity_score[i])
```

`norm(x)` = min-max normalisation, range fitted on benign events only.

**Warmup events** (first 19 per host, `recon_error = NaN`):
`RECON_WEIGHT` is redistributed proportionally among the remaining active
signals so the composite still spans [0, 1].

---

## 7. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Unsupervised / benign-only training** | Attack samples are rare and evolve constantly; modelling normality generalises to novel attacks |
| **Four-signal ensemble** | Each signal has blind spots; fusion reduces both false positives and false negatives |
| **Frequency + hash encoding for high-cardinality columns** | OHE explodes with hundreds of machines/ports; hash encoding gives fixed-width features; frequency encoding provides useful signal (rare host = anomaly) |
| **OHE only for low-cardinality columns** | Bounded unique values (EventID, boolean flags, time parts) — OHE is safe and expressive |
| **Benign-only scaler + normalisation** | Prevents attack feature distributions from widening the mean/range and making attack values appear closer to normal |
| **Streaming CSV ingestion** | Never holds more than one page (~5,000 rows) in memory; handles multi-million event datasets |
| **`--eval-only` mode** | Allows adding new evaluation periods without re-ingesting training data |
| **Label=2 as routing marker** | Not a ground-truth annotation — tells the pipeline "score but don't train on these rows" |
| **Dense AE as primary signal (weight=0.50)** | Validated at ROC-AUC 0.9950 standalone; sharp per-event signal without sequence dilution |
| **Temporal burst features** | `event_burst_count` detects process-creation storms; `log_time_delta` flags sudden bursts after idle periods |
| **Sinusoidal PE (not learned)** | Stable on short sequences (20 events); no over-fitting of position parameters |
| **True TransformerDecoder with cross-attention** | Richer reconstruction than a second encoder; each decoded position attends to full encoder memory |
| **Compressed bottleneck (embed_dim//4 = 24)** | Forces meaningful compression; prevents identity shortcut |
| **GNN weight = 0** | Ablation ROC-AUC = 0.31 (worse than random); architecture retained for future improvement |
| **Memmap for large datasets** | Two-pass algorithm; peak RAM = one host's events, not all windows |
| **Jeffreys smoothing in Rarity Engine** | Prevents zero probabilities for unseen patterns; stable on small datasets |
| **Host-scoped alert chains** | Events from different machines never merge; consistent with incident response scoping |
| **Investigation report always generated** | Enables manual triage even without ground-truth labels |

---

## 8. Artifacts on Disk

### ARTIFACTS_DIR (model weights, reports — typically < 1 GB)

| File | Produced by | Description |
|---|---|---|
| `scaler.pkl` | `normaliser.fit_scaler` | Fitted MinMaxScaler (joblib) |
| `ohe_encoder.pkl` | `feature_engineering` | Fitted OHE for low-cardinality columns |
| `freq_maps.pkl` | `feature_engineering` | Frequency maps for high-cardinality columns |
| `gnn_encoder.pt` | `gnn_encoder.train_gnn` | Trained GNN state dict |
| `dense_autoencoder.pt` | `dense_autoencoder.train_dense_ae` | Trained dense AE state dict |
| `transformer_autoencoder.pt` | `train.train_model` | Trained Transformer state dict |
| `alerts.csv` | `alert_aggregator` | Attack chain summaries |
| `anomaly_report.txt` | `anomaly_report` | Narrative investigation report |
| `flagged_events.csv` | `anomaly_report` | Flagged events with full context |
| `metrics.json` | `metrics.evaluate` | All scalar evaluation metrics |
| `roc_curve.csv` | `metrics.evaluate` | ROC curve data points |
| `pr_curve.csv` | `metrics.evaluate` | PR curve data points |
| `score_distributions.csv` | `metrics.evaluate` | Per-class score statistics |
| `threshold_sweep.csv` | `metrics.evaluate` | Full 101-row threshold sweep |

### DATA_DIR (bulk data — potentially tens of GB)

| File | Produced by | Description |
|---|---|---|
| `elastic_data.csv` | `elastic_ingest.py` | Combined training + eval event CSV |
| `anomaly_scores.csv` | `main.py` | Per-event scores (all components + label) |
| `sequences.dat` | `sequence_builder` | Sliding-window sequences (np.memmap) |
| `seq_labels.dat` | `sequence_builder` | Sequence labels (np.memmap) |

---

## 9. Configuration Quick Reference

All pipeline behaviour is controlled by **`config.py`** exclusively.
Ingestion is controlled by **`elastic_config.yml`**.

### Pipeline parameters (`config.py`)

| Parameter | Default | Effect of increasing |
|---|---|---|
| `SEQUENCE_LENGTH` | 20 | More temporal context; more warmup events per host |
| `EMBED_DIM` | 96 | More capacity; slower; risk of over-fitting on benign data |
| `EPOCHS` | 20 | More training; early stopping usually triggers before this |
| `BATCH_SIZE` | 512 | Larger GPU batches; may need reduction on small VRAM |
| `DENSE_WEIGHT` | 0.50 | More weight on dense AE per-event signal |
| `RECON_WEIGHT` | 0.20 | More weight on Transformer sequence signal |
| `RARITY_WEIGHT` | 0.30 | More weight on rarity signal |
| `GRAPH_WEIGHT` | 0.00 | Re-enable GNN contribution (re-validate first) |
| `ALERT_PERCENTILE` | 97.5 | Higher = fewer alerts, higher precision, lower recall |
| `ALERT_WINDOW` | 300 s | Wider = longer chains; may merge unrelated events |
| `LARGE_DATASET_THRESHOLD` | 200,000 | Lower to force memmap earlier |
| `HAS_GROUND_TRUTH` | False | Set True after relabelling to enable AUC/F1 metrics |

### Ingestion parameters (`elastic_config.yml`)

| Parameter | Description |
|---|---|
| `host` | Elasticsearch URL |
| `sources` | List of data sources: `defend`, `sysmon`, `wazuh` |
| `train_start` / `train_end` | Training (benign) date window |
| `eval_start` / `eval_end` | Evaluation date window |
| `page_size` | Hits per search_after page (default 5000) |
| `output_path` | Path for the merged CSV |

---

## 10. How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure elastic_config.yml with your ES connection + date windows

# 3. Ingest data from Elasticsearch
python elastic_ingest.py                    # fresh ingestion
python elastic_ingest.py --eval-only        # add eval window to existing CSV

# 4. Run the full pipeline
python main.py

# 5. Review results
#    → anomaly_report.txt    (narrative triage report)
#    → flagged_events.csv    (flagged events for spreadsheet review)
#    → alerts.csv            (attack chain summaries)

# 6. (Optional) Relabel confirmed attacks for evaluation metrics
python relabel_anomalies.py --host WORKSTATION-04
python relabel_anomalies.py --process "mimikatz"
# Then set HAS_GROUND_TRUTH = True in config.py and re-run main.py
```

**To diagnose empty results from Elasticsearch:**
```bash
python diagnose_elastic.py --config elastic_config.yml
```
