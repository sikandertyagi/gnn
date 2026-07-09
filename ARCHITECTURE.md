# Architecture Reference — Sysmon Anomaly Detection Pipeline

This document is the authoritative description of the pipeline. It is written
so that any reader — human or AI tool — with no prior context can understand
**what every file does, what it consumes, what it produces, and how all pieces
connect**. Every section reflects the current code exactly.

---

## Table of Contents

1. [System Goal](#1-system-goal)
2. [End-to-End Data Flow](#2-end-to-end-data-flow)
3. [Feature Vector Specification](#3-feature-vector-specification)
4. [File Reference](#4-file-reference)
5. [Composite Scoring Formula](#5-composite-scoring-formula)
6. [Key Design Decisions](#6-key-design-decisions)
7. [Artifacts on Disk](#7-artifacts-on-disk)
8. [Configuration Quick Reference](#8-configuration-quick-reference)
9. [How to Run](#9-how-to-run)

---

## 1. System Goal

Detect cyber-attack activity (malware execution, C2 communication, privilege
escalation, lateral movement) in Windows Sysmon event logs — **without relying
on signatures and without requiring labelled attack data at training time**.

The fundamental principle:

> *Train every model exclusively on normal (benign) behaviour. Anything the
> model cannot reconstruct or has never seen before is, by definition, anomalous.*

Three complementary anomaly signals are computed independently and fused into a
single `composite_score` in [0, 1] per event.

| Signal | Module | What it measures |
|---|---|---|
| `recon_error` | `transformer_autoencoder.py` | How poorly the Transformer reconstructs a 20-event sequence |
| `rarity_score` | `rarity_engine.py` | How rare the process-lineage / destination-IP patterns are vs benign baseline |
| `graph_score` | `gnn_encoder.py` | How anomalous a process node is in the system-call graph (weight=0, see §5) |

---

## 2. End-to-End Data Flow

```
sysmondataless.csv  (raw Sysmon telemetry, one row per event)
        │
        ▼  main.py — step 1: LOAD + EventID filter
        │
        │  Keep only EventID 1 (process creation) and EventID 3 (network
        │  connection).  All three engines, training, inference, and evaluation
        │  operate on this same filtered subset.  Other event types (module loads,
        │  registry writes, terminations) dilute signal — transformer AUC ≈ 0.50
        │  without the filter.
        │  DataFrame index is reset to a contiguous 0..N-1 RangeIndex.
        │
        ▼  main.py — step 2: FEATURE ENGINEERING  (feature_engineering.py)
        │
        │  Raw Sysmon columns → 57-column numeric feature matrix.
        │  Sections:
        │    · Process identity  (CRC32-hashed names, rare_process_score)
        │    · Handcrafted command-line flags  (entropy, base64, IP, download, -enc)
        │    · Semantic command-line embeddings  cmd_emb_0…31  ← NEW
        │      (all-MiniLM-L6-v2 → 384-d → PCA-32, cached on disk)
        │    · EventID, path depth, binary metadata, network, time-of-day
        │
        ▼  main.py — step 3: NORMALISATION  (normaliser.py)
        │
        │  StandardScaler fitted on benign rows (Label==0) only.
        │  First 6 columns (CRC32 hashes + rare_process_score) are skipped
        │  because they are already in [0, 1].
        │  cmd_emb_* and all other numeric cols are z-scored.
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
        │  Inference: benign node features + full topology.   │
        │  Per-process MSE → mapped back to events.            │
        │                                                      │
        │  → event_graph_scores (N_events,) ∈ [0,1]          │
        │                                                      │
        └──────────────────┬───────────────────────────────────┘
                           │
                           ▼  main.py — step 8: TRANSFORMER PATH
                           │
                           │  sequence_builder.py
                           │  1. _add_temporal_features():
                           │       · time_delta_seconds  (diff per host, clipped ≥ 0)
                           │       · log_time_delta       (log1p of above)
                           │       · event_burst_count   (events on host in prev 60 s)
                           │       All three z-scored on benign rows.
                           │       feature_cols grows from 57 → 60.
                           │
                           │  2. Sliding windows (seq_len=20) per host:
                           │       Small path  (<200k events): in-memory ndarray
                           │       Large path  (≥200k events): np.memmap on disk
                           │
                           │  transformer_autoencoder.py  (TransformerAutoencoder)
                           │  Trained on benign sequences only.  MSE loss.
                           │  Architecture:
                           │    input_proj + LayerNorm
                           │    → sinusoidal PE
                           │    → TransformerEncoder (2 layers, 4 heads)
                           │    → mean-pool → bottleneck (96 → 24)
                           │    → expand (24 → 96)
                           │    → positional query tokens → query_proj
                           │    → TransformerDecoder (cross-attention on encoder memory)
                           │    → output_proj
                           │
                           │  evaluate.py
                           │  Batched inference → MSE per sequence.
                           │  Score alignment: seq j belongs to last event in window j.
                           │  First 19 events per host → NaN (warmup, no full window).
                           │
                           │  → event_recon_errors (N_events,)  NaN for warmup events
                           │
                           ▼  main.py — step 9: COMPOSITE SCORING
                           │
                           │  anomaly_engine.py
                           │  Each signal is min-max normalised on benign rows.
                           │  composite = 0.45·recon + 0.00·graph + 0.55·rarity
                           │  Warmup events: composite = rarity_score
                           │
                           ▼  alert_aggregator.py
                           │
                           │  Flag events with score ≥ 0.40.
                           │  Group consecutive flagged events (gap ≤ 300 s) per host
                           │  → "attack chains"
                           │  → alerts.csv
                           │
                           ▼  metrics.py
                           │
                           │  ROC-AUC, PR-AUC, F1, threshold sweep,
                           │  per-component ablation AUCs, alert-level metrics.
                           │  → metrics.json, roc_curve.csv, pr_curve.csv, …
```

---

## 3. Feature Vector Specification

Total: **60 features** per event, in this exact order in `feature_cols`.

### Section A — CRC32-hashed categoricals (indices 0–5)
Not z-scored. Already in [0, 1]. Skipped by the StandardScaler
(`N_CATEGORICAL_FEATURES = 6` in `feature_engineering.py`).

| Index | Name | Source | How computed |
|---|---|---|---|
| 0 | `process_name` | `Image` | Backslash-split + lowercase + CRC32/0xFFFFFFFF |
| 1 | `parent_process` | `ParentImage` | Same |
| 2 | `parent_child` | `ParentImage`, `Image` | `"parent->child"` string → CRC32 |
| 3 | `User` | `User` | CRC32 of username |
| 4 | `IntegrityLevel` | `IntegrityLevel` | CRC32 of level string |
| 5 | `rare_process_score` | `Image` | `1 / freq(process_name)` in dataset; (0, 1] |

### Section B — Handcrafted command-line features (indices 6–13)
Z-scored by StandardScaler (benign fit).

| Index | Name | Description |
|---|---|---|
| 6 | `cmd_length` | Character length of `CommandLine` |
| 7 | `cmd_token_count` | Space-delimited token count |
| 8 | `has_base64` | 1 if regex matches base64 blob ≥ 20 chars |
| 9 | `has_http` | 1 if "http" (case-insensitive) in command |
| 10 | `has_ip` | 1 if bare IPv4 address (`\b\d+\.\d+\.\d+\.\d+\b`) found |
| 11 | `has_download` | 1 if wget / curl / iwr / DownloadString / BITSAdmin etc. found |
| 12 | `has_encodedcommand` | 1 if `-enc` or `-EncodedCommand` flag present |
| 13 | `cmd_entropy` | Shannon entropy of character distribution |

### Section C — Semantic command-line embeddings (indices 14–45) ← NEW
Z-scored by StandardScaler. Added by `commandline_embedding.embed_commandlines`.

| Index | Name | Description |
|---|---|---|
| 14–45 | `cmd_emb_0` … `cmd_emb_31` | PCA-32 projection of all-MiniLM-L6-v2 384-d sentence embedding |

### Section D — Execution path features (indices 46–49)
Z-scored.

| Index | Name | Description |
|---|---|---|
| 46 | `path_depth` | Slash/backslash count in `Image` |
| 47 | `is_system_bin` | 1 if path matches system directories (system32, /usr/bin, /usr/sbin, /bin, /sbin) |
| 48 | `is_users_dir` | 1 if path matches user directories (\\users\\, /home/) |
| 49 | `is_temp_exec` | 1 if path matches temp/writable directories (temp, /tmp, /var/tmp, /dev/shm, appdata, downloads, programdata) |

### Section E — Binary metadata (indices 50–51)
Z-scored.

| Index | Name | Description |
|---|---|---|
| 50 | `is_signed` | 1 if `Signed == "true"` |
| 51 | `missing_company` | 1 if `Company` is NaN |

### Section F — Network features (indices 52–53)
Z-scored.

| Index | Name | Description |
|---|---|---|
| 52 | `dest_port` | `DestinationPort` numeric (0 for non-network events) |
| 53 | `dest_external` | 1 if destination IP is outside RFC 1918 / loopback |

### Section G — Time-of-day features (indices 54–55)
Z-scored.

| Index | Name | Description |
|---|---|---|
| 54 | `hour` | Hour of day from `SystemTime` (0–23) |
| 55 | `is_after_hours` | 1 if `hour < 7` or `hour > 19` |

### Section H — Event type (index 56)
Z-scored.

| Index | Name | Description |
|---|---|---|
| 56 | `eventid` | Sysmon EventID integer (1 or 3 after global filter) |

### Section I — Temporal behaviour features (indices 57–59) ← NEW
Added by `sequence_builder._add_temporal_features` **after** the StandardScaler
runs. Z-scored on benign rows inside `_add_temporal_features` using the same
convention.

| Index | Name | Formula | Notes |
|---|---|---|---|
| 57 | `time_delta_seconds` | `diff(SystemTime)` per host, clipped ≥ 0 | 0 for first event per host |
| 58 | `log_time_delta` | `log1p(time_delta_seconds)` | Compresses multi-hour idle gaps |
| 59 | `event_burst_count` | Events on same host in [t−60s, t) | O(n log n) via `numpy.searchsorted` |

---

## 4. File Reference

---

### `config.py`

**Role:** Single source of truth for all hyperparameters, weights, and file paths.
**Imported by:** Every other module.
**Never modify any other file to change a number** — edit `config.py` instead.

```
DATA_PATH                = "sysmondataless.csv"
RANDOM_SEED              = 42
SEQUENCE_LENGTH          = 20         sliding-window length
TRAIN_LABEL              = 0          label meaning "benign"
BATCH_SIZE               = 512
EPOCHS                   = 20         max transformer training epochs
LEARNING_RATE            = 1e-3
VAL_RATIO                = 0.15       fraction of benign seqs for validation
EARLY_STOPPING_PATIENCE  = 5
EMBED_DIM                = 96         transformer internal width
NUM_HEADS                = 4
NUM_LAYERS               = 2
FF_DIM                   = 256        feed-forward sublayer width
HIGH_SIGNAL_EVENTIDS     = [1, 3]
GNN_EMBED_DIM            = 64
GNN_EPOCHS               = 15
GNN_LR                   = 1e-3
GNN_EARLY_STOPPING_PAT   = 5
RECON_WEIGHT             = 0.45
GRAPH_WEIGHT             = 0.00       disabled (GNN AUC 0.31 < random)
RARITY_WEIGHT            = 0.55
ALERT_THRESHOLD          = 0.40
ALERT_WINDOW             = 300        seconds
LARGE_DATASET_THRESHOLD  = 200_000    events above which memmap mode is used
INFER_BATCH_SIZE         = 2048
CMD_EMBED_MODEL          = "all-MiniLM-L6-v2"
CMD_EMBED_N_COMPONENTS   = 32         PCA output dimension
CMD_EMBED_BATCH_SIZE     = 256
CMD_EMBED_CACHE_DIR      = ".cmd_embed_cache"
CMD_EMBED_PCA_PATH       = "cmd_pca.pkl"
```

---

### `main.py`

**Role:** Orchestrates all nine pipeline steps from CSV load to metric output.
**Entry point:** `python main.py`

**Step summary:**

| Step | Action | Module called |
|---|---|---|
| 1 | Load CSV; filter to EventID 1 & 3; reset index | — |
| 2 | Feature engineering | `feature_engineering.feature_engineering` |
| 3 | Fit + apply StandardScaler (benign-only) | `normaliser.fit_scaler`, `apply_scaler` |
| 4 | Build full + benign-only heterogeneous graphs | `graph_builder.build_event_graph` |
| 5 | Train GNN on benign graph | `gnn_encoder.train_gnn` |
| 6 | Compute per-event graph anomaly scores | `gnn_encoder.graph_anomaly_scores` |
| 7 | Fit rarity engine; score all events | `rarity_engine.RarityEngine` |
| 8 | Build sequences; train Transformer; compute recon errors | `sequence_builder`, `transformer_autoencoder`, `train`, `evaluate` |
| 9 | Composite score; alerts; evaluation | `anomaly_engine`, `alert_aggregator`, `metrics` |

**Score alignment (step 8):**
`build_sequences` and `build_sequences_memmap` iterate hosts in the order they
appear in the DataFrame. Window `j` on a host ends at host event index
`j + SEQUENCE_LENGTH − 1`. The code maps each window score to that last event's
position in the global `event_recon_errors` array. The first
`SEQUENCE_LENGTH − 1 = 19` events per host are warmup and receive `NaN`.

**Scaling decision:** `use_memmap = n_events > LARGE_DATASET_THRESHOLD (200 000)`

---

### `feature_engineering.py`

**Role:** Converts raw Sysmon rows into the 57-column numeric feature matrix.
**Key output:** `(df, feature_cols)` where `feature_cols` is the ordered list
of column names to pass to the model.
**Module-level constant:** `N_CATEGORICAL_FEATURES = 6` — read by `normaliser.py`.

**Missing-column safety:** `_OPTIONAL_COLS` dict defines every optional Sysmon
column with a safe default. Any column absent from the CSV is filled before
feature computation, so the pipeline runs on any Sysmon config.

**Categorical encoding:** CRC32 hashing
```python
hash_value = (zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF) / 0xFFFFFFFF
```
Stable across runs (no `PYTHONHASHSEED`), handles unseen categories at
inference without error, always in [0, 1].

**Bug fixes in this file:**
- `dest_external`: original bitwise `~` on an int Series yielded −1/−2.
  Fixed by inverting the **bool** Series before `.astype(int)`.
- RFC 1918: `172.*` captured public addresses.
  Correct regex: `^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|127\.)`.

**Semantic embeddings** (section 2b):
```python
_embs = embed_commandlines(df["CommandLine"].fillna("").astype(str))  # (N, 32)
for i in range(CMD_EMBED_N_COMPONENTS):
    df[f"cmd_emb_{i}"] = _embs[:, i]
```
The 32 columns are appended to `feature_cols` in the numeric section so they
are z-scored by the StandardScaler.

---

### `commandline_embedding.py`

**Role:** Provides PCA-compressed semantic representations of `CommandLine`
strings using a pre-trained sentence-transformer model.

**Public function:**
```python
embeddings = embed_commandlines(
    cmd_series: pd.Series,
    pca_path:   str = CMD_EMBED_PCA_PATH,
    cache_dir:  str = CMD_EMBED_CACHE_DIR,
) -> np.ndarray  # (N, 32), float32
```

**Internal pipeline:**
```
cmd_series  (N strings)
    │
    ▼  _cache_key(): SHA-256 of all strings in order → hex string
    │
    ├─ cache hit  → load (N, 384) raw embeddings from .cmd_embed_cache/<hash>.npy
    └─ cache miss → SentenceTransformer("all-MiniLM-L6-v2").encode(
                        texts, batch_size=256, convert_to_numpy=True)
                    → save raw.npy for next run
    │
    ▼  PCA model
    ├─ pca_path exists → joblib.load(pca_path)
    └─ pca_path absent → PCA(n_components=32, random_state=42).fit(raw)
                         → joblib.dump(pca, pca_path)
    │
    ▼  pca.transform(raw).astype(float32)  →  (N, 32)
```

**Why two tiers of caching:**
- Raw 384-d embeddings take minutes to compute for large datasets.
  A SHA-256 cache avoids re-encoding the same dataset on repeated runs.
- The PCA model is stable after first fit; reloading it avoids non-determinism
  from re-fitting on a subset in a different run.

**Determinism guarantees:**
- Model weights fixed (downloaded once from HuggingFace Hub).
- `random_state=42` in PCA.
- Cache key is order-sensitive; only exact same data in same order produces a hit.

---

### `normaliser.py`

**Role:** Fits and applies a `StandardScaler` to the numeric portion of the
feature matrix, leaving the CRC32-hashed categorical columns untouched.

**Public API:**
```python
scaler = fit_scaler(df, feature_cols)          # returns + saves scaler.pkl
df     = apply_scaler(df, feature_cols, scaler)
scaler = load_scaler()                          # reload for inference
```

**What is scaled:** `feature_cols[N_CATEGORICAL_FEATURES:]`
(indices 6–56 — everything except the first 6 CRC32/rare-score columns).

**What is NOT scaled here:**
- Columns 0–5: already in [0, 1]; z-scoring hash values is meaningless.
- Columns 57–59 (temporal): added inside `sequence_builder` after this step.
  They are z-scored there on benign rows using the same convention.

**Benign-only fitting:** `scaler.fit(df[df["Label"] == TRAIN_LABEL][numeric_cols])`.
This prevents attack feature distributions from widening the mean/std and
making attack values appear closer to normal.

---

### `sequence_builder.py`

**Role:** Injects temporal features and builds overlapping sliding-window
sequences of events per host.

**Public API:**
```python
# Small datasets (< LARGE_DATASET_THRESHOLD events)
X, y = build_sequences(df, feature_cols, seq_len)
#   X: ndarray (N_seq, seq_len, n_features)  float32
#   y: ndarray (N_seq,)                       int32

# Large datasets (memmap path)
n_seq, seq_shape = build_sequences_memmap(
    df, feature_cols, seq_len, seq_path="sequences.dat", labels_path="seq_labels.dat"
)
y_mm   = load_seq_labels(labels_path, n_sequences)
X_mm   = load_seq_memmap(seq_path, seq_shape)
```

**Temporal feature injection — `_add_temporal_features(df, feature_cols)`:**

Called at the top of both build functions. Sorts df by `[Computer, SystemTime]`
(required for `diff()` and `searchsorted` to be meaningful), computes three
features, z-scores them on benign rows, then appends the names to `feature_cols`.

```python
_TEMPORAL_COLS = ["time_delta_seconds", "log_time_delta", "event_burst_count"]
```

`_burst_count` helper (per host):
```python
t_ns      = SystemTime.astype("datetime64[ns]").view(np.int64)
window_ns = 60_000_000_000   # 60 s in nanoseconds
counts    = searchsorted(t_ns, t_ns, "left")
          - searchsorted(t_ns, t_ns - window_ns, "left")
```
O(n log n), no DatetimeIndex required, NaT timestamps replaced with epoch.

**Sliding windows** use `numpy.lib.stride_tricks.sliding_window_view` (zero-copy).
Shape: `(n, n_features, seq_len)` → transposed to `(n, seq_len, n_features)`.
Label: label of the **last** event in the window.

**Memmap algorithm (large datasets):**
1. Pass 1 — count total windows per host (O(n_hosts), no data loaded).
2. Allocate two memmap files with the exact required shape.
3. Pass 2 — write host-by-host. Peak RAM = one host's event matrix, not all
   windows simultaneously.
4. Flush to disk; delete memmap handles.

---

### `transformer_autoencoder.py`

**Role:** Defines the `TransformerAutoencoder` nn.Module.

**Architecture:**
```
Input  (B, T=20, F=60)
  │
  ├─ input_proj: Linear(F, 96) + LayerNorm(96)
  ├─ pos_enc: SinusoidalPositionalEncoding(96)   fixed, Vaswani et al. 2017
  ├─ encoder: TransformerEncoder(d=96, heads=4, layers=2, ff=256)
  ├─ mean-pool over T → (B, 96)
  └─ bottleneck: Linear(96, 24)        ← forces compact representation

Bottleneck z  (B, 24)
  │
  ├─ bottleneck_expand: Linear(24, 96)
  ├─ broadcast to (B, T, 96) → memory
  ├─ query_pos_enc(zeros(B, T, 96)) → positional query tokens
  ├─ query_proj: Linear(96, 96)       ← dedicated query space for cross-attn
  ├─ decoder_transformer: TransformerDecoder(d=96, heads=4, layers=2, ff=256)
  │   cross-attends to memory
  └─ output_proj: Linear(96, F)

Output  (B, T=20, F=60)
```

**Key model parameters:**

| Symbol | Value | Meaning |
|---|---|---|
| `EMBED_DIM` | 96 | Internal width |
| `NUM_HEADS` | 4 | Attention heads |
| `NUM_LAYERS` | 2 | Encoder and decoder layer count |
| `FF_DIM` | 256 | Feed-forward sublayer width |
| `bottleneck_dim` | `max(96//4, 16) = 24` | Compression bottleneck |

**Public methods:**
```python
z   = model.encode(x)              # (B,T,F) → (B, 24)
out = model.decode(z, seq_len=T)  # (B, 24)  → (B, T, F)
out = model(x)                    # encode + decode
```

**Training objective:** `MSELoss(output, input)` on benign sequences only.
High MSE at inference = model cannot explain the sequence = anomaly.

**Why `query_proj`:** Without it, the TransformerDecoder's queries are raw
sinusoidal embeddings — fixed, with no learned content. `query_proj` gives the
decoder a learned linear projection that acts as a position-aware "question" to
ask the encoder memory, improving reconstruction specificity for unusual events.

---

### `train.py`

**Role:** Two training entry points for the Transformer autoencoder.

**`train_model(model, X_train, epochs, batch_size, lr, val_ratio, patience)`**
In-memory path.

- Shuffles `X_train` with `RANDOM_SEED`; holds out `val_ratio` fraction for validation.
- **Optimiser:** `AdamW(lr, weight_decay=1e-4)`.
- **Scheduler:** Linear warmup over first 10% of steps → cosine annealing to 0.
  Stepped once per batch.
- **AMP:** `torch.amp.autocast("cuda")` on GPU (no-op on CPU).
- **Gradient clipping:** `clip_grad_norm_(max_norm=1.0)`.
- **Early stopping:** Monitors val loss (or train loss if no val set). Saves
  best weights with `copy.deepcopy`. Restores on exit.

**`train_model_large(model, seq_path, seq_shape, train_indices, ...)`**
Disk-backed path.

- Uses `MemmapDataset` (custom `torch.utils.data.Dataset`) that reads from a
  read-only `np.memmap`. `train_indices` selects benign rows without loading
  the full file. Each `__getitem__` calls `.copy()` because PyTorch cannot own
  memmap-backed memory.
- Otherwise identical optimiser, scheduler, AMP, clipping, and early-stopping.
- Peak RAM = one batch of sequences at a time.

---

### `evaluate.py`

**Role:** Batched inference — computes per-sequence MSE reconstruction error.

**Public API:**
```python
scores = anomaly_scores(model, X, batch_size=512)
# → ndarray (N_seq,)  float32 — MSE per sequence
```

- `X` may be `np.ndarray` (in-memory) or `np.memmap` (disk-backed); both are
  sliced chunk-by-chunk with `np.array(slice)` to materialise memmap chunks.
- MSE computed in float32 (`recon.float()`) even under AMP to avoid FP16
  precision loss in the error metric.
- AMP autocast is active during inference for consistent GPU behaviour.
- Progress printed every 100 batches.

---

### `rarity_engine.py`

**Role:** Scores how rarely each event's behavioural patterns appeared during
benign training, using three complementary sub-signals.

**Class `RarityEngine`:**
```python
engine = RarityEngine().fit(df)          # fit on Label==0 rows
scores = engine.score_dataframe(df)      # → ndarray (N,)  float32 ∈ [0, 1]
score  = engine.score_row(row)           # single event (calls score_dataframe)
```

**Three sub-signals (per-event mean over whichever are non-NaN):**

| Signal | Pair tracked | Non-NaN condition |
|---|---|---|
| `parent_child` | `(parent_process_name, child_process_name)` | Both `Image` and `ParentImage` non-null |
| `proc_ip` | `(process_name, DestinationIp)` | Both non-null |
| `network_dest` | `DestinationIp` | Non-null |

**Scoring formula (Jeffreys / α=0.5 smoothing):**
```
p(pattern) = (count + 0.5) / (total_count + 0.5 × vocab_size)
rarity     = 1 − p(pattern)
```
Unseen patterns get `count=0` → high rarity close to 1 without being exactly 1.
Smoothing prevents zero-division and gives stable scores on small datasets.

**Scalability:** All operations are vectorised via pandas `groupby` + `merge`.
No Python-level row iteration. Safe for millions of events.

---

### `graph_builder.py`

**Role:** Converts the event DataFrame into a `torch_geometric.data.HeteroData`
heterogeneous graph for the GNN.

**Node types:** `process`, `ip`, `user`, `host`.
Each node is identified by a unique string key (image path, IP string, username,
computer name). Node feature vectors aggregate event statistics for that entity.

**Edge types (bidirectional):**

| Forward | Reverse |
|---|---|
| `(process, parent_of, process)` | `(process, rev_parent_of, process)` |
| `(process, connects_to, ip)` | `(ip, rev_connects_to, process)` |
| `(process, runs_as, user)` | `(user, rev_runs_as, process)` |
| `(process, runs_on, host)` | `(host, rev_runs_on, process)` |

Reverse edges are required for bidirectional message-passing in GraphSAGE.

**Shared encoders:** `build_event_graph(df, encoders=None)` returns
`(HeteroData, encoders_dict)`. When called twice — once for the full graph and
once for the benign-only graph — the **same `encoders` dict** is passed to the
second call so both graphs have identical node orderings. This is critical for
mapping GNN process embeddings back to event rows.

**`node_feature_dims(graph) → dict[str, int]`** returns the input feature
dimension per node type; used to construct `HeteroGNNEncoder`.

---

### `gnn_encoder.py`

**Role:** Defines `HeteroGNNEncoder`, its training, and per-event score
computation.

**Architecture — two-layer heterogeneous GraphSAGE:**
```
x_dict  (per node type: Tensor(N_type, fdim))
  │
  ├─ input_projs[ntype]: Linear(fdim, 64)   project to common embedding space
  │
  ├─ conv1: HeteroConv(SAGEConv per edge type, aggr="mean") + ReLU
  │    nodes with no incoming edges fall back to projected input features
  │
  ├─ conv2: HeteroConv(SAGEConv per edge type, aggr="mean") + LayerNorm
  │
Embeddings h_dict  (per node type: Tensor(N_type, 64))
  │
  └─ decoders[ntype]: Linear(64, fdim)   reconstruct original features
```

**Training (`train_gnn`):**
- Trains on the **benign-only graph** only.
- Loss: `sum(MSE(decoded[ntype], x_dict[ntype]))` across all node types.
- Optimiser: Adam. AMP + gradient clipping. Early stopping on training loss.

**Inference (`graph_anomaly_scores`):**
- `x_dict` = **benign node features** (attack-only processes → zero vectors).
- `edge_index_dict` = **full graph topology** (attack edges are the signal).
- Returns per-process-node MSE, normalised to [0, 1]:
  `(recon_err − min) / (max − min + 1e-8)`.
- Scores are mapped to events via the process node encoder.

**Current status:** `GRAPH_WEIGHT = 0.00`. Ablation showed ROC-AUC = 0.31,
worse than random (0.50). The code is retained and can be re-enabled by
setting a non-zero weight in `config.py`.

---

### `anomaly_engine.py`

**Role:** Fuses the three per-event score arrays into one composite score.

**Public API:**
```python
composite = compute_anomaly_scores(
    recon_errors,   # (N,) float32 — NaN for warmup events
    graph_scores,   # (N,) float32
    rarity_scores,  # (N,) float32
    labels,         # (N,) int  — for benign-only normalisation fit
) -> np.ndarray   # (N,) float32 ∈ [0, 1]
```

**Normalisation (`_normalise`):**
- Min-max, fitted on benign rows only (prevents attack ranges from compressing scores).
- `max == min` → return zeros (constant signal).
- NaN entries (recon_error warmup) → fill 0 after normalisation.

**Composite formula:**
```
scored events   →  0.45·r + 0.00·g + 0.55·s
warmup events   →  s
```
Where `r`, `g`, `s` are the normalised reconstruction, graph, and rarity
signals respectively. `remain = GRAPH_WEIGHT + RARITY_WEIGHT = 0.55`.
Warmup events get `(RARITY_WEIGHT / remain)·s = s`, preserving the [0, 1]
ceiling across both groups.

---

### `alert_aggregator.py`

**Role:** Groups high-scoring events into human-readable attack-chain records.

**Public API:**
```python
alerts_df = aggregate_alerts(df, scores, threshold=ALERT_THRESHOLD,
                             window_sec=ALERT_WINDOW)
# → pd.DataFrame  columns:
#   chain_id, host, start_time, end_time, duration_s, num_events,
#   max_score, mean_score, processes, dest_ips, labels
```

**Algorithm:**
1. Flag events where `score >= threshold` (default 0.40).
2. Drop events with NaT `SystemTime`.
3. Process each host independently (events from different machines never merge).
4. Within each host: sort by `SystemTime`; iterate and start a new chain
   whenever the time gap to the previous flagged event exceeds `window_sec`.
5. Summarise each chain:
   - `processes`: deduplicated process names in order of appearance.
   - `dest_ips`: deduplicated destination IPs.
   - `labels`: deduplicated labels (reveals if the chain contains attack events).

---

### `metrics.py`

**Role:** Research-paper-grade evaluation of anomaly detection performance.

**Public API:**
```python
metrics = evaluate(df_scores, df_alerts=None, df_events=None,
                   report_path="threshold_sweep.csv") -> dict
```

`df_scores` must contain: `score`, `recon_error`, `graph_score`,
`rarity_score`, `label`.

**Label convention:** 0 = benign, 1 = confirmed attack, 2 = suspicious.
Binary evaluation: `label > 0` is positive.

**Metrics computed:**

| Category | Metrics |
|---|---|
| Ranking | ROC-AUC, PR-AUC |
| Per-component ablation | ROC-AUC for `recon_error`, `graph_score`, `rarity_score`, `score` |
| F1-optimal threshold | Precision, Recall, F1, Accuracy, MCC, Cohen's Kappa, G-Mean, TP/FP/FN/TN, FPR, FNR |
| Youden-J threshold | Same as above |
| Operational TPR table | TPR at FPR = 0.1%, 0.5%, 1%, 5%, 10% |
| Threshold sweep | All binary metrics at every threshold 0.00→1.00 step 0.01 |
| Alert-level | Alert Precision, Alert Recall, chain count |
| Per-label stats | mean, std, median, p25, p75, p95, p99 per class |

**NaN handling:** `_safe_auc` filters NaN score rows before calling sklearn
(warmup events have `recon_error = NaN`), preventing `ValueError`.

**Output files:**

| File | Contents |
|---|---|
| `metrics.json` | All scalar metrics — import into paper Table 1 |
| `roc_curve.csv` | `fpr, tpr, threshold` — Figure: ROC curve |
| `pr_curve.csv` | `precision, recall, threshold` — Figure: PR curve |
| `score_distributions.csv` | Per-class score statistics |
| `threshold_sweep.csv` | Full 101-row sweep table |

---

## 5. Composite Scoring Formula

```
composite_score[i] = RECON_WEIGHT  × norm(recon_error[i])
                   + GRAPH_WEIGHT  × norm(graph_score[i])
                   + RARITY_WEIGHT × norm(rarity_score[i])

                   = 0.45 × norm(recon_error[i])
                   + 0.00 × norm(graph_score[i])
                   + 0.55 × norm(rarity_score[i])
```

`norm(x)` = min-max normalisation, range fitted on benign events only.

**Warmup events** (first 19 per host, `recon_error = NaN`):
```
composite_score[i] = norm(rarity_score[i])
```

This preserves the same [0, 1] ceiling as fully-scored events so the
`ALERT_THRESHOLD = 0.40` applies uniformly.

---

## 6. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Unsupervised / benign-only training** | Attack samples are rare and evolve constantly; modelling normality generalises to novel attacks |
| **Three-signal ensemble** | Each signal has blind spots; fusion reduces both false positives and false negatives |
| **Global EventID 1 & 3 filter** | These event types carry the strongest attack signal. Applied once at load so all engines, training, and evaluation operate on the same consistent subset |
| **CRC32 categorical hashing** | Deterministic across runs; handles unseen categories at inference; no fit/transform step |
| **Benign-only scaler + normalisation** | Prevents attack feature distributions from distorting the baseline, which would compress anomaly gaps and reduce separability |
| **Semantic command-line embeddings** | Captures similarity between functionally-equivalent commands that binary flags miss entirely (e.g. different base64-encoded payloads) |
| **Temporal burst features** | `event_burst_count` detects process-creation storms common in exploit chains; `log_time_delta` flags sudden bursts after long idle periods |
| **Sinusoidal PE (not learned)** | Stable on short sequences (20 events); no over-fitting of position parameters |
| **True TransformerDecoder with cross-attention** | Each decoded position attends to the full encoder memory, giving a richer reconstruction signal than a second encoder |
| **Compressed bottleneck (embed_dim//4 = 24)** | Forces meaningful compression; prevents the identity shortcut where all sequences are trivially reconstructed |
| **`query_proj` before cross-attention** | Gives the decoder a learned linear projection as "questions" to ask the encoder, improving specificity for unusual event patterns |
| **Warmup event handling** | Redistributes weight so warmup events have the same composite score ceiling as fully-scored events; alert threshold is fair for all events |
| **GNN weight = 0** | Ablation ROC-AUC = 0.31 (worse than random); architecture retained for future improvement |
| **Memmap for large datasets** | Two-pass algorithm: count windows first (O(n_hosts)), then write host-by-host. Peak RAM = one host's events |
| **Benign features + full GNN topology** | Attack-only processes have zero feature vectors and appear anomalous even in the benign feature space; attack-introduced edges are the structural signal |
| **Host-scoped alert chains** | Events from different machines are never merged; consistent with real-world incident response scoping |
| **Jeffreys smoothing in Rarity Engine** | Prevents zero probabilities for unseen patterns; stable on small datasets |

---

## 7. Artifacts on Disk

| File | Produced by | Description |
|---|---|---|
| `scaler.pkl` | `normaliser.fit_scaler` | Fitted StandardScaler (joblib) |
| `cmd_pca.pkl` | `commandline_embedding` | Fitted PCA model 384→32 (joblib) |
| `.cmd_embed_cache/<sha256>.npy` | `commandline_embedding` | Cached raw 384-d embeddings |
| `gnn_encoder.pt` | `gnn_encoder.train_gnn` | Trained GNN state dict (PyTorch) |
| `transformer_autoencoder.pt` | `train.train_model` | Trained Transformer state dict (PyTorch) |
| `sequences.dat` | `sequence_builder` | All sliding-window sequences as np.memmap (large datasets) |
| `seq_labels.dat` | `sequence_builder` | Corresponding sequence labels as np.memmap |
| `anomaly_scores.csv` | `main` | Per-event: score, recon_error, graph_score, rarity_score, label |
| `alerts.csv` | `alert_aggregator` | Attack chain summaries |
| `metrics.json` | `metrics.evaluate` | All scalar evaluation metrics |
| `roc_curve.csv` | `metrics.evaluate` | ROC curve data points |
| `pr_curve.csv` | `metrics.evaluate` | PR curve data points |
| `score_distributions.csv` | `metrics.evaluate` | Per-class score statistics |
| `threshold_sweep.csv` | `metrics.evaluate` | Full 101-row threshold sweep |

---

## 8. Configuration Quick Reference

All pipeline behaviour is controlled by **`config.py`** exclusively.

| Parameter | Default | Effect of increasing |
|---|---|---|
| `SEQUENCE_LENGTH` | 20 | More temporal context; more warmup events per host |
| `EMBED_DIM` | 96 | More capacity; slower; risk of over-fitting on benign data |
| `EPOCHS` | 20 | More training; early stopping usually triggers before this |
| `BATCH_SIZE` | 512 | Larger GPU batches; may need reduction on small VRAM |
| `RECON_WEIGHT` | 0.45 | More weight on Transformer signal |
| `RARITY_WEIGHT` | 0.55 | More weight on rarity signal |
| `GRAPH_WEIGHT` | 0.00 | Re-enable GNN contribution (re-validate first) |
| `ALERT_THRESHOLD` | 0.40 | Higher = fewer alerts, higher precision, lower recall |
| `ALERT_WINDOW` | 300 s | Wider = longer chains; may merge unrelated events |
| `LARGE_DATASET_THRESHOLD` | 200,000 | Lower to force memmap earlier |
| `CMD_EMBED_N_COMPONENTS` | 32 | Richer embeddings; slightly wider feature vector |

---

## 9. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Place your Sysmon CSV at the DATA_PATH in config.py (default: sysmondataless.csv)
# Run the full pipeline
python main.py
```

The pipeline will print step-by-step progress and write all artifacts listed
in §7 to the current directory.

**To change any setting** (threshold, weights, model size, embedding dimensions):
edit `config.py` only. No other file needs modification.
