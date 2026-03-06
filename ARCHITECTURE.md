# Architecture: GNN + Transformer Ensemble Anomaly Detection Pipeline

## What Are We Trying to Achieve?

Modern endpoint security teams face a needle-in-a-haystack problem: a single
machine can generate millions of Windows Sysmon events per day, and a real
attack may touch only a few hundred of them. The goal is to **automatically
surface malicious activity** — malware execution, command-and-control
communication, privilege escalation, lateral movement — without relying on
pre-written attack signatures that would miss novel threats.

The fundamental design principle is:

> *Train every model exclusively on normal (benign) behaviour. Anything
> the model finds surprising is, by definition, anomalous.*

This is **unsupervised anomaly detection**. It requires no labelled attack
samples during training and can detect attack patterns it has never seen before.

### Why Three Signals?

A single detector has blind spots. This pipeline fuses **three complementary
anomaly signals**:

| Signal | What it catches | Example |
|--------|-----------------|---------|
| **Graph anomaly** (GNN) | Unusual process-to-process or process-to-IP relationships in the network of all processes | `notepad.exe` connecting to an external IP it has never connected to |
| **Temporal sequence anomaly** (Transformer Autoencoder) | Unusual sequences of events over time | A sequence of `cmd.exe → powershell.exe → curl` that never appeared in training |
| **Rarity anomaly** (Rarity Engine) | Individual events involving rare processes or rare destinations | A process that has run only once across the entire dataset |

Fusing all three into a **composite score** dramatically reduces both false
positives (normal events flagged as attacks) and false negatives (attacks
missed entirely).

---

## High-Level Pipeline

```
Sysmon CSV (raw endpoint telemetry)
        │
        ▼  [0] EventID Filter
        │       Retain only EventID 1 (process creation) and
        │       EventID 3 (network connection) — applied once at
        │       load time; all engines, training, validation,
        │       inference, and evaluation use this same subset
        │
        ▼  [1] Feature Engineering
        │       Convert raw log columns → 20 numeric features per event
        │
        ▼  [2] Normalisation
        │       StandardScaler fitted on benign rows only
        │
        ├─────────────────────────────────────────────┐
        │                                             │
        ▼  [3] Graph Construction                     ▼  [5] Sequence Building
        │       Heterogeneous graph:                  │       Sliding window of
        │       process / ip / user / host nodes      │       20 events per host
        │                                             │
        ▼  [4] GNN Encoder Training                   ▼  [6] Transformer Autoencoder
        │       Self-supervised on benign graph       │       Trained on benign sequences
        │       → graph anomaly score per event       │       → reconstruction error per event
        │                                             │
        └──────────────────┬──────────────────────────┘
                           │
                           │  + [7] Rarity Engine
                           │        Bayesian frequency scoring
                           │        → rarity score per event
                           │
                           ▼  [8] Composite Anomaly Scoring
                           │       weighted sum: 0.5×recon + 0.3×graph + 0.2×rarity
                           │
                           ▼  [9] Alert Aggregation
                           │       Group consecutive high-score events
                           │       → attack chain alerts CSV
                           │
                           ▼  [10] Evaluation
                                   AUROC, PR-AUC, F1, threshold sweep
```

---

## Step-by-Step Explanation

### [1] Feature Engineering — `feature_engineering.py`

Raw Sysmon log columns (file paths, command-line strings, IPs) cannot be fed
directly to neural networks. This step converts each event row into 20 numeric
features across five categories.

#### Categorical Encoding: Deterministic CRC32 Hashing

The five categorical columns (`process_name`, `parent_process`, `parent_child`,
`User`, `IntegrityLevel`) are encoded using **CRC32 hashing** instead of
`sklearn.LabelEncoder`:

```python
hash_value = (zlib.crc32(s.encode("utf-8")) & 0xFFFFFFFF) / 0xFFFFFFFF
```

This is stable across runs (no `PYTHONHASHSEED` dependency), requires no
fit/transform step, always produces values in [0, 1], and handles unseen
categories at inference without error.  These hashed columns come first in
`feature_cols` and are skipped by the StandardScaler (they are already in
[0, 1]; z-scoring them is semantically meaningless).

#### Bug Fixes in Feature Engineering

- **`dest_external`**: The original code applied bitwise `~` to an int Series,
  yielding -1/-2.  Fixed by inverting the bool Series before `.astype(int)`.
- **RFC 1918 detection**: `172.*` was too broad (captured public addresses).
  The correct 172.16.0.0/12 range is now matched with:
  `re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.|127\.)")`

#### Process Identity
| Feature | What it captures |
|---------|-----------------|
| `process_name` | Which executable ran (CRC32 hash, in [0, 1]) |
| `parent_process` | Which process spawned it (CRC32 hash) |

#### Command-Line Behaviour
| Feature | What it detects |
|---------|----------------|
| `cmd_length` | Abnormally long commands (obfuscation indicators) |
| `cmd_entropy` | Shannon entropy — high entropy = encoded/obfuscated payload |
| `has_base64` | Base64 strings longer than 20 characters in command line |
| `has_http` | HTTP/HTTPS URLs embedded in command line |
| `has_ip` | Bare IP addresses in arguments (C2 / reverse shells) |
| `has_download` | wget, curl, Invoke-WebRequest, BITSAdmin, etc. |
| `has_encodedcommand` | PowerShell `-EncodedCommand` / `-enc` flag |

#### Binary Execution Path
| Feature | What it detects |
|---------|----------------|
| `path_depth` | Very shallow or very deep execution paths |
| `is_system32` | Binary in System32 (legitimacy signal) |
| `is_users_dir` | Binary running from user profile (suspicious for executables) |
| `is_temp_exec` | Binary running from Temp (classic dropper pattern) |

#### Binary Metadata
| Feature | What it detects |
|---------|----------------|
| `is_signed` | Unsigned binaries are more suspicious |
| `missing_company` | Missing company metadata is common in compiled malware |

#### Network Behaviour
| Feature | What it detects |
|---------|----------------|
| `dest_port` | Raw destination port number |
| `dest_external` | Whether the connection target is outside the internal network |

#### Temporal Behaviour
| Feature | What it detects |
|---------|----------------|
| `hour` | Hour of day (0–23) |
| `is_after_hours` | Activity before 7am or after 7pm |

#### Event Type
| Feature | What it captures |
|---------|----------------|
| `eventid` | EventID numeric value |

---

### [2] Normalisation — `normaliser.py`

Neural networks are sensitive to the scale of input features. A `StandardScaler`
is fitted **exclusively on benign (label=0) rows**, then applied to all rows.

Fitting on benign-only data is critical: including attack events in the
scaling step would contaminate the baseline and make anomalous feature values
appear more "normal" than they actually are.

The scaler is applied **only to the numeric columns** (skipping the first
`N_CATEGORICAL_FEATURES = 5` CRC32-hashed columns, which are already in
[0, 1]).  Z-scoring hash values is semantically meaningless and would destroy
the uniform distribution they are designed to produce.

---

### [3] Graph Construction — `graph_builder.py`

This is the most distinctive part of the pipeline. Instead of treating events
as independent rows, we model the **entire dataset as a graph** that captures
relationships between entities.

#### Why a Graph?

An attacker rarely acts in isolation. A typical attack chain involves:
- `word.exe` spawning `cmd.exe`
- `cmd.exe` spawning `powershell.exe`
- `powershell.exe` connecting to `185.220.101.x`

A graph captures this multi-hop relationship explicitly. A single event view
would miss it.

#### Graph Structure (Heterogeneous)

We build a **heterogeneous graph** — one with multiple types of nodes and edges:

```
Node types:
  process  – unique executable images (e.g. "powershell.exe")
  ip       – unique destination IP addresses
  user     – unique user accounts
  host     – unique machine names

Edge types:
  (process)  ──parent_of──►  (process)   parent spawned child
  (process)  ──connects_to►  (ip)        process made a network connection
  (process)  ──runs_as──►    (user)       process was running under this account
  (process)  ──runs_on──►    (host)       process ran on this machine
  + all reverse edges (for bidirectional message passing)
```

Each node carries a numeric feature vector derived from the events it
participated in (e.g. how many times a process ran, average token count of
its command lines, whether it ever connected externally).

Two separate graphs are built:
- **Benign graph**: only events with label = 0 (used for training the GNN)
- **Full graph**: all events (used for inference / scoring)

Both graphs share the same node encoders so that process names map to the
same integer IDs in both.

#### Bug Fixes in Graph Construction

- **Deterministic `name_hash`**: The original `hash()` call used Python's
  session-randomised hash seed.  Replaced with CRC32, identical to
  `feature_engineering.py`, producing stable values in [0, 1].
- **`cmd_token_mean` rename**: The per-process aggregation field was named
  `cmd_entropy_mean` but actually aggregated `_cmd_tok` (token count, not
  Shannon entropy).  Renamed to `cmd_token_mean` for clarity.
- **Edge deduplication**: Multiple events between the same process pair
  created parallel edges.  Removed with `torch.unique(edge.t(), dim=0)`.
- **RFC 1918 `is_external` fix**: `startswith("172.")` captured public IPs
  in the 172.0–172.15 and 172.32–172.255 ranges.  Replaced with the same
  `_RFC1918` regex pattern used in `feature_engineering.py`.

---

### [4] GNN Encoder — `gnn_encoder.py`

#### What is a Graph Neural Network?

A GNN is a neural network that operates on graph-structured data. Each node
aggregates feature information from its neighbours, then from their neighbours,
and so on. After several rounds, each node's embedding captures not just its
own features but the structural context of its local neighbourhood.

#### Architecture: Two-Layer Heterogeneous GraphSAGE

```
For each node type (process, ip, user, host):
  Input: node feature vector  x ∈ R^d
        │
        ▼  HeteroConv Layer 1  (SAGEConv per edge type) + ReLU
  Embedding: h ∈ R^64
        │
        ▼  HeteroConv Layer 2  (SAGEConv per edge type) + LayerNorm
  Embedding: z ∈ R^64
        │
        ▼  Linear Decoder
  Reconstruction: x̂ ∈ R^d
```

GraphSAGE (Graph Sample and Aggregate) is used because it handles variable-
sized neighbourhoods gracefully and scales to large graphs.

#### Training Objective: Self-Supervised Reconstruction

The GNN is trained only on the benign graph using a **self-supervised node
feature reconstruction** objective:
- Encode each node → embedding `z`
- Decode `z` → reconstructed feature vector `x̂`
- Loss: `MSE(x, x̂)` across all benign nodes

After training, the model has learned what "normal" graph structure looks like.
A node whose neighbourhood looks anomalous will produce a high reconstruction
error at inference time.

#### Inference: Benign Features + Full Graph Topology

At inference, the GNN receives:
- **Node features** from the **benign graph** (not the full graph)
- **Edge topology** from the **full graph**

This separation is critical.  If node features were aggregated from all events
(including attacks), attack statistics (e.g. very long command lines) would
leak into the feature vectors, making the reconstruction error noisy and
uncorrelated with actual anomalies.  By using benign-only features, attack
processes that only appear in attack events will have zero-filled feature
vectors (via `reindex(fill_value=0.0)`), making them structurally distinctive.

The **edge topology** from the full graph is kept because attack-introduced
relationships (new parent-child process chains, novel IP connections) are the
primary structural signal.

These node-level scores are mapped back to events via the process image name.

---

### [5] Sequence Building — `sequence_builder.py`

Events are grouped by host, sorted by time, and windowed:

```
Host A events (sorted by SystemTime):
  [e1, e2, e3, e4, e5 ... eN]

Sequences (window=20, stride=1):
  seq_1 = [e1 .. e20]   label = label(e20)
  seq_2 = [e2 .. e21]   label = label(e21)
  ...
  seq_{N-19} = [e_{N-19} .. eN]
```

Windows never cross machine boundaries.

**Off-by-one fix**: `numpy.lib.stride_tricks.sliding_window_view(values, seq_len, axis=0)` on an array of length `M` produces exactly `M - seq_len + 1` windows (not `M - seq_len`).  The original code discarded the last valid window.  Both the in-memory and memmap paths are corrected.

#### Scalability: Memmap Mode for Large Datasets

For datasets above 200,000 events, the pipeline automatically switches to a
**disk-backed numpy memmap** strategy:
- Sequences are written to `sequences.dat` on disk rather than held in RAM
- Training reads mini-batches directly from disk
- Inference scores are computed in batches of 512

This allows the pipeline to handle datasets of millions of events on a machine
with modest RAM.

---

### [6] Transformer Autoencoder — `transformer_autoencoder.py` + `train.py`

#### What is an Autoencoder?

An autoencoder compresses its input into a lower-dimensional representation
and then reconstructs it. Trained only on normal data, it becomes very good
at reconstructing normal patterns and very bad at reconstructing anomalous ones.
High reconstruction error = anomalous sequence.

#### Why a Transformer (not LSTM)?

A Transformer uses **self-attention** to model relationships between any two
events in the sequence regardless of distance. An LSTM would struggle to
connect event position 3 with event position 18 in a sequence of 20. The
Transformer treats all positions equally and learns which pairs of events
are related.

#### Architecture

```
Input:  (batch, 20 events, 20 features)
        │
        ▼  Linear projection  →  embed_dim=128
(batch, 20, 128)
        │
        ▼  + Sinusoidal Positional Encoding (fixed, from Vaswani et al. 2017)
(batch, 20, 128)   ← order information preserved without learned parameters
        │
        ▼  Transformer Encoder (2 layers, 4 attention heads, ffn_dim=256)
(batch, 20, 128)   ← each event contextualised by all others
        │
        ▼  Mean-pool + bottleneck Linear  →  embed_dim//4 = 32
(batch, 32)        ← compressed representation; forces meaningful compression
        │
        ▼  Expand Linear  →  embed_dim=128  (broadcast over T time steps)
(batch, 20, 128)
        │
        ▼  Transformer Decoder (cross-attention on encoder memory)
(batch, 20, 128)   ← each position attends to full encoder context
        │
        ▼  Linear projection
(batch, 20, 20)    ← reconstructed feature sequence
```

**Architecture improvements:**
- **Sinusoidal PE** (fixed, not learned): stable on short sequences; prevents
  over-fitting of position embeddings on small datasets.
- **True TransformerDecoder with cross-attention**: each decoded position
  attends to the full encoder memory, giving a richer reconstruction signal
  than a second encoder (which has no cross-attention).
- **Compressed bottleneck** (embed_dim → embed_dim//4): forces the model to
  learn a compact representation of normal behaviour; prevents the trivial
  identity shortcut.

#### Training

- **AdamW** (`weight_decay=1e-4`) for transformer weight regularisation.
- **LR schedule**: 10% linear warmup then cosine annealing to zero (per-batch).
- **Gradient clipping** (`max_norm=1.0`) to prevent exploding gradients.
- 85/15 train/validation split from benign data.
- Early stopping with patience=5.

---

### [7] Rarity Engine — `rarity_engine.py`

The GNN and Transformer both learn patterns from the training data as a whole.
But some events are anomalous simply because they are **extremely rare** —
a process that runs only once across millions of events, or an IP address
that has never been seen before.

The Rarity Engine scores this using **Bayesian (Jeffreys) smoothing**:

```
P(pattern) = (count + 0.5) / (total_count + 0.5 × vocab_size)
rarity_score = 1 − P(pattern)
```

The 0.5 smoothing prevents division-by-zero for unseen patterns and gives
them a rarity score close to 1 (very rare) without being exactly 1.

#### Three Frequency Patterns Tracked

| Pattern | What is counted |
|---------|----------------|
| `parent → child` | How often has this specific parent-child process pair been seen? |
| `process → IP` | How often has this process connected to this specific IP? |
| `destination IP` | How often has this IP appeared as a destination at all? |

All three are fitted on benign-only events, then applied to score all events.
The three sub-scores are averaged into a single rarity score per event.

---

### [8] Composite Anomaly Scoring — `anomaly_engine.py`

The three signals are combined into a single score per event:

```
composite_score = 0.5 × recon_error   +  0.3 × graph_score  +  0.2 × rarity_score
```

Before weighting, each signal is **min-max normalised** using the range
observed on **benign events only**.  This prevents attack scores from
compressing the normalised range and making themselves appear less anomalous
(test-set leakage via the normalisation step).

Because the EventID filter is applied globally at load time, every event in the
dataset is a candidate for transformer scoring. The only events that remain
unscored are the **warmup events** — the first `(SEQUENCE_LENGTH - 1) = 19`
events per host that do not yet have a full sliding window. For these warmup
events, `RECON_WEIGHT` (0.5) is redistributed proportionally to the two
available signals so the composite score still sums to 1.0:

```
warmup composite = (GRAPH_WEIGHT / remain) × graph_score
                 + (RARITY_WEIGHT / remain) × rarity_score
  where remain = GRAPH_WEIGHT + RARITY_WEIGHT = 0.5
```

**Weight rationale:**

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Reconstruction error | 0.5 | Strongest signal; captures temporal attack patterns directly |
| Graph anomaly | 0.3 | Structural relationships are highly informative but noisier |
| Rarity | 0.2 | Strong for novel processes; weaker for common attack tools |

Weights are configurable in `config.py`.

---

### [9] Alert Aggregation — `alert_aggregator.py`

Raw per-event scores are hard for an analyst to act on. This step groups
high-scoring events into **attack chains**:

1. Flag all events with `composite_score >= 0.6` (configurable threshold)
2. **Group by host** (`Computer` column) so events from different machines
   are never merged into the same chain
3. Within each host, sort flagged events by timestamp
4. Group consecutive events whose time gap is **≤ 300 seconds** into one chain
5. For each chain, produce a summary:
   - Start/end time and duration
   - Number of events involved
   - Unique processes involved (in sequence: `cmd.exe → powershell.exe → ...`)
   - Destination IPs contacted
   - Max and mean score across the chain

This transforms a list of millions of scored events into a small table of
human-readable attack narratives.

---

### [10] Evaluation — `metrics.py`

The pipeline produces research-grade evaluation artefacts:

| Output | Description |
|--------|-------------|
| `metrics.json` | All scalar metrics; importable into paper tables |
| `roc_curve.csv` | FPR/TPR at every threshold → ROC curve figure |
| `pr_curve.csv` | Precision/Recall at every threshold → PR curve figure |
| `threshold_sweep.csv` | F1/P/R for every threshold value |
| `anomaly_scores.csv` | Per-event composite score + all three sub-signals |
| `alerts.csv` | Aggregated attack chain alerts |

Key metrics reported:

| Metric | Meaning |
|--------|---------|
| **AUROC** | Area under ROC curve — threshold-independent discrimination ability |
| **PR-AUC** | Area under Precision-Recall curve — more meaningful when attacks are rare |
| **Best F1** | F1 at the threshold that maximises it (Youden's J on ROC) |
| **Precision** | Of flagged events, what fraction are real attacks? |
| **Recall** | Of all real attacks, what fraction did we catch? |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Unsupervised / benign-only training** | Attack samples are rare and evolve constantly; training on normal behaviour generalises to novel attacks |
| **Three-signal ensemble** | Each signal has different blind spots; fusion reduces both false positives and false negatives |
| **Global EventID 1 & 3 filter at load time** | EventID 1 (process creation) and 3 (network connection) carry the strongest attack signal. Filtering once at load time ensures all three engines, training, validation, inference, and evaluation metrics operate on the same consistent population, eliminating any possibility of engine-to-engine scope mismatch |
| **Heterogeneous graph (not homogeneous)** | Processes, IPs, users, and hosts have fundamentally different semantics; mixing them into one node type would destroy information |
| **GraphSAGE (not GCN or GAT)** | Scales to large graphs without requiring full-batch training; inductive (generalises to unseen nodes) |
| **Transformer (not LSTM)** | Self-attention captures arbitrary-range event dependencies; LSTMs struggle with long-range patterns |
| **Sinusoidal PE (not learned)** | Fixed encoding is stable on short sequences and requires no extra parameters |
| **TransformerDecoder (not second encoder)** | Cross-attention on encoder memory produces richer reconstruction signal |
| **Compressed bottleneck (embed_dim//4)** | Forces meaningful compression; prevents identity shortcut that would make all sequences easy to reconstruct |
| **CRC32 categorical hashing** | Deterministic and stable across runs; no fit/transform mismatch at inference |
| **Benign-only scaler fitting** | Prevents attack data from distorting the normalisation baseline |
| **Benign-only normalisation in anomaly engine** | Prevents attack score ranges from compressing the composite score and hiding anomalies |
| **Benign features + full topology for GNN inference** | Avoids contaminated node features while preserving structural attack signals in edge topology |
| **Host-scoped alert chains** | Prevents events from different machines being merged into a single chain |
| **Memmap for large datasets** | Allows processing of multi-million-event datasets on machines with limited RAM |
| **Bayesian smoothing in Rarity Engine** | Prevents zero probabilities for unseen patterns; gives stable scores across rare events |
| **Attack chain aggregation** | Reduces analyst workload from millions of event scores to tens of actionable chains |

---

## Scalability

| Component | Small dataset (<200k events) | Large dataset (>200k events) |
|-----------|------------------------------|------------------------------|
| Sequences | Held in RAM as numpy array | Written to disk as `numpy.memmap` |
| Training | Full-batch from RAM | Mini-batched from disk |
| Inference | Full-pass GPU | Batched in chunks of 512 |
| Graph | Fully in-memory | Fully in-memory (PyG sparse) |

---

## File Map

| File | Role |
|------|------|
| `config.py` | All hyperparameters, weights, paths, and thresholds |
| `feature_engineering.py` | Raw log → 20 numeric features |
| `normaliser.py` | StandardScaler wrapper (benign-fit) |
| `graph_builder.py` | Build heterogeneous HeteroData graph from events |
| `gnn_encoder.py` | HeteroGNNEncoder model, training, centroid scoring |
| `rarity_engine.py` | Bayesian frequency-based rarity scoring |
| `sequence_builder.py` | Sliding-window sequences; memmap path for large datasets |
| `transformer_autoencoder.py` | Transformer autoencoder model definition |
| `train.py` | Training loops for Transformer (in-memory and memmap paths) |
| `evaluate.py` | Batched anomaly score inference |
| `anomaly_engine.py` | Fuse three signals into composite score |
| `alert_aggregator.py` | Group high-score events into attack chain alerts |
| `metrics.py` | Research-grade evaluation: AUROC, PR-AUC, F1, curves |
| `main.py` | End-to-end orchestration |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Place your Sysmon CSV at the path in config.py (default: sysmondataless.csv)
# Run the full pipeline
python main.py
```

### Outputs

| File | Contents |
|------|----------|
| `gnn_encoder.pt` | Trained GNN encoder weights |
| `transformer_autoencoder.pt` | Trained Transformer autoencoder weights |
| `scaler.pkl` | Fitted StandardScaler |
| `anomaly_scores.csv` | Per-event composite + sub-signal scores |
| `alerts.csv` | Attack chain alert table |
| `metrics.json` | All evaluation metrics |
| `roc_curve.csv` | ROC curve data |
| `pr_curve.csv` | PR curve data |
| `threshold_sweep.csv` | F1/P/R at every threshold |

---

## Comparison with the Transformer-Only Pipeline

The `claude/analyze-code-correctness-GmXxV` branch uses a single signal
(Transformer reconstruction error). This full pipeline extends that with two
additional signals and alert aggregation:

| Capability | Transformer-only | This pipeline |
|-----------|-----------------|---------------|
| Temporal sequence anomaly | Yes | Yes |
| Graph / structural anomaly | No | Yes (GNN) |
| Frequency / rarity anomaly | No | Yes (Rarity Engine) |
| Attack chain grouping | No | Yes (Alert Aggregator) |
| Large dataset (>200k) support | No | Yes (memmap) |
| Evaluation artefacts | Basic | Research-grade (curves + sweep) |
| Reproducibility (seeds) | No | Yes (fixed RANDOM_SEED) |
