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

#### Process Identity
| Feature | What it captures |
|---------|-----------------|
| `process_name` | Which executable ran (label-encoded integer) |
| `parent_process` | Which process spawned it |

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
participated in (e.g. how many times a process ran, average entropy of its
command lines, whether it ever connected externally).

Two separate graphs are built:
- **Benign graph**: only events with label = 0 (used for training the GNN)
- **Full graph**: all events (used for inference / scoring)

Both graphs share the same node encoders so that process names map to the
same integer IDs in both.

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

#### Benign Centroids and Anomaly Scoring

After training, we compute the **centroid** (mean embedding) for each node
type over the benign graph. At inference on the full graph:

```
graph_anomaly_score(node) = ||embedding(node) - centroid||²
```

A process node whose embedding is far from the centroid of all benign process
embeddings is structurally anomalous.

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
```

Windows never cross machine boundaries.

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
        ▼  + Learned Positional Embedding
(batch, 20, 128)   ← order information preserved
        │
        ▼  Transformer Encoder (2 layers, 4 attention heads, ffn_dim=256)
(batch, 20, 128)   ← each event now contextualised by all others
        │
        ▼  Linear decoder
(batch, 20, 20)    ← reconstructed feature sequence
```

#### Training

- Trained only on **benign sequences**
- 85/15 train/validation split (both from benign data)
- Early stopping with patience=5 (stops when validation loss stops improving)
- GPU-accelerated when available (auto-detected via `torch.cuda.is_available()`)

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

Before weighting, each signal is **min-max normalised to [0, 1]** so that
no single signal dominates due to scale differences.

Leading events that do not have a full 20-event window (and therefore no
reconstruction error) are treated as baseline (score=0) rather than being
penalised.

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
2. Sort flagged events by timestamp
3. Group consecutive events whose time gap is **≤ 300 seconds** into one chain
4. For each chain, produce a summary:
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
| **Heterogeneous graph (not homogeneous)** | Processes, IPs, users, and hosts have fundamentally different semantics; mixing them into one node type would destroy information |
| **GraphSAGE (not GCN or GAT)** | Scales to large graphs without requiring full-batch training; inductive (generalises to unseen nodes) |
| **Transformer (not LSTM)** | Self-attention captures arbitrary-range event dependencies; LSTMs struggle with long-range patterns |
| **Benign-only scaler fitting** | Prevents attack data from distorting the normalisation baseline |
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
