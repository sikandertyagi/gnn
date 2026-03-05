# Architecture: Transformer Autoencoder Anomaly Detection Pipeline

## What Are We Trying to Achieve?

Modern endpoint security tools generate enormous volumes of Windows event logs
(Sysmon). Manually reviewing millions of events is impossible. The goal of this
pipeline is to **automatically identify malicious activity** — things like
malware execution, command-and-control communication, and lateral movement —
buried inside that volume of normal day-to-day events, **without needing a
labelled training set of attacks**.

The core insight is:

> *Train a model only on normal behaviour. Anything the model cannot
> reconstruct well must be unusual — and therefore suspicious.*

This is called **unsupervised anomaly detection** (specifically, autoencoder-
based). It means the pipeline can detect attacks it has never seen before,
rather than relying on pre-written signatures.

---

## High-Level Pipeline

```
Sysmon CSV (raw endpoint telemetry)
        │
        ▼  [1] EventID Filter
Keep only high-signal event types
  EventID 1 → Process Creation
  EventID 3 → Network Connection
        │
        ▼  [2] Feature Engineering
Convert raw log columns into 23 numeric features
        │
        ▼  [3] Sequence Building
Group events by host machine, sort by time,
slide a window of 20 consecutive events → one training sample
        │
        ▼  [4] Normalisation
Fit StandardScaler on benign (label=0) rows only
        │
        ▼  [5] Transformer Autoencoder — Train on benign sequences
Encoder compresses the sequence → compressed representation
Decoder attempts to reconstruct the original sequence
Trained to minimise reconstruction error on normal data
        │
        ▼  [6] Score All Events
Reconstruction error per sequence (MSE)
High error = sequence looks nothing like normal behaviour
        │
        ▼  [7] Evaluation
AUROC, PR-AUC, Precision, Recall, F1 @ best threshold
```

---

## Step-by-Step Explanation

### [1] EventID Filter — `main.py`

Sysmon logs dozens of event types. Most carry little security signal.

| EventID | Meaning | Why we keep it |
|---------|---------|----------------|
| 1 | Process Creation | Every piece of malware spawns a process |
| 3 | Network Connection | C2 beaconing and lateral movement show here |

Filtering to these two event types reduces noise and speeds up training
significantly. In a large dataset this can reduce millions of rows to the most
actionable subset.

---

### [2] Feature Engineering — `feature_engineering.py`

Raw log columns (file paths, command lines, IPs) are not directly usable by
neural networks. This step converts them into 23 numeric features across five
categories:

#### Process Identity
| Feature | What it captures |
|---------|-----------------|
| `process_name` | Which executable ran (label-encoded integer) |
| `parent_process` | Which process spawned it |
| `parent_child` | The parent→child relationship as a pair |
| `rare_process_score` | 1/frequency — rare processes score higher |

#### Command-Line Behaviour
| Feature | What it detects |
|---------|----------------|
| `cmd_length` | Abnormally long commands (obfuscation) |
| `cmd_token_count` | Unusual argument counts |
| `cmd_entropy` | High entropy = encoded/obfuscated payload |
| `has_base64` | Base64 strings in command line |
| `has_http` | HTTP URLs in command line |
| `has_ip` | Bare IP addresses (C2 / reverse shells) |
| `has_download` | wget, curl, Invoke-WebRequest, BITSAdmin etc. |
| `has_encodedcommand` | PowerShell `-EncodedCommand` / `-enc` flag |

#### Binary Execution Path
| Feature | What it detects |
|---------|----------------|
| `path_depth` | Very shallow or very deep paths |
| `is_system32` | Binary in System32 (legitimacy signal) |
| `is_users_dir` | Binary running from user profile (suspicious) |
| `is_temp_exec` | Binary running from Temp (classic dropper pattern) |

#### Binary Metadata
| Feature | What it detects |
|---------|----------------|
| `is_signed` | Unsigned binaries are more suspicious |
| `missing_company` | Missing company metadata (unsigned malware) |

#### Network Behaviour (EventID 3)
| Feature | What it detects |
|---------|----------------|
| `dest_port` | Unusual destination ports |
| `dest_external` | Connection going outside the internal network |

#### Temporal Behaviour
| Feature | What it detects |
|---------|----------------|
| `hour` | Hour of day (0–23) |
| `is_after_hours` | Activity before 7am or after 7pm |

#### Event Type
| Feature | What it captures |
|---------|----------------|
| `eventid` | 1 (process) or 3 (network) |

---

### [3] Sequence Building — `sequence_builder.py`

A single log event in isolation is rarely conclusive. Attackers leave patterns
across sequences of events — a process spawns, executes a command, makes a
network connection. This step captures that temporal context.

**How it works:**

1. Events are grouped by **host machine** (`Computer` column)
2. Within each host, events are **sorted by timestamp** (`SystemTime`)
3. A **sliding window** of 20 events is moved one step at a time
4. Each window becomes one training sample: shape `(20, 23)`
5. The **label of the last event** in the window becomes the sequence label

```
Events:  [e1, e2, e3, e4, e5, e6 ...]
Window 1: [e1, e2, ... e20]  → label = label(e20)
Window 2: [e2, e3, ... e21]  → label = label(e21)
...
```

Windows never cross host boundaries, so the model only learns patterns within
a single machine's activity.

---

### [4] Normalisation — `main.py`

Neural networks train poorly on features with very different scales (e.g.,
`cmd_length` in thousands vs. `has_base64` in {0, 1}).

- A `StandardScaler` is fitted **only on benign (label=0) rows**
- This is critical: fitting on attack rows would teach the scaler what "normal
  attack scale" looks like, contaminating the baseline
- The same scaler is then applied to all rows (including anomalous ones)

---

### [5] Transformer Autoencoder — `transformer_autoencoder.py` + `train.py`

#### What is an Autoencoder?

An autoencoder is a neural network trained to compress its input and then
reconstruct it. If trained only on normal data, it learns to compress and
reconstruct normal patterns efficiently. When it encounters an anomalous
pattern, it cannot reconstruct it well — the reconstruction error is high.

#### Why a Transformer?

Traditional autoencoders treat each time step independently. A Transformer
uses **self-attention** to model relationships between events at different
positions in the sequence. This lets the model learn that, for example, a
`cmd.exe` spawning from `winword.exe` (position 5) combined with a subsequent
external network connection (position 12) is a suspicious pattern — even
though those events are 7 steps apart.

#### Architecture

```
Input:  (batch, 20 events, 23 features)
        │
        ▼  Linear projection
(batch, 20, embed_dim=96)
        │
        ▼  + Positional Embedding
(batch, 20, 96)   ← each position gets a learned offset
        │
        ▼  Transformer Encoder (2 layers, 4 attention heads)
(batch, 20, 96)   ← each event is now contextualised by all others
        │
        ▼  Linear decoder
(batch, 20, 23)   ← reconstructed feature sequence
```

#### Training

- Trained only on **benign sequences** (label = 0)
- Loss function: **Mean Squared Error** between input and reconstruction
- 80/20 train/validation split (both benign) — validation loss monitors
  overfitting
- Optimiser: Adam

---

### [6] Scoring — `evaluate.py`

After training, every sequence (benign and anomalous) is passed through the
frozen autoencoder.

```
Reconstruction Error = mean((original - reconstructed)²)
                       averaged over all 20 time steps and 23 features
```

- **Benign sequences**: the model has seen this pattern → low error
- **Attack sequences**: the model hasn't learned this → high error

---

### [7] Evaluation — `evaluate.py`

The pipeline computes standard intrusion-detection metrics:

| Metric | What it tells us |
|--------|-----------------|
| **AUROC** | How well the score separates normal from anomalous at every threshold |
| **AUPRC** | Precision-Recall trade-off (more meaningful when attacks are rare) |
| **Best threshold** | Chosen via Youden's J = max(TPR − FPR) on the ROC curve |
| **Precision** | Of events flagged as attacks, how many actually are? |
| **Recall** | Of all actual attacks, how many did we catch? |
| **F1** | Harmonic mean of precision and recall |
| **Confusion matrix** | Full breakdown of TP/FP/TN/FN |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Unsupervised (benign-only training)** | Attack samples are rare and change constantly; a model trained only on normal behaviour generalises to novel attacks |
| **Sequence windows (not single events)** | Attacks are multi-step; temporal context is essential to distinguish malicious from benign behaviour |
| **Transformer (not LSTM)** | Self-attention captures long-range dependencies between events that RNNs struggle with |
| **Benign-only scaler fitting** | Prevents attack data from distorting the normalisation baseline |
| **EventID 1 & 3 only** | Removes noisy, low-signal events; process creation and network connection are the strongest attack indicators in Sysmon |

---

## File Map

| File | Role |
|------|------|
| `config.py` | All hyperparameters and file paths in one place |
| `feature_engineering.py` | Raw log → 23 numeric features |
| `sequence_builder.py` | Sliding-window sequences per host |
| `transformer_autoencoder.py` | Model definition |
| `train.py` | Training loop with validation |
| `evaluate.py` | Anomaly scoring and full metric computation |
| `main.py` | Orchestration: runs all steps end-to-end |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Place your Sysmon CSV at the path in config.py (default: sysmondataless.csv)
# Run the full pipeline
python main.py
```

Outputs:
- `transformer_autoencoder.pt` — trained model weights
- `scaler.pkl` — fitted normalisation scaler
- `anomaly_scores.csv` — per-sequence score and label
- Console: full evaluation metrics

---

## Limitations and Where the Full Pipeline (GNN branch) Improves

This pipeline uses a **single signal** — sequence reconstruction error. It has
two known limitations:

1. **No graph context**: it cannot see that a process is connecting to an IP
   that 10 other suspicious processes also connected to
2. **No frequency awareness**: a process that runs 10,000 times a day and one
   that runs once look the same to the model

The `claude/gnn-full-architecture-GmXxV` branch addresses both by adding a
Graph Neural Network (structural anomaly) and a Rarity Engine (frequency
anomaly) as two additional signals, fused into a composite score.
