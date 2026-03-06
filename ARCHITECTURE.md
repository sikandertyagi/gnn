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

This is called **unsupervised anomaly detection** (autoencoder-based). It means
the pipeline can detect attacks it has never seen before, rather than relying on
pre-written signatures.

---

## High-Level Pipeline

```
Sysmon CSV (raw endpoint telemetry)
        │
        ▼  [1] EventID Filter
        │   Keep only high-signal event types
        │   EventID 1 → Process Creation
        │   EventID 3 → Network Connection
        │
        ▼  [2] Feature Engineering
        │   Convert raw log columns → 24 numeric features
        │   CRC32-hashed categoricals (stable, [0,1]) + numeric features
        │
        ▼  [3] Sequence Building
        │   Group events by host machine, sort by time,
        │   slide a window of 20 consecutive events → one sample
        │
        ▼  [4] Normalisation
        │   StandardScaler fitted on benign (label=0) rows only
        │   (CRC32 categorical columns skipped — already in [0,1])
        │
        ▼  [5] Transformer Autoencoder — Train on benign sequences only
        │   Sinusoidal PE → Encoder → bottleneck
        │   → TransformerDecoder (cross-attention) → reconstruction
        │   Loss: MSE(input, reconstruction)
        │
        ▼  [6] Score All Sequences
        │   Reconstruction MSE per sequence
        │   High error = sequence looks nothing like normal behaviour
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

Filtering to these two event types reduces noise and speeds up training.

---

### [2] Feature Engineering — `feature_engineering.py`

Raw log columns (file paths, command lines, IPs) are converted into 24 numeric
features across five categories.

#### Categorical Encoding

Categorical features (`process_name`, `parent_process`, `parent_child`) are
encoded using **deterministic CRC32 hashing** normalised to [0, 1]:

```python
hash_value = (zlib.crc32(value.encode("utf-8")) & 0xFFFFFFFF) / 0xFFFFFFFF
```

**Why CRC32 instead of `sklearn.LabelEncoder`?**

| Issue | LabelEncoder | CRC32 hashing |
|-------|-------------|---------------|
| Stability across runs | No (order-dependent, PYTHONHASHSEED) | Yes (always deterministic) |
| Fit/transform mismatch at inference | Yes (new categories crash or corrupt) | Never |
| Scaling | Raw integers (0–N) | Always [0, 1] |

#### Process Identity
| Feature | What it captures |
|---------|-----------------|
| `process_name` | Which executable ran (CRC32 hash) |
| `parent_process` | Which process spawned it |
| `parent_child` | The parent→child relationship as a pair |
| `rare_process_score` | `1/frequency` — rare processes score higher |

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

#### Network Behaviour
| Feature | What it detects |
|---------|----------------|
| `dest_port` | Unusual destination ports |
| `dest_external` | Connection going outside the internal network |

**RFC 1918 fix**: private ranges correctly checked as `10.0.0.0/8`,
`192.168.0.0/16`, `172.16.0.0/12` (i.e. `172.16.x.x`–`172.31.x.x`),
and loopback `127.x.x.x`. The original `startswith("172.")` was incorrect
and treated public addresses in `172.0–172.15` and `172.32–172.255` as private.

**`dest_external` bug fix**: the original code applied bitwise `~` to an int
Series, yielding `-1`/`-2` instead of `1`/`0`. Fixed by inverting the bool
Series before `.astype(int)`.

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

1. Events are grouped by **host machine** (`Computer` column)
2. Within each host, events are **sorted by timestamp** (`SystemTime`)
3. A **sliding window** of 20 events is moved one step at a time
4. Each window becomes one training sample: shape `(20, 24)`
5. The **label of the last event** in the window becomes the sequence label

```
Events:  [e1, e2, e3, e4, e5, e6 ...]
Window 1: [e1, e2, ... e20]  → label = label(e20)
Window 2: [e2, e3, ... e21]  → label = label(e21)
...
```

Windows never cross host boundaries.

---

### [4] Normalisation — `main.py`

Neural networks train poorly on features with very different scales.

- A `StandardScaler` is fitted **only on benign (label=0) rows**
- Fitting on attack rows would teach the scaler what "normal attack scale"
  looks like, contaminating the baseline
- **CRC32-hashed categorical columns are skipped**: they are already in [0, 1];
  z-scoring arbitrary hash values is semantically meaningless
- The same scaler is applied to all rows at inference time

---

### [5] Transformer Autoencoder — `transformer_autoencoder.py` + `train.py`

#### What is an Autoencoder?

An autoencoder learns to compress its input into a lower-dimensional
representation and then reconstruct it. Trained only on normal data, it becomes
very good at reconstructing normal patterns and very bad at reconstructing
anomalous ones. **High reconstruction error = anomalous sequence.**

#### Architecture

```
Input:  (batch, 20 events, 24 features)
        │
        ▼  Linear projection + LayerNorm  →  embed_dim=96
(batch, 20, 96)
        │
        ▼  + Sinusoidal Positional Encoding (+ Dropout)
(batch, 20, 96)   ← event ordering is now encoded
        │
        ▼  Transformer Encoder (2 layers, 4 heads, dropout)
(batch, 20, 96)   ← each event contextualised by all others
        │
        ▼  Mean pool → Bottleneck (96 → 24)
(batch, 24)       ← compressed sequence representation
        │
        ▼  Expand (24 → 96) → Sinusoidal Positional Queries
(batch, 20, 96)
        │
        ▼  Transformer Decoder (cross-attention to encoder memory)
(batch, 20, 96)   ← each position reconstructed from full context
        │
        ▼  Linear output projection
(batch, 20, 24)   ← reconstructed feature sequence
```

**Key fixes vs. original implementation:**

| Issue | Fix |
|-------|-----|
| Learned positional embedding (unstable for long sequences) | Replaced with fixed sinusoidal PE — stable for any sequence length |
| Decoder was a single `nn.Linear` — no temporal structure | Replaced with `nn.TransformerDecoder` + cross-attention to encoder output |
| Bottleneck was same dimension as encoder output (identity shortcut) | Compressed to `embed_dim//4` to force meaningful representation |
| No dropout | Added `dropout=0.1` in all transformer layers and PE |
| No gradient clipping | Added `clip_grad_norm_(max_norm=1.0)` |

#### Training Improvements — `train.py`

| Issue | Fix |
|-------|-----|
| Plain `Adam` with no regularisation | **AdamW** with `weight_decay=1e-4` |
| Constant learning rate (can diverge early or stagnate late) | **Linear warmup** (10% of steps) **→ cosine decay** to 0 |
| No early stopping | **Patience=5**: restores best checkpoint when val loss stagnates |
| Non-reproducible shuffle (`np.random.permutation`) | Fixed seed via `np.random.default_rng(RANDOM_SEED)` |
| Full validation batch (OOM risk) | Validation computed in mini-batches |

---

### [6] Scoring — `evaluate.py`

After training, every sequence is passed through the frozen autoencoder:

```
reconstruction_error(seq) = mean((original - reconstructed)²)
                             averaged over all 20 time steps and 24 features
```

- **Benign sequences**: model has seen this pattern → low error
- **Attack sequences**: model hasn't learned this → high error

---

### [7] Evaluation — `evaluate.py`

| Metric | What it tells us |
|--------|-----------------|
| **AUROC** | How well the score separates normal from anomalous at every threshold |
| **AUPRC** | Precision-Recall trade-off (more meaningful when attacks are rare) |
| **Best threshold** | Chosen via Youden's J = max(TPR − FPR) on the ROC curve |
| **Precision** | Of events flagged as attacks, how many actually are? |
| **Recall** | Of all actual attacks, how many did we catch? |
| **F1** | Harmonic mean of precision and recall |
| **Confusion matrix** | Full breakdown of TP/FP/TN/FN |

Edge case handled: if the evaluation set contains no anomalies (all label=0),
metric computation is skipped gracefully instead of raising a `ValueError`.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Unsupervised (benign-only training)** | Attack samples are rare and change constantly; a model trained only on normal behaviour generalises to novel attacks |
| **Sequence windows (not single events)** | Attacks are multi-step; temporal context is essential to distinguish malicious from benign behaviour |
| **CRC32 categorical hashing** | Deterministic, run-stable encoding; no fit/transform mismatch at inference; always [0,1] |
| **Categorical columns skipped in StandardScaler** | CRC32 hashes are already in [0,1]; z-scoring arbitrary IDs is semantically harmful |
| **Sinusoidal positional encoding** | Fixed (not learned); stable for any sequence length; injects temporal ordering into the transformer |
| **True TransformerDecoder with cross-attention** | Each decoded position attends to the full encoded sequence, unlike the original single linear layer |
| **Compressed bottleneck (embed_dim//4)** | Forces meaningful compression; prevents identity shortcut that collapses all reconstruction errors to near zero |
| **AdamW + weight decay** | Regularises large transformer weight matrices; reduces overfitting on small benign training sets |
| **LR warmup + cosine decay** | Prevents unstable updates in the first epoch; improves final convergence |
| **Early stopping** | Avoids overfitting; saves compute; restores the best-seen checkpoint |
| **Benign-only scaler fitting** | Prevents attack data from distorting the normalisation baseline |
| **RFC 1918 correct 172.16/12 check** | 172.0–172.15 and 172.32–172.255 are public; the old `startswith("172.")` was incorrect |
| **EventID 1 & 3 only** | Process creation and network connection are the strongest attack indicators in Sysmon; other events add noise |

---

## File Map

| File | Role |
|------|------|
| `config.py` | All hyperparameters, paths, and seeds |
| `feature_engineering.py` | Raw log → 24 numeric features; CRC32 categorical hashing; RFC 1918 fix |
| `sequence_builder.py` | Sliding-window sequences per host (host-scoped, no cross-host leakage) |
| `transformer_autoencoder.py` | Model: sinusoidal PE + true TransformerDecoder + compressed bottleneck |
| `train.py` | Training loop: AdamW, warmup+cosine LR, grad clipping, early stopping, seed |
| `evaluate.py` | Batched anomaly scoring + metrics (handles edge cases gracefully) |
| `main.py` | Orchestration: EventID filter, feature engineering, normalisation, train, score |

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
| `transformer_autoencoder.pt` | Trained model weights |
| `scaler.pkl` | Fitted StandardScaler (numeric columns only) |
| `anomaly_scores.csv` | Per-sequence reconstruction error and label |
| Console | Full evaluation metrics (AUROC, AUPRC, F1, confusion matrix) |
