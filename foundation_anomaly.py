"""
foundation_anomaly.py
─────────────────────
Foundation-model approach to Sysmon anomaly detection.

Instead of handcrafted features → autoencoder, this pipeline:

  1. Converts each Sysmon event into a structured text string
  2. Fine-tunes DistilBERT's Masked Language Model head on benign events
  3. Scores every event by its MLM loss (how "surprising" the text is)
  4. High loss = the model has never seen this pattern in normal data = anomaly

Why this can outperform the dense AE:
  · The language model learns contextual patterns ("powershell launched from
    temp dir with encoded command targeting external IP" is anomalous as a
    *combination*, even if each field individually looks normal)
  · Pre-trained weights already encode general language understanding
  · No information loss from hashing / bucketing categorical fields

Architecture:  DistilBERT (66M params) + MLM head
Training:      Masked Language Modeling on benign events only
Scoring:       Average cross-entropy on masked tokens (multiple passes)

Usage:
    python foundation_anomaly.py
    python foundation_anomaly.py --epochs 5 --batch-size 32
"""

import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import (
        DistilBertTokenizerFast,
        DistilBertForMaskedLM,
        get_linear_schedule_with_warmup,
    )
except ImportError:
    print("ERROR: transformers not installed.")
    print("  pip install transformers")
    sys.exit(1)

from config import (
    DATA_PATH, ARTIFACTS_DIR,
    RANDOM_SEED, TRAIN_LABEL,
    ALERT_PERCENTILE,
)

# ── defaults ─────────────────────────────────────────────────────────────────
MODEL_NAME       = "distilbert-base-uncased"
MAX_LENGTH       = 256
MLM_PROBABILITY  = 0.15
TRAIN_SUBSAMPLE  = 300_000
FT_EPOCHS        = 3
FT_BATCH_SIZE    = 64
FT_LR            = 2e-5
FT_WARMUP_RATIO  = 0.1
SCORE_BATCH_SIZE = 128
NUM_SCORE_PASSES = 3
FILTER_EVENTIDS  = [1, 3]

SAVE_DIR = os.path.join(ARTIFACTS_DIR, "foundation_model")


# ─────────────────────────────────────────────────────────────────────────────
# Event → text conversion
# ─────────────────────────────────────────────────────────────────────────────

def _clean(val, fallback="") -> str:
    """Return cleaned string or fallback for NaN / zero-fill / empty."""
    if val is None or pd.isna(val):
        return fallback
    s = str(val).strip()
    if s in ("", "0", "nan", "-"):
        return fallback
    return s


def _proc_name(path_str: str) -> str:
    """Extract lowercase process name from a Windows/Linux path."""
    if not path_str:
        return "unknown"
    return path_str.replace("\\", "/").split("/")[-1].lower()


def event_to_text(row: pd.Series) -> str:
    """Convert a Sysmon event row into a structured text string.

    Uses the actual column names from the Sysmon CSV. Includes all
    high-signal fields for anomaly detection:
      - process + parent process (name, path, command line)
      - user context (user, integrity, working directory)
      - binary metadata (signed, company, original filename)
      - network info (protocol, source, destination)
      - host identity

    Example output:
        host:laptop-01 event:process_create process:powershell.exe
        parent:cmd.exe user:system\admin integrity:high
        cmdline:powershell -enc ... parentcmd:cmd /c ...
        cwd:c:\users\admin\appdata\local\temp signed:false
        company:unknown originalname:powershell.exe
        path:c:\windows\system32\windowspowershell\v1.0\powershell.exe
        dest:192.168.1.1:443 protocol:tcp
    """
    parts = []

    # host
    host = _clean(row.get("Computer"), "unknown")
    parts.append(f"host:{host.lower()}")

    # event type
    eid = int(float(_clean(row.get("EventID"), "0") or "0"))
    eid_map = {
        1: "process_create", 3: "network_connect", 5: "process_terminate",
        7: "image_load", 10: "process_access", 11: "file_create",
        12: "registry_add", 13: "registry_set", 14: "registry_rename",
    }
    parts.append(f"event:{eid_map.get(eid, f'type_{eid}')}")

    # process info
    image = _clean(row.get("Image"), "unknown")
    parts.append(f"process:{_proc_name(image)}")

    parent_image = _clean(row.get("ParentImage"), "unknown")
    parts.append(f"parent:{_proc_name(parent_image)}")

    # user context
    user = _clean(row.get("User"), "unknown")
    parts.append(f"user:{user.lower()}")

    integrity = _clean(row.get("IntegrityLevel"), "unknown")
    parts.append(f"integrity:{integrity.lower()}")

    # command line — the single richest field for attack detection
    cmdline = _clean(row.get("CommandLine"))
    if cmdline:
        parts.append(f"cmdline:{cmdline[:400]}")

    # parent command line — crucial for detecting lateral movement / LOLBins
    parent_cmd = _clean(row.get("ParentCommandLine"))
    if parent_cmd:
        parts.append(f"parentcmd:{parent_cmd[:300]}")

    # working directory — temp/downloads dirs are suspicious
    cwd = _clean(row.get("CurrentDirectory"))
    if cwd:
        parts.append(f"cwd:{cwd.lower()[:150]}")

    # binary metadata
    signed = _clean(row.get("Signed"), "unknown")
    parts.append(f"signed:{signed.lower()}")

    company = _clean(row.get("Company"))
    if company:
        parts.append(f"company:{company.lower()[:80]}")

    # original filename — detects renamed binaries (e.g. mimikatz → svchost)
    orig = _clean(row.get("OriginalFileName"))
    if orig:
        parts.append(f"originalname:{orig.lower()}")

    # full image path
    if image != "unknown":
        parts.append(f"path:{image.lower()[:200]}")

    # network fields (EventID 3)
    protocol = _clean(row.get("Protocol"))
    if protocol:
        parts.append(f"protocol:{protocol.lower()}")

    src_ip = _clean(row.get("SourceIp"))
    src_port = _clean(row.get("SourcePort"))
    if src_ip:
        parts.append(f"src:{src_ip}:{src_port}")

    dest_ip = _clean(row.get("DestinationIp"))
    dest_port = _clean(row.get("DestinationPort"))
    if dest_ip:
        parts.append(f"dest:{dest_ip}:{dest_port}")

    dest_host = _clean(row.get("DestinationHostname"))
    if dest_host:
        parts.append(f"desthost:{dest_host.lower()[:100]}")

    return " ".join(parts)


def events_to_texts(df: pd.DataFrame) -> list[str]:
    """Convert all events to text strings (vectorised where possible)."""
    print(f"  Converting {len(df):,} events to text...")
    t0 = time.time()
    texts = df.apply(event_to_text, axis=1).tolist()
    print(f"  Done in {time.time() - t0:.1f}s")
    # sample
    print(f"  Example: {texts[0][:120]}...")
    return texts


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class EventTextDataset(Dataset):
    """Pre-tokenized event dataset for MLM training or inference."""

    def __init__(self, encodings):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def tokenize_texts(texts: list[str], tokenizer, max_length: int = MAX_LENGTH,
                   batch_size: int = 10000) -> dict:
    """Tokenize texts in batches to avoid OOM on large datasets."""
    all_ids = []
    all_masks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        all_ids.append(enc["input_ids"])
        all_masks.append(enc["attention_mask"])
        if (i // batch_size) % 10 == 0:
            print(f"    Tokenized {min(i + batch_size, len(texts)):,}/{len(texts):,}")
    return {
        "input_ids": torch.cat(all_ids, dim=0),
        "attention_mask": torch.cat(all_masks, dim=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MLM masking
# ─────────────────────────────────────────────────────────────────────────────

def mask_tokens(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                tokenizer, mlm_prob: float = MLM_PROBABILITY,
                generator: torch.Generator = None):
    """Create MLM labels: mask mlm_prob of tokens, return (masked_ids, labels).

    Labels are -100 for non-masked positions (ignored by cross-entropy).
    """
    labels = input_ids.clone()
    probability_matrix = torch.full(input_ids.shape, mlm_prob)

    # don't mask special tokens or padding
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for sp_id in tokenizer.all_special_ids:
        special_mask |= (input_ids == sp_id)
    pad_mask = (attention_mask == 0)
    probability_matrix.masked_fill_(special_mask | pad_mask, 0.0)

    masked_indices = torch.bernoulli(probability_matrix, generator=generator).bool()
    labels[~masked_indices] = -100

    # 80% → [MASK], 10% → random, 10% → keep
    indices_replaced = masked_indices & torch.bernoulli(
        torch.full(input_ids.shape, 0.8), generator=generator
    ).bool()
    input_ids[indices_replaced] = tokenizer.mask_token_id

    indices_random = masked_indices & ~indices_replaced & torch.bernoulli(
        torch.full(input_ids.shape, 0.5), generator=generator
    ).bool()
    random_words = torch.randint(
        len(tokenizer), input_ids.shape, dtype=torch.long, generator=generator,
    )
    input_ids[indices_random] = random_words[indices_random]

    return input_ids, labels


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_mlm(model, train_dataset, tokenizer, epochs, batch_size, lr,
              warmup_ratio, val_dataset=None):
    """Fine-tune DistilBERT MLM on benign event texts."""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = model.to(device)

    loader_kw = dict(
        num_workers=4 if use_cuda else 0,
        pin_memory=use_cuda,
        persistent_workers=use_cuda and True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED), **loader_kw,
    )

    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size * 2, shuffle=False, **loader_kw,
        )

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    print(f"\n  Fine-tuning {MODEL_NAME} MLM on {device}")
    print(f"  {len(train_dataset):,} train samples, {epochs} epochs, "
          f"batch_size={batch_size}")
    print(f"  {total_steps:,} total steps, {warmup_steps:,} warmup steps")

    best_val_loss = float("inf")
    best_state = None
    gen = torch.Generator().manual_seed(RANDOM_SEED)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            ids = batch["input_ids"].clone()
            attn = batch["attention_mask"]

            masked_ids, labels = mask_tokens(ids, attn, tokenizer,
                                             generator=gen)
            masked_ids = masked_ids.to(device, non_blocking=True)
            attn = attn.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_cuda):
                outputs = model(input_ids=masked_ids, attention_mask=attn,
                                labels=labels)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        train_loss = total_loss / n_batches
        elapsed = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]

        # validation
        val_str = ""
        if val_loader is not None:
            model.eval()
            val_total = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    ids = batch["input_ids"].clone()
                    attn = batch["attention_mask"]
                    masked_ids, labels = mask_tokens(ids, attn, tokenizer,
                                                     generator=gen)
                    masked_ids = masked_ids.to(device, non_blocking=True)
                    attn = attn.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_cuda):
                        out = model(input_ids=masked_ids, attention_mask=attn,
                                    labels=labels)
                    val_total += out.loss.item()
                    val_n += 1
            val_loss = val_total / val_n
            val_str = f"  val={val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
        else:
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_state = copy.deepcopy(model.state_dict())

        print(f"  Epoch {epoch+1}/{epochs}  train={train_loss:.4f}{val_str}  "
              f"lr={cur_lr:.2e}  ({elapsed:.0f}s)")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Training complete — best loss: {best_val_loss:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Scoring — per-event MLM loss
# ─────────────────────────────────────────────────────────────────────────────

def score_events_mlm(model, dataset, tokenizer, batch_size=SCORE_BATCH_SIZE,
                     num_passes=NUM_SCORE_PASSES):
    """Compute per-event anomaly score as average MLM loss over multiple passes.

    Each pass masks a different random 15% of tokens. Averaging over passes
    gives a stable score that doesn't depend on which tokens happen to be masked.
    """
    use_cuda = torch.cuda.is_available()
    device = next(model.parameters()).device
    model.eval()

    n = len(dataset)
    all_scores = np.zeros(n, dtype=np.float64)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4 if use_cuda else 0, pin_memory=use_cuda,
    )

    for pass_idx in range(num_passes):
        gen = torch.Generator().manual_seed(RANDOM_SEED + pass_idx + 100)
        scores_pass = []

        with torch.no_grad():
            for batch in loader:
                ids = batch["input_ids"].clone()
                attn = batch["attention_mask"]

                masked_ids, labels = mask_tokens(ids, attn, tokenizer,
                                                 generator=gen)
                masked_ids = masked_ids.to(device, non_blocking=True)
                attn = attn.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    logits = model(input_ids=masked_ids,
                                   attention_mask=attn).logits

                # per-token cross-entropy, then average over masked tokens per event
                # logits: (B, seq_len, vocab_size)
                # labels: (B, seq_len) with -100 for non-masked
                B, S, V = logits.shape
                ce = F.cross_entropy(
                    logits.view(-1, V), labels.view(-1), reduction="none",
                ).view(B, S)

                # average only over masked positions (labels != -100)
                mask = (labels != -100).float()
                n_masked = mask.sum(dim=1).clamp(min=1)
                event_loss = (ce * mask).sum(dim=1) / n_masked
                scores_pass.append(event_loss.cpu().numpy())

        scores_pass = np.concatenate(scores_pass)
        all_scores += scores_pass
        print(f"    Pass {pass_idx+1}/{num_passes}  "
              f"mean_loss={scores_pass.mean():.4f}")

    return (all_scores / num_passes).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_scores(labels, scores):
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

    binary = (labels > 0).astype(int)
    roc_auc = roc_auc_score(binary, scores)

    prec_arr, rec_arr, pr_thresholds = precision_recall_curve(binary, scores)
    pr_auc = auc(rec_arr, prec_arr)

    f1_scores = 2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-12)
    best_idx = np.argmax(f1_scores)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "best_f1": f1_scores[best_idx],
        "best_f1_threshold": float(pr_thresholds[best_idx]) if best_idx < len(pr_thresholds) else float(pr_thresholds[-1]),
        "best_f1_precision": prec_arr[best_idx],
        "best_f1_recall": rec_arr[best_idx],
    }


def evaluate_at_threshold(labels, scores, threshold):
    binary = (labels > 0).astype(int)
    preds = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (binary == 1)).sum())
    fp = int(((preds == 1) & (binary == 0)).sum())
    fn = int(((preds == 0) & (binary == 1)).sum())
    tn = int(((preds == 0) & (binary == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {"threshold": threshold, "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Foundation model anomaly detection")
    p.add_argument("--epochs", type=int, default=FT_EPOCHS)
    p.add_argument("--batch-size", type=int, default=FT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=FT_LR)
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--train-subsample", type=int, default=TRAIN_SUBSAMPLE)
    p.add_argument("--score-passes", type=int, default=NUM_SCORE_PASSES)
    p.add_argument("--all-events", action="store_true",
                   help="Use all event types (default: EventID 1 & 3 only)")
    p.add_argument("--score-only", action="store_true",
                   help="Skip training, load saved model and score")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.benchmark = True

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── 1. load data ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("Foundation Model Anomaly Detection (DistilBERT MLM)")
    print("=" * 70)

    print("\n[1/6] Loading data...")
    df = pd.read_csv(DATA_PATH)
    if "Label" not in df.columns:
        df["Label"] = 0
    n_raw = len(df)

    if not args.all_events:
        df["EventID"] = pd.to_numeric(df["EventID"], errors="coerce")
        df = df[df["EventID"].isin(FILTER_EVENTIDS)].reset_index(drop=True)
        print(f"  {n_raw:,} → {len(df):,} events (EventIDs {FILTER_EVENTIDS})")
    else:
        print(f"  {n_raw:,} events (all event types)")

    labels = df["Label"].values
    benign_mask = labels == TRAIN_LABEL
    n_benign = int(benign_mask.sum())
    n_pos = int((labels > 0).sum())
    print(f"  {n_benign:,} benign, {n_pos:,} positive (label > 0)")

    # ── 2. convert events to text ────────────────────────────────────────────
    print("\n[2/6] Converting events to text...")
    texts = events_to_texts(df)

    # ── 3. tokenize ──────────────────────────────────────────────────────────
    print("\n[3/6] Tokenizing...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    # split benign texts for training
    benign_indices = np.where(benign_mask)[0]
    if args.train_subsample and len(benign_indices) > args.train_subsample:
        rng = np.random.default_rng(RANDOM_SEED)
        train_indices = rng.choice(benign_indices, size=args.train_subsample,
                                   replace=False)
    else:
        train_indices = benign_indices

    # val split from training subset
    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(train_indices))
    n_val = max(1, int(len(train_indices) * 0.1))
    val_idx = train_indices[perm[:n_val]]
    tr_idx = train_indices[perm[n_val:]]

    train_texts = [texts[i] for i in tr_idx]
    val_texts = [texts[i] for i in val_idx]

    print(f"  Training: {len(train_texts):,} benign events")
    print(f"  Validation: {len(val_texts):,} benign events")

    print("  Tokenizing training set...")
    train_enc = tokenize_texts(train_texts, tokenizer, args.max_length)
    print("  Tokenizing validation set...")
    val_enc = tokenize_texts(val_texts, tokenizer, args.max_length)

    train_dataset = EventTextDataset(train_enc)
    val_dataset = EventTextDataset(val_enc)

    # ── 4. train or load model ───────────────────────────────────────────────
    model_path = os.path.join(SAVE_DIR, "model")

    if args.score_only and os.path.exists(model_path):
        print(f"\n[4/6] Loading saved model from {model_path}...")
        model = DistilBertForMaskedLM.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    else:
        print(f"\n[4/6] Fine-tuning {MODEL_NAME}...")
        model = DistilBertForMaskedLM.from_pretrained(MODEL_NAME)
        model = train_mlm(
            model, train_dataset, tokenizer,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, warmup_ratio=FT_WARMUP_RATIO,
            val_dataset=val_dataset,
        )
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        print(f"  Model saved → {model_path}")

    # ── 5. score all events ──────────────────────────────────────────────────
    print(f"\n[5/6] Scoring all {len(texts):,} events "
          f"({args.score_passes} passes)...")

    print("  Tokenizing full dataset...")
    all_enc = tokenize_texts(texts, tokenizer, args.max_length)
    all_dataset = EventTextDataset(all_enc)

    t0 = time.time()
    scores = score_events_mlm(
        model, all_dataset, tokenizer,
        batch_size=SCORE_BATCH_SIZE, num_passes=args.score_passes,
    )
    print(f"  Scoring done in {time.time() - t0:.1f}s")

    # ── 6. evaluate ──────────────────────────────────────────────────────────
    print(f"\n[6/6] Evaluation...")

    # score summary by label
    for lbl, name in [(0, "Benign"), (1, "Attack"), (2, "Suspicious")]:
        mask = labels == lbl
        if mask.any():
            s = scores[mask]
            print(f"  {name:>10}: n={mask.sum():>8,}  "
                  f"mean={s.mean():.4f}  median={np.median(s):.4f}  "
                  f"p95={np.percentile(s, 95):.4f}  "
                  f"p99={np.percentile(s, 99):.4f}")

    # threshold
    benign_scores = scores[benign_mask]
    threshold = float(np.percentile(benign_scores, ALERT_PERCENTILE))
    print(f"\n  Threshold (p{ALERT_PERCENTILE} benign): {threshold:.6f}")

    if labels.max() > 0:
        metrics = evaluate_scores(labels, scores)
        print(f"\n  ROC-AUC  : {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC   : {metrics['pr_auc']:.4f}")
        print(f"  Best F1  : {metrics['best_f1']:.4f}  "
              f"@ threshold={metrics['best_f1_threshold']:.6f}  "
              f"P={metrics['best_f1_precision']:.4f}  "
              f"R={metrics['best_f1_recall']:.4f}")

        pct = evaluate_at_threshold(labels, scores, threshold)
        print(f"\n  @ p{ALERT_PERCENTILE} threshold ({threshold:.6f}):")
        print(f"    Precision: {pct['precision']:.4f}")
        print(f"    Recall   : {pct['recall']:.4f}")
        print(f"    F1       : {pct['f1']:.4f}")
        print(f"    TP={pct['tp']:,}  FP={pct['fp']:,}  "
              f"FN={pct['fn']:,}  TN={pct['tn']:,}")

        bf1 = evaluate_at_threshold(labels, scores, metrics["best_f1_threshold"])
        print(f"\n  @ best-F1 threshold ({metrics['best_f1_threshold']:.6f}):")
        print(f"    Precision: {bf1['precision']:.4f}")
        print(f"    Recall   : {bf1['recall']:.4f}")
        print(f"    F1       : {bf1['f1']:.4f}")
        print(f"    TP={bf1['tp']:,}  FP={bf1['fp']:,}  "
              f"FN={bf1['fn']:,}  TN={bf1['tn']:,}")

    # save scores
    out_path = os.path.join(ARTIFACTS_DIR, "foundation_scores.csv")
    pd.DataFrame({
        "label": labels,
        "mlm_loss": scores,
    }).to_csv(out_path, index=False)
    print(f"\n  Scores saved → {out_path}")

    # save metrics
    if labels.max() > 0:
        import json
        metrics_path = os.path.join(ARTIFACTS_DIR, "foundation_metrics.json")
        metrics["threshold_percentile"] = threshold
        metrics["pct_results"] = pct
        metrics["best_f1_results"] = bf1
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Metrics saved → {metrics_path}")

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
