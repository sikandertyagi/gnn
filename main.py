"""
main.py
───────
Full end-to-end pipeline:

  Sysmon CSV
    │
    ▼  feature_engineering.py      (vectorised; missing columns filled safely)
  Tabular feature matrix  +  enriched DataFrame
    │
    ▼  normaliser.py               (StandardScaler fitted on benign rows only)
  Normalised feature matrix
    │
    ├─► graph_builder.py           → HeteroData graph
    │       │
    │       ▼  gnn_encoder.py      (early stopping)
    │   HeteroGNNEncoder           trained on benign graph
    │       │
    │       ▼
    │   graph_anomaly_scores       (N_events,)
    │
    ├─► rarity_engine.py           fitted on benign events  [vectorised]
    │       │
    │       ▼
    │   rarity_scores              (N_events,)
    │
    ├─► EventID filter [1, 3] only  ← transformer sees only high-signal types
    │       │                          so it can distinguish attack vs benign
    │       ▼  sequence_builder.py
    │   sliding-window sequences   small → in-memory ndarray
    │                              large → np.memmap on disk
    │       │
    │       ▼  train.py            (val split + early stopping)
    │   TransformerAutoencoder     trained on benign sequences
    │       │
    │       ▼  evaluate.py         (batched inference)
    │   reconstruction_errors      (N_seq_events,) → index-aligned → (N_events,)
    │
    ▼  anomaly_engine.py
  Composite anomaly scores         (N_events,)
    │
    ▼  alert_aggregator.py
  Attack-chain alerts CSV
    │
    ▼  metrics.py                  (research-paper grade evaluation)
  JSON + ROC/PR curve CSVs + threshold sweep table
"""

import random

import numpy as np
import pandas as pd
import torch

from config import (
    DATA_PATH, SEQUENCE_LENGTH, TRAIN_LABEL,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, VAL_RATIO, EARLY_STOPPING_PATIENCE,
    GNN_EPOCHS, GNN_LR,
    MODEL_PATH, GNN_MODEL_PATH,
    SCORES_PATH, ALERTS_PATH,
    LARGE_DATASET_THRESHOLD, SEQ_MEMMAP_PATH, LABELS_MEMMAP_PATH,
    INFER_BATCH_SIZE,
    RANDOM_SEED,
    HIGH_SIGNAL_EVENTIDS,
)
from feature_engineering import feature_engineering
from normaliser import fit_scaler, apply_scaler
from graph_builder import build_event_graph, node_feature_dims
from gnn_encoder import (HeteroGNNEncoder, train_gnn, graph_anomaly_scores)
from rarity_engine import RarityEngine
from sequence_builder import (build_sequences, build_sequences_memmap,
                               load_seq_memmap, load_seq_labels)
from transformer_autoencoder import TransformerAutoencoder
from train import train_model, train_model_large
from evaluate import anomaly_scores
from anomaly_engine import compute_anomaly_scores
from alert_aggregator import aggregate_alerts
from metrics import evaluate


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Fix #11: deterministic behaviour for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main():

    _set_seeds(RANDOM_SEED)

    # ── 1. load data ──────────────────────────────────────────────────────────
    print("\n[1/9] Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # ensure Label column exists (allows inference on unlabelled CSVs)
    if "Label" not in df.columns:
        print("      WARNING: 'Label' column not found — defaulting to 0 (benign).")
        print("      Evaluation metrics will not be meaningful.")
        df["Label"] = 0

    n_events   = len(df)
    use_memmap = n_events > LARGE_DATASET_THRESHOLD
    print(f"      {n_events:,} events — "
          f"{'large (memmap)' if use_memmap else 'small (in-memory)'} mode")

    # ── 2. feature engineering ────────────────────────────────────────────────
    print("\n[2/9] Feature engineering...")
    df, feature_cols = feature_engineering(df)
    print(f"      {len(feature_cols)} feature columns")

    # ── 3. normalise features (benign-fit StandardScaler) ─────────────────────
    print("\n[3/9] Fitting StandardScaler on benign rows...")
    scaler = fit_scaler(df, feature_cols)
    df     = apply_scaler(df, feature_cols, scaler)
    print(f"      Scaler saved → scaler.pkl")

    # ── 4. build heterogeneous event graph ────────────────────────────────────
    print("\n[4/9] Building event graph...")
    df_benign = df[df["Label"] == TRAIN_LABEL]

    # Fix #1: build full_graph first to get shared encoders, then reuse them
    # for benign_graph so both graphs have identical feature dimensions.
    full_graph, encoders = build_event_graph(df)
    benign_graph, _      = build_event_graph(df_benign, encoders=encoders)

    # Fix #5: guard against empty node types
    for ntype in full_graph.node_types:
        if full_graph[ntype].x.shape[0] == 0:
            raise ValueError(f"Empty node type '{ntype}' in full_graph – "
                             "check that the dataset contains the required columns.")

    n_proc   = full_graph["process"].x.shape[0]
    n_ip     = full_graph["ip"].x.shape[0]
    pp_edges = full_graph["process", "parent_of",   "process"].edge_index.shape[1]
    pi_edges = full_graph["process", "connects_to", "ip"     ].edge_index.shape[1]
    print(f"      {n_proc:,} process nodes  {n_ip:,} IP nodes")
    print(f"      {pp_edges:,} parent->child edges  {pi_edges:,} process->IP edges")

    # ── 5. train GNN encoder (benign graph, early stopping) ───────────────────
    print("\n[5/9] Training GNN encoder on benign graph...")
    # Fix #1: use full_graph's feat_dims so model matches inference graph dims
    feat_dims = node_feature_dims(full_graph)
    gnn_model = HeteroGNNEncoder(feat_dims)
    gnn_model = train_gnn(gnn_model, benign_graph, epochs=GNN_EPOCHS, lr=GNN_LR)
    torch.save(gnn_model.state_dict(), GNN_MODEL_PATH)
    print(f"      Saved -> {GNN_MODEL_PATH}")

    # ── 6. per-event graph anomaly scores ─────────────────────────────────────
    print("\n[6/9] Computing graph anomaly scores...")
    proc_graph_scores = graph_anomaly_scores(
        gnn_model, full_graph, encoders["process"]
    )
    proc_enc           = encoders["process"]
    event_images       = df["Image"].fillna("unknown").astype(str).values
    proc_ids           = proc_enc.transform(event_images)
    event_graph_scores = proc_graph_scores[proc_ids].numpy()
    print(f"      mean graph score : {event_graph_scores.mean():.4f}")

    # ── 7. rarity scoring ─────────────────────────────────────────────────────
    print("\n[7/9] Fitting rarity engine & scoring...  [vectorised]")
    rarity              = RarityEngine().fit(df)
    event_rarity_scores = rarity.score_dataframe(df)
    print(f"      mean rarity score: {event_rarity_scores.mean():.4f}")

    # ── 8. sequences + transformer autoencoder ────────────────────────────────
    #
    # WHY we filter here and not globally:
    #   When trained on ALL event types, the transformer learns to reconstruct
    #   benign module-loads, registry writes, and terminations — events that look
    #   identical in attacks and normal operation.  The resulting model reconstructs
    #   attack sequences just as well as benign ones → AUC ≈ 0.50 (non-discriminative).
    #
    #   EventID 1 (process creation) and 3 (network connection) carry the strongest
    #   attack signal.  Restricting the transformer to these two types focuses its
    #   anomaly signal exactly where attacks deviate from normal behaviour.
    #
    #   The GNN (steps 4-6) and rarity engine (step 7) intentionally keep all events:
    #   they extract structural and frequency-based signals that benefit from full context.
    #
    print(f"\n[8/9] Building sequences & training Transformer autoencoder...")
    df["EventID"] = pd.to_numeric(df["EventID"], errors="coerce")
    df_seq        = df[df["EventID"].isin(HIGH_SIGNAL_EVENTIDS)]
    n_seq_events  = len(df_seq)
    use_memmap_seq = n_seq_events > LARGE_DATASET_THRESHOLD
    print(f"      EventID filter: {n_events:,} → {n_seq_events:,} rows "
          f"(kept EventIDs {HIGH_SIGNAL_EVENTIDS})")

    if use_memmap_seq:
        # ── large-dataset path ────────────────────────────────────────────────
        print(f"      Writing sequences to {SEQ_MEMMAP_PATH} ...")
        n_seq, seq_shape = build_sequences_memmap(
            df_seq, feature_cols, SEQUENCE_LENGTH,
            SEQ_MEMMAP_PATH, LABELS_MEMMAP_PATH,
        )
        print(f"      {n_seq:,} sequences of length {SEQUENCE_LENGTH} (on disk)")

        y_seq         = load_seq_labels(LABELS_MEMMAP_PATH, n_seq)
        train_indices = np.where(y_seq == TRAIN_LABEL)[0]
        print(f"      {len(train_indices):,} benign training sequences")

        feature_dim = seq_shape[2]
        ta_model    = TransformerAutoencoder(feature_dim)
        ta_model    = train_model_large(
            ta_model, SEQ_MEMMAP_PATH, seq_shape, train_indices,
            EPOCHS, BATCH_SIZE, LEARNING_RATE,
            val_ratio=VAL_RATIO, patience=EARLY_STOPPING_PATIENCE,
        )
        torch.save(ta_model.state_dict(), MODEL_PATH)
        print(f"      Saved -> {MODEL_PATH}")

        X_mm             = load_seq_memmap(SEQ_MEMMAP_PATH, seq_shape)
        seq_recon_errors = anomaly_scores(ta_model, X_mm, INFER_BATCH_SIZE)

    else:
        # ── small-dataset path ────────────────────────────────────────────────
        X, y_s = build_sequences(df_seq, feature_cols, SEQUENCE_LENGTH)
        print(f"      {X.shape[0]:,} sequences of length {SEQUENCE_LENGTH}")

        X_train = X[y_s == TRAIN_LABEL]
        print(f"      {X_train.shape[0]:,} benign training sequences")

        feature_dim = X.shape[2]
        ta_model    = TransformerAutoencoder(feature_dim)
        ta_model    = train_model(
            ta_model, X_train, EPOCHS, BATCH_SIZE, LEARNING_RATE,
            val_ratio=VAL_RATIO, patience=EARLY_STOPPING_PATIENCE,
        )
        torch.save(ta_model.state_dict(), MODEL_PATH)
        print(f"      Saved -> {MODEL_PATH}")

        seq_recon_errors = anomaly_scores(ta_model, X, INFER_BATCH_SIZE)

    # ── align sequence scores back to the full event array ────────────────────
    #
    # df_seq is a filtered subset of df.  Its .index values are original row
    # positions in df (RangeIndex 0..n_events-1, never reset after filtering).
    # Sequence i spans df_seq rows i..i+SEQUENCE_LENGTH-1, associated with the
    # LAST event in the window: df_seq.index[i + SEQUENCE_LENGTH - 1].
    # Events not in df_seq (non-1/3 EventIDs) and events before the first full
    # window receive NaN — the anomaly engine treats NaN as 0 after normalising.
    #
    event_recon_errors = np.full(n_events, np.nan, dtype=np.float32)
    n_seq              = len(seq_recon_errors)
    last_event_idx     = df_seq.index[SEQUENCE_LENGTH - 1 : SEQUENCE_LENGTH - 1 + n_seq]
    event_recon_errors[last_event_idx] = seq_recon_errors
    n_valid = (~np.isnan(event_recon_errors)).sum()
    print(f"      mean recon error : {np.nanmean(event_recon_errors):.4f}  "
          f"({n_valid:,} events scored, {n_events - n_valid:,} set to NaN)")

    # ── 9. composite scores, alerts, evaluation ────────────────────────────────
    print("\n[9/9] Composite scoring, alerts & evaluation...")
    composite = compute_anomaly_scores(
        event_recon_errors,
        event_graph_scores,
        event_rarity_scores,
    )

    df_scores = pd.DataFrame({
        "score":        composite,
        "recon_error":  event_recon_errors,
        "graph_score":  event_graph_scores,
        "rarity_score": event_rarity_scores,
        "label":        df["Label"].values,
    })
    df_scores.to_csv(SCORES_PATH, index=False)
    print(f"      Scores saved -> {SCORES_PATH}")

    print("\n  Score summary by label:")
    print(df_scores.groupby("label")[
        ["score", "recon_error", "graph_score", "rarity_score"]
    ].mean().to_string())

    alerts = aggregate_alerts(df, composite)
    alerts.to_csv(ALERTS_PATH, index=False)
    print(f"\n  Alerts saved -> {ALERTS_PATH}")

    if not alerts.empty:
        print("\n  Top-5 alert chains:")
        print(alerts.nlargest(5, "max_score")[
            ["chain_id", "num_events", "max_score", "processes", "dest_ips"]
        ].to_string(index=False))

    # only run metrics if we have labelled data
    has_labels = df_scores["label"].nunique() > 1
    if not has_labels:
        print("\n  No anomalous labels found — skipping evaluation metrics.")
        return

    metrics = evaluate(
        df_scores   = df_scores,
        df_alerts   = alerts,
        df_events   = df,
        report_path = "threshold_sweep.csv",
    )

    print(f"  ROC-AUC : {metrics.get('roc_auc', float('nan')):.4f}")
    print(f"  PR-AUC  : {metrics.get('pr_auc',  float('nan')):.4f}")
    best = metrics.get("best_f1_threshold", {})
    print(f"  Best F1 : {best.get('f1', 0):.4f}  "
          f"@ threshold={best.get('threshold', '?')}"
          f"  P={best.get('precision', 0):.4f}  R={best.get('recall', 0):.4f}")


if __name__ == "__main__":
    main()
