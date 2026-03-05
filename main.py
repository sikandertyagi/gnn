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
    ├─► sequence_builder.py        sliding-window sequences
    │       │                      small dataset → in-memory ndarray
    │       │                      large dataset → np.memmap on disk
    │       ▼  train.py            (val split + early stopping)
    │   TransformerAutoencoder     trained on benign sequences
    │       │
    │       ▼  evaluate.py         (batched inference)
    │   reconstruction_errors      (N_sequences,)
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
)
from feature_engineering import feature_engineering
from normaliser import fit_scaler, apply_scaler
from graph_builder import build_event_graph, node_feature_dims
from gnn_encoder import (HeteroGNNEncoder, train_gnn,
                         compute_benign_centroids, graph_anomaly_scores)
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
    df_benign        = df[df["Label"] == TRAIN_LABEL]
    full_graph,   encoders = build_event_graph(df)
    benign_graph, _        = build_event_graph(df_benign)

    n_proc   = full_graph["process"].x.shape[0]
    n_ip     = full_graph["ip"].x.shape[0]
    pp_edges = full_graph["process", "parent_of",   "process"].edge_index.shape[1]
    pi_edges = full_graph["process", "connects_to", "ip"     ].edge_index.shape[1]
    print(f"      {n_proc:,} process nodes  {n_ip:,} IP nodes")
    print(f"      {pp_edges:,} parent->child edges  {pi_edges:,} process->IP edges")

    # ── 5. train GNN encoder (benign graph, early stopping) ───────────────────
    print("\n[5/9] Training GNN encoder on benign graph...")
    feat_dims = node_feature_dims(benign_graph)
    gnn_model = HeteroGNNEncoder(feat_dims)
    gnn_model = train_gnn(gnn_model, benign_graph, epochs=GNN_EPOCHS, lr=GNN_LR)
    torch.save(gnn_model.state_dict(), GNN_MODEL_PATH)
    print(f"      Saved -> {GNN_MODEL_PATH}")

    benign_centroids = compute_benign_centroids(gnn_model, benign_graph)

    # ── 6. per-event graph anomaly scores ─────────────────────────────────────
    print("\n[6/9] Computing graph anomaly scores...")
    proc_graph_scores = graph_anomaly_scores(
        gnn_model, full_graph, benign_centroids, encoders["process"]
    )
    proc_enc           = encoders["process"]
    event_images       = df["Image"].fillna("unknown").values
    proc_ids           = proc_enc.transform(event_images)
    event_graph_scores = proc_graph_scores[proc_ids].numpy()
    print(f"      mean graph score : {event_graph_scores.mean():.4f}")

    # ── 7. rarity scoring ─────────────────────────────────────────────────────
    print("\n[7/9] Fitting rarity engine & scoring...  [vectorised]")
    rarity              = RarityEngine().fit(df)
    event_rarity_scores = rarity.score_dataframe(df)
    print(f"      mean rarity score: {event_rarity_scores.mean():.4f}")

    # ── 8. sequences + transformer autoencoder ────────────────────────────────
    print(f"\n[8/9] Building sequences & training Transformer autoencoder...")

    if use_memmap:
        # ── large-dataset path ────────────────────────────────────────────────
        print(f"      Writing sequences to {SEQ_MEMMAP_PATH} ...")
        n_seq, seq_shape = build_sequences_memmap(
            df, feature_cols, SEQUENCE_LENGTH,
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
        n_seq_out        = n_seq

    else:
        # ── small-dataset path ────────────────────────────────────────────────
        X, y = build_sequences(df, feature_cols, SEQUENCE_LENGTH)
        print(f"      {X.shape[0]:,} sequences of length {SEQUENCE_LENGTH}")

        X_train = X[y == TRAIN_LABEL]
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
        n_seq_out        = len(X)

    # align sequence-level scores back to events
    event_recon_errors        = np.empty(n_events, dtype=np.float32)
    pad                       = n_events - n_seq_out
    event_recon_errors[:pad]  = seq_recon_errors[0]
    event_recon_errors[pad:]  = seq_recon_errors
    print(f"      mean recon error : {event_recon_errors.mean():.4f}")

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
