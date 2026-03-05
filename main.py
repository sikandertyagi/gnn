"""
main.py  —  Fleet-Scale Pipeline
──────────────────────────────────
Scalable end-to-end anomaly detection for 1000+ machine networks.

Key architectural differences from the original single-graph pipeline
─────────────────────────────────────────────────────────────────────
  ① Per-machine graphs (machine_graph_builder)
      · One small HeteroData per machine instead of one massive global graph
      · Edges deduplicated per machine (repeated parent→child pairs collapse)
      · Removes signal dilution: an anomalous process on 1 machine is not
        drowned out by 999 benign machines sharing the same process name

  ② Graph mini-batch training (PyG DataLoader over machine graphs)
      · GNN processes MACHINE_GNN_BATCH_SIZE machine graphs per step
      · Peak RAM ∝ batch_size × avg_nodes_per_machine  (constant at fleet scale)
      · No full-graph OOM regardless of how many machines exist

  ③ Per-machine anomaly scoring
      · GNN scores each machine's graph independently vs a fleet-wide centroid
      · Scores mapped back to events via (machine, process_image) key

  ④ Transformer + Rarity (unchanged, already scale-safe)
      · Sequences written to disk with np.memmap above LARGE_DATASET_THRESHOLD
      · Rarity engine uses vectorised pandas merge — O(N) at any fleet size

Full pipeline
─────────────
  Sysmon CSV
    │
    ▼  [1] feature_engineering.py    vectorised; missing columns filled safely
  Enriched DataFrame + 20 feature columns
    │
    ▼  [2] normaliser.py             StandardScaler fitted on benign rows only
  Normalised DataFrame
    │
    ├─► [3] machine_graph_builder.py → list[HeteroData]  (one per machine)
    │         │
    │         ▼  [4] gnn_encoder.train_gnn_fleet()       graph mini-batch training
    │     HeteroGNNEncoder  trained on benign machine graphs
    │         │
    │         ▼  [5] gnn_encoder.score_machine_graphs()
    │     graph_scores[machine][process] → map to events
    │
    ├─► [6] rarity_engine.py         vectorised Bayesian frequency scoring
    │
    ├─► [7] sequence_builder.py      sliding-window per host
    │         │                      large → disk memmap; small → in-memory
    │         ▼  [8] train.py        Transformer autoencoder (benign-only)
    │     reconstruction_errors per sequence → aligned to events
    │
    ▼  [9] anomaly_engine.py
  Composite scores  (0.5×recon + 0.3×graph + 0.2×rarity)
    │
    ▼  [10] alert_aggregator.py
  Attack-chain alerts CSV
    │
    ▼  [11] metrics.py
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
    MACHINE_GNN_BATCH_SIZE, MIN_EVENTS_PER_MACHINE,
)
from feature_engineering import feature_engineering
from normaliser import fit_scaler, apply_scaler

# ── fleet-scale GNN components ─────────────────────────────────────────────
from machine_graph_builder import build_machine_graphs, get_node_feature_dims
from gnn_encoder import (
    HeteroGNNEncoder,
    train_gnn_fleet,
    compute_fleet_centroids,
    score_machine_graphs,
    map_graph_scores_to_events,
)

# ── remaining components (unchanged) ──────────────────────────────────────
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def main():

    _set_seeds(RANDOM_SEED)

    # ── 1. load data ──────────────────────────────────────────────────────────
    print("\n[1/11] Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if "Label" not in df.columns:
        print("      WARNING: 'Label' column not found — defaulting to 0 (benign).")
        df["Label"] = 0

    n_events   = len(df)
    use_memmap = n_events > LARGE_DATASET_THRESHOLD
    n_machines = df["Computer"].nunique() if "Computer" in df.columns else "?"
    print(f"      {n_events:,} events across {n_machines} machines — "
          f"{'large (memmap)' if use_memmap else 'small (in-memory)'} mode")

    # ── 2. feature engineering ────────────────────────────────────────────────
    print("\n[2/11] Feature engineering...")
    df, feature_cols = feature_engineering(df)
    print(f"      {len(feature_cols)} feature columns")

    # ── 3. normalise (benign-fit StandardScaler) ──────────────────────────────
    print("\n[3/11] Fitting StandardScaler on benign rows...")
    scaler = fit_scaler(df, feature_cols)
    df     = apply_scaler(df, feature_cols, scaler)
    print(f"      Scaler saved → scaler.pkl")

    # ── 4. build per-machine graphs ───────────────────────────────────────────
    print(f"\n[4/11] Building per-machine graphs  "
          f"(min_events={MIN_EVENTS_PER_MACHINE})...")
    all_graphs    = build_machine_graphs(df, min_events=MIN_EVENTS_PER_MACHINE)
    benign_graphs = [g for g in all_graphs if g.machine_label == 0]

    n_all    = len(all_graphs)
    n_benign = len(benign_graphs)
    print(f"      {n_all} machine graphs built  "
          f"({n_benign} fully benign for GNN training)")

    if n_benign == 0:
        raise RuntimeError(
            "No fully-benign machine graphs found.  "
            "Check that TRAIN_LABEL=0 rows exist and that machines have "
            f">={MIN_EVENTS_PER_MACHINE} events."
        )

    # ── 5. train GNN encoder (fleet mini-batch) ───────────────────────────────
    print(f"\n[5/11] Training GNN encoder  "
          f"(batch={MACHINE_GNN_BATCH_SIZE} machines/step)...")
    feat_dims = get_node_feature_dims(all_graphs)
    gnn_model = HeteroGNNEncoder(feat_dims)
    gnn_model = train_gnn_fleet(
        gnn_model, benign_graphs,
        batch_size=MACHINE_GNN_BATCH_SIZE,
        epochs=GNN_EPOCHS, lr=GNN_LR,
    )
    torch.save(gnn_model.state_dict(), GNN_MODEL_PATH)
    print(f"      Saved → {GNN_MODEL_PATH}")

    # ── 6. fleet anomaly scores (GNN) ─────────────────────────────────────────
    print("\n[6/11] Computing per-machine graph anomaly scores...")
    fleet_centroids    = compute_fleet_centroids(gnn_model, benign_graphs,
                                                  batch_size=MACHINE_GNN_BATCH_SIZE)
    machine_scores     = score_machine_graphs(gnn_model, all_graphs, fleet_centroids)
    event_graph_scores = map_graph_scores_to_events(df, machine_scores)
    print(f"      mean graph score : {event_graph_scores.mean():.4f}")

    # ── 7. rarity scoring ─────────────────────────────────────────────────────
    print("\n[7/11] Fitting rarity engine & scoring...  [vectorised]")
    rarity              = RarityEngine().fit(df)
    event_rarity_scores = rarity.score_dataframe(df)
    print(f"      mean rarity score: {event_rarity_scores.mean():.4f}")

    # ── 8. sequences + transformer autoencoder ────────────────────────────────
    print(f"\n[8/11] Building sequences & training Transformer autoencoder...")

    if use_memmap:
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
        print(f"      Saved → {MODEL_PATH}")

        X_mm             = load_seq_memmap(SEQ_MEMMAP_PATH, seq_shape)
        seq_recon_errors = anomaly_scores(ta_model, X_mm, INFER_BATCH_SIZE)
        n_seq_out        = n_seq

    else:
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
        print(f"      Saved → {MODEL_PATH}")

        seq_recon_errors = anomaly_scores(ta_model, X, INFER_BATCH_SIZE)
        n_seq_out        = len(X)

    # align sequence-level scores back to events
    event_recon_errors       = np.full(n_events, np.nan, dtype=np.float32)
    pad                      = n_events - n_seq_out
    event_recon_errors[pad:] = seq_recon_errors
    print(f"      mean recon error : {np.nanmean(event_recon_errors):.4f}  "
          f"({pad} leading events set to NaN)")

    # ── 9. composite scores ────────────────────────────────────────────────────
    print("\n[9/11] Computing composite anomaly scores...")
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
    print(f"      Scores saved → {SCORES_PATH}")

    print("\n  Score summary by label:")
    print(df_scores.groupby("label")[
        ["score", "recon_error", "graph_score", "rarity_score"]
    ].mean().to_string())

    # ── 10. alert aggregation ──────────────────────────────────────────────────
    print("\n[10/11] Aggregating alerts into attack chains...")
    alerts = aggregate_alerts(df, composite)
    alerts.to_csv(ALERTS_PATH, index=False)
    print(f"      Alerts saved → {ALERTS_PATH}")

    if not alerts.empty:
        print("\n  Top-5 alert chains:")
        print(alerts.nlargest(5, "max_score")[
            ["chain_id", "num_events", "max_score", "processes", "dest_ips"]
        ].to_string(index=False))

    # ── 11. evaluation metrics ────────────────────────────────────────────────
    print("\n[11/11] Evaluating detection performance...")
    has_labels = df_scores["label"].nunique() > 1
    if not has_labels:
        print("      No anomalous labels found — skipping evaluation metrics.")
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
