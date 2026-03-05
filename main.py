"""
main.py
───────
Full end-to-end pipeline:

  Sysmon CSV
    │
    ▼  feature_engineering.py
  Tabular feature matrix  +  enriched DataFrame
    │
    ├─► graph_builder.py          → HeteroData graph
    │       │
    │       ▼  gnn_encoder.py
    │   HeteroGNNEncoder          trained on benign graph
    │       │
    │       ▼
    │   graph_anomaly_scores      (N_events,)
    │
    ├─► rarity_engine.py          fitted on benign events
    │       │
    │       ▼
    │   rarity_scores             (N_events,)
    │
    ├─► sequence_builder.py       sliding-window sequences
    │       │
    │       ▼  transformer_autoencoder.py + train.py
    │   TransformerAutoencoder    trained on benign sequences
    │       │
    │       ▼  evaluate.py
    │   reconstruction_errors     (N_sequences,)
    │
    ▼  anomaly_engine.py
  Composite anomaly scores        (N_events,)
    │
    ▼  alert_aggregator.py
  Attack-chain alerts CSV
"""

import numpy as np
import pandas as pd
import torch

from config import (
    DATA_PATH, SEQUENCE_LENGTH, TRAIN_LABEL,
    EPOCHS, BATCH_SIZE, LEARNING_RATE,
    GNN_EPOCHS, GNN_LR,
    MODEL_PATH, GNN_MODEL_PATH,
    SCORES_PATH, ALERTS_PATH,
)
from feature_engineering import feature_engineering
from graph_builder import build_event_graph, node_feature_dims
from gnn_encoder import HeteroGNNEncoder, train_gnn, compute_benign_centroids, graph_anomaly_scores
from rarity_engine import RarityEngine
from sequence_builder import build_sequences
from transformer_autoencoder import TransformerAutoencoder
from train import train_model
from evaluate import anomaly_scores
from anomaly_engine import compute_anomaly_scores
from alert_aggregator import aggregate_alerts
from metrics import evaluate


def main():

    # ── 1. load data ──────────────────────────────────────────────────────────
    print("\n[1/8] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"      {len(df):,} events loaded")

    # ── 2. feature engineering ────────────────────────────────────────────────
    print("\n[2/8] Feature engineering...")
    df, feature_cols = feature_engineering(df)
    print(f"      {len(feature_cols)} feature columns")

    # ── 3. build heterogeneous event graph ────────────────────────────────────
    print("\n[3/8] Building event graph...")
    df_benign     = df[df["Label"] == TRAIN_LABEL]
    full_graph, encoders = build_event_graph(df)
    benign_graph, _      = build_event_graph(df_benign)

    n_proc = full_graph["process"].x.shape[0]
    n_ip   = full_graph["ip"].x.shape[0]
    print(f"      {n_proc} process nodes, {n_ip} IP nodes")
    pp_edges = full_graph["process", "parent_of", "process"].edge_index.shape[1]
    pi_edges = full_graph["process", "connects_to", "ip"].edge_index.shape[1]
    print(f"      {pp_edges} parent→child edges, {pi_edges} process→IP edges")

    # ── 4. train GNN encoder (benign graph only) ──────────────────────────────
    print("\n[4/8] Training GNN encoder on benign graph...")
    feat_dims  = node_feature_dims(benign_graph)
    gnn_model  = HeteroGNNEncoder(feat_dims)
    gnn_model  = train_gnn(gnn_model, benign_graph, epochs=GNN_EPOCHS, lr=GNN_LR)
    torch.save(gnn_model.state_dict(), GNN_MODEL_PATH)
    print(f"      Saved → {GNN_MODEL_PATH}")

    benign_centroids = compute_benign_centroids(gnn_model, benign_graph)

    # ── 5. compute per-event graph anomaly scores ─────────────────────────────
    print("\n[5/8] Computing graph anomaly scores...")
    proc_graph_scores = graph_anomaly_scores(
        gnn_model, full_graph, benign_centroids,
        encoders["process"]
    )  # shape: (N_process_nodes,)

    # map each event row → its process node score
    proc_enc      = encoders["process"]
    event_images  = df["Image"].fillna("unknown").values
    proc_ids      = proc_enc.transform(event_images)
    event_graph_scores = proc_graph_scores[proc_ids].numpy()
    print(f"      mean graph score: {event_graph_scores.mean():.4f}")

    # ── 6. fit rarity engine & score all events ───────────────────────────────
    print("\n[6/8] Fitting rarity engine & scoring...")
    rarity = RarityEngine().fit(df)
    event_rarity_scores = rarity.score_dataframe(df)
    print(f"      mean rarity score: {event_rarity_scores.mean():.4f}")

    # ── 7. train transformer autoencoder (benign sequences) ───────────────────
    print("\n[7/8] Building sequences & training Transformer autoencoder...")
    X, y = build_sequences(df, feature_cols, SEQUENCE_LENGTH)
    print(f"      {X.shape[0]:,} sequences of length {SEQUENCE_LENGTH}")

    X_train = X[y == TRAIN_LABEL]
    print(f"      {X_train.shape[0]:,} benign training sequences")

    feature_dim = X.shape[2]
    ta_model    = TransformerAutoencoder(feature_dim)
    ta_model    = train_model(ta_model, X_train, EPOCHS, BATCH_SIZE, LEARNING_RATE)
    torch.save(ta_model.state_dict(), MODEL_PATH)
    print(f"      Saved → {MODEL_PATH}")

    # reconstruction errors per sequence
    seq_recon_errors = anomaly_scores(ta_model, X)  # (N_seq,)

    # align sequences back to events:
    # build_sequences produces one sequence per event (offset by seq_len-1)
    # we pad the first (seq_len-1) events with the first sequence's score
    n_events = len(df)
    event_recon_errors = np.empty(n_events)
    pad = n_events - len(seq_recon_errors)
    event_recon_errors[:pad] = seq_recon_errors[0]
    event_recon_errors[pad:] = seq_recon_errors

    print(f"      mean recon error: {event_recon_errors.mean():.4f}")

    # ── 8. composite anomaly scoring ──────────────────────────────────────────
    print("\n[8/8] Computing composite anomaly scores & aggregating alerts...")
    composite = compute_anomaly_scores(
        event_recon_errors,
        event_graph_scores,
        event_rarity_scores,
    )

    # save per-event scores
    df_scores = pd.DataFrame({
        "score":        composite,
        "recon_error":  event_recon_errors,
        "graph_score":  event_graph_scores,
        "rarity_score": event_rarity_scores,
        "label":        df["Label"].values,
    })
    df_scores.to_csv(SCORES_PATH, index=False)
    print(f"      Scores saved → {SCORES_PATH}")

    # print per-label summary
    print("\n  Score summary by label:")
    print(df_scores.groupby("label")[["score", "recon_error", "graph_score", "rarity_score"]].mean().to_string())

    # aggregate alerts
    alerts = aggregate_alerts(df, composite)
    alerts.to_csv(ALERTS_PATH, index=False)
    print(f"\n  Alerts saved → {ALERTS_PATH}")

    if not alerts.empty:
        print("\n  Top-5 alert chains:")
        print(alerts.nlargest(5, "max_score")[
            ["chain_id", "num_events", "max_score", "processes", "dest_ips"]
        ].to_string(index=False))

    # ── 9. evaluation & metrics ───────────────────────────────────────────────
    metrics = evaluate(
        df_scores  = df_scores,
        df_alerts  = alerts,
        df_events  = df,
        report_path= "threshold_sweep.csv",
    )

    print(f"  ROC-AUC  : {metrics.get('roc_auc', float('nan')):.4f}")
    print(f"  PR-AUC   : {metrics.get('pr_auc',  float('nan')):.4f}")
    best = metrics.get("best_f1_threshold", {})
    print(f"  Best F1  : {best.get('f1', 0):.4f}  "
          f"@ threshold={best.get('threshold', '?')}"
          f"  P={best.get('precision',0):.4f}  R={best.get('recall',0):.4f}")


if __name__ == "__main__":
    main()
