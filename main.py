import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH, SEQUENCE_LENGTH, TRAIN_LABEL, BATCH_SIZE,
    EPOCHS, LEARNING_RATE, VAL_RATIO, MODEL_PATH, SCALER_PATH,
)
from feature_engineering import feature_engineering, N_CATEGORICAL_FEATURES
from sequence_builder import build_sequences
from transformer_autoencoder import TransformerAutoencoder
from train import train_model
from evaluate import anomaly_scores, evaluate_metrics


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # ------------------------------------------------------------------
    # Keep only high-signal event types:
    #   EventID 1 — process creation (strongest attack signal)
    #   EventID 3 — network connection (C2 / lateral movement)
    # ------------------------------------------------------------------
    df["EventID"] = pd.to_numeric(df["EventID"], errors="coerce")
    before = len(df)
    df = df[df["EventID"].isin([1, 3])].reset_index(drop=True)
    print(f"EventID filter  : {before} → {len(df)} rows (kept 1 & 3)")

    print("Feature engineering...")
    df, feature_cols = feature_engineering(df)

    print("Building sequences...")
    X, y = build_sequences(df, feature_cols, SEQUENCE_LENGTH)
    print(f"Total sequences : {X.shape}")

    # ------------------------------------------------------------------
    # Split: train only on benign (label 0)
    # ------------------------------------------------------------------
    X_train = X[y == TRAIN_LABEL]
    print(f"Benign (train)  : {X_train.shape}")
    print(f"Anomalous       : {X[y != TRAIN_LABEL].shape}")

    # ------------------------------------------------------------------
    # Feature normalisation — fit ONLY on benign training data.
    # The first N_CATEGORICAL_FEATURES columns are CRC32-hashed to [0,1]
    # already; z-scoring them is semantically meaningless, so the scaler
    # is applied only to the remaining numeric columns.
    # ------------------------------------------------------------------
    print("Fitting StandardScaler on benign numeric features...")
    n_samples, seq_len, n_features = X_train.shape

    numeric_start = N_CATEGORICAL_FEATURES   # skip the CRC32-hashed columns
    scaler = StandardScaler()

    # Fit on benign numeric features only (flattened over time steps)
    scaler.fit(
        X_train[:, :, numeric_start:].reshape(-1, n_features - numeric_start)
    )

    def _scale(X_arr):
        """Apply scaler to numeric columns; leave categorical columns as-is."""
        out = X_arr.copy()
        flat = X_arr[:, :, numeric_start:].reshape(-1, n_features - numeric_start)
        out[:, :, numeric_start:] = scaler.transform(flat).reshape(
            X_arr.shape[0], seq_len, n_features - numeric_start
        )
        return out

    X_train_scaled = _scale(X_train)
    X_scaled       = _scale(X)

    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved → {SCALER_PATH}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    model = TransformerAutoencoder(n_features)

    model = train_model(
        model,
        X_train_scaled,
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE,
        VAL_RATIO,
    )

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved  → {MODEL_PATH}")

    # ------------------------------------------------------------------
    # Score all sequences (normal + anomalous)
    # ------------------------------------------------------------------
    print("Scoring all sequences...")
    scores = anomaly_scores(model, X_scaled)

    # Mean score per label (quick sanity check)
    df_results = pd.DataFrame({"score": scores, "label": y})
    print("\nMean reconstruction error by label:")
    print(df_results.groupby("label")["score"].mean().to_string())

    # ------------------------------------------------------------------
    # Full evaluation metrics  (label 0 = normal, 1 & 2 = anomalous)
    # ------------------------------------------------------------------
    metrics = evaluate_metrics(scores, y)

    df_results.to_csv("anomaly_scores.csv", index=False)
    print("Results saved → anomaly_scores.csv")


if __name__ == "__main__":
    main()
