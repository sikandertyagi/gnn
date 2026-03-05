import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler

from config import *
from feature_engineering import feature_engineering
from sequence_builder import build_sequences
from transformer_autoencoder import TransformerAutoencoder
from train import train_model
from evaluate import anomaly_scores, evaluate_metrics


def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

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
    # Feature normalisation — fit ONLY on benign training data
    # ------------------------------------------------------------------
    print("Fitting StandardScaler on benign data...")
    n_samples, seq_len, n_features = X_train.shape

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))

    X_train_scaled = scaler.transform(
        X_train.reshape(-1, n_features)
    ).reshape(X_train.shape)

    X_scaled = scaler.transform(
        X.reshape(-1, n_features)
    ).reshape(X.shape)

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
