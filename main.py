import pandas as pd
import torch

from config import *
from feature_engineering import feature_engineering
from sequence_builder import build_sequences
from transformer_autoencoder import TransformerAutoencoder
from train import train_model
from evaluate import anomaly_scores

def main():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("Feature engineering...")

    df, feature_cols = feature_engineering(df)

    print("Building sequences...")

    X, y = build_sequences(df, feature_cols, SEQUENCE_LENGTH)

    print("Train size:", X.shape)

    # Train only benign

    X_train = X[y == TRAIN_LABEL]

    print("Training samples:", X_train.shape)

    feature_dim = X.shape[2]

    model = TransformerAutoencoder(feature_dim)

    model = train_model(
        model,
        X_train,
        EPOCHS,
        BATCH_SIZE,
        LEARNING_RATE
    )

    torch.save(model.state_dict(), MODEL_PATH)

    print("Scoring anomalies...")

    scores = anomaly_scores(model, X)

    df_results = pd.DataFrame({
        "score": scores,
        "label": y
    })

    print(df_results.groupby("label").mean())

    df_results.to_csv("anomaly_scores.csv", index=False)

if __name__ == "__main__":
    main()