import numpy as np

def build_sequences(df, feature_cols, seq_len):

    sequences = []
    labels = []

    df = df.sort_values("SystemTime")

    hosts = df["Computer"].unique()

    for host in hosts:

        host_df = df[df["Computer"] == host]

        values = host_df[feature_cols].values
        lab = host_df["Label"].values

        for i in range(len(values) - seq_len + 1):

            seq = values[i:i+seq_len]

            sequences.append(seq)

            labels.append(lab[i+seq_len-1])

    return np.array(sequences), np.array(labels)