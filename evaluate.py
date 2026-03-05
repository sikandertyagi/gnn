import torch
import numpy as np

def anomaly_scores(model, X):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    scores = []

    with torch.no_grad():

        for seq in X:

            x = torch.tensor(seq).float().unsqueeze(0).to(device)

            recon = model(x)

            loss = torch.mean((x - recon) ** 2).item()

            scores.append(loss)

    return np.array(scores)