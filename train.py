import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def train_model(model, X_train, epochs, batch_size, lr, val_ratio=0.2):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Train / validation split (both from benign data)
    n = len(X_train)
    n_val = int(n * val_ratio)
    n_train = n - n_val

    idx = np.random.permutation(n)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_tr  = X_train[train_idx]
    X_val = X_train[val_idx]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_tr).float()),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val).float()),
        batch_size=batch_size, shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):

        # --- training ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon = model(x)
                val_loss += criterion(recon, x).item()

        print(
            f"Epoch {epoch+1}/{epochs}  "
            f"Train Loss: {train_loss/len(train_loader):.4f}  "
            f"Val Loss: {val_loss/len(val_loader):.4f}"
        )

    return model
