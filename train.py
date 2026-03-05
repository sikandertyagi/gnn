import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_model(model, X_train, epochs, batch_size, lr):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    dataset = TensorDataset(torch.tensor(X_train).float())

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):

        total_loss = 0

        for batch in loader:

            x = batch[0].to(device)

            optimizer.zero_grad()

            recon = model(x)

            loss = criterion(recon, x)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss {total_loss/len(loader):.4f}")

    return model