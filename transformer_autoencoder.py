import torch
import torch.nn as nn

class TransformerAutoencoder(nn.Module):

    def __init__(self, feature_dim, embed_dim=96, num_heads=4, num_layers=2, ff_dim=256):

        super().__init__()

        self.input_proj = nn.Linear(feature_dim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.decoder = nn.Linear(embed_dim, feature_dim)

    def forward(self, x):

        x = self.input_proj(x)

        z = self.encoder(x)

        out = self.decoder(z)

        return out