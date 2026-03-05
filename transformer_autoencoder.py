import torch
import torch.nn as nn

class TransformerAutoencoder(nn.Module):

    def __init__(self, feature_dim, embed_dim=96, num_heads=4, num_layers=2, ff_dim=256):

        super().__init__()

        self.input_proj = nn.Linear(feature_dim, embed_dim)

        # Positional encoding so the model can distinguish event order in a sequence
        self.pos_embedding = nn.Embedding(512, embed_dim)

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

        # Add positional embeddings so temporal order is captured
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)

        z = self.encoder(x)

        out = self.decoder(z)

        return out