"""
transformer_autoencoder.py
──────────────────────────
Transformer-based sequence autoencoder.

Architecture
────────────
  Input  : (B, T, feature_dim)
  Encoder: linear projection → TransformerEncoder → mean-pool → bottleneck
  Decoder: expand bottleneck → TransformerEncoder → linear projection
  Output : (B, T, feature_dim)

Training loss: MSE(output, input) on benign sequences only.
"""

import torch
import torch.nn as nn

from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM


class TransformerAutoencoder(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        embed_dim:   int = EMBED_DIM,
        num_heads:   int = NUM_HEADS,
        num_layers:  int = NUM_LAYERS,
        ff_dim:      int = FF_DIM,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.embed_dim   = embed_dim

        # ── encoder ───────────────────────────────────────────────────────────
        self.input_proj = nn.Linear(feature_dim, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # compress sequence to a single bottleneck vector
        self.bottleneck = nn.Linear(embed_dim, embed_dim)

        # ── decoder ───────────────────────────────────────────────────────────
        self.expand_proj = nn.Linear(embed_dim, embed_dim)

        dec_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, batch_first=True,
        )
        self.decoder_transformer = nn.TransformerEncoder(dec_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, feature_dim)

    # ── public API ────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) → (B, embed_dim)  sequence-level embedding."""
        h = self.input_proj(x)
        h = self.encoder(h)
        z = self.bottleneck(h.mean(dim=1))
        return z

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """(B, embed_dim) → (B, T, F)."""
        h = self.expand_proj(z).unsqueeze(1).expand(-1, seq_len, -1)
        h = self.decoder_transformer(h)
        return self.output_proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z, seq_len=x.size(1))
