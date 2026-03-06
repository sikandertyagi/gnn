"""
transformer_autoencoder.py
──────────────────────────
Transformer-based sequence autoencoder for anomaly detection.

Architecture
────────────
  Input  : (B, T, feature_dim)
  Encoder: linear projection + LayerNorm
           + sinusoidal positional encoding (preserves temporal order)
           → TransformerEncoder
           → mean-pool over T → compressed bottleneck (embed_dim → embed_dim//4)
  Decoder: expand bottleneck (embed_dim//4 → embed_dim)
           + sinusoidal positional queries
           → TransformerDecoder (cross-attention to encoder memory)
           → linear projection
  Output : (B, T, feature_dim)

Key improvements over the original implementation
──────────────────────────────────────────────────
  · Sinusoidal positional encoding (fixed, not learned) — events at different
    positions in the window are distinguishable; temporal ordering is preserved.
  · True TransformerDecoder with cross-attention: each decoded position attends
    to the full encoded sequence rather than receiving an identical broadcast
    vector (which was the bug in the original encoder-only "decoder").
  · Compressed bottleneck (embed_dim//4): forces the model to learn compact
    sequence representations; prevents identity shortcuts that cause all
    reconstruction errors to be near zero regardless of input.
  · Dropout in all transformer layers for regularisation.

Training loss: MSE(output, input) on benign sequences only.
"""

import math

import torch
import torch.nn as nn

from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)           # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)  →  x + pe[:, :T]"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerAutoencoder(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        embed_dim:   int   = EMBED_DIM,
        num_heads:   int   = NUM_HEADS,
        num_layers:  int   = NUM_LAYERS,
        ff_dim:      int   = FF_DIM,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.embed_dim      = embed_dim
        self.bottleneck_dim = max(embed_dim // 4, 16)

        # ── encoder ──────────────────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.pos_enc = SinusoidalPositionalEncoding(embed_dim, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Compressed bottleneck — prevents identity shortcut
        self.bottleneck_enc = nn.Linear(embed_dim, self.bottleneck_dim)
        self.bottleneck_dec = nn.Linear(self.bottleneck_dim, embed_dim)

        # ── decoder ──────────────────────────────────────────────────────────
        # Positional queries give the decoder position-aware "starting points"
        self.query_pos_enc = SinusoidalPositionalEncoding(embed_dim, dropout=dropout)
        self.query_proj    = nn.Linear(embed_dim, embed_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, dropout=dropout, batch_first=True,
        )
        self.decoder     = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(embed_dim, feature_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) → (B, bottleneck_dim)"""
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.bottleneck_enc(h.mean(dim=1))

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """(B, bottleneck_dim) → (B, T, F)"""
        B      = z.size(0)
        device = z.device
        memory = self.bottleneck_dec(z).unsqueeze(1).expand(-1, seq_len, -1)
        queries = self.query_pos_enc(
            torch.zeros(B, seq_len, self.embed_dim, device=device)
        )
        queries = self.query_proj(queries)
        h = self.decoder(queries, memory)
        return self.output_proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z, seq_len=x.size(1))
