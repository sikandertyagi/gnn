"""
transformer_autoencoder.py
──────────────────────────
Transformer-based sequence autoencoder.

Architecture
────────────
  Input  : (B, T, feature_dim)
  Encoder: linear projection → sinusoidal PE → TransformerEncoder
           → mean-pool → bottleneck (embed_dim → embed_dim//4)
  Decoder: expand bottleneck → TransformerDecoder (cross-attention on memory)
           → linear projection
  Output : (B, T, feature_dim)

Key design choices
──────────────────
  · Sinusoidal positional encoding (fixed, from "Attention Is All You Need"):
    preserves event ordering without learned parameters; stable on short seqs.
  · True TransformerDecoder with cross-attention: each decoded position
    attends to the full encoder memory, giving a richer reconstruction signal
    than a second encoder.
  · Compressed bottleneck (embed_dim → embed_dim//4): forces meaningful
    compression and prevents the model from learning an identity shortcut.
  · Dropout on the PE layer for regularisation.

Training loss: MSE(output, input) on benign sequences only.
"""

import math

import torch
import torch.nn as nn

from config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_DIM


# ── sinusoidal positional encoding ────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal PE as in Vaswani et al. (2017)."""

    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ── main model ─────────────────────────────────────────────────────────────────

class TransformerAutoencoder(nn.Module):

    def __init__(
        self,
        feature_dim: int,
        embed_dim:   int = EMBED_DIM,
        num_heads:   int = NUM_HEADS,
        num_layers:  int = NUM_LAYERS,
        ff_dim:      int = FF_DIM,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.feature_dim    = feature_dim
        self.embed_dim      = embed_dim
        bottleneck_dim      = embed_dim // 4

        # ── shared positional encoding ─────────────────────────────────────────
        self.pos_enc    = SinusoidalPositionalEncoding(embed_dim, dropout=dropout)

        # ── encoder ───────────────────────────────────────────────────────────
        self.input_proj = nn.Linear(feature_dim, embed_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, batch_first=True, dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # compress sequence mean-pool to a bottleneck vector
        self.bottleneck        = nn.Linear(embed_dim, bottleneck_dim)
        self.bottleneck_expand = nn.Linear(bottleneck_dim, embed_dim)

        # ── decoder ───────────────────────────────────────────────────────────
        dec_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=ff_dim, batch_first=True, dropout=dropout,
        )
        self.decoder_transformer = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        # Projects positional query tokens before cross-attention; forces the
        # decoder to learn a dedicated query space rather than attending directly
        # with raw sinusoidal embeddings, improving reconstruction specificity.
        self.query_proj  = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, feature_dim)

    # ── public API ────────────────────────────────────────────────────────────

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) → (B, bottleneck_dim)  sequence-level embedding."""
        h = self.pos_enc(self.input_proj(x))
        h = self.encoder(h)
        z = self.bottleneck(h.mean(dim=1))
        return z

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        """(B, bottleneck_dim) → (B, T, F)."""
        # expand bottleneck back to embed_dim; broadcast over time steps
        memory = self.bottleneck_expand(z).unsqueeze(1).expand(-1, seq_len, -1)
        # positional query tokens for the decoder, projected to query space
        tgt = self.pos_enc(
            torch.zeros(z.size(0), seq_len, self.embed_dim, device=z.device)
        )
        tgt = self.query_proj(tgt)
        h = self.decoder_transformer(tgt, memory)
        return self.output_proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z, seq_len=x.size(1))
