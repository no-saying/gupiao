"""Attention-based stock ranking model.

Architecture:
  1. TimeEncoder (bidirectional GRU) — encodes each stock's time-series
  2. CrossSectionalTransformer — stocks attend to each other
  3. ScoreHead — MLP outputs one scalar per stock

Loss functions:
  - pairwise_ranking_loss  (hinge)
  - lambdarank_loss         (NDCG-weighted pairwise)
  - listnet_loss            (top-K distribution matching)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    D_MODEL, N_HEADS, N_GRU_LAYERS, DROPOUT,
    N_TRANSFORMER_LAYERS, D_FF, DEVICE, RANKING_MARGIN,
)


# ---------------------------------------------------------------------------
# Time Encoder
# ---------------------------------------------------------------------------

class TimeEncoder(nn.Module):
    """Bidirectional GRU that maps (T, F) -> d_model per stock."""

    def __init__(
        self,
        n_features: int,
        d_model: int = D_MODEL,
        n_layers: int = N_GRU_LAYERS,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.gru = nn.GRU(
            d_model, d_model, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*N, T, F) -> (B*N, d_model)."""
        x = self.input_proj(x)
        _, h_n = self.gru(x)
        h_fwd = h_n[-2, :, :]
        h_bwd = h_n[-1, :, :]
        h = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.dropout(self.output_proj(h))


# ---------------------------------------------------------------------------
# Cross-Sectional Transformer Block
# ---------------------------------------------------------------------------

class CrossSectionalTransformer(nn.Module):
    """One transformer encoder block operating over the stock axis."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        d_ff: int = D_FF,
        dropout: float = DROPOUT,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N, d_model) -> (B, N, d_model), attn_weights: (B, N, N)."""
        key_padding = (mask == 0) if mask is not None else None
        attn_out, attn_weights = self.self_attn(
            x, x, x, key_padding_mask=key_padding,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class PortfolioPredictor(nn.Module):
    """TimeEncoder -> CrossSectionalTransformer -> ScoreHead."""

    def __init__(
        self,
        n_features: int,
        n_transformer_layers: int = N_TRANSFORMER_LAYERS,
    ) -> None:
        super().__init__()
        self.time_encoder = TimeEncoder(n_features)
        self.transformers = nn.ModuleList([
            CrossSectionalTransformer() for _ in range(n_transformer_layers)
        ])
        self.score_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, 1),
        )

    def encode_stocks(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get transformer embeddings without scores. x: (B,N,T,F) -> (B,N,d_model)."""
        B, N, T, F = x.shape
        emb = self.time_encoder(x.reshape(B * N, T, F))
        emb = emb.reshape(B, N, -1)
        for transformer in self.transformers:
            emb, _ = transformer(emb, mask)
        return emb

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """x: (B,N,T,F) -> scores: (B,N), attn_weights: (B,N,N)."""
        B, N, T, F = x.shape
        emb = self.time_encoder(x.reshape(B * N, T, F))
        emb = emb.reshape(B, N, -1)

        attn_weights = None
        for transformer in self.transformers:
            emb, attn_weights = transformer(emb, mask)

        scores = self.score_head(emb).squeeze(-1)
        if mask is not None:
            scores = scores * mask + (1 - mask) * (-1e9)
        return scores, attn_weights


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def pairwise_ranking_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    margin: float = RANKING_MARGIN,
) -> torch.Tensor:
    """Hinge ranking loss: for r_i > r_j, want s_i > s_j + margin."""
    score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)        # (B, N, N)
    target_diff = targets.unsqueeze(2) - targets.unsqueeze(1)

    sign = torch.sign(target_diff)
    loss_ij = F.relu(margin - sign * score_diff)
    pair_valid = (sign.abs() > 1e-6).float()
    if valid_mask is not None:
        pair_valid *= valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)

    n = pair_valid.sum() + 1e-8
    return (loss_ij * pair_valid).sum() / n


def _ndcg_gain(returns: torch.Tensor) -> torch.Tensor:
    return 2.0 ** returns - 1.0


def lambdarank_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    """LambdaRank: pairwise logistic loss weighted by |delta NDCG|."""
    B, N = scores.shape
    device = scores.device

    masked_t = targets.clone()
    if valid_mask is not None:
        masked_t[valid_mask < 0.5] = -float("inf")

    _, true_rank = torch.sort(masked_t, dim=1, descending=True)
    positions = torch.zeros_like(targets).scatter(
        1, true_rank,
        torch.arange(N, device=device).float().unsqueeze(0).expand(B, -1),
    )

    gd = _ndcg_gain(targets) / torch.log2(positions + 2.0)
    delta_ndcg = torch.abs(gd.unsqueeze(2) - gd.unsqueeze(1))
    score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    target_diff = targets.unsqueeze(2) - targets.unsqueeze(1)

    pair_loss = delta_ndcg * F.softplus(-sigma * score_diff)
    pair_valid = (target_diff.abs() > 1e-6).float()
    if valid_mask is not None:
        pair_valid *= valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)

    n = pair_valid.sum() + 1e-8
    return (pair_loss * pair_valid).sum() / n


def listnet_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """ListNet: KL divergence between predicted top-K distribution and target distribution.

    Converts both scores and targets to probability distributions over the top-K
    stocks via softmax, then minimizes KL(target_dist || pred_dist).
    """
    if valid_mask is not None:
        scores = scores.clone()
        scores[valid_mask < 0.5] = -float("inf")

    pred_dist = F.softmax(scores / temperature, dim=-1)
    target_dist = F.softmax(targets / temperature, dim=-1)

    # KL(target || pred) = sum(target * log(target / pred))
    kl = target_dist * (torch.log(target_dist + 1e-9) - F.log_softmax(scores / temperature, dim=-1))
    return kl.sum(dim=-1).mean()
