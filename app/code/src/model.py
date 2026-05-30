"""Attention-based stock ranking model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import D_MODEL, N_HEADS, N_GRU_LAYERS, DROPOUT, N_TRANSFORMER_LAYERS, D_FF


class TimeEncoder(nn.Module):
    """Bidirectional GRU: (B*N, T, F) -> (B*N, d_model)."""

    def __init__(self, n_features: int):
        super().__init__()
        self.input_proj = nn.Linear(n_features, D_MODEL)
        self.gru = nn.GRU(D_MODEL, D_MODEL, num_layers=N_GRU_LAYERS,
                          batch_first=True, bidirectional=True,
                          dropout=DROPOUT if N_GRU_LAYERS > 1 else 0.0)
        self.output_proj = nn.Linear(D_MODEL * 2, D_MODEL)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        _, h_n = self.gru(x)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.dropout(self.output_proj(h))


class CrossSectionalTransformer(nn.Module):
    """One transformer block over the stock axis."""

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True)
        self.norm1 = nn.LayerNorm(D_MODEL)
        self.norm2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(
            nn.Linear(D_MODEL, D_FF), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(D_FF, D_MODEL), nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        kp = (mask == 0) if mask is not None else None
        a_out, a_w = self.attn(x, x, x, key_padding_mask=kp)
        x = self.norm1(x + a_out)
        x = self.norm2(x + self.ffn(x))
        return x, a_w


class PortfolioPredictor(nn.Module):
    """TimeEncoder -> CrossSectionalTransformer -> ScoreHead."""

    def __init__(self, n_features: int):
        super().__init__()
        self.time_encoder = TimeEncoder(n_features)
        self.transformers = nn.ModuleList([CrossSectionalTransformer()
                                           for _ in range(N_TRANSFORMER_LAYERS)])
        self.score_head = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL // 2), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL // 2, 1),
        )

    def encode_stocks(self, x, mask=None):
        B, N, T, F = x.shape
        e = self.time_encoder(x.reshape(B * N, T, F)).reshape(B, N, -1)
        for t in self.transformers:
            e, _ = t(e, mask)
        return e

    def forward(self, x, mask=None):
        B, N, T, F = x.shape
        e = self.time_encoder(x.reshape(B * N, T, F)).reshape(B, N, -1)
        for t in self.transformers:
            e, _ = t(e, mask)
        scores = self.score_head(e).squeeze(-1)
        if mask is not None:
            scores = scores * mask + (1 - mask) * (-1e9)
        return scores, None


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def listnet_loss(scores, targets, valid_mask=None, temperature=1.0):
    """KL divergence between softmax distributions of scores and targets."""
    if valid_mask is not None:
        scores = scores.clone()
        scores[valid_mask < 0.5] = -float("inf")
    p = F.softmax(targets / temperature, dim=-1)
    q = F.log_softmax(scores / temperature, dim=-1)
    return (p * (torch.log(p + 1e-9) - q)).sum(dim=-1).mean()


def lambdarank_loss(scores, targets, valid_mask=None, sigma=1.0):
    """LambdaRank: pairwise logistic loss weighted by |delta NDCG|."""
    B, N = scores.shape
    mt = targets.clone()
    if valid_mask is not None:
        mt[valid_mask < 0.5] = -float("inf")
    _, rank = torch.sort(mt, dim=1, descending=True)
    pos = torch.zeros_like(targets).scatter(
        1, rank, torch.arange(N, device=scores.device).float().unsqueeze(0).expand(B, -1))
    gd = (2.0 ** targets - 1.0) / torch.log2(pos + 2.0)
    dndcg = torch.abs(gd.unsqueeze(2) - gd.unsqueeze(1))
    sd = scores.unsqueeze(2) - scores.unsqueeze(1)
    td = targets.unsqueeze(2) - targets.unsqueeze(1)
    pl = dndcg * F.softplus(-sigma * sd)
    pv = (td.abs() > 1e-6).float()
    if valid_mask is not None:
        pv *= valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
    return (pl * pv).sum() / (pv.sum() + 1e-8)


def pairwise_ranking_loss(scores, targets, valid_mask=None, margin=0.05):
    """Hinge pairwise ranking loss."""
    sd = scores.unsqueeze(2) - scores.unsqueeze(1)
    td = targets.unsqueeze(2) - targets.unsqueeze(1)
    sign = torch.sign(td)
    loss = F.relu(margin - sign * sd)
    pv = (sign.abs() > 1e-6).float()
    if valid_mask is not None:
        pv *= valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
    return (loss * pv).sum() / (pv.sum() + 1e-8)
