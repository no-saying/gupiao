"""
core/nn_model.py — Mamba 时序编码 + GAT 截面图网络 + MAE 预训练

架构:
  ┌─ Mamba 时序编码（60天序列 → embedding per stock）
  └─ GAT 截面编码（300只股票当天关系图 → embedding per stock）
        ↓ Concat
  ┌─ 预训练头: MAE 重建被遮掩的特征
  └─ 微调头:   全连接 → 收益排序分数
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 一、Mamba SSM 核心（纯 PyTorch 实现，无 CUDA 依赖）
# ==============================================================================

class SelectiveScan(nn.Module):
    """时序混合器：纯 Linear 实现（无 conv/RNN，不依赖 cuDNN）。

    沿时间维度做 channel mixing: (B, L, D) → Linear(D, 4D) → GELU → Linear(4D, D)
    每个时间步独立（类似 Transformer FFN），然后残差连接。
    """

    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) → (B, L, D)"""
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x + residual


class MambaBlock(nn.Module):
    """Mamba 块 = RMSNorm → GRU → 残差 + RMSNorm → FFN → 残差"""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.ssm = SelectiveScan(d_model, d_state)
        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expand * 2),
            nn.GELU(),
            nn.Linear(d_model * expand * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class MambaEncoder(nn.Module):
    """时序编码器：GRU 堆叠 → per-stock embedding"""

    def __init__(self, d_input: int, d_model: int = 128, n_layers: int = 4,
                 d_state: int = 16):
        super().__init__()
        self.input_proj = nn.Linear(d_input, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state) for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, F) → (B, L, D)"""
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ==============================================================================
# 二、图注意力网络（GAT）— 截面编码
# ==============================================================================

class GATLayer(nn.Module):
    """单头图注意力层：股票之间的边权重由特征相似度+行业关系动态生成"""

    def __init__(self, d_model: int, d_head: int = 32):
        super().__init__()
        self.d_head = d_head
        self.q_proj = nn.Linear(d_model, d_head)
        self.k_proj = nn.Linear(d_model, d_head)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_head ** -0.5

    def forward(self, x: torch.Tensor,
                adj_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, N, D) — batch, num_stocks, d_model
        adj_mask: (B, N, N) — 可选的图邻接矩阵（同行业=1）
        """
        B, N, D = x.shape

        q = self.q_proj(x).view(B, N, self.d_head)   # (B, N, h)
        k = self.k_proj(x).view(B, N, self.d_head)
        v = self.v_proj(x)                             # (B, N, D)

        # 注意力分数
        attn = torch.einsum('bnh,bmh->bnm', q, k) * self.scale  # (B, N, N)

        # 可选：用行业邻接矩阵 bias 注意力
        if adj_mask is not None:
            attn = attn + adj_mask * 0.5  # 同行业加 0.5 bias

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)

        out = torch.einsum('bnm,bmd->bnd', attn, v)
        return self.out_proj(out) + x  # 残差连接


class GATEncoder(nn.Module):
    """GAT 截面编码器：300只股票 → 经图注意力聚合邻居信息"""

    def __init__(self, d_model: int = 128, n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        d_head = d_model // n_heads

        self.layers = nn.ModuleList([
            nn.ModuleList([GATLayer(d_model, d_head) for _ in range(n_heads)])
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor,
                adj_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, N, D) → (B, N, D)"""
        for head_layers in self.layers:
            # 多头并行
            outs = [head(x, adj_mask) for head in head_layers]
            x = torch.stack(outs).mean(0)  # 多头平均
        return x


# ==============================================================================
# 三、MAE 预训练（掩码自编码器）
# ==============================================================================

class MAEPretrain(nn.Module):
    """掩码自编码器：遮掩 40% 股票的特征，训练模型重建。

    预训练用，不依赖标签。学会后才微调到排序任务。
    """

    def __init__(self, mamba_encoder: MambaEncoder, d_model: int = 128):
        super().__init__()
        self.encoder = mamba_encoder
        self.mask_token = nn.Parameter(torch.randn(1, 1, mamba_encoder.input_proj.in_features) * 0.02)

        # 解码器：轻量级，从 embedding 重建原始特征
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, mamba_encoder.input_proj.in_features),
        )

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.4):
        """MAE forward: 遮掩 → 编码 → 解码 → 重建损失。"""
        B, N, L, nF = x.shape
        mask = torch.rand(B, N, device=x.device) < mask_ratio

        # 遮掩
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
        x_masked = torch.where(mask_expanded, self.mask_token.expand(B, N, L, nF), x)

        # 编码
        x_flat = x_masked.reshape(B * N, L, nF)
        encoded = self.encoder(x_flat)
        encoded_last = encoded[:, -1, :].reshape(B, N, -1)

        # 解码 + loss
        x_pred = self.decoder(encoded_last)
        x_target = x[:, :, -1, :]
        mask_2d = mask.unsqueeze(-1).expand(-1, -1, nF)
        loss = F.mse_loss(x_pred[mask_2d], x_target[mask_2d])

        return x_pred, mask, loss


# ==============================================================================
# 四、完整模型 — 时序+截面 双编码 + 排序头
# ==============================================================================

class StockRankingModel(nn.Module):
    """完整选股模型 = Mamba 时序 + GAT 截面 + 排序头"""

    def __init__(self, n_features: int, d_model: int = 128,
                 n_stocks: int = 300, n_mamba_layers: int = 4,
                 n_gat_layers: int = 2, n_gat_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_stocks = n_stocks

        # 时序编码
        self.mamba = MambaEncoder(n_features, d_model, n_mamba_layers)

        # 截面编码
        self.gat = GATEncoder(d_model, n_gat_layers, n_gat_heads)

        # 融合投影
        self.fusion = nn.Linear(d_model * 2, d_model)

        # 排序头
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor,
                adj_mask: torch.Tensor | None = None,
                return_embedding: bool = False):
        """
        x: (B, N, L, F) — batch, stocks, seq_len, features

        返回: scores (B, N) 或 (scores, embeddings)
        """
        B, N, L, nF = x.shape

        # 时序编码
        x_flat = x.reshape(B * N, L, nF)
        temporal_emb = self.mamba(x_flat)          # (B*N, L, D)
        temporal_emb = temporal_emb[:, -1, :]       # (B*N, D) — 最后时刻
        temporal_emb = temporal_emb.reshape(B, N, -1)  # (B, N, D)

        # 截面编码（聚合邻居信息）
        cross_emb = self.gat(temporal_emb, adj_mask)  # (B, N, D)

        # 融合
        emb = self.fusion(torch.cat([temporal_emb, cross_emb], dim=-1))  # (B, N, D)

        # 排序分数
        scores = self.score_head(emb).squeeze(-1)  # (B, N)

        if return_embedding:
            return scores, emb
        return scores

    def get_encoder(self) -> MambaEncoder:
        """返回 Mamba 编码器（用于 MAE 预训练）"""
        return self.mamba


# ==============================================================================
# 五、Ranking Loss — 直接优化排序
# ==============================================================================

def lambdarank_loss(scores: torch.Tensor, targets: torch.Tensor,
                    top_k: int = 5) -> torch.Tensor:
    """LambdaRank loss：对 top-k 的排序错误施加更大惩罚。

    scores:  (B, N) — 预测分数
    targets: (B, N) — 真实收益
    """
    B, N = scores.shape
    loss = 0.0

    for b in range(B):
        s = scores[b]
        t = targets[b]

        # 所有 pair 的分数差和收益差
        s_diff = s.unsqueeze(1) - s.unsqueeze(0)   # (N, N)
        t_diff = t.unsqueeze(1) - t.unsqueeze(0)   # (N, N)

        # 只关心收益更高的股票被排在后面的情况
        wrong_order = (t_diff > 0).float()  # i < j but t_i > t_j → penalty
        # 对 top-K 的 pair 加大权重
        top_rank = max(N - top_k, 0)
        weight = torch.sigmoid(s_diff)

        pair_loss = weight * wrong_order * torch.log1p(torch.exp(-s_diff))
        loss += pair_loss.mean()

    return loss / B


def pairwise_ranking_loss(scores: torch.Tensor, targets: torch.Tensor,
                          margin: float = 0.05) -> torch.Tensor:
    """简化的 pairwise hinge ranking loss。"""
    B, N = scores.shape
    loss = 0.0

    for b in range(B):
        s = scores[b]
        t = targets[b]
        s_diff = s.unsqueeze(1) - s.unsqueeze(0)
        t_diff = (t.unsqueeze(1) - t.unsqueeze(0) > 0).float()
        loss += F.relu(margin - s_diff * (2 * t_diff - 1)).mean()

    return loss / B
