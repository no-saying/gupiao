"""
=============================================================================
  模型模块 —— 基于注意力机制的股票排序预测
=============================================================================

整体架构（三阶段流水线）：

  输入 X: (Batch, N_stocks=300, T=60, F=22)
      │
      ▼
  ┌─────────────────────────────────────────────────────┐
  │ 阶段 1: TimeEncoder（时序编码器）                     │
  │   - 双向 GRU 独立处理每只股票的 60 天时序             │
  │   - 300 只股票共享同一套 GRU 参数                     │
  │   - 输出: (B, N, d_model=128) — 每只股票的时序摘要   │
  └─────────────────────────────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────────────────────────────┐
  │ 阶段 2: CrossSectionalTransformer（截面自注意力）     │
  │   - 300 只股票互相做 Multi-Head Self-Attention       │
  │   - 每只股票的 embedding 会融合其他股票的信息         │
  │   - 自动学习哪些股票"应该一起涨/跌"                  │
  │   - 2 层 Transformer 堆叠以学习更复杂的关联           │
  │   - 输出: (B, N, d_model) — 注意力增强后的向量       │
  └─────────────────────────────────────────────────────┘
      │
      ▼
  ┌─────────────────────────────────────────────────────┐
  │ 阶段 3: ScoreHead（评分头）                           │
  │   - MLP 将 d_model 维向量压缩成 1 个标量分数          │
  │   - 输出: (B, N) — 每只股票的预测分数                 │
  │   - 分数越高 = 模型认为未来一周涨得越多                │
  └─────────────────────────────────────────────────────┘

为什么这个架构适合本问题？

  1. 共享 GRU 让模型能提取任意数量股票的时序特征
     不需要为每只股票训练独立的编码器

  2. 截面自注意力（Cross-Sectional Attention）是核心创新：
     - 传统方法：每只股票独立预测 → 容易选出高度相关的股票
     - 本方法：每只股票的分数在计算时已经"看"了所有其他股票
       → 模型自动学会避免选中冗余标的
     - 注意力权重矩阵 (N×N) 可以解读为股票间的"模型认为的关联强度"

  3. 排序损失（LambdaRank）直接优化排序质量：
     - 我们不需要每只股票的精确收益率，只需要知道"A 是否比 B 更值得买"
     - LambdaRank 对 top 位置的排序错误惩罚更重（NDCG delta 加权）

损失函数说明：

  1. Pairwise Ranking Loss（Hinge 形式）
     对每对 (i, j)：如果 r_i > r_j，要求 s_i > s_j + margin
     简单直观，但所有 pair 权重相同

  2. LambdaRank Loss（推荐使用）
     将每对 (i, j) 的 loss 乘以 |ΔNDCG_ij|
     排名靠前的 pair 权重更大，直接优化 top-5 的排序质量
     使用 softplus 实现数值稳定

  3. Diversity Penalty（可选附加损失）
     惩罚 top-K 预测股票在 embedding 空间过于相近的情况
     鼓励模型分散选股

=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    D_MODEL, N_HEADS, N_GRU_LAYERS, DROPOUT,
    N_TRANSFORMER_LAYERS, D_FF, DEVICE, RANKING_MARGIN,
)


# =============================================================================
# 时序编码器：将 (T, F) 序列压缩为一个 d_model 维向量
# =============================================================================

class TimeEncoder(nn.Module):
    """
    双向 GRU 编码器，将每只股票的 t 天特征序列映射为一个固定维度的向量。

    为什么选 GRU 而不是 LSTM？
      - GRU 参数量更少，训练更快
      - 对 60 天这种中等长度的序列，GRU 和 LSTM 表现相当
      - 双向可以同时捕捉过去和未来的局部上下文（对已知序列没问题）

    为什么用双向？
      - 前向捕捉"从过去到现在"的趋势
      - 后向捕捉"从后往前看"的模式（如 V 型反转）
      - 拼接前向+后向的最终隐态获得更丰富的表示
    """

    def __init__(self, n_features: int, d_model: int = D_MODEL,
                 n_layers: int = N_GRU_LAYERS, dropout: float = DROPOUT,
                 use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        self.input_proj = nn.Linear(n_features, d_model)

        self.gru = nn.GRU(
            d_model, d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )

        # Bahdanau 时序注意力（可选）
        if use_attention:
            self.attn_hidden = nn.Linear(d_model, d_model, bias=False)
            self.attn_query  = nn.Linear(d_model, d_model, bias=False)
            self.attn_v      = nn.Linear(d_model, 1, bias=False)

        self.output_proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*N, T, F)

        Returns:
            (B*N, d_model)
        """
        x = self.input_proj(x)                           # (BN, T, F) → (BN, T, d_model)

        # GRU: 同时保留所有时间步输出（用于 attention）和最后隐态
        output, h_n = self.gru(x)
        # output: (BN, T, d_model*2)  全部时间步
        # h_n:    (2*layers, BN, d_model)

        h_fwd = h_n[-2, :, :]                            # (BN, d_model)
        h_bwd = h_n[-1, :, :]                            # (BN, d_model)

        if self.use_attention:
            # 用前向输出做 key，最后后向隐态做 query
            hs = self.gru.hidden_size                     # = d_model
            forward_outs = output[:, :, :hs]             # (BN, T, d_model)
            query = h_bwd                                 # (BN, d_model)

            # Bahdanau score: v^T tanh(W_h·hiddens + W_q·query)
            e = self.attn_v(torch.tanh(
                self.attn_hidden(forward_outs) + self.attn_query(query).unsqueeze(1)
            )).squeeze(-1)                                # (BN, T)

            attn_w = torch.softmax(e, dim=1)             # (BN, T)
            context = (forward_outs * attn_w.unsqueeze(-1)).sum(dim=1)  # (BN, d_model)
            h = torch.cat([context, h_bwd], dim=-1)      # (BN, 2*d_model)
        else:
            h = torch.cat([h_fwd, h_bwd], dim=-1)        # (BN, 2*d_model)

        return self.dropout(self.output_proj(h))          # (BN, d_model)


# =============================================================================
# 截面 Transformer：在股票维度做自注意力
# =============================================================================

class CrossSectionalTransformer(nn.Module):
    """
    单层 Transformer Encoder，作用在股票维度上。

    输入：(B, N, d_model)  — N 只股票各用一个 d_model 维向量表示
    输出：(B, N, d_model)  — 每只股票融合了其他股票信息后的向量

    内部结构（标准 Transformer Encoder Block）：
      1. Multi-Head Self-Attention
         - Query/Key/Value 都来自同一组 stock embeddings
         - 每只股票"查询"哪些股票和自己相关
         - 注意力权重矩阵 (N×N) 即股票间的关联强度

      2. Add & LayerNorm（残差连接 + 层归一化）
         - 残差连接防止深层网络退化
         - LayerNorm 稳定训练

      3. Feed-Forward Network
         - 两层 MLP + GELU 激活
         - 提供非线性变换能力

      4. Add & LayerNorm（第二次残差连接）

    为什么用 LayerNorm 而不是 BatchNorm？
      - BatchNorm 在 batch 维度做归一化，对小 batch 不稳定
      - LayerNorm 在特征维度做归一化，对 batch size 不敏感
      - Transformer 架构标准就是用 LayerNorm
    """

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 d_ff: int = D_FF, dropout: float = DROPOUT):
        super().__init__()

        # Multi-Head Self-Attention
        # n_heads=8, d_model=128 → 每个 head 处理 16 维的子空间
        # 不同 head 可能学到不同的关联模式：
        #   例如 head 1 关注同行业、head 2 关注同市值、head 3 关注同动量
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )

        # Layer Normalization（两次，分别用于 attention 和 FFN 之后）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-Forward Network
        # d_model → d_ff → d_model
        # d_ff 设为 d_model 的 2 倍（256），提供足够的变换容量
        # GELU 是 Transformer 标配激活函数，比 ReLU 更平滑
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x:    (B, N, d_model)  — 股票 embeddings
            mask: (B, N)           — 1 = 有效股票, 0 = 屏蔽（数据不足的股票）

        Returns:
            out:          (B, N, d_model)  — 注意力增强后的向量
            attn_weights: (B, N, N)         — 注意力权重矩阵
                           attn_weights[b, i, j] = 股票 i 对股票 j 的关注度
                           可用于事后分析股票间的关联关系
        """
        # key_padding_mask=True 的位置会被屏蔽
        key_padding = (mask == 0) if mask is not None else None

        # Pre-LN 残差连接: Norm → Sublayer → Add
        # 相比 Post-LN（Add → Norm），Pre-LN 梯度更稳定，可训练更深网络
        attn_out, attn_weights = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=key_padding
        )
        x = x + attn_out

        # Feed-Forward 同样 Pre-LN
        x = x + self.ffn(self.norm2(x))

        return x, attn_weights


# =============================================================================
# 完整模型：时序编码 → 截面注意力 → 评分
# =============================================================================

class PortfolioPredictor(nn.Module):
    """
    股票组合预测模型。

    完整前向传播流程：
      (B, N, T, F)
          │
          ├─ reshape → (B*N, T, F)
          ├─ TimeEncoder → (B*N, d_model)
          ├─ reshape → (B, N, d_model)        ← 此时每只股票有了独立表示
          │
          ├─ Transformer Layer 1 → (B, N, d_model)  相互注意
          ├─ Transformer Layer 2 → (B, N, d_model)  更深层关联
          │
          ├─ ScoreHead MLP → (B, N, 1)               评分
          ├─ squeeze → (B, N)                         每个股票的标量分数
          │
          └─ mask 处理：无效股票分数设为 -1e9

    额外方法：
      encode_stocks(): 只跑编码器和 Transformer，不跑评分头
                       用于获取股票 embedding（做多样化选股等）
    """

    def __init__(self, n_features: int,
                 n_transformer_layers: int = N_TRANSFORMER_LAYERS,
                 d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_gru_layers: int = N_GRU_LAYERS, d_ff: int = D_FF,
                 dropout: float = DROPOUT,
                 use_attention: bool = False):
        super().__init__()
        self.d_model = d_model

        self.time_encoder = TimeEncoder(n_features, d_model, n_gru_layers, dropout, use_attention)

        # 堆叠多层 Transformer
        self.transformers = nn.ModuleList([
            CrossSectionalTransformer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_transformer_layers)
        ])

        # 评分头：将 d_model 维的股票表示映射为一个标量分数
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def encode_stocks(self, x: torch.Tensor,
                      mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        获取 Transformer 编码后的股票 embedding（不做评分）。

        用途：
          - 在 predict.py 中用于多样化选股（用 embedding 相似度判断股票相关性）
          - 可以做 embedding 可视化（如 t-SNE 看股票在特征空间的分布）
          - 可以计算股票间的"模型认为的关联距离"

        Args:
            x:    (B, N, T, F)
            mask: (B, N)

        Returns:
            (B, N, d_model)  — 每只股票的 embedding 向量
        """
        B, N, T, F = x.shape

        # 时序编码
        x_flat = x.reshape(B * N, T, F)               # 合并 batch 和 stock 维度
        emb = self.time_encoder(x_flat)                # (B*N, d_model)
        emb = emb.reshape(B, N, -1)                    # (B, N, d_model)

        # 通过所有 Transformer 层
        for transformer in self.transformers:
            emb, _ = transformer(emb, mask)

        return emb  # (B, N, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x:    (B, N, T, F)  — 一批样本，每个样本是 N 只股票的 T 天 F 维特征
            mask: (B, N)        — 1 = 该股票有效, 0 = 数据缺失

        Returns:
            scores:       (B, N)     — 每只股票的预测分数
            attn_weights: (B, N, N)  — 最后一层 Transformer 的注意力权重
        """
        B, N, T, F = x.shape

        # ── 阶段 1: 时序编码 ─────────────────────────────────
        # 把 B 和 N 合并成 batch 维，让 GRU 同时处理所有股票
        # 这样做的好处：GPU 并行效率高，且所有股票共享同一套时序编码器参数
        x_flat = x.reshape(B * N, T, F)               # (B*N, T, F)
        emb = self.time_encoder(x_flat)               # (B*N, d_model)
        emb = emb.reshape(B, N, -1)                   # (B, N, d_model)

        # ── 阶段 2: 截面自注意力 ─────────────────────────────
        # 此时 emb 的每一行 (N, d_model) 代表同一样本内 N 只股票的向量
        # Transformer 在这 N 只股票之间做自注意力
        attn_weights = None
        for transformer in self.transformers:
            emb, attn_weights = transformer(emb, mask)

        # ── 阶段 3: 评分 ────────────────────────────────────
        scores = self.score_head(emb).squeeze(-1)     # (B, N, 1) → (B, N)

        # ── 屏蔽无效股票 ───────────────────────────────────
        # 无效股票（数据不足的）分数设为 -1e9
        # 这样在后续排序/选股时会自然排到最后
        if mask is not None:
            scores = scores * mask + (1 - mask) * (-1e9)

        return scores, attn_weights


# =============================================================================
# 损失函数
# =============================================================================

# ── 辅助函数：NDCG 的 gain ─────────────────────────────────

def ndcg_gain(returns: torch.Tensor) -> torch.Tensor:
    """
    将收益率映射为 NDCG 的 gain 值。

    使用指数映射：gain = 2^r - 1
    - 收益越大的股票，gain 越大
    - 这样模型会更关注高收益股票的正确排序
    - r=0 时 gain=0，负收益的 gain 接近 -1
    """
    return 2.0 ** returns - 1.0


# ── 损失 1: Pairwise Hinge Ranking ─────────────────────────

def pairwise_ranking_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    margin: float = RANKING_MARGIN,
) -> torch.Tensor:
    """
    Pairwise Hinge 排序损失。

    核心思想：
      对于每一对股票 (i, j)，如果真实收益 r_i > r_j，
      那么预测分数 s_i 应该比 s_j 大至少 margin（默认 0.05）。

    数学形式：
      loss_ij = max(0, margin - sign(r_i - r_j) * (s_i - s_j))

    当 s_i - s_j > margin 且 r_i > r_j 时，loss_ij = 0（正确排序，无惩罚）
    当 s_i < s_j 但 r_i > r_j 时，loss_ij = margin + (s_j - s_i)（排序错误，有惩罚）

    Pairwise hinge 简单高效，但对所有 pair 一视同仁。
    对于 top-5 选股问题，LambdaRank（下个函数）通常效果更好。

    Args:
        scores:     (B, N)  — 模型预测分数
        targets:    (B, N)  — 真实未来收益
        valid_mask: (B, N)  — 哪些股票是有效的（mask=0 的 pair 不参与 loss）
        margin:     float   — hinge 的边界值

    Returns:
        scalar loss
    """
    B, N = scores.shape

    # 构造 (B, N, N) 的 pairwise 差异矩阵
    # score_diff[b, i, j] = s_i - s_j
    # target_diff[b, i, j] = r_i - r_j
    score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)
    target_diff = targets.unsqueeze(2) - targets.unsqueeze(1)

    # sign = +1 当 r_i > r_j, -1 当 r_i < r_j, 0 当 r_i == r_j
    sign = torch.sign(target_diff)

    # Hinge: loss = max(0, margin - sign * score_diff)
    # 当 r_i > r_j (sign=+1): loss = max(0, margin - (s_i-s_j))
    #   → 如果 s_i-s_j>margin，loss=0 ✓
    # 当 r_i < r_j (sign=-1): loss = max(0, margin + (s_i-s_j))
    #   → 等价于 max(0, margin - (s_j-s_i))，对称正确 ✓
    loss_ij = F.relu(margin - sign * score_diff)

    # 构建有效性 mask：需要 target 不同（有意义比较）且两只股票都有效
    pair_valid = (sign.abs() > 1e-6).float()                    # 排除 target 完全相等的情况
    if valid_mask is not None:
        both_valid = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
        pair_valid = pair_valid * both_valid                    # 排除任一股票无效的情况

    n_pairs = pair_valid.sum() + 1e-8                           # +1e-8 防止除零
    return (loss_ij * pair_valid).sum() / n_pairs


# ── 损失 2: LambdaRank ─────────────────────────────────────

def lambdarank_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    LambdaRank 排序损失（推荐用于训练）。

    与 Pairwise Hinge 的区别：
      每对 (i, j) 的 loss 被 |ΔNDCG_ij| 加权。
      排在顶部的 pair 的排序错误会被惩罚得更重。

    为什么这对我们的任务很重要？
      - 我们只需要选 top-5 的股票
      - 第 1 名和第 2 名排错位置，比第 100 名和第 101 名排错的代价大得多
      - LambdaRank 通过 NDCG delta 加权重现了这一点

    LambdaRank 的梯度解释：
      λ_ij = -σ / (1 + exp(σ*(s_i - s_j))) * |ΔNDCG_ij|
      直观理解：模型会更努力地把"真正好的"股票推到排序顶部

    数值稳定性：
      使用 F.softplus(-σ*score_diff) 代替 log(1+exp(-σ*score_diff))
      softplus 在数值上更稳定，不会溢出

    Args:
        scores:     (B, N)
        targets:    (B, N)
        valid_mask: (B, N)
        sigma:      控制 logistic 的陡峭程度，默认 1.0

    Returns:
        scalar loss
    """
    B, N = scores.shape
    device = scores.device

    # ── 步骤 1: 计算排序位置 ─────────────────────────────
    # 按真实收益降序排列
    # 无效股票的目标值设为 -inf，排序时自然排在最后
    if valid_mask is not None:
        masked_targets = targets.clone()
        masked_targets[valid_mask < 0.5] = -float("inf")
    else:
        masked_targets = targets

    # 按真实收益降序 → true_rank 给出排序索引
    _, true_rank = torch.sort(masked_targets, dim=1, descending=True)

    # 计算每个位置在排序中的排名（从 0 开始）
    # positions[i] = 股票 i 在全量排序中的位置
    positions = torch.zeros_like(targets)
    positions = positions.scatter(
        1, true_rank,
        torch.arange(N, device=device).float().unsqueeze(0).expand(B, -1),
    )

    # ── 步骤 2: 计算 NDCG 折扣 ────────────────────────────
    # gain = 2^r - 1（指数映射，让高收益股票贡献更大）
    # discount = 1 / log2(pos + 2)（位置越靠后，折扣越大）
    gains = ndcg_gain(targets)                                # (B, N)
    discounts = 1.0 / torch.log2(positions + 2.0)            # (B, N)
    gd = gains * discounts                                    # (B, N) 每只股票的增益×折扣

    # ── 步骤 3: Pairwise NDCG delta ────────────────────────
    score_diff = scores.unsqueeze(2) - scores.unsqueeze(1)   # (B, N, N)
    target_diff = targets.unsqueeze(2) - targets.unsqueeze(1)

    # |ΔNDCG_ij| = |gd_i - gd_j| → 如果交换 i 和 j 的位置，NDCG 会变化多少
    delta_ndcg = torch.abs(gd.unsqueeze(2) - gd.unsqueeze(1))  # (B, N, N)

    # ── 步骤 4: Logistic Loss × Delta NDCG ────────────────
    # softplus(x) = log(1 + exp(x))，数值稳定
    # softplus(-σ * score_diff) = log(1 + exp(-σ*(s_i - s_j)))
    # 当 s_i > s_j 且 r_i > r_j 时，score_diff > 0，softplus → 0 ✓
    pair_loss = delta_ndcg * F.softplus(-sigma * score_diff)

    # 只计算有效 pair
    pair_valid = (target_diff.abs() > 1e-6).float()            # 收益不同的 pair
    if valid_mask is not None:
        both_valid = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
        pair_valid = pair_valid * both_valid

    n_pairs = pair_valid.sum() + 1e-8
    return (pair_loss * pair_valid).sum() / n_pairs


# ── 附加损失: Diversity Penalty ────────────────────────────

def diversity_penalty(
    scores: torch.Tensor,
    features: torch.Tensor,
    top_k: int = 5,
) -> torch.Tensor:
    """
    多样化惩罚：防止模型选出的 top-K 股票在 embedding 空间过于集中。

    工作原理：
      1. 取分数最高的 K 只股票
      2. 计算这 K 只股票在 embedding 空间的 pairwise cosine similarity
      3. 惩罚非对角线上的高相似度

    这相当于给模型一个信号："不要把所有高分都给同一个行业的股票"。
    配合截面的自注意力机制，模型会学会在评分时就考虑到分散化。

    Args:
        scores:   (B, N)       — 预测分数
        features: (B, N, d)    — 股票 embedding（来自 Transformer 输出）
        top_k:    int          — 关注 top-K

    Returns:
        scalar penalty
    """
    B, N, D = features.shape
    if N < top_k:
        return torch.tensor(0.0, device=scores.device)

    # 选 top-K
    _, top_idx = torch.topk(scores, top_k, dim=1)             # (B, K)

    # 取出 top-K 的 embedding
    top_feats = torch.gather(
        features, 1,
        top_idx.unsqueeze(-1).expand(-1, -1, D)
    )                                                          # (B, K, D)

    # L2 归一化（让内积等于 cosine similarity）
    top_feats = F.normalize(top_feats, dim=-1)

    # pairwise cosine similarity
    sim = torch.bmm(top_feats, top_feats.transpose(1, 2))     # (B, K, K)

    # 只惩罚非对角线（ii 位置是自己和自己 = 1，不惩罚）
    off_diag_mask = 1.0 - torch.eye(top_k, device=scores.device).unsqueeze(0)
    penalty = (sim.abs() * off_diag_mask).sum() / (off_diag_mask.sum() + 1e-8) / B

    return penalty


def combined_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    features: torch.Tensor | None = None,
    lambda_diverse: float = 0.01,
) -> torch.Tensor:
    """
    LambdaRank + 可选多样化惩罚的组合损失。

    lambda_diverse 控制多样化惩罚的强度：
      0     = 纯 LambdaRank（不考虑分散化）
      0.01  = 轻微偏向分散化（推荐默认值）
      0.1   = 强分散化约束

    这个值需要在回测中调试，找到收益和分散化的最佳平衡。
    """
    rank_loss = lambdarank_loss(scores, targets, valid_mask)
    if features is not None and lambda_diverse > 0:
        div_loss = diversity_penalty(scores, features)
        return rank_loss + lambda_diverse * div_loss
    return rank_loss


# =============================================================================
# 损失 4: ListNet（Listwise 排序损失）—— 直接优化排序分布
# =============================================================================

def listnet_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    temperature: float = 2.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    ListNet 排序损失（基于 Plackett-Luce 模型的 listwise 方法）。

    支持 sample_weight 实现时间衰减加权：
      近期样本权重高，远期样本权重低，让模型更关注当前市场规律。

    Args:
        scores:     (B, N)
        targets:    (B, N)
        valid_mask: (B, N)
        temperature: float  — 较高的温度使梯度更平滑，避免 NaN
        sample_weight: (B,) — 每个样本的权重（None = 等权）
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(targets)

    B, N = scores.shape
    device = scores.device
    losses = []
    kept_weights = []

    for i in range(B):
        valid_idx = (valid_mask[i] > 0.5).nonzero(as_tuple=True)[0]
        n_valid = len(valid_idx)
        if n_valid < 2:
            continue

        s = scores[i, valid_idx]       # (n_valid,)
        t = targets[i, valid_idx]      # (n_valid,)

        s_shift = s - s.max()
        t_shift = t - t.max()

        P = torch.softmax(s_shift / temperature, dim=0)
        Q = torch.softmax(t_shift / temperature, dim=0)

        loss_i = -(Q * torch.log(P + 1e-10)).sum()
        losses.append(loss_i)
        if sample_weight is not None:
            kept_weights.append(sample_weight[i])

    if not losses:
        return torch.tensor(0.0, device=device)

    losses_t = torch.stack(losses)
    if sample_weight is not None and kept_weights:
        w_t = torch.tensor(kept_weights, device=device)
        return (losses_t * w_t).sum() / w_t.sum()
    return losses_t.mean()


# =============================================================================
# 损失 5: Top-K ListNet —— 只关注前 K 只股票的排序
# =============================================================================

def topk_listnet_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    k: int = 10,
    temperature: float = 1.0,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Top-K ListNet 损失（支持时间衰减加权）。

    只对分数最高的 K 只股票计算 ListNet 损失。

    Args:
        sample_weight: (B,) — 时间衰减权重，最近的样本权重高
    """
    if valid_mask is None:
        valid_mask = torch.ones_like(targets)

    B, N = scores.shape
    device = scores.device

    masked_scores = scores.clone()
    masked_scores[valid_mask < 0.5] = -1e9
    masked_targets = targets.clone()
    masked_targets[valid_mask < 0.5] = -1e9

    losses, kept_w = [], []
    for i in range(B):
        valid_idx = (valid_mask[i] > 0.5).nonzero(as_tuple=True)[0]
        n_valid = len(valid_idx)
        if n_valid < 2:
            continue

        k_actual = min(k, n_valid)
        topk_idx = torch.topk(masked_scores[i, valid_idx], k_actual).indices
        topk_scores = scores[i, valid_idx[topk_idx]]
        topk_targets = targets[i, valid_idx[topk_idx]]

        log_P = torch.log_softmax(topk_scores / temperature, dim=0)
        Q = torch.softmax(topk_targets / temperature, dim=0)
        loss_i = -(Q * log_P).sum()
        losses.append(loss_i)
        if sample_weight is not None:
            kept_w.append(sample_weight[i])

    if not losses:
        return torch.tensor(0.0, device=device)
    losses_t = torch.stack(losses)
    if sample_weight is not None and kept_w:
        w_t = torch.tensor(kept_w, device=device)
        return (losses_t * w_t).sum() / w_t.sum()
    return losses_t.mean()
