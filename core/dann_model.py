"""
DANN (Domain-Adversarial Neural Network) — 域自适应选股

核心思路：500只训练股 + 300只选股池 → 对抗训练剥离市值偏差
特征提取层学出"分不清是哪只池子"的通用量价特征。

架构：
  Feature Extractor (Mamba) → Task Head (收益预测)
                            → Domain Classifier (300 vs 500, 梯度反转)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from core.nn_model import MambaEncoder


class GradientReversal(torch.autograd.Function):
    """梯度反转层：前向不变，反向乘 -lambda"""

    @staticmethod
    def forward(ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradRevLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversal.apply(x, self.lambda_)


class DANNStockModel(nn.Module):
    """DANN 选股模型 = 特征提取 + 任务头 + 域分类器。

    Args:
        n_features: 输入特征数
        d_model: 隐藏维度
        n_stocks: 股票总数（用于 GAT）
        n_mamba_layers: Mamba 层数
        lambda_domain: 域损失的梯度反转强度
    """

    def __init__(self, n_features: int, d_model: int = 256,
                 n_stocks: int = 886, n_mamba_layers: int = 4,
                 lambda_domain: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.lambda_domain = lambda_domain

        # 特征提取器（共享）
        self.mamba = MambaEncoder(n_features, d_model, n_layers=n_mamba_layers)

        # 任务头：预测收益率
        self.task_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
        )

        # 域分类器：判断 300 or 500
        self.grad_rev = GradRevLayer(lambda_domain)
        self.domain_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),  # 二分类 logit
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, N, L, F) — batch, stocks, seq_len, features

        Returns:
            scores: (B, N) — 收益预测
            domain_logits: (B*N,) — 域分类 logits
        """
        B, N, L, nF = x.shape

        # 特征提取
        x_flat = x.reshape(B * N, L, nF)
        features = self.mamba(x_flat)           # (B*N, L, D)
        features_last = features[:, -1, :]       # (B*N, D)

        # 任务头 → 收益率
        scores = self.task_head(features_last).reshape(B, N)  # (B, N)

        # 域分类 → 300 or 500（梯度反转）
        reversed_features = self.grad_rev(features_last)
        domain_logits = self.domain_head(reversed_features).squeeze(-1)  # (B*N,)

        return scores, domain_logits


def dann_loss(task_scores: torch.Tensor, task_targets: torch.Tensor,
              domain_logits: torch.Tensor, domain_labels: torch.Tensor,
              lambda_domain: float = 1.0) -> dict:
    """DANN 组合损失。

    Args:
        task_scores: (B, N) — 预测收益
        task_targets: (B, N) — 真实收益
        domain_logits: (B*N,) — 域分类 logits
        domain_labels: (B*N,) — 域标签 (0=300, 1=500)

    Returns:
        dict with total_loss, task_loss, domain_loss
    """
    # 任务损失：MSE on returns
    task_loss = F.mse_loss(task_scores, task_targets)

    # 域损失：二分类 BCE（希望分不清 → 越大越好 → 对抗训练）
    domain_loss = F.binary_cross_entropy_with_logits(
        domain_logits, domain_labels.float())

    # 总损失 = 任务 - lambda * 域（梯度反转已在层中处理）
    total_loss = task_loss + lambda_domain * domain_loss

    return {
        'total': total_loss,
        'task': task_loss.detach(),
        'domain': domain_loss.detach(),
    }


def build_domain_labels(batch_size: int, n_300: int, n_500: int,
                         device: torch.device) -> torch.Tensor:
    """构建域标签：0 = CSI 300, 1 = CSI 500。

    Args:
        batch_size: 批次数
        n_300: 每批中 300 的股票数
        n_500: 每批中 500 的股票数

    Returns:
        domain_labels: (B * (n_300 + n_500),) — 0/1 标签
    """
    labels_300 = torch.zeros(batch_size * n_300, device=device)
    labels_500 = torch.ones(batch_size * n_500, device=device)
    return torch.cat([labels_300, labels_500])
