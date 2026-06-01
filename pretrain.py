"""
自监督预训练 —— Denoising Autoencoder + Masked Feature Prediction

两阶段训练：
  阶段1（本脚本）：预训练编码器（GRU + Transformer）
    输入 → 加噪声/掩码 → 编码 → 解码重构原始输入
    无标签，纯自监督，学到股票时序模式

  阶段2（train.py --load-pretrained）：加载预训练权重，微调排名任务
"""
import argparse, pickle, numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import MODEL_DIR, PROCESSED_DIR, DEVICE, D_MODEL, BATCH_SIZE, N_EPOCHS, LR
from model import PortfolioPredictor, TimeEncoder, CrossSectionalTransformer


# =============================================================================
# 预训练模型：编码器 + 解码器
# =============================================================================

class PretrainModel(nn.Module):
    """
    自监督预训练模型。

    架构：
      Encoder（共享 TimeEncoder + Transformer）→ (B, N, d_model)
      Decoder（轻量 MLP）→ (B, N, T, F) 重构原始输入

    预训练后，Encoder 权重 → train.py 微调排名任务。
    """
    def __init__(self, n_features: int, lookback: int = 60,
                 d_model: int = D_MODEL, n_transformer_layers: int = 2):
        super().__init__()
        self.n_features = n_features
        self.lookback = lookback
        self.d_model = d_model

        # 编码器（与推理模型共享）
        self.time_encoder = TimeEncoder(n_features, d_model)
        self.transformers = nn.ModuleList([
            CrossSectionalTransformer(d_model) for _ in range(n_transformer_layers)
        ])

        # 解码器：d_model → T × F（重构每个时间步的特征）
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, lookback * n_features),
        )

    def encode(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """提取股票 embeddings（供下游使用）"""
        B, N, T, F = x.shape
        x_flat = x.reshape(B * N, T, F)
        emb = self.time_encoder(x_flat)
        emb = emb.reshape(B, N, -1)
        for transformer in self.transformers:
            emb, _ = transformer(emb, mask)
        return emb

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, N, T, F) — 输入特征
            mask: (B, N) — 有效股票掩码

        Returns:
            (B, N, T, F) — 重构的特征
        """
        emb = self.encode(x, mask)                        # (B, N, d_model)
        recon = self.decoder(emb)                          # (B, N, T*F)
        B, N, T, F = x.shape
        recon = recon.reshape(B, N, T, F)                  # (B, N, T, F)
        return recon

    def save_encoder(self, path: Path):
        """保存编码器权重（供 train.py --load-pretrained 加载）"""
        torch.save({
            "time_encoder": self.time_encoder.state_dict(),
            "transformers": self.transformers.state_dict(),
            "config": {
                "n_features": self.n_features,
                "d_model": self.d_model,
            },
        }, path)
        print(f"[pretrain] Encoder saved to {path}")


# =============================================================================
# 数据增广：加噪声 + 时间步掩码
# =============================================================================

def corrupt_input(x: torch.Tensor, noise_std: float = 0.1,
                  mask_ratio: float = 0.15, mask_value: float = 0.0):
    """
    对输入添加两种噪声：
      1. 高斯噪声（全特征加小噪声）
      2. 时间步掩码（随机置零整行）

    Args:
        x:         原始输入 (B, N, T, F)
        noise_std: 高斯噪声标准差
        mask_ratio: 时间步掩码比例
        mask_value: 掩码填充值

    Returns:
        带噪输入 (B, N, T, F)
    """
    corrupted = x.clone()

    # 1. 高斯噪声
    if noise_std > 0:
        noise = torch.randn_like(corrupted) * noise_std
        corrupted = corrupted + noise

    # 2. 时间步掩码（对每只股票独立随机掩码部分时间步）
    if mask_ratio > 0:
        B, N, T, F = x.shape
        # 对每只股票生成掩码
        mask_t = torch.rand(B, N, T, 1, device=x.device) < mask_ratio
        corrupted = corrupted.masked_fill(mask_t, mask_value)

    return corrupted


# =============================================================================
# 训练循环
# =============================================================================

def train_epoch(model, loader, optimizer, epoch):
    model.train()
    total_loss = 0
    n = 0

    pbar = tqdm(loader, desc=f"Pretrain Epoch {epoch:3d}")
    for Xb, _, mb in pbar:
        Xb = Xb.to(DEVICE)
        mb = mb.to(DEVICE)

        # 加噪声
        X_corrupt = corrupt_input(Xb, noise_std=0.1, mask_ratio=0.15)

        # 前向
        X_recon = model(X_corrupt, mb)

        # 损失：只对有效股票计算
        loss = ((X_recon - Xb) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * Xb.size(0)
        n += Xb.size(0)
        pbar.set_postfix(loss=f"{total_loss / max(n, 1):.6f}")

    return total_loss / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="自监督预训练编码器")
    parser.add_argument("--epochs", type=int, default=50,
                        help="预训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批次大小（预训练内存更大）")
    args = parser.parse_args()

    # 加载数据
    print("Loading data ...")
    processed_path = PROCESSED_DIR / "samples.pkl"
    if not processed_path.exists():
        print("[ERROR] Run train.py first to build samples.pkl")
        return

    with open(processed_path, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    mask = data["mask"]
    n_features = X.shape[-1]
    print(f"  X: {X.shape}, features: {n_features}")

    # 创建数据集
    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(np.zeros((len(X), 300))),  # dummy y
        torch.from_numpy(mask),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    # 创建预训练模型
    model = PretrainModel(n_features=n_features).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Pretrain model: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, epoch)
        scheduler.step()
        print(f"  train_loss={loss:.6f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if loss < best_loss:
            best_loss = loss
            model.save_encoder(MODEL_DIR / "pretrained_encoder.pt")

    print(f"\n[pretrain] Done! Best loss: {best_loss:.6f}")
    print(f"[pretrain] Saved: {MODEL_DIR / 'pretrained_encoder.pt'}")
    print("[pretrain] Use: python train.py --load-pretrained")


if __name__ == "__main__":
    main()
