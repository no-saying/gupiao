"""
train_nn.py — GPU 优化训练管线 (AMP + torch.compile + 梯度累积)

用法:
  python train_nn.py --pretrain --epochs 200    # MAE 预训练
  python train_nn.py --finetune --epochs 100    # CVXPY 微调
  python train_nn.py --full                     # 全流程
"""

import argparse, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
# AMP removed — pure fp32 training to avoid OOM
from pathlib import Path
from tqdm import tqdm

from config import MODEL_DIR, DEVICE
from core.nn_model import (MambaEncoder, StockRankingModel,
                            MAEPretrain, lambdarank_loss)
from core.cvxpy_layer import (DifferentiablePortfolio, portfolio_loss)


# ==============================================================================
# 数据准备（向量化：从 lgbm_df 直接构建，10 秒完成）
# ==============================================================================

def build_window_samples(lgbm_df, feature_cols, lookback=20, step=5):
    """向量化窗口构建：从 lgbm_df 一步到位。

    lgbm_df 格式: date, stock_id, target, feature1, feature2, ...
    每日期 300 行（每只股票一行）。

    Returns:
        X: (N_stocks, T_windows, L_lookback, F_features)
        y: (N_stocks, T_windows) — 每周收益率标签
        pred_dates: 每个窗口的预测日期
        stock_ids: 股票代码列表
    """
    stock_ids = sorted(lgbm_df['stock_id'].unique())
    dates = sorted(lgbm_df['date'].unique())
    n_stocks = len(stock_ids)
    F = len(feature_cols)
    L = lookback

    # ── 全网格填充 → numpy（处理日期/股票缺失）──
    n_dates = len(dates)

    # 创建完整 date×stock_id 索引
    full_idx = pd.MultiIndex.from_product([dates, stock_ids], names=['date', 'stock_id'])
    full_df = pd.DataFrame(index=full_idx).reset_index()

    # Left join: 保持所有 date×stock 组合，缺失填 0
    merged = full_df.merge(lgbm_df[['date', 'stock_id'] + feature_cols + ['target']],
                           on=['date', 'stock_id'], how='left')
    merged[feature_cols] = merged[feature_cols].fillna(0.0)
    merged['target'] = merged['target'].fillna(0.0)

    feature_matrix = merged[feature_cols].values.astype(np.float32).reshape(
        n_dates, n_stocks, F)
    target_matrix = merged['target'].values.astype(np.float32).reshape(
        n_dates, n_stocks)

    # ── 逐窗口提取（避免 sliding_window_view OOM for 886 stocks）──
    pred_indices = np.arange(L, n_dates, step)
    pred_dates = [dates[i] for i in pred_indices if i < n_dates]

    X_list, y_list = [], []
    for idx in pred_indices:
        if idx < n_dates and idx - L >= 0:
            # 直接切片 feature_matrix[idx-L:idx] → (L, N, F)
            win = feature_matrix[idx - L:idx].transpose(1, 0, 2)  # (N, L, F)
            X_list.append(win.astype(np.float32))
            y_list.append(target_matrix[idx])

    if not X_list:
        raise ValueError("No valid windows generated")

    X = np.stack(X_list).transpose(1, 0, 2, 3)  # (N, T, L, F)
    y = np.stack(y_list).T.astype(np.float32)     # (N, T)

    print(f"  Windows: {n_stocks}s × {X.shape[1]}w × {L}d × {F}f "
          f"= {X.nbytes/1e9:.1f}GB")
    return X, y, pred_dates, stock_ids


# ==============================================================================
# MAE 预训练（GPU 优化）
# ==============================================================================

def pretrain_mae(lgbm_df, feature_cols, d_model=256, epochs=200,
                  batch_size=16, mask_ratio=0.4, lr=3e-4,
                  grad_accum=2):
    """MAE 预训练 — AMP + compile + 梯度累积。"""
    print("=" * 60)
    print(f"  MAE Pre-training | d_model={d_model} | batch={batch_size}×{grad_accum} | AMP")
    print("=" * 60)

    X, _, _, _ = build_window_samples(lgbm_df, feature_cols)
    n_stocks, n_windows, L, F = X.shape
    print(f"  Data: {n_stocks} stocks × {n_windows} windows × {L}d × {F}f")

    device = torch.device(DEVICE)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"  GPU before model: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    encoder = MambaEncoder(F, d_model, n_layers=4).to(device)
    mae = MAEPretrain(encoder, d_model).to(device)
    print(f"  GPU after model: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    # torch.compile 与动态 view 不兼容，用 AMP 加速

    optimizer = torch.optim.AdamW(mae.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    # no AMP scaler

    # 数据放 CPU 内存（503GB 够用），每 batch 传 GPU
    X_cpu = torch.FloatTensor(X)  # (N, T, L, F) on CPU
    print(f"  Data on CPU: {X_cpu.element_size() * X_cpu.numel() / 1e9:.1f} GB")
    print(f"  GPU before data: {torch.cuda.memory_allocated()/1e9:.2f}GB")

    best_loss = float('inf')
    t0 = time.time()

    for epoch in range(epochs):
        mae.train()
        epoch_loss = 0.0
        indices = torch.randperm(n_windows)

        for start in range(0, n_windows, batch_size):
            batch_idx = indices[start:start + batch_size]
            x_batch = X_cpu[:, batch_idx].permute(1, 0, 2, 3).to(device, non_blocking=True)

            _, _, loss = mae(x_batch, mask_ratio)
            loss_val = loss.item()
            loss = loss / grad_accum
            loss.backward()

            if (start // batch_size + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(mae.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            del x_batch, loss
            epoch_loss += loss_val / grad_accum

        scheduler.step()
        avg_loss = epoch_loss / max(1, n_windows // batch_size)

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 20 == 0:
            elapsed = time.time() - t0
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            print(f"  E{epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | "
                  f"best={best_loss:.4f} | GPU={gpu_mem:.1f}GB | {elapsed:.0f}s", flush=True)

    save_path = MODEL_DIR / "mae_pretrained.pt"
    torch.save({'encoder': encoder.state_dict(), 'd_model': d_model, 'n_features': F}, save_path)
    print(f"  Saved: {save_path} | Best loss: {best_loss:.4f}")
    return encoder


# ==============================================================================
# CVXPY 微调（GPU 优化）
# ==============================================================================

def finetune_with_cvxpy(lgbm_df, feature_cols, encoder, d_model=256,
                         epochs=100, batch_size=8, grad_accum=4,
                         lr=1e-4, lambda_cvar=2.0):
    """CVXPY 微调 — AMP + compile。"""
    print("=" * 60)
    print(f"  CVXPY Fine-tuning | λ_cvar={lambda_cvar} | AMP")
    print("=" * 60)

    F = len(feature_cols)
    X, y, pred_dates, stock_ids = build_window_samples(lgbm_df, feature_cols)
    n_stocks, n_windows = y.shape

    device = torch.device(DEVICE)

    model = StockRankingModel(F, d_model, n_stocks, n_mamba_layers=4).to(device)
    model.mamba.load_state_dict(encoder.state_dict())
    # torch.compile 与动态 view 不兼容，用 AMP 加速

    portfolio = DifferentiablePortfolio(n_stocks)

    # 分层学习率 + warmup
    base_lr = lr
    optimizer = torch.optim.AdamW([
        {'params': model.mamba.parameters(), 'lr': base_lr * 0.5},   # 0.5x (was 0.1x)
        {'params': model.gat.parameters(), 'lr': base_lr},
        {'params': model.fusion.parameters(), 'lr': base_lr},
        {'params': model.score_head.parameters(), 'lr': base_lr * 2},  # 2x for new head
    ], weight_decay=0.01)

    # Warmup scheduler: 前 10 epoch 线性增长，之后余弦衰减
    warmup_epochs = 10
    def warmup_cosine(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine)
    # no AMP scaler

    # 历史收益（CVaR 用）
    hist_returns = _build_hist_returns(y, window=20)
    hist_gpu = torch.FloatTensor(hist_returns).to(device)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)

    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_ret = 0.0, 0.0
        indices = torch.randperm(n_windows)

        for start in range(0, n_windows, batch_size):
            batch_idx = indices[start:start + batch_size]
            x_batch = X_tensor[:, batch_idx].permute(1, 0, 2, 3).to(device)
            y_batch = y_tensor[:, batch_idx].T.to(device)
            h_batch = hist_gpu[:, batch_idx].permute(1, 0, 2)

            # CVaR 渐进升温: epoch 0-50 从 0 线性增长到目标值
            current_lambda = lambda_cvar * min(1.0, epoch / 50)

            scores = model(x_batch)
            result = portfolio_loss(scores, y_batch,
                                     hist_returns=h_batch,
                                     lambda_cvar=current_lambda,
                                     optimizer=portfolio)
            loss = result['loss'] / grad_accum
            loss.backward()

            if (start // batch_size + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum
            epoch_ret += result['mean_return'].item()

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - t0
            gpu = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            n_batch = max(1, n_windows // batch_size)
            print(f"  E{epoch+1:3d}/{epochs} | "
                  f"loss={epoch_loss/n_batch:.4f} | "
                  f"ret={epoch_ret/n_batch:+.4f} | "
                  f"λ={current_lambda:.1f} | "
                  f"GPU={gpu:.1f}GB | {elapsed:.0f}s", flush=True)

    save_path = MODEL_DIR / "portfolio_model.pt"
    torch.save(model.state_dict(), save_path)
    print(f"  Saved: {save_path}")
    return model


def _build_hist_returns(y, window=20):
    """历史收益序列 (N, T, window)。"""
    N, T = y.shape
    hist = np.zeros((N, T, window), dtype=np.float32)
    for t in range(T):
        start = max(0, t - window)
        length = min(window, t - start)
        if length > 0:
            hist[:, t, -length:] = y[:, start:t]
    return hist


# ==============================================================================
# 主入口
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPU-optimized NN training")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=200)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lambda-cvar", type=float, default=2.0)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")

    # 数据
    from tushare_loader import fetch_csi300_stocks, build_tushare_only_panel
    from core.features import engineer_features
    from core.alpha158 import add_alpha158_features
    from core.data import build_lgbm_data

    stock_ids = fetch_csi300_stocks()
    panel = build_tushare_only_panel(stock_ids)
    panel = engineer_features(panel, stock_ids)
    panel = add_alpha158_features(panel)
    lgbm_df, feature_cols = build_lgbm_data(panel)

    if args.full:
        args.pretrain = True
        args.finetune = True

    if args.pretrain:
        encoder = pretrain_mae(lgbm_df, feature_cols,
                                d_model=args.d_model,
                                epochs=args.pretrain_epochs,
                                batch_size=args.batch_size)
    else:
        ckpt = torch.load(MODEL_DIR / "mae_pretrained.pt", map_location='cpu')
        encoder = MambaEncoder(ckpt['n_features'], ckpt['d_model'])
        encoder.load_state_dict(ckpt['encoder'])

    if args.finetune:
        finetune_with_cvxpy(lgbm_df, feature_cols, encoder,
                             d_model=args.d_model,
                             epochs=args.finetune_epochs,
                             batch_size=args.batch_size // 2,
                             lambda_cvar=args.lambda_cvar)


if __name__ == "__main__":
    main()
