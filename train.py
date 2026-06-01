"""
=============================================================================
  训练脚本 —— 端到端的模型训练与回测验证
=============================================================================

训练流程概览：

  1. 数据获取       fetch_csi300_stocks() + build_panel()
     ↓
  2. 特征工程       engineer_features() → 22 个因子
     ↓
  3. 构造样本       make_window_samples() → (X, y, mask) 张量
     ↓
  4. 数据划分       按时间顺序切分 train/val/test
     ↓
  5. 模型训练       LambdaRank loss + AdamW + ReduceLROnPlateau
     ↓
  6. 模型验证       验证集指标 + Early Stopping
     ↓
  7. 回测评估       在测试集上模拟 top-5 等权组合的收益
     ↓
  8. 模型保存       保存权重 + 配置到 models/

关键设计决策：

  - 时间序列必须按时间切分（不能用 shuffle split）
    避免用"未来"数据预测"过去"（data leakage）

  - 验证集和测试集都按时间切分在最近的数据上
    因为市场环境随时间变化，在最近的数据上验证更有意义

  - 回测模拟真实交易：
    买 T+1 开盘价，卖 T+5 开盘价，手续费未考虑（按赛题要求）

使用方法：
  python train.py                     # 默认配置训练
  python train.py --loss pairwise     # 用 pairwise hinge loss
  python train.py --epochs 200        # 自定义训练轮数
  python train.py --lr 5e-5           # 自定义学习率
  python train.py --download          # 强制重新下载数据

=============================================================================
"""

import os
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import (
    MODEL_DIR, PROCESSED_DIR, DEVICE, D_MODEL,
    N_HEADS, N_GRU_LAYERS, N_TRANSFORMER_LAYERS, D_FF, DROPOUT,
    BATCH_SIZE, N_EPOCHS, LR, LR_PEAK, WEIGHT_DECAY, GRAD_CLIP,
    LR_PATIENCE, EARLY_STOP_PATIENCE, VAL_RATIO, TEST_RATIO,
    LOOKBACK_DAYS, PREDICT_HORIZON, STEP_DAYS,
    MAX_STOCKS, TOP_K_CANDIDATES, TEMPERATURE, N_WORKERS,
    BIG_D_MODEL, BIG_N_HEADS, BIG_N_GRU_LAYERS,
    BIG_N_TRANSFORMER_LAYERS, BIG_D_FF, USE_ATTENTION,
)
from data_loader import get_official_stock_ids, build_panel_from_official
from features import engineer_features, make_window_samples, get_norm_stats
from model import PortfolioPredictor, lambdarank_loss, pairwise_ranking_loss, listnet_loss, topk_listnet_loss, pcc_loss


# =============================================================================
# 时序交叉验证：4折，折间留5天空白
# =============================================================================

def make_tscv_dataloaders(
    X: np.ndarray, y: np.ndarray, mask: np.ndarray,
    n_folds: int = 4,
    gap: int = 5,
    val_size: int = 50,
    batch_size: int = BATCH_SIZE,
    num_workers: int = N_WORKERS,
    decay_rate: float = 0.0,
) -> list[tuple[DataLoader, DataLoader]]:
    """
    时间序列交叉验证（expanding window）。

    每折：
      Fold 1: train [0:end_1]             val [end_1+gap : end_1+gap+val_size]
      Fold 2: train [0:end_2]             val [end_2+gap : end_2+gap+val_size]
      ...

    折间留 gap 天空白防止数据泄漏。
    验证集大小固定为 val_size 个样本。

    Returns:
        [(train_loader, val_loader), ...] 共 n_folds 对
    """
    n = len(X)
    folds = []

    # 等分区间
    step = (n - val_size - gap) // n_folds

    for k in range(n_folds):
        train_end = (k + 1) * step
        val_start = train_end + gap
        val_end = min(val_start + val_size, n - 5)  # 留5个给测试

        if train_end < 100 or val_end > n - 5:
            break

        X_tr, y_tr, m_tr = X[:train_end], y[:train_end], mask[:train_end]
        X_va, y_va, m_va = X[train_end:val_end], y[train_end:val_end], mask[train_end:val_end]

        # 时间衰减
        if decay_rate > 0:
            pos = np.arange(len(X_tr), dtype=np.float32)
            w_tr = np.exp(-decay_rate * (len(X_tr) - 1 - pos) / max(len(X_tr) - 1, 1))
            w_tr = w_tr / w_tr.mean()
        else:
            w_tr = np.ones(len(X_tr), dtype=np.float32)

        def _dl(Xa, ya, ma, wa, shuffle):
            ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya),
                               torch.from_numpy(ma), torch.from_numpy(wa))
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                              drop_last=False, num_workers=num_workers,
                              pin_memory=True, prefetch_factor=2)

        folds.append((
            _dl(X_tr, y_tr, m_tr, w_tr, shuffle=True),
            _dl(X_va, y_va, m_va, np.ones(len(X_va)), shuffle=False),
        ))

    print(f"  [tscv] {len(folds)} folds, gap={gap} days, val_size={val_size}")
    return folds


# =============================================================================
# 数据加载器创建：按时间顺序切分
# =============================================================================

def make_dataloaders(
    X: np.ndarray, y: np.ndarray, mask: np.ndarray,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    num_workers: int = N_WORKERS,
    decay_rate: float = 0.0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    将样本按时间顺序切分为训练集、验证集、测试集。

    为什么不能用随机切分？
      时间序列数据中，相邻样本高度相关。
      如果随机切分，模型可能在"见过类似样本"的验证集上表现好，
      但在真正的新时间点上表现差。这就是数据泄露（data leakage）。

    切分方式：
      假设有 1000 个样本（按时间排序）：
        test_start = 900 (后 10%)
        val_start  = 750 (后 15% + 10%)

        训练集:    [0, 750)   ← 最早的数据
        验证集:    [750, 900) ← 中间
        测试集:    [900, 1000] ← 最新（用于最终评估）

    训练集做 shuffle=True 增加随机性，验证/测试集不做 shuffle。
    """
    n = len(X)
    test_start = int(n * (1 - test_ratio))
    val_start = int(n * (1 - test_ratio - val_ratio))

    X_train, y_train, m_train = X[:val_start], y[:val_start], mask[:val_start]
    X_val,   y_val,   m_val   = X[val_start:test_start], y[val_start:test_start], mask[val_start:test_start]
    X_test,  y_test,  m_test  = X[test_start:], y[test_start:], mask[test_start:]

    # 时间衰减权重：越新的样本权重越高
    if decay_rate > 0:
        n_train = len(X_train)
        # weight[i] = exp(-decay_rate * (n_train-1-i) / (n_train-1))
        # 最新样本 weight=1.0, 最旧 weight=exp(-decay_rate)
        pos = np.arange(n_train, dtype=np.float32)
        w_train = np.exp(-decay_rate * (n_train - 1 - pos) / max(n_train - 1, 1))
        w_train = w_train / w_train.mean()  # 归一化到平均=1，不影响学习率
        print(f"  [train] Time decay: rate={decay_rate}, weight range=[{w_train.min():.3f}, {w_train.max():.3f}]")
    else:
        w_train = np.ones(len(X_train), dtype=np.float32)

    def to_loader(Xa, ya, ma, shuffle=False, wa=None):
        ds = TensorDataset(
            torch.from_numpy(Xa),
            torch.from_numpy(ya),
            torch.from_numpy(ma),
            torch.from_numpy(wa) if wa is not None else torch.ones(len(Xa)),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          drop_last=False, num_workers=num_workers,
                          pin_memory=True, prefetch_factor=2)

    print(f"[train] Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return to_loader(X_train, y_train, m_train, shuffle=True, wa=w_train), \
           to_loader(X_val,   y_val,   m_val),                              \
           to_loader(X_test,  y_test,  m_test)


# =============================================================================
# 验证/测试评估
# =============================================================================

def evaluate(model: PortfolioPredictor, loader: DataLoader, loss_fn) -> dict:
    """
    在验证集或测试集上计算损失和排序准确率。

    Pairwise Accuracy（排序准确率）：
      对所有有效 pair (i, j)，判断模型给出的排序方向是否正确。
      如果 r_i > r_j 且 s_i > s_j → 正确
      如果 r_i > r_j 但 s_i < s_j → 错误

      这比 MSE 等回归指标更能反映模型的实际效果，
      因为我们最终只需要选出排名靠前的股票。
    """
    model.eval()
    total_loss = 0.0
    total_correct_pairs = 0.0
    n_samples = 0
    n_pairs = 0

    with torch.no_grad():
        for batch in loader:
            Xb, yb, mb = batch[0], batch[1], batch[2]
            Xb, yb, mb = Xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)
            scores, _ = model(Xb, mb)                         # 前向传播
            loss = loss_fn(scores, yb, mb)                    # 带 mask 的排序损失
            total_loss += loss.item() * Xb.size(0)
            n_samples += Xb.size(0)

            # ---- 计算 pairwise accuracy（逐样本） ----
            for i in range(Xb.size(0)):
                s, t, m = scores[i], yb[i], mb[i]            # 单个样本
                valid_idx = (m > 0.5).nonzero(as_tuple=True)[0]
                if len(valid_idx) < 2:
                    continue

                # 只考虑有效股票之间的 pair
                s_v, t_v = s[valid_idx], t[valid_idx]
                score_diff = s_v.unsqueeze(1) - s_v.unsqueeze(0)     # (V, V)
                target_diff = t_v.unsqueeze(1) - t_v.unsqueeze(0)    # (V, V)

                # sign(score_diff) == sign(target_diff) → 排序方向正确
                correct = (torch.sign(score_diff) == torch.sign(target_diff)).float()

                # 排除 target 完全相等的 pair
                valid_pair = (target_diff.abs() > 1e-6).float()
                total_correct_pairs += (correct * valid_pair).sum().item()
                n_pairs += valid_pair.sum().item()

    return {
        "loss": total_loss / max(n_samples, 1),
        "pair_accuracy": total_correct_pairs / max(n_pairs, 1),
    }


# =============================================================================
# 单轮训练
# =============================================================================

def augment_input(x: torch.Tensor, noise_std: float = 0.05,
                  mask_ratio: float = 0.1) -> torch.Tensor:
    """
    训练数据增强：高斯噪声 + 时间步掩码，提升泛化能力。
    对归一化后的特征加小噪声，随机遮住部分时间步。
    """
    if noise_std <= 0 and mask_ratio <= 0:
        return x
    corrupted = x.clone()
    if noise_std > 0:
        corrupted = corrupted + torch.randn_like(corrupted) * noise_std
    if mask_ratio > 0:
        B, N, T, F = x.shape
        mask = torch.rand(B, N, T, 1, device=x.device) < mask_ratio
        corrupted = corrupted.masked_fill(mask, 0.0)
    return corrupted


def labels_to_ranks(y: torch.Tensor, mask: torch.Tensor, gaussian: bool = True) -> torch.Tensor:
    """
    将原始收益率转换为排名分数（用于 ListNet 训练）。

    模式：
      gaussian=True:  百分位数 → 高斯分位数（z-score），尾部更突出
      gaussian=False: 百分位数（0~1 均匀分布）

    为什么 Gaussian 可能更好？
      百分位数线性均匀分布，但 top-5 选股关注头部。
      高斯变换让 top 5% 的股票获得更大的概率质量。
    """
    B, N = y.shape
    ranks = y.clone()
    for i in range(B):
        valid_idx = (mask[i] > 0.5).nonzero(as_tuple=True)[0]
        n_valid = len(valid_idx)
        if n_valid < 2:
            ranks[i] = -float("inf")
            continue
        valid_vals = y[i, valid_idx]
        sorted_idx = valid_vals.argsort()
        # 百分位数 [0, 1]
        percentile = torch.zeros(n_valid, device=y.device)
        percentile[sorted_idx] = torch.arange(n_valid, device=y.device).float() / (n_valid - 1)
        # 可选：高斯分位数（尾部拉伸）
        if gaussian and n_valid > 2:
            from torch.distributions import Normal
            # 将 [0,1] → [-0.999, 0.999] → 高斯分位数
            clipped = percentile.clamp(0.001, 0.999)
            normal = Normal(0.0, 1.0)
            ranks[i, valid_idx] = normal.icdf(clipped)  # z-score
        else:
            ranks[i, valid_idx] = percentile
        ranks[i, ~((mask[i] > 0.5))] = -float("inf")
    return ranks


_AUGMENT_ENABLED = False  # 数据增强，在 main 中设置


def train_epoch(model, loader, optimizer, scheduler, loss_fn, epoch: int,
                use_rank_labels: bool = False,
                scaler=None) -> float:
    """
    执行一个 epoch 的训练（支持混合精度 AMP + OneCycleLR）。
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}")
    for batch in pbar:
        Xb, yb, mb, wb = batch
        Xb, yb, mb, wb = Xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE), wb.to(DEVICE)

        yb_target = labels_to_ranks(yb, mb) if use_rank_labels else yb

        # 数据增强：加噪声 + 时间步掩码，防止过拟合
        if _AUGMENT_ENABLED:
            Xb = augment_input(Xb, noise_std=0.03, mask_ratio=0.05)

        with torch.amp.autocast(device_type="cuda", enabled=scaler is not None):
            scores, _ = model(Xb, mb)
            # sample_weight 只对 listnet/topk_listnet 生效
            kw = {"sample_weight": wb} if loss_fn.__name__ in ("listnet_loss", "topk_listnet_loss") else {}
            loss = loss_fn(scores, yb_target, mb, **kw)

        # 反向传播（支持 AMP）
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # OneCycleLR: 每个 batch 后更新学习率
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}",
                         lr=f"{scheduler.get_last_lr()[0]:.2e}")

    return total_loss / max(n_batches, 1)


# =============================================================================
# 完整训练流程
# =============================================================================

def train_model(
    model: PortfolioPredictor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = N_EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    loss_name: str = "lambdarank",
    use_rank_labels: bool = False,
    curriculum: bool = False,
    lr_patience: int = LR_PATIENCE,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
) -> PortfolioPredictor:
    """
    完整训练循环，包含：
      1. AdamW 优化器（带解耦权重衰减，比 Adam 更好的泛化能力）
      2. ReduceLROnPlateau 学习率调度（验证损失不降时自动减半学习率）
      3. Early Stopping（验证损失连续 N 轮不降则停止）
      4. Best Model Checkpointing（始终保存验证集上最优的权重）
    """
    model = model.to(DEVICE)

    # AdamW: Adam + 解耦的 L2 正则化
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # OneCycleLR: 先预热到峰值学习率，再余弦退火到接近0
    # 比 ReduceLROnPlateau 更激进，让模型在更多 epoch 里持续学习
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR_PEAK, total_steps=total_steps,
        pct_start=0.1,  # 前10% step 预热
        anneal_strategy="cos",
        div_factor=25.0,    # 初始 lr = max_lr / 25 = 1.2e-5
        final_div_factor=1e4,  # 最终 lr = max_lr / (25*1e4) = 1.2e-9
    )

    # 选择损失函数
    loss_map = {
        "lambdarank": lambdarank_loss,
        "pairwise": pairwise_ranking_loss,
        "listnet": listnet_loss,
        "topk_listnet": topk_listnet_loss,
        "pcc": pcc_loss,
    }
    loss_fn = loss_map.get(loss_name, listnet_loss)

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    # 混合精度缩放器（GPU 可用时启用 FP16）
    scaler = torch.amp.GradScaler(device="cuda") if DEVICE == "cuda" else None
    if scaler:
        print(f"  [train] Mixed precision (FP16) enabled")

    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, epoch,
                                 use_rank_labels=use_rank_labels, scaler=scaler)

        # ---- 验证 ----
        val_metrics = evaluate(model, val_loader, loss_fn)
        current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else lr
        print(f"  train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_pair_acc={val_metrics['pair_accuracy']:.3f}  "
              f"lr={current_lr:.2e}")

        # ---- 模型保存 & Early Stopping ----
        if val_metrics["loss"] < best_val_loss - 1e-7:
            best_val_loss = val_metrics["loss"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  → New best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # 恢复最优权重
    if best_weights is not None:
        model.load_state_dict(best_weights)
    else:
        print("  WARNING: No improvement found, using final weights")

    # ── 课程微调：用最近数据精调 ──────────────────────────────────
    if curriculum and hasattr(train_loader, 'dataset'):
        full_ds = train_loader.dataset
        n_full = len(full_ds)
        for ratio, ft_epochs in [(0.3, 8), (0.15, 8)]:
            n_sub = int(n_full * ratio)
            if n_sub < 50:
                break
            # 取最后 n_sub 个样本（最近的数据）
            indices = list(range(n_full - n_sub, n_full))
            subset = torch.utils.data.Subset(full_ds, indices)
            sub_loader = DataLoader(subset, batch_size=train_loader.batch_size,
                                    shuffle=True, num_workers=0)

            ft_lr = lr * 0.1  # 微调学习率减为 1/10
            ft_opt = torch.optim.AdamW(model.parameters(), lr=ft_lr, weight_decay=weight_decay)
            ft_sch = torch.optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=ft_epochs)

            print(f"\n  [curriculum] Fine-tuning on last {n_sub} samples ({ratio*100:.0f}%) for {ft_epochs} epochs")
            for ep in range(1, ft_epochs + 1):
                train_epoch(model, sub_loader, ft_opt, ft_sch, loss_fn, ep,
                            use_rank_labels=use_rank_labels, scaler=scaler)
                val_metrics = evaluate(model, val_loader, loss_fn)
                ft_sch.step()
                if ep % 4 == 0:
                    print(f"    ft_epoch={ep}  val_loss={val_metrics['loss']:.4f}  val_pair_acc={val_metrics['pair_accuracy']:.3f}")

    return model


# =============================================================================
# 回测：模拟交易评估模型效果
# =============================================================================

def portfolio_return(
    scores: np.ndarray,
    true_returns: np.ndarray,
    stock_mask: np.ndarray,
    max_stocks: int = MAX_STOCKS,
) -> float:
    """
    给定一个样本的预测分数和真实收益，计算 top-K 等权组合的实际收益。

    模拟过程：
      1. 找出模型评分最高的 K 只有效股票
      2. 等权分配（每只 1/K）
      3. 组合收益 = 各股票真实收益的平均值

    注意：这里用"真实收益"是回测——假设我们在 T+1 开盘买入、T+5 开盘卖出，
    用事后知道的收益率来评估决策质量。

    等权 vs 加权：
      对于 top-5 的选股问题，等权通常已经足够好，
      因为入选的股票已经是模型最看好的，权重微调带来的提升有限。
    """
    valid_idx = np.where(stock_mask > 0.5)[0]                 # 只考虑有效股票

    valid_scores = scores[valid_idx]
    valid_returns = true_returns[valid_idx]

    k = min(max_stocks, len(valid_scores))
    if k == 0:
        return 0.0

    # argsort 默认升序，取 [-k:] 得到最高的 k 个
    top_k_local = np.argsort(valid_scores)[-k:]
    selected_returns = valid_returns[top_k_local]

    return float(np.mean(selected_returns))


def backtest(model: PortfolioPredictor, loader: DataLoader) -> dict:
    """
    在测试集上做完整回测。

    对测试集中每个时间窗口：
      1. 用模型给所有股票打分
      2. 选 top-5 等权持有
      3. 记录实际收益

    汇总统输出：
      - mean_return  : 平均每周收益率
      - std_return   : 收益的标准差
      - sharpe       : 周度夏普比率（简化版，无风险利率假设为 0）
      - win_rate     : 正收益的比例（>50% 说明多数时候赚钱）
    """
    model.eval()
    portfolio_returns = []

    with torch.no_grad():
        for batch in loader:
            Xb, yb, mb = batch[0], batch[1], batch[2]
            Xb = Xb.to(DEVICE)
            scores, _ = model(Xb, mb.to(DEVICE))
            scores = scores.cpu().numpy()
            yb = yb.cpu().numpy()
            mb = mb.cpu().numpy()

            for i in range(len(scores)):
                ret = portfolio_return(scores[i], yb[i], mb[i])
                portfolio_returns.append(ret)

    arr = np.array(portfolio_returns)
    return {
        "mean_return": float(np.mean(arr)),
        "std_return": float(np.std(arr)),
        "sharpe": float(np.mean(arr) / (np.std(arr) + 1e-9)),  # +1e-9 防止除零
        "win_rate": float((arr > 0).mean()),
        "n_samples": len(arr),
    }


# =============================================================================
# 强化学习微调 — 直接优化组合收益率
# =============================================================================

def rl_finetune(model, train_loader, val_loader, test_loader, stock_ids,
                epochs=20, lr=1e-5, rank_labels=False):
    """
    强化学习微调：用 Policy Gradient (REINFORCE) 直接最大化组合收益。

    原理：
      监督学习优化的是"排序准确率"，RL 优化的是"组合收益率"。
      后者才是竞赛真正关心的指标。

    每步：
      1. 模型对当前 batch 打分 → scores
      2. 选 top-5 股票 + 分配权重 → portfolio
      3. 用真实收益计算组合收益 → reward
      4. Policy Gradient: loss = -reward * log_prob(selected stocks)

    Args:
        model: 已训练的模型
        train_loader: 训练数据
        epochs: RL 微调轮数
        lr: RL 学习率（通常比监督学习小）
    """
    from predict import select_diverse_portfolio

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    print(f"\n{'='*60}")
    print(f"RL Fine-tuning: {epochs} epochs")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        total_reward = 0.0
        n_batches = 0

        for batch in train_loader:
            Xb, yb, mb = batch[0], batch[1], batch[2]
            Xb = Xb.to(DEVICE, dtype=torch.float32)
            yb = yb.to(DEVICE)
            mb = mb.to(DEVICE)

            # 前向：获取分数
            scores, _ = model(Xb, mb)

            # 对 batch 内每个样本计算策略梯度
            policy_loss = 0.0
            batch_reward = 0.0
            n_valid_samples = 0

            for i in range(Xb.size(0)):
                s = scores[i]       # (N,)
                t = yb[i]           # (N,) 真实收益
                m = mb[i]           # (N,) 有效掩码

                valid_idx = (m > 0.5).nonzero(as_tuple=True)[0]
                if len(valid_idx) < 5:
                    continue

                # 模型选股（用分数 + embedding）
                with torch.no_grad():
                    emb = model.encode_stocks(
                        Xb[i:i+1], mb[i:i+1]
                    ).squeeze(0).cpu()

                # 选 top-5（用模型分数）
                sel = select_diverse_portfolio(
                    s.detach().cpu().numpy(),
                    stock_ids,
                    emb,
                    max_stocks=5, top_k_candidates=30,
                )
                sel_indices = [stock_ids.index(sid) for sid, _ in sel]

                if len(sel_indices) < 2:
                    continue

                # 组合收益（等权）
                portfolio_return = t[sel_indices].mean()

                # 策略梯度：最大化组合收益
                # log_prob = log_softmax(scores)[selected_stocks] 之和
                log_probs = torch.log_softmax(s, dim=0)
                selected_log_prob = log_probs[sel_indices].mean()

                # gradient = reward * grad(log_prob)
                policy_loss = policy_loss - portfolio_return * selected_log_prob

                batch_reward += portfolio_return.item()
                n_valid_samples += 1

            if n_valid_samples == 0:
                continue

            policy_loss = policy_loss / n_valid_samples

            optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_reward += batch_reward / n_valid_samples
            n_batches += 1

        avg_reward = total_reward / max(n_batches, 1)
        print(f"  RL Epoch {epoch:3d}: avg_portfolio_return={avg_reward:.6f}")

    print(f"\n  RL fine-tuning done.")
    return model


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="训练基于注意力机制的股票组合预测模型"
    )
    parser.add_argument("--download", action="store_true",
                        help="强制重新下载所有数据（清除缓存）")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS,
                        help=f"训练轮数（默认: {N_EPOCHS}）")
    parser.add_argument("--loss", choices=["lambdarank", "pairwise", "listnet", "topk_listnet", "pcc"],
                        default="listnet",
                        help="损失函数: lambdarank/pairwise/listnet(推荐)/topk_listnet")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"学习率（默认: {LR}）")
    parser.add_argument("--rank-labels", action="store_true",
                        help="将收益率转为百分位数排名作为训练目标（推荐用于 listnet）")
    parser.add_argument("--load-pretrained", type=str, default=None,
                        help="加载预训练编码器权重（来自 pretrain.py）")
    parser.add_argument("--rl-finetune", action="store_true",
                        help="强化学习微调：直接优化组合收益率（需先有训练好的模型）")
    parser.add_argument("--rl-epochs", type=int, default=20,
                        help="RL 微调轮数")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（保证可复现 + 多样性）")
    parser.add_argument("--output", type=str, default=None,
                        help="模型保存路径（默认: models/portfolio_model.pt）")
    parser.add_argument("--big-model", action="store_true",
                        help="使用大模型（D_MODEL=256, 4层Transformer, 3层GRU）")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"DataLoader 工作线程数（默认: {N_WORKERS}）")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="批次大小（覆盖 config 默认值）")
    parser.add_argument("--no-rank-labels", action="store_true",
                        help="禁用 rank labels（使用原始收益率）")
    parser.add_argument("--decay", type=float, default=0.0,
                        help="时间衰减系数: 0=等权, 1=中度衰减, 2=强衰减, 3=极强衰减")
    parser.add_argument("--curriculum", action="store_true",
                        help="课程微调：主训练后用最近数据精调")
    parser.add_argument("--augment", action="store_true",
                        help="数据增强：训练时加高斯噪声+时间步掩码")
    parser.add_argument("--tscv", action="store_true",
                        help="时序交叉验证：4折训练，保存4个模型")
    args = parser.parse_args()

    # ---- 可选：强制重新下载 ----
    if args.download:
        import shutil
        from config import RAW_DIR
        shutil.rmtree(RAW_DIR, ignore_errors=True)
        RAW_DIR.mkdir(exist_ok=True)
        print("[train] Cleared data cache, will re-download.")

    # ==================================================================
    # Step 1-2: 数据获取（沪深300 + 中证500 = ~800 只股票）
    # ==================================================================
    print("=" * 60)
    print("Step 1: Load official THU-BDC data")
    stock_ids = get_official_stock_ids()
    panel = build_panel_from_official(stock_ids)

    # ==================================================================
    # Step 2: 特征工程
    # ==================================================================
    print("\nStep 2: Feature engineering")
    panel = engineer_features(panel, stock_ids)

    # ==================================================================
    # Step 3: 构造训练样本
    # ==================================================================
    print("\nStep 3: Build training windows")

    # 从 Panel 中动态提取特征列名（跨市场指数特征在 engineer_features 中添加）
    EXCLUDE_COLS = {"open", "high", "low", "close", "preclose", "volume",
                    "amount", "adjustflag", "turn", "tradestatus",
                    "pctChg", "peTTM", "pbMRQ", "market_ret",
                    "stock_id", "index_name", "vol_ma5", "vol_ma20",
                    "turn_ma5", "turn_ma20",
                    "amplitude", "change",
                    "roe_ttm", "np_margin", "gp_margin", "eps_ttm",
                    "current_ratio", "debt_to_asset", "profit_yoy", "equity_yoy",
                    "nn_score"}
    feature_cols = [c for c in panel.columns if c not in EXCLUDE_COLS]

    X, y, mask, dates = make_window_samples(panel, stock_ids)
    n_features = X.shape[-1]
    print(f"  X: {X.shape}, y: {y.shape}, mask: {mask.shape}")
    print(f"  Feature cols ({len(feature_cols)}): {feature_cols[:8]}... +{len(feature_cols)-8} more")

    # 缓存处理后的数据（predict.py 可以直接加载）
    processed = {"X": X, "y": y, "mask": mask,
                 "dates": dates, "stock_ids": stock_ids}
    with open(PROCESSED_DIR / "samples.pkl", "wb") as f:
        pickle.dump(processed, f)
    print(f"  Cached to {PROCESSED_DIR / 'samples.pkl'}")

    # ==================================================================
    # Step 4: 划分数据 & 训练
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 4: Training")

    # 数据增强开关
    global _AUGMENT_ENABLED
    _AUGMENT_ENABLED = args.augment
    if args.augment:
        print("  [train] Data augmentation enabled (noise=0.05, mask=0.1)")

    # 设置随机种子
    if args.seed is not None:
        import random
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"  Seed: {args.seed}")

    # 高性能模型
    use_big = args.big_model
    if use_big:
        print(f"  [train] BIG MODEL: d_model={BIG_D_MODEL}, "
              f"n_transformer={BIG_N_TRANSFORMER_LAYERS}, "
              f"n_gru={BIG_N_GRU_LAYERS}")

    # 批次大小
    batch_size = args.batch_size or BATCH_SIZE
    print(f"  Batch size: {batch_size}")

    # rank_labels: 默认对 listnet 启用，除非 --no-rank-labels
    use_rank = not args.no_rank_labels if args.loss in ("listnet", "topk_listnet") else args.rank_labels

    # ==================================================================
    # 时序交叉验证（4折训练，每折保存一个模型）
    # ==================================================================
    if args.tscv:
        print("\n  [tscv] Time-series cross validation (4 folds) ...")
        folds = make_tscv_dataloaders(
            X, y, mask, n_folds=4, gap=5, val_size=50,
            batch_size=batch_size, num_workers=args.workers,
            decay_rate=args.decay,
        )
        base_name = Path(args.output).stem if args.output else "portfolio_model"
        out_dir = Path(args.output).parent if args.output else MODEL_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        for k, (tr_ld, va_ld) in enumerate(folds):
            fold_seed = (args.seed or 42) + k * 100
            torch.manual_seed(fold_seed)
            np.random.seed(fold_seed)

            print(f"\n  {'='*50}")
            print(f"  [tscv] Fold {k+1}/{len(folds)}: train={len(tr_ld.dataset)} val={len(va_ld.dataset)}")
            print(f"  {'='*50}")

            fold_model = PortfolioPredictor(n_features=n_features, use_attention=USE_ATTENTION, use_market_gate=USE_MARKET_GATE)
            if args.load_pretrained:
                pt_path = Path(args.load_pretrained)
                if pt_path.exists():
                    pt_state = torch.load(pt_path, map_location=DEVICE, weights_only=False)
                    fold_model.time_encoder.load_state_dict(pt_state["time_encoder"])
                    fold_model.transformers.load_state_dict(pt_state["transformers"])

            fold_model = train_model(
                fold_model, tr_ld, va_ld,
                epochs=args.epochs, lr=args.lr, loss_name=args.loss,
                use_rank_labels=use_rank,
            )

            fold_path = out_dir / f"{base_name}_fold{k+1}.pt"
            norm_stats = get_norm_stats()
            torch.save({
                "model_state_dict": fold_model.state_dict(),
                "feature_cols": feature_cols,
                "stock_ids": stock_ids,
                "feat_mean": np.array(norm_stats[0]) if norm_stats else None,
                "feat_std": np.array(norm_stats[1]) if norm_stats else None,
                "config": {"n_features": n_features, "d_model": D_MODEL,
                           "n_transformer_layers": 2, "n_heads": N_HEADS,
                           "n_gru_layers": N_GRU_LAYERS, "d_ff": D_FF,
                           "use_attention": USE_ATTENTION,
                           "use_market_gate": USE_MARKET_GATE},
            }, fold_path)
            print(f"  [tscv] Fold {k+1} saved to {fold_path}")

        print(f"\n  [tscv] All {len(folds)} folds trained!")
        return

    # DataLoader（多进程加载，减少 CPU 瓶颈）
    train_loader, val_loader, test_loader = make_dataloaders(
        X, y, mask, batch_size=batch_size, num_workers=args.workers,
        decay_rate=args.decay,
    )
    if args.workers > 0:
        print(f"  DataLoader workers: {args.workers}")

    # 创建模型（big model 或标准）
    if use_big:
        model = PortfolioPredictor(
            n_features=n_features,
            d_model=BIG_D_MODEL, n_heads=BIG_N_HEADS,
            n_gru_layers=BIG_N_GRU_LAYERS,
            n_transformer_layers=BIG_N_TRANSFORMER_LAYERS,
            d_ff=BIG_D_FF,
            use_attention=USE_ATTENTION,
            use_market_gate=USE_MARKET_GATE,
        )
    else:
        model = PortfolioPredictor(n_features=n_features, use_attention=USE_ATTENTION,
                                   use_market_gate=USE_MARKET_GATE)

    # 加载预训练编码器权重（如果有）
    if args.load_pretrained:
        pt_path = Path(args.load_pretrained)
        if pt_path.exists():
            pt_state = torch.load(pt_path, map_location=DEVICE, weights_only=False)
            model.time_encoder.load_state_dict(pt_state["time_encoder"])
            model.transformers.load_state_dict(pt_state["transformers"])
            print(f"  Loaded pretrained encoder from {pt_path}")
        else:
            print(f"  [WARN] Pretrained weights not found: {pt_path}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    print(f"  Device: {DEVICE}")
    print(f"  Loss: {args.loss}")

    # rank_labels: 默认对 listnet 启用，除非 --no-rank-labels
    model = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, loss_name=args.loss,
        use_rank_labels=use_rank,
        curriculum=args.curriculum,
    )

    # ---- RL 微调（可选） ----
    if args.rl_finetune:
        print("\n" + "=" * 60)
        print("RL Fine-tuning Phase")
        print("=" * 60)
        model = rl_finetune(
            model, train_loader, val_loader, test_loader, stock_ids,
            epochs=args.rl_epochs, lr=args.lr * 0.1,
            rank_labels=use_rank,
        )

    # ==================================================================
    # Step 5: 评估 & 回测
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 5: Evaluation")

    # 测试集指标
    test_metrics = evaluate(model, test_loader, lambdarank_loss)
    print(f"  Test loss:        {test_metrics['loss']:.4f}")
    print(f"  Test pair acc:    {test_metrics['pair_accuracy']:.3f} "
          f"(random=0.500, perfect=1.000)")

    # 模拟交易回测
    bt = backtest(model, test_loader)
    print(f"\n  ── Backtest (top-{MAX_STOCKS} equal-weight) ──")
    print(f"  Mean weekly return:  {bt['mean_return']:.6f} "
          f"({'POSITIVE' if bt['mean_return'] > 0 else 'NEGATIVE'})")
    print(f"  Std weekly return:   {bt['std_return']:.6f}")
    print(f"  Weekly Sharpe:       {bt['sharpe']:.4f}")
    print(f"  Win rate:            {bt['win_rate']:.3f} "
          f"({bt['win_rate']*100:.1f}% weeks profitable)")
    print(f"  Test samples:        {bt['n_samples']}")

    # ==================================================================
    # Step 6: 保存模型
    # ==================================================================
    model_path = Path(args.output) if args.output else MODEL_DIR / "portfolio_model.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    norm_stats = get_norm_stats()
    feat_mean = np.array(norm_stats[0]) if norm_stats is not None else None
    feat_std  = np.array(norm_stats[1]) if norm_stats is not None else None
    # 模型架构参数（确保大模型能正确加载）
    actual_d_model = model.d_model if hasattr(model, 'd_model') else D_MODEL
    # 从模型结构中提取实际参数
    first_tf = model.transformers[0] if model.transformers else None
    actual_n_heads = first_tf.self_attn.num_heads if first_tf else N_HEADS
    actual_d_ff = first_tf.ffn[0].out_features if first_tf else D_FF
    actual_n_gru_layers = model.time_encoder.gru.num_layers
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "stock_ids": stock_ids,
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "config": {
            "n_features": n_features,
            "d_model": actual_d_model,
            "n_transformer_layers": len(model.transformers),
            "n_heads": actual_n_heads,
            "n_gru_layers": actual_n_gru_layers,
            "d_ff": actual_d_ff,
            "use_attention": USE_ATTENTION,
            "use_market_gate": USE_MARKET_GATE,
        },
    }, model_path)
    print(f"\n  Model saved to {model_path}")


if __name__ == "__main__":
    main()
