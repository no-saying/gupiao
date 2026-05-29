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
    MODEL_DIR, PROCESSED_DIR, DEVICE,
    BATCH_SIZE, N_EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP,
    LR_PATIENCE, EARLY_STOP_PATIENCE, VAL_RATIO, TEST_RATIO,
    LOOKBACK_DAYS, PREDICT_HORIZON, STEP_DAYS,
    MAX_STOCKS, TOP_K_CANDIDATES, TEMPERATURE,
)
from data_loader import fetch_csi300_stocks, build_panel
from features import engineer_features, make_window_samples
from model import PortfolioPredictor, lambdarank_loss, pairwise_ranking_loss


# =============================================================================
# 数据加载器创建：按时间顺序切分
# =============================================================================

def make_dataloaders(
    X: np.ndarray, y: np.ndarray, mask: np.ndarray,
    batch_size: int = BATCH_SIZE,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
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
    test_start = int(n * (1 - test_ratio))                     # 测试集起始位置
    val_start = int(n * (1 - test_ratio - val_ratio))          # 验证集起始位置

    X_train, y_train, m_train = X[:val_start], y[:val_start], mask[:val_start]
    X_val,   y_val,   m_val   = X[val_start:test_start], y[val_start:test_start], mask[val_start:test_start]
    X_test,  y_test,  m_test  = X[test_start:], y[test_start:], mask[test_start:]

    def to_loader(Xa, ya, ma, shuffle=False):
        """辅助函数：numpy → TensorDataset → DataLoader"""
        ds = TensorDataset(
            torch.from_numpy(Xa),
            torch.from_numpy(ya),
            torch.from_numpy(ma),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    print(f"[train] Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # 训练集 shuffle=True: 每个 epoch 看到不同顺序的数据，防止模型记住顺序
    return to_loader(X_train, y_train, m_train, shuffle=True), \
           to_loader(X_val,   y_val,   m_val),                   \
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

    with torch.no_grad():                                     # 不计算梯度，节省内存
        for Xb, yb, mb in loader:
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

def train_epoch(model, loader, optimizer, loss_fn, epoch: int) -> float:
    """
    执行一个 epoch 的训练。

    每个 epoch 遍历全部训练样本一次，更新模型参数。

    使用梯度裁剪 (Gradient Clipping) 防止梯度爆炸：
      当梯度的 L2 范数超过阈值 (GRAD_CLIP=1.0) 时，
      等比例缩小所有梯度，避免参数更新过大。
      这对 RNN/Transformer 模型在长序列上尤其重要。
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}")
    for Xb, yb, mb in pbar:
        Xb, yb, mb = Xb.to(DEVICE), yb.to(DEVICE), mb.to(DEVICE)

        # 标准训练循环
        optimizer.zero_grad()                                 # 清空梯度累积
        scores, _ = model(Xb, mb)                             # 前向传播
        loss = loss_fn(scores, yb, mb)                        # 计算损失
        loss.backward()                                       # 反向传播（计算梯度）
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # 梯度裁剪
        optimizer.step()                                      # 更新参数

        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

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
    # 相比 Adam，AdamW 的权重衰减直接作用在参数上而非梯度上，效果更好
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ReduceLROnPlateau: 验证损失 plateau 时自动降低学习率
    # factor=0.5 → 减半, patience=10 → 等 10 个 epoch
    # 这是训练深度模型常用的技巧：先用大学习率快速收敛，再减小学习率精调
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=LR_PATIENCE, verbose=True,
    )

    # 选择损失函数
    loss_fn = lambdarank_loss if loss_name == "lambdarank" else pairwise_ranking_loss

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, epoch)

        # ---- 验证 ----
        val_metrics = evaluate(model, val_loader, loss_fn)
        print(f"  train_loss={train_loss:.4f}  "
              f"val_loss={val_metrics['loss']:.4f}  "
              f"val_pair_acc={val_metrics['pair_accuracy']:.3f}")

        # ---- 学习率调度 ----
        scheduler.step(val_metrics["loss"])

        # ---- 模型保存 & Early Stopping ----
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            # 深拷贝最优权重（不能用 model.state_dict() 直接赋值，会被后续更新覆盖）
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  → New best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # 恢复最优权重
    model.load_state_dict(best_weights)
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
        for Xb, yb, mb in loader:
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
    parser.add_argument("--loss", choices=["lambdarank", "pairwise"],
                        default="lambdarank",
                        help="损失函数: lambdarank（推荐） 或 pairwise")
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"学习率（默认: {LR}）")
    args = parser.parse_args()

    # ---- 可选：强制重新下载 ----
    if args.download:
        import shutil
        from config import RAW_DIR
        shutil.rmtree(RAW_DIR, ignore_errors=True)
        RAW_DIR.mkdir(exist_ok=True)
        print("[train] Cleared data cache, will re-download.")

    # ==================================================================
    # Step 1-2: 数据获取
    # ==================================================================
    print("=" * 60)
    print("Step 1: Fetch CSI 300 stocks & daily data")
    stock_ids = fetch_csi300_stocks()
    panel = build_panel(stock_ids)

    # ==================================================================
    # Step 3: 特征工程
    # ==================================================================
    print("\nStep 2: Feature engineering")
    panel = engineer_features(panel)

    # ==================================================================
    # Step 3: 构造训练样本
    # ==================================================================
    print("\nStep 3: Build training windows")

    # 从 Panel 中动态提取特征列名（跨市场指数特征在 engineer_features 中添加）
    EXCLUDE_COLS = {"open", "high", "low", "close", "preclose", "volume",
                    "amount", "adjustflag", "turn", "tradestatus",
                    "pctChg", "peTTM", "pbMRQ", "market_ret",
                    "stock_id", "index_name", "vol_ma5", "vol_ma20",
                    "turn_ma5", "turn_ma20"}
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
    # Step 5: 划分数据 & 训练
    # ==================================================================
    print("\n" + "=" * 60)
    print("Step 4: Training")
    train_loader, val_loader, test_loader = make_dataloaders(X, y, mask)

    model = PortfolioPredictor(n_features=n_features)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    print(f"  Device: {DEVICE}")
    print(f"  Loss: {args.loss}")

    model = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr, loss_name=args.loss,
    )

    # ==================================================================
    # Step 6: 评估 & 回测
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
    # Step 7: 保存模型
    # ==================================================================
    model_path = MODEL_DIR / "portfolio_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "stock_ids": stock_ids,
        "config": {
            "n_features": n_features,
            "d_model": D_MODEL,
        },
    }, model_path)
    print(f"\n  Model saved to {model_path}")


if __name__ == "__main__":
    main()
