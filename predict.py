"""
=============================================================================
  预测脚本 —— 使用训练好的模型生成 result.csv 提交文件
=============================================================================

预测流程：

  1. 加载训练时保存的模型权重和配置
  2. 获取最新数据，构造最新的一个时间窗口
  3. 模型前向传播，得到每只股票的预测分数
  4. 用 Transformer embedding 做贪心多样化选股
  5. 等权分配，输出 result.csv

多样化的贪心选股算法（select_diverse_portfolio）：

  目标是选出既"分数高"又"彼此不太像"的 5 只股票。

  算法：
    1. 取模型打分最高的 K 只（默认 30 只）进入候选池
    2. 第 1 只：选分数最高的
    3. 后续每只：
       - 计算该候选与已选股票在 embedding 空间的 cosine similarity
       - 取最大相似度（即它与"已选中最像的那只"有多像）
       - adjusted_score = original_score - similarity * original_score * 0.5
       - 选 adjusted_score 最高的
    4. 重复直到选满 5 只

  为什么用 embedding 相似度而不是行业标签？
    - embedding 是模型"内化的关联认知"，可能比行业标签更精细
    - 模型可能发现跨行业的联动（如"白酒"和"消费"的关联）
    - 不依赖外部行业分类数据

使用方法：
  python predict.py                            # 默认输出到 result.csv
  python predict.py --model models/best.pt     # 指定模型路径
  python predict.py --output my_submission.csv  # 自定义输出路径

=============================================================================
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import (
    MODEL_DIR, PROCESSED_DIR, SUBMISSION_PATH, DEVICE,
    LOOKBACK_DAYS, PREDICT_HORIZON, MAX_STOCKS,
    TOP_K_CANDIDATES, TEMPERATURE, D_MODEL,
    N_TRANSFORMER_LAYERS, N_GRU_LAYERS, D_FF,
)
from data_loader import build_panel_from_official
from features import engineer_features, make_window_samples
from model import PortfolioPredictor


# =============================================================================
# 多样化选股算法
# =============================================================================

def select_diverse_portfolio(
    scores: np.ndarray,
    stock_ids: list[str],
    embeddings: torch.Tensor | None = None,
    max_stocks: int = MAX_STOCKS,
    top_k_candidates: int = TOP_K_CANDIDATES,
    temperature: float = TEMPERATURE,
) -> list[tuple[str, float]]:
    """
    从高分股票中选出一个多样化的组合。

    Args:
        scores            : (N,)  — 模型给每只股票的预测分，越高越好
        stock_ids         : [str]  — 所有股票代码
        embeddings        : (N, d_model) — 股票的 Transformer embedding
                            用于计算相似度、实现多样化选股
        max_stocks        : int  — 最多选几只
        top_k_candidates  : int  — 候选池大小（从分最高的 K 只中选）
        temperature       : float — 分数缩放参数（预留，当前用等权）

    Returns:
        [(stock_id, weight), ...]  — 选中的股票和权重
    """
    n = len(stock_ids)

    # ---- 1. 按分数排序，取候选池 ----
    order = np.argsort(scores)[::-1]
    candidates = order[:min(top_k_candidates, n)]

    # ---- 2. 贪心多样选股 ----
    # 第1只：选分数最高的（无条件）
    first = candidates[0]
    selected_indices = [first]
    remaining = list(candidates[1:])

    # 后续：每次选"adjusted_score = score - 相似度惩罚"最高的
    DIV_PENALTY = 0.5  # 多样性惩罚系数

    while len(selected_indices) < max_stocks and remaining:
        best_adj = -float("inf")
        best_idx = None

        for idx in remaining:
            # embedding 余弦相似度
            sim = max(
                float(torch.cosine_similarity(
                    embeddings[idx].unsqueeze(0),
                    embeddings[s].unsqueeze(0), dim=-1,
                )) for s in selected_indices
            ) if embeddings is not None else 0.0

            # 调整分数：越相似折扣越大，但高分股仍能入选
            adj = scores[idx] - sim * abs(scores[idx]) * DIV_PENALTY

            if adj > best_adj:
                best_adj = adj
                best_idx = idx

        if best_idx is None:
            break
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    # ---- 3. 等权分配 ----
    # 对于 top-5 选股问题，等权通常不会太差：
    #   入选的已经是模型最看好的股票，精确的权重优化对 4 天收益影响有限
    k = len(selected_indices)
    weights = np.ones(k) / k
    selected = [(stock_ids[i], float(w)) for i, w in zip(selected_indices, weights)]

    return selected


# =============================================================================
# 生成提交文件
# =============================================================================

def generate_submission(
    model: PortfolioPredictor,
    stock_ids: list[str],
    panel: pd.DataFrame,
    output_path: Path = SUBMISSION_PATH,
) -> None:
    """
    用训练好的模型生成 result.csv 提交文件。

    步骤：
      1. 从 Panel 构造最新的样本窗口
      2. 模型打分
      3. 多样化选股 + 等权
      4. 写入 CSV
    """
    model = model.to(DEVICE)
    model.eval()

    # ---- 1. 构造最新窗口 ----
    print("[predict] Building latest feature window ...")
    X, y, mask, dates = make_window_samples(panel, stock_ids)

    if len(X) == 0:
        raise RuntimeError(
            "No valid samples could be constructed. "
            "Check that data is available and dates are recent."
        )

    # 取最后一个样本（最新时间点）
    X_latest = torch.from_numpy(X[-1:]).to(DEVICE, dtype=torch.float32)
    mask_latest = torch.from_numpy(mask[-1:]).to(DEVICE)       # (1, N)

    print(f"[predict] Latest sample date: {dates[-1]}")

    # ---- 2. 模型打分 ----
    with torch.no_grad():
        # 获取分数
        scores, attn_weights = model(X_latest, mask_latest)
        scores = scores.squeeze(0).cpu().numpy()               # (N,)

        # 获取 embedding（用于多样化选股）
        embeddings = model.encode_stocks(X_latest, mask_latest)
        embeddings = embeddings.squeeze(0)                      # (N, d_model)

        mask_np = mask_latest.squeeze(0).cpu().numpy()          # (N,)

    valid_idx = np.where(mask_np > 0.5)[0]
    print(f"[predict] Valid stocks: {len(valid_idx)} / {len(stock_ids)}")

    # ---- 3. 多样化选股 ----
    selected = select_diverse_portfolio(scores, stock_ids, embeddings.cpu())

    # ---- 4. 输出 CSV ----
    df = pd.DataFrame(selected, columns=["stock_id", "weight"])

    # 验证输出格式
    total_weight = df["weight"].sum()
    n_stocks = len(df)
    print(f"[predict] Selected {n_stocks} stocks, total weight = {total_weight:.4f}")

    if n_stocks > MAX_STOCKS:
        print(f"  WARNING: {n_stocks} > {MAX_STOCKS} max stocks!")
    if total_weight > 1.0001:
        print(f"  WARNING: total weight {total_weight:.4f} > 1.0!")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[predict] Submission saved to {output_path}")
    print(df.to_string(index=False))


# =============================================================================
# 主入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="使用训练好的模型生成股票组合提交文件"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="模型权重路径（默认: models/portfolio_model.pt）")
    parser.add_argument("--output", type=str, default=str(SUBMISSION_PATH),
                        help="输出 CSV 路径（默认: result.csv）")
    args = parser.parse_args()

    # ---- 加载模型 ----
    model_path = args.model or str(MODEL_DIR / "portfolio_model.pt")
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please run 'python train.py' first to train a model."
        )

    print(f"[predict] Loading model from {model_path}")
    # weights_only=False: PyTorch 2.6+ 默认开启，但我们保存了 dict 需要关掉
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    cfg = checkpoint["config"]
    n_features = cfg["n_features"]
    stock_ids = checkpoint["stock_ids"]

    # 重建模型架构（兼容大模型）
    model = PortfolioPredictor(
        n_features=n_features,
        d_model=cfg.get("d_model", D_MODEL),
        n_transformer_layers=cfg.get("n_transformer_layers", N_TRANSFORMER_LAYERS),
        n_gru_layers=cfg.get("n_gru_layers", N_GRU_LAYERS),
        d_ff=cfg.get("d_ff", D_FF),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[predict] Model loaded: {n_params:,} params")

    # ---- 加载官方数据（THU-BDC 格式） ----
    print("[predict] Loading official THU-BDC data ...")
    panel = build_panel_from_official(stock_ids)
    panel = engineer_features(panel, stock_ids)

    # ---- 应用标准化参数 ----
    feat_mean = checkpoint.get("feat_mean", None)
    feat_std = checkpoint.get("feat_std", None)

    X, y, mask, dates = make_window_samples(panel, stock_ids, normalize=False)
    if feat_mean is not None and feat_std is not None:
        X = (X - feat_mean) / feat_std
        print("[predict] Applied saved normalization stats")
    elif len(X) > 0:
        m = X.mean(axis=(0, 1, 2), keepdims=True)
        s = X.std(axis=(0, 1, 2), keepdims=True)
        X = (X - m) / (s + 1e-8)
        print("[predict] WARNING: Applied data-level normalization")

    # ---- 预测最新样本 ----
    X_latest = torch.from_numpy(X[-1:]).to(DEVICE)
    mask_latest = torch.from_numpy(mask[-1:]).to(DEVICE)
    print(f"[predict] Latest sample date: {dates[-1]}")

    model.eval()
    with torch.no_grad():
        scores, attn_weights = model(X_latest, mask_latest)
        scores = scores.squeeze(0).cpu().numpy()
        embeddings = model.encode_stocks(X_latest, mask_latest)
        embeddings = embeddings.squeeze(0).cpu()

    mask_np = mask_latest.squeeze(0).cpu().numpy()
    valid_idx = np.where(mask_np > 0.5)[0]
    print(f"[predict] Valid stocks: {len(valid_idx)} / {len(stock_ids)}")

    # ---- 多样化选股 ----
    selected = select_diverse_portfolio(scores, stock_ids, embeddings)
    df = pd.DataFrame(selected, columns=["stock_id", "weight"])
    # 转换为整数股票代码（匹配官方 test.csv 格式：1,2,3...）
    df["stock_id"] = df["stock_id"].astype(int)
    total_weight = df["weight"].sum()
    n_stocks = len(df)
    print(f"[predict] Selected {n_stocks} stocks, total weight = {total_weight:.4f}")
    output_path = Path(args.output)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[predict] Submission saved to {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
