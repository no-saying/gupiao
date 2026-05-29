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
    TOP_K_CANDIDATES, TEMPERATURE,
)
from data_loader import fetch_csi300_stocks, build_panel
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
    order = np.argsort(scores)[::-1]                            # 降序排列
    candidates = order[:min(top_k_candidates, n)]               # 取前 K 个

    # ---- 2. 贪心多样选股 ----
    if embeddings is not None:
        selected_indices = []
        remaining = list(candidates)

        # 策略：分数最高的直接入选（无论它是什么行业）
        first = remaining.pop(0)
        selected_indices.append(first)

        # 后续每个位置：在"分数"和"不像已选"之间平衡
        while len(selected_indices) < max_stocks and remaining:
            best_score = -np.inf
            best_idx = None

            for idx in remaining:
                # 计算与所有已选股票的最大余弦相似度
                # sim = 1.0 表示几乎完全一样（同行业龙头股可能很相似）
                # sim = 0.0 表示完全不相关（不同行业）
                sim_to_selected = max(
                    float(torch.cosine_similarity(
                        embeddings[idx].unsqueeze(0),
                        embeddings[s].unsqueeze(0),
                        dim=-1,
                    ))
                    for s in selected_indices
                )

                # 如果已选股票中有一个和它非常像（sim > 0.9），
                # 那选它的边际价值就很小 → 调整后分数大幅降低
                # 0.5 是多样性惩罚系数，可调
                adjusted_score = scores[idx] - sim_to_selected * scores[idx] * 0.5

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_idx = idx

            if best_idx is None:
                break
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
    else:
        # 不做多样化：直接取分数最高的
        selected_indices = list(candidates[:max_stocks])

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
    X_latest = torch.from_numpy(X[-1:]).to(DEVICE)             # (1, N, T, F)
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

    n_features = checkpoint["config"]["n_features"]
    stock_ids = checkpoint["stock_ids"]

    # 重建模型架构并加载权重
    model = PortfolioPredictor(n_features=n_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[predict] Model loaded: {n_params:,} params")

    # ---- 加载数据 ----
    # 优先从缓存加载处理后的数据，否则从头构建
    processed_path = PROCESSED_DIR / "samples.pkl"
    if processed_path.exists():
        # 缓存命中：直接读取
        with open(processed_path, "rb") as f:
            data = pickle.load(f)
        print(f"[predict] Loaded processed data: {data['X'].shape[0]} samples")
        # 用已有数据构建 Panel
        panel = build_panel(stock_ids)
        panel = engineer_features(panel)
    else:
        # 从头构建（可能要花几分钟）
        print("[predict] No cached data, building from scratch ...")
        print("[predict] This may take a few minutes ...")
        panel = build_panel(stock_ids)
        panel = engineer_features(panel)

    # ---- 生成提交 ----
    generate_submission(model, stock_ids, panel, Path(args.output))


if __name__ == "__main__":
    main()
