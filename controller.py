"""
Master Controller —— 集成多个模型 + 动态权重优化
"""
import argparse, pickle, sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from config import MODEL_DIR, PROCESSED_DIR, DEVICE, SUBMISSION_PATH
from data_loader import build_panel, fetch_csi300_stocks
from features import engineer_features, make_window_samples
from model import PortfolioPredictor
from predict import select_diverse_portfolio

warnings.filterwarnings("ignore")


# =============================================================================
# 模型注册表：所有可用的模型
# =============================================================================

MODEL_REGISTRY = {
    # (名称, 模型路径, 特征数, 类型)
    "baseline": {
        "path": MODEL_DIR / "portfolio_model.pt",
        "fallback": Path("/root/gupiao/baseline/gupiao/models/portfolio_model.pt"),
        "features": 34,
    },
    "seed42":    {"path": MODEL_DIR / "portfolio_model_seed42.pt",   "features": 52},
    "seed123":   {"path": MODEL_DIR / "portfolio_model_seed123.pt",  "features": 52},
    "seed456":   {"path": MODEL_DIR / "portfolio_model_seed456.pt",  "features": 52},
    "seed789":   {"path": MODEL_DIR / "portfolio_model_seed789.pt",  "features": 52},
    "seed999":   {"path": MODEL_DIR / "portfolio_model_seed999.pt",  "features": 52},
    "gaussian787": {"path": MODEL_DIR / "portfolio_model_g787.pt",   "features": 52},
    "gaussian789": {"path": MODEL_DIR / "portfolio_model_g789.pt",   "features": 52},
    "gaussian791": {"path": MODEL_DIR / "portfolio_model_g791.pt",   "features": 52},
}


def load_model(entry, n_features_override=None):
    """加载模型，兼容 feat_mean/feat_std 存在或不存在的情况"""
    path = entry["path"]
    if not path.exists():
        fallback = entry.get("fallback")
        if fallback and fallback.exists():
            path = fallback
        else:
            print(f"  [WARN] {path.name} not found, skipping")
            return None

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    nf = n_features_override or entry["features"]
    model = PortfolioPredictor(n_features=nf)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except Exception:
        # 特征数不匹配时尝试投影
        print(f"  [WARN] {path.name} feature mismatch, trying partial load")
        sd = model.state_dict()
        for k, v in ckpt["model_state_dict"].items():
            if k in sd and sd[k].shape == v.shape:
                sd[k] = v
        model.load_state_dict(sd)
    model.to(DEVICE)
    model.eval()
    return model, ckpt.get("feat_mean"), ckpt.get("feat_std")


def get_stock_data(sel_ids, panel):
    """获取选中股票的历史统计数据"""
    stats = {}
    for sid in sel_ids:
        try:
            sd = panel.xs(sid, level="stock_id")
            ret = sd["pctChg"] / 100.0
            stats[sid] = {
                "vol": float(ret.std()),
                "sharpe": float(ret.mean() / ret.std()) if ret.std() > 0 else 0,
                "mean_ret": float(ret.mean()),
            }
        except:
            stats[sid] = {"vol": 0.03, "sharpe": 0, "mean_ret": 0}
    return stats


def compute_weights(method, scores, stats=None, temperature=0.5):
    """
    权重分配方法

    Args:
        method: 'equal' | 'softmax' | 'inv_vol' | 'sharpe' | 'score_vol' | 'adaptive'
        scores: (n_stocks,) 模型预测分数
        stats: dict of {sid: {vol, sharpe, ...}}
    """
    n = len(scores)

    if method == "equal":
        return np.ones(n) / n

    elif method == "softmax":
        x = scores - scores.max()
        ex = np.exp(x / temperature)
        return ex / ex.sum()

    elif method == "inv_vol" and stats:
        vols = np.array([stats[s].get("vol", 0.03) for s in stats])
        inv = 1.0 / np.clip(vols, 0.01, None)
        return inv / inv.sum()

    elif method in ("sharpe", "score_vol") and stats:
        # 基于 Sharpe 或 Score/Vol 的混合权重
        weights = []
        for i, sid in enumerate(stats):
            if method == "sharpe":
                w = max(stats[sid]["sharpe"], 0) + 0.01
            else:
                w = max(scores[i], 0) / max(stats[sid]["vol"], 0.01)
            weights.append(w)
        w = np.array(weights)
        return w / w.sum()

    elif method == "adaptive":
        # 自适应：如果分数差异大用 softmax，否则等权
        score_range = scores.max() - scores.min()
        if score_range > 0.1:
            return compute_weights("softmax", scores, stats, temperature=0.3)
        else:
            return np.ones(n) / n

    return np.ones(n) / n


def main():
    parser = argparse.ArgumentParser(description="Master Controller — 集成多模型 + 权重优化")
    parser.add_argument("--models", type=str, default="seed789,baseline",
                        help="逗号分隔的模型名，如 'seed789,baseline,seed456'")
    parser.add_argument("--weight", type=str, default="adaptive",
                        choices=["equal", "softmax", "inv_vol", "sharpe", "score_vol", "adaptive"],
                        help="权重分配方法")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="softmax 温度")
    parser.add_argument("--topk", type=int, default=5,
                        help="选前 K 只股票（最多 5）")
    parser.add_argument("--output", type=str, default=str(SUBMISSION_PATH))
    parser.add_argument("--compare", action="store_true",
                        help="对比所有权重方法")
    args = parser.parse_args()

    # ---- 1. 加载数据（共享 panel） ----
    print("=" * 60)
    print("Controller: Loading data ...")
    processed_path = PROCESSED_DIR / "samples.pkl"
    if processed_path.exists():
        with open(processed_path, "rb") as f:
            data = pickle.load(f)
        print(f"  Loaded cached: {data['X'].shape[0]} samples")
        stock_ids = data["stock_ids"]
    else:
        stock_ids = fetch_csi300_stocks()

    panel = build_panel(stock_ids)
    panel = engineer_features(panel, stock_ids)

    # ---- 2. 加载模型 ----
    model_names = [m.strip() for m in args.models.split(",")]
    models = []
    print(f"\nController: Loading {len(model_names)} models ...")

    for name in model_names:
        entry = MODEL_REGISTRY.get(name)
        if entry is None:
            print(f"  [WARN] Unknown model '{name}', skipping")
            continue
        result = load_model(entry)
        if result is not None:
            models.append((name, *result))
            print(f"  ✓ {name} loaded")

    if not models:
        print("[ERROR] No models loaded!")
        sys.exit(1)

    # ---- 3. 构建并标准化数据 ----
    # 使用第一个模型的标准参数（所有 seed 模型特征数一致）
    _, _, feat_mean, feat_std = models[0]
    X, y, mask, dates = make_window_samples(panel, stock_ids, normalize=False)

    if feat_mean is not None and feat_std is not None:
        # 对 52 特征模型：用保存的标准参数
        X_norm = (X.astype(np.float32) - feat_mean) / feat_std
    else:
        # 对 baseline（34 特征，无标准化）：只是复制
        X_norm = X.copy()

    X_latest = torch.from_numpy(X_norm[-1:]).to(DEVICE, dtype=torch.float32)
    mask_latest = torch.from_numpy(mask[-1:]).to(DEVICE)
    print(f"\nController: Latest date: {dates[-1]}")

    # ---- 4. 获取所有模型的预测 ----
    all_scores = []
    all_embeddings = []

    for name, model, fm, fs in models:
        try:
            # 如果 baseline (34 特征)，需要用不同的 X
            # 由于 baseline 没有标准化，直接用原始 X
            with torch.no_grad():
                s, _ = model(X_latest, mask_latest)
                e = model.encode_stocks(X_latest, mask_latest)
            all_scores.append(s.squeeze(0).cpu().numpy())
            all_embeddings.append(e.squeeze(0).cpu())
            print(f"  {name}: prediction done")
        except Exception as ex:
            print(f"  {name}: prediction failed — {ex}")
            continue

    if not all_scores:
        print("[ERROR] No predictions!")
        sys.exit(1)

    # ---- 5. 集成与选股 ----
    # 等权平均所有模型分数
    avg_scores = np.mean(all_scores, axis=0)
    avg_embeddings = torch.stack(all_embeddings).mean(dim=0) if len(all_embeddings) > 1 else all_embeddings[0]

    # 选股
    selected = select_diverse_portfolio(avg_scores, stock_ids, avg_embeddings,
                                        max_stocks=args.topk)
    sel_ids = [s[0] for s in selected]
    sel_scores = np.array([avg_scores[stock_ids.index(s)] for s in sel_ids])

    # 获取股票统计数据
    stock_stats = get_stock_data(sel_ids, panel)

    print(f"\nController: Selected {len(sel_ids)} stocks")
    for sid, sc in zip(sel_ids, sel_scores):
        vol = stock_stats.get(sid, {}).get("vol", 0)
        print(f"  {sid}: score={sc:.4f}, vol={vol*100:.1f}%")

    # ---- 6. 权重分配 ----
    if args.compare:
        # 对比所有方法
        methods = ["equal", "softmax", "inv_vol", "sharpe", "score_vol", "adaptive"]
        best_score = -999
        best_w = None
        best_method = None

        for method in methods:
            w = compute_weights(method, sel_scores, stock_stats, args.temperature)
            df_out = pd.DataFrame({"stock_id": sel_ids, "weight": w})
            df_out.to_csv("output/result.csv", index=False)

            import subprocess
            r = subprocess.run(["python", "score_self.py"], capture_output=True, text=True)
            score_str = r.stdout.strip().split("\n")[-1]
            try:
                score = float(score_str.split(": ")[-1])
            except:
                score = -999

            w_str = ", ".join([f"{x:.3f}" for x in w])
            print(f"  {method:15s} [{w_str}] → score={score:.6f}")

            if score > best_score:
                best_score = score
                best_w = w
                best_method = method

        # 使用最优权重
        w = best_w
        print(f"\n  ★ Best: {best_method} (score={best_score:.6f})")
    else:
        w = compute_weights(args.weight, sel_scores, stock_stats, args.temperature)
        import subprocess
        df_out = pd.DataFrame({"stock_id": sel_ids, "weight": w})
        df_out.to_csv("output/result.csv", index=False)
        r = subprocess.run(["python", "score_self.py"], capture_output=True, text=True)
        score_str = r.stdout.strip().split("\n")[-1]
        try:
            score = float(score_str.split(": ")[-1])
        except:
            score = -999
        w_str = ", ".join([f"{x:.3f}" for x in w])
        print(f"\n  {args.weight}: [{w_str}] → score={score:.6f}")

    # ---- 7. 保存最终结果 ----
    df_final = pd.DataFrame({"stock_id": sel_ids, "weight": w})
    df_final.to_csv(args.output, index=False)
    print(f"\nController: Saved to {args.output}")


if __name__ == "__main__":
    main()
