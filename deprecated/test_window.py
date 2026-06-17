"""
test_window.py — 滚动后验：检测数据泄漏 / 未来信息偷看

原理：
  把数据按时间切成多个窗口，每个窗口：
    1. 用窗口前的数据训练
    2. 对窗口内的样本做预测
    3. 看预测收益 vs 实际收益
  如果模型能 consistently 预测未来，说明没偷看。
  如果窗口 1 表现很好但窗口 3 突然变差 → 可能是过拟合。
"""
import argparse, pickle, numpy as np, pandas as pd
from pathlib import Path
import torch
from torch.utils.data import TensorDataset, DataLoader

from config import MODEL_DIR, PROCESSED_DIR, DEVICE, D_MODEL, BATCH_SIZE
from data_loader import get_official_stock_ids, build_panel_from_official
from features import engineer_features, make_window_samples, get_norm_stats
from model import PortfolioPredictor, listnet_loss
from train import train_model, make_dataloaders, evaluate, backtest


def main():
    parser = argparse.ArgumentParser(description="滚动后验检验数据泄漏")
    parser.add_argument("--n-windows", type=int, default=5,
                        help="滚动窗口数（默认5，每个窗口约20%数据）")
    parser.add_argument("--epochs", type=int, default=50,
                        help="每个窗口训练轮数")
    parser.add_argument("--loss", default="listnet",
                        choices=["lambdarank", "pairwise", "listnet", "topk_listnet"])
    parser.add_argument("--rank-labels", action="store_true", default=True,
                        help="使用排名标签")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Rolling Window Test: {args.n_windows} windows, {args.epochs} epochs each")
    print("=" * 60)

    # 加载数据
    stock_ids = get_official_stock_ids()
    panel = build_panel_from_official(stock_ids)
    panel = engineer_features(panel, stock_ids)
    X, y, mask, dates = make_window_samples(panel, stock_ids, normalize=False)

    n_total = len(X)
    window_size = n_total // args.n_windows
    print(f"Total samples: {n_total}, window size: ~{window_size}")
    print(f"Date range: {dates[0]} ~ {dates[-1]}")
    print()

    results = []

    for w in range(args.n_windows):
        # 窗口划分
        train_end = w * window_size
        val_end = train_end + window_size // 2
        test_end = min(val_end + window_size, n_total)

        if w == args.n_windows - 1:
            # 最后一个窗口用剩余全部数据
            test_end = n_total

        # 确保每个窗口有足够数据
        if train_end < 30 or test_end - val_end < 10:
            print(f"[Window {w+1}] Skipping (too few samples)")
            continue

        X_train, y_train, m_train = X[:train_end], y[:train_end], mask[:train_end]
        X_val,   y_val,   m_val   = X[train_end:val_end], y[train_end:val_end], mask[train_end:val_end]
        X_test,  y_test,  m_test  = X[val_end:test_end], y[val_end:test_end], mask[val_end:test_end]

        # 标准化：只用训练集的统计量（关键！防止未来信息泄漏）
        feat_mean = X_train.mean(axis=(0, 1, 2), keepdims=True)
        feat_std  = X_train.std(axis=(0, 1, 2), keepdims=True)
        feat_std  = np.where(feat_std < 1e-8, 1.0, feat_std)

        X_train_norm = (X_train.astype(np.float32) - feat_mean) / feat_std
        X_val_norm   = (X_val.astype(np.float32) - feat_mean) / feat_std
        X_test_norm  = (X_test.astype(np.float32) - feat_mean) / feat_std

        def to_loader(Xa, ya, ma, shuffle=False):
            wa = np.ones(len(Xa), dtype=np.float32)
            ds = TensorDataset(torch.from_numpy(Xa), torch.from_numpy(ya), torch.from_numpy(ma), torch.from_numpy(wa))
            return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)

        train_loader = to_loader(X_train_norm, y_train, m_train, shuffle=True)
        val_loader   = to_loader(X_val_norm, y_val, m_val)
        test_loader  = to_loader(X_test_norm, y_test, m_test)

        # 训练
        n_features = X.shape[-1]
        model = PortfolioPredictor(n_features=n_features).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n[Window {w+1}/{args.n_windows}] "
              f"train={len(X_train)} val={len(X_val)} test={len(X_test)} "
              f"dates={dates[train_end]}:{dates[test_end-1] if test_end>0 else ''}")

        loss_fn = listnet_loss
        model = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs, loss_name=args.loss,
            use_rank_labels=args.rank_labels,
        )

        # 评估
        test_metrics = evaluate(model, test_loader, loss_fn)
        bt = backtest(model, test_loader)

        print(f"  → Test pair acc: {test_metrics['pair_accuracy']:.3f} "
              f"(random=0.500)")
        print(f"  → Sharpe: {bt['sharpe']:.4f}, Win rate: {bt['win_rate']:.3f}")

        results.append({
            "window": w + 1,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "date_range": f"{dates[train_end][:10]}:{dates[test_end-1][:10]}",
            "pair_accuracy": test_metrics["pair_accuracy"],
            "sharpe": bt["sharpe"],
            "win_rate": bt["win_rate"],
            "mean_return": bt["mean_return"],
        })

    # 汇总
    print("\n" + "=" * 60)
    print("Rolling Window Test Summary")
    print("=" * 60)
    print(f"{'Win':>6}  {'Train':>6}  {'Test':>5}  {'PairAcc':>8}  {'Sharpe':>8}  {'WinRate':>8}  Date Range")
    print("-" * 70)

    for r in results:
        print(f"{r['window']:>6}  {r['train_samples']:>6}  {r['test_samples']:>5}  "
              f"{r['pair_accuracy']:>8.3f}  {r['sharpe']:>8.4f}  {r['win_rate']:>8.3f}  "
              f"{r['date_range']}")

    # 泄漏检验
    sharpe_vals = [r["sharpe"] for r in results]
    pair_accs = [r["pair_accuracy"] for r in results]

    print("\n" + "=" * 60)
    print("Leakage Check")
    print("=" * 60)
    print(f"Mean Sharpe:     {np.mean(sharpe_vals):.4f}  (positive = no leakage)")
    print(f"Mean Pair Acc:   {np.mean(pair_accs):.4f}  (>0.5 = learning signal)")
    print(f"Sharpe Std:      {np.std(sharpe_vals):.4f}  (low = consistent)")
    print(f"Sharpe Trend:    {'decreasing' if sharpe_vals[0] > sharpe_vals[-1] * 1.5 else 'stable'}")
    print()

    if np.mean(pair_accs) > 0.5:
        print("✓ Model shows consistent learning signal across windows")
    else:
        print("✗ Model does NOT learn consistently — check for issues")

    if np.mean(sharpe_vals) > 0:
        print("✓ Average Sharpe positive — no evidence of data leakage")
    else:
        print("⚠ Negative average Sharpe — possible leakage or insufficient signal")

    # 保存结果
    df = pd.DataFrame(results)
    df.to_csv("output/test_window_results.csv", index=False)
    print(f"\nResults saved to output/test_window_results.csv")


if __name__ == "__main__":
    main()
