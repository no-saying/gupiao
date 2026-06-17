"""DANN 模型回测 — 12 周 walk-forward

Usage:
  python backtest_dann.py                         # dann_mse.pt (默认)
  python backtest_dann.py --model dann_margin.pt   # Margin Ranking Loss 模型
  python backtest_dann.py --model dann_mse.pt      # MSE 模型
"""
import argparse, sys, torch, numpy as np, pandas as pd, time
sys.path.insert(0, '.')
from config import MODEL_DIR, DEVICE
from core.dann_model import DANNStockModel

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="dann_mse.pt",
                    help="Model file in models/")
args = parser.parse_args()

device = torch.device(DEVICE)
torch.cuda.empty_cache()
print(f"Model: {args.model}")

# ── 1. 加载数据 ──
from tushare_loader import TUSHARE_DIR, fetch_csi300_stocks, fetch_csi500_stocks
from core.features import engineer_features
from core.alpha158 import add_alpha158_features
from core.data import build_lgbm_data

s300 = set(fetch_csi300_stocks())
s500 = set(fetch_csi500_stocks())
union = sorted(s300 | s500)
print(f"Stocks: {len(s300)} CSI300 + {len(s500)} CSI500 = {len(union)} total")

cache_path = TUSHARE_DIR / "tushare_panel_886.parquet"
panel = pd.read_parquet(cache_path)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index(["date", "stock_id"])
print(f"Panel: {panel.shape}")

panel = engineer_features(panel, union)
panel = add_alpha158_features(panel)
print(f"Features: {panel.shape}")

lgbm_df, feature_cols = build_lgbm_data(panel)
F = len(feature_cols)
print(f"Data: {len(lgbm_df)} rows, {F} features")

# ── 2. 构建窗口 ──
from train_nn import build_window_samples
X, y, pred_dates_all, stock_ids_all = build_window_samples(lgbm_df, feature_cols)
n_stocks, n_windows = y.shape
stock_to_idx = {s: i for i, s in enumerate(stock_ids_all)}
print(f"Windows: {n_stocks}s × {n_windows}w")

# ── 3. 加载 DANN 模型 ──
model = DANNStockModel(F, d_model=256, n_stocks=n_stocks, lambda_domain=0.5).to(device)
model.load_state_dict(torch.load(MODEL_DIR / args.model, map_location=device))
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# ── 4. Walk-forward 回测 ──
panel_dates = sorted(panel.index.get_level_values('date').unique())
n_weeks = 12

# 按周采样（非重叠）
valid_dates = [d for d in pred_dates_all if d in pred_dates_all]
weekly_dates = []
last_used = None
for d in pred_dates_all:
    if last_used is None or (d - last_used).days >= 5:
        weekly_dates.append(d)
        last_used = d

if len(weekly_dates) < n_weeks:
    n_weeks = max(4, len(weekly_dates))
pred_dates = weekly_dates[-n_weeks:]

print(f"\n{'='*60}")
print(f"  DANN Walk-Forward — {n_weeks} weeks")
print(f"  Period: {pred_dates[0].date()} → {pred_dates[-1].date()}")
print(f"{'='*60}")

X_cpu = torch.FloatTensor(X)  # (N, T, L, F)
weekly_scores = []

for wi, pred_date in enumerate(pred_dates):
    # 找到对应窗口索引
    w_idx = pred_dates_all.index(pred_date)

    # DANN 推理（单窗口）
    x_input = X_cpu[:, w_idx:w_idx+1].permute(1, 0, 2, 3).to(device)  # (1, N, L, F)
    with torch.no_grad():
        dann_scores, _ = model(x_input)
    dann_scores = dann_scores.squeeze(0).cpu().numpy()

    # 只在 CSI300 里选
    s300_scores = {}
    for i, sid in enumerate(stock_ids_all):
        if sid in s300:
            s300_scores[sid] = dann_scores[i]

    top5 = sorted(s300_scores.items(), key=lambda x: -x[1])[:5]
    sel_ids = [s for s, _ in top5]

    # 计算真实收益 (T+1 → T+5)
    eval_dates = [d for d in panel_dates if d > pred_date][:5]
    returns = []
    panel_flat = panel.reset_index()
    edf = panel_flat[panel_flat['date'].isin(eval_dates)][['date', 'stock_id', 'open']]

    for sid in sel_ids:
        try:
            sd = edf[edf['stock_id'] == sid].sort_values('date')
            if len(sd) >= 3:
                ret = float(sd['open'].iloc[-1]) / float(sd['open'].iloc[0]) - 1
                returns.append(ret)
        except Exception:
            continue

    sc = float(np.mean(returns)) if returns else None
    sc_str = f"{sc:.4f}" if sc is not None else "N/A"
    picks = ", ".join(sel_ids)
    print(f"  [W{wi+1:2d}] {pred_date.date()} | picks={picks} | SCORE={sc_str}")
    if sc is not None:
        weekly_scores.append(sc)

# ── 5. 汇总 ──
if weekly_scores:
    arr = np.array(weekly_scores)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0
    sharpe = mean_v / std_v * np.sqrt(52) if std_v > 1e-8 else 0
    win_rate = float((arr > 0).mean())

    print(f"\n{'='*60}")
    print(f"  DANN Backtest Results ({len(weekly_scores)} weeks)")
    print(f"{'='*60}")
    print(f"  Mean:     {mean_v:+.4f}")
    print(f"  Std:      {std_v:.4f}")
    print(f"  Min:      {np.min(arr):+.4f}")
    print(f"  Max:      {np.max(arr):+.4f}")
    print(f"  Sharpe:   {sharpe:.2f}")
    print(f"  WinRate:  {win_rate:.0%}")
    print(f"  Weekly:  ", [f"{s:+.4f}" for s in weekly_scores])
