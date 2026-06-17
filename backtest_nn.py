"""NN 4周 Walk-Forward 回测 — 每周期微调10 epoch"""
import torch, numpy as np, sys, time
sys.path.insert(0, '.')
from config import MODEL_DIR, DEVICE
from core.nn_model import StockRankingModel
from core.cvxpy_layer import DifferentiablePortfolio, portfolio_loss

from tushare_loader import fetch_csi300_stocks, build_tushare_only_panel
from core.features import engineer_features
from core.alpha158 import add_alpha158_features
from core.data import build_lgbm_data
from train_nn import build_window_samples, _build_hist_returns

# 全量数据（每个回测周截断）
stock_ids = fetch_csi300_stocks()
panel_full = build_tushare_only_panel(stock_ids)
panel_full = engineer_features(panel_full, stock_ids)
panel_full = add_alpha158_features(panel_full)
lgbm_df_full, feature_cols = build_lgbm_data(panel_full)
all_dates = sorted(lgbm_df_full['date'].unique())
panel_dates = sorted(panel_full.index.get_level_values('date').unique())

# 有效预测日期（按周采样，持仓不重叠）
valid_dates = []
for d in all_dates:
    eval_dates = [ed for ed in panel_dates if ed > d][:5]
    if len(eval_dates) >= 3:
        valid_dates.append(d)
weekly_dates = []
last_used = None
for d in valid_dates:
    if last_used is None or (d - last_used).days >= 5:
        weekly_dates.append(d)
        last_used = d
pred_dates = weekly_dates[-4:]

F = len(feature_cols)
d_model = 256
n_stocks = len(stock_ids)
device = torch.device(DEVICE)
results = []

for wi, pred_date in enumerate(pred_dates):
    print(f"\n{'='*60}")
    print(f"[Week {wi+1}/4] pred_date={pred_date.date()}")
    print(f"{'='*60}")

    # 截断数据到预测日之前
    mask = lgbm_df_full['date'] <= pred_date
    lgbm_df = lgbm_df_full[mask].copy()
    panel = panel_full[panel_full.index.get_level_values('date') <= pred_date].copy()

    # 构建窗口
    X, y, _, _ = build_window_samples(lgbm_df, feature_cols)
    n_windows = y.shape[1]

    # 加载预训练编码器 + 微调
    torch.cuda.empty_cache()
    model = StockRankingModel(F, d_model, n_stocks, n_mamba_layers=4).to(device)
    ckpt = torch.load(MODEL_DIR / "mae_pretrained.pt", map_location='cpu')
    model.mamba.load_state_dict(ckpt['encoder'])
    portfolio = DifferentiablePortfolio(n_stocks)

    optimizer = torch.optim.AdamW([
        {'params': model.mamba.parameters(), 'lr': 5e-5},
        {'params': model.gat.parameters(), 'lr': 1e-4},
        {'params': model.fusion.parameters(), 'lr': 1e-4},
        {'params': model.score_head.parameters(), 'lr': 2e-4},
    ], weight_decay=0.01)

    hist_returns = _build_hist_returns(y, window=20)
    hist_gpu = torch.FloatTensor(hist_returns).to(device)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    batch_size = 4

    # 微调 10 epoch
    t0 = time.time()
    for epoch in range(10):
        model.train()
        indices = torch.randperm(n_windows)
        for start in range(0, n_windows, batch_size):
            batch_idx = indices[start:start + batch_size]
            x_batch = X_tensor[:, batch_idx].permute(1, 0, 2, 3).to(device)
            y_batch = y_tensor[:, batch_idx].T.to(device)
            h_batch = hist_gpu[:, batch_idx].permute(1, 0, 2)

            scores = model(x_batch)
            result = portfolio_loss(scores, y_batch, hist_returns=h_batch,
                                     lambda_cvar=0.0, optimizer=portfolio)
            (result['loss'] / 2).backward()
            optimizer.step(); optimizer.zero_grad()

    dt = time.time() - t0
    print(f"  Fine-tuned 10 epochs in {dt:.0f}s")

    # 预测最新日
    model.eval()
    x_pred = torch.FloatTensor(X[:, -1:]).permute(1, 0, 2, 3).to(device)
    with torch.no_grad():
        scores = model(x_pred).squeeze(0).cpu().numpy()

    # 选股 top-5
    scored = sorted(zip(stock_ids, scores), key=lambda x: -x[1])
    sel_ids = [s for s, _ in scored[:5]]

    # 计算真实收益
    eval_dates = [d for d in panel_dates if d > pred_date][:5]
    returns = []
    for sid in sel_ids:
        try:
            sd = panel_full.xs(sid, level='stock_id')
            t1 = float(sd.loc[eval_dates[0], 'open']) if eval_dates[0] in sd.index else 0
            t5 = float(sd.loc[eval_dates[-1], 'open']) if eval_dates[-1] in sd.index else 0
            returns.append((t5 - t1) / max(t1, 1e-8) if t1 > 0 else 0)
        except:
            returns.append(0)
    score = float(np.mean(returns))
    results.append(score)

    print(f"  Selected: {', '.join(sel_ids)}")
    print(f"  SCORE={score:+.4f}")
    torch.cuda.empty_cache()

# 汇总
print(f"\n{'='*60}")
print(f"  NN 4-Week Results")
print(f"{'='*60}")
arr = np.array(results)
print(f"  Mean:    {float(np.mean(arr)):+.4f}")
print(f"  Min:     {float(np.min(arr)):+.4f}")
print(f"  Max:     {float(np.max(arr)):+.4f}")
print(f"  WinRate: {float((arr > 0).mean()):.0%}")
