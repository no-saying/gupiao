"""NN 单周预测脚本 — 加载预训练模型，预测最新日期，选股输出。"""
import torch, numpy as np, pandas as pd, sys
from pathlib import Path
sys.path.insert(0, '.')

from config import MODEL_DIR, DEVICE, SUBMISSION_PATH
from core.nn_model import StockRankingModel
from core.cvxpy_layer import DifferentiablePortfolio

# 数据
from tushare_loader import fetch_csi300_stocks, build_tushare_only_panel
from core.features import engineer_features
from core.alpha158 import add_alpha158_features
from core.data import build_lgbm_data
from train_nn import build_window_samples

stock_ids = fetch_csi300_stocks()
panel = build_tushare_only_panel(stock_ids)
panel = engineer_features(panel, stock_ids)
panel = add_alpha158_features(panel)
lgbm_df, feature_cols = build_lgbm_data(panel)

# 加载模型
d_model = 256
F = len(feature_cols)
n_stocks = len(stock_ids)
model = StockRankingModel(F, d_model, n_stocks, n_mamba_layers=4)
ckpt = torch.load(MODEL_DIR / "portfolio_model.pt", map_location='cpu')
model.load_state_dict(ckpt)
model.eval()
model.cuda()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} params")

# 构建最新窗口（只用最后一个预测日期）
X, y, pred_dates, _ = build_window_samples(lgbm_df, feature_cols)
latest_idx = X.shape[1] - 1
x_latest = torch.FloatTensor(X[:, latest_idx:latest_idx+1]).permute(1, 0, 2, 3).cuda()
print(f"Input shape: {x_latest.shape}")

# 预测
with torch.no_grad():
    scores = model(x_latest).squeeze(0).cpu().numpy()

# 选股：分数排序 top-5
scored = sorted(zip(stock_ids, scores), key=lambda x: -x[1])
top5 = scored[:5]

print(f"\nPrediction date: {pred_dates[latest_idx].date()}")
print(f"Top-5 NN picks:")
for sid, sc in top5:
    print(f"  {sid}: score={sc:+.4f}")

# 输出
weights = np.ones(5) / 5
df_out = pd.DataFrame({"stock_id": [str(s).zfill(6) for s, _ in top5], "weight": weights})
df_out.to_csv("output/nn_result.csv", index=False)
print(f"\nSaved to output/nn_result.csv")
