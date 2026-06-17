"""DANN 域自适应训练 — 500训练 + 300选股"""
import torch, numpy as np, time, sys
sys.path.insert(0, '.')
from config import MODEL_DIR, DEVICE
from core.dann_model import DANNStockModel, dann_loss, build_domain_labels

device = torch.device(DEVICE)
torch.cuda.empty_cache()

# 数据
from tushare_loader import TUSHARE_DIR, fetch_csi300_stocks, fetch_csi500_stocks
from core.features import engineer_features
from core.alpha158 import add_alpha158_features
from core.data import build_lgbm_data
import pandas as pd

s300 = fetch_csi300_stocks()
s500 = fetch_csi500_stocks()
union = sorted(set(s300) | set(s500))
n_300 = len(s300)
n_500 = len(union) - n_300
print(f"Stocks: {n_300} CSI300 + {n_500} CSI500 = {len(union)} total")

# 加载 886 panel
# stk_factor_pro 列名已统一，无需额外处理
cache_path = TUSHARE_DIR / "tushare_panel_886.parquet"
panel = pd.read_parquet(cache_path)
panel["date"] = pd.to_datetime(panel["date"])
panel = panel.set_index(["date", "stock_id"])
print(f"Panel: {panel.shape}")

# 特征工程
panel = engineer_features(panel, union)
panel = add_alpha158_features(panel)
print(f"Features: {panel.shape}")

# 构建 LGBM 数据 + 域标签
lgbm_df, feature_cols = build_lgbm_data(panel)
lgbm_df["domain"] = lgbm_df["stock_id"].apply(lambda s: 0 if s in set(s300) else 1)
print(f"LGBM data: {len(lgbm_df)} rows, {len(feature_cols)} features")

# 构建窗口样本（用 train_nn 的函数）
from train_nn import build_window_samples
X, y, pred_dates, stock_ids_all = build_window_samples(lgbm_df, feature_cols)
domain_arr = np.array([0 if s in set(s300) else 1 for s in stock_ids_all], dtype=np.float32)
n_stocks, n_windows = y.shape
F = len(feature_cols)
print(f"Windows: {n_stocks}s × {n_windows}w × 60d × {F}f")

# DANN 模型
d_model = 256
model = DANNStockModel(F, d_model, n_stocks, lambda_domain=0.5).to(device)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

# 域标签张量（每 batch 复用）
domain_labels_all = torch.FloatTensor(domain_arr).to(device)  # (N,)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)

X_cpu = torch.FloatTensor(X)
y_cpu = torch.FloatTensor(y)
batch_size = 4

t0 = time.time()
for epoch in range(200):
    model.train()
    epoch_task, epoch_dom = 0.0, 0.0
    indices = torch.randperm(n_windows)

    for start in range(0, n_windows, batch_size):
        batch_idx = indices[start:start + batch_size]
        x_batch = X_cpu[:, batch_idx].permute(1, 0, 2, 3).to(device)
        y_batch = y_cpu[:, batch_idx].T.to(device)
        B = len(batch_idx)

        scores, domain_logits = model(x_batch)
        domain_labels = domain_labels_all.unsqueeze(0).expand(B, -1).reshape(-1)

        result = dann_loss(scores, y_batch, domain_logits, domain_labels, lambda_domain=0.5)
        result['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); optimizer.zero_grad()

        epoch_task += result['task'].item()
        epoch_dom += result['domain'].item()

    scheduler.step()

    if (epoch + 1) % 10 == 0:
        n_batch = max(1, n_windows // batch_size)
        print(f"  E{epoch+1:3d}/200 | task={epoch_task/n_batch:.4f} | "
              f"domain={epoch_dom/n_batch:.4f} | {time.time()-t0:.0f}s", flush=True)

# 保存
torch.save(model.state_dict(), MODEL_DIR / "dann_model.pt")
print(f"Saved to dann_model.pt")

# 只在 CSI 300 上评估
model.eval()
x_latest = X_cpu[:, -1:].permute(1, 0, 2, 3).to(device)
with torch.no_grad():
    scores, _ = model(x_latest)
scores = scores.squeeze(0).cpu().numpy()

s300_scores = {s: scores[i] for i, s in enumerate(stock_ids_all) if s in set(s300)}
top5 = sorted(s300_scores.items(), key=lambda x: -x[1])[:5]
print(f"\nTop-5 CSI 300 picks:")
for sid, sc in top5:
    print(f"  {sid}: +{sc:.4f}")
