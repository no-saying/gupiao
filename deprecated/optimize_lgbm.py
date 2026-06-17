"""
LightGBM 超参数搜索 — Optuna
搜索目标: 最大化验证集 RankIC
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import get_official_stock_ids, build_panel_from_official
from features import engineer_features
from controller import build_lgbm_data, compute_rankic, EXCLUDE

warnings.filterwarnings("ignore")
MODEL_DIR = Path("models")
N_TRIALS = 50

# 加载数据
print("[1/3] Loading data...")
stock_ids = get_official_stock_ids()
panel = build_panel_from_official(stock_ids)
panel = engineer_features(panel, stock_ids)
lgbm_df, feat_cols = build_lgbm_data(panel)
print(f"  LGBM data: {len(lgbm_df)} rows, {len(feat_cols)} features")

# 切分: 最后 20 天做验证（模拟回测最近窗口）
dates = sorted(lgbm_df['date'].unique())
train_dates = dates[:-20]
val_dates = dates[-20:]

train_df = lgbm_df[lgbm_df['date'].isin(train_dates)].copy()
val_df = lgbm_df[lgbm_df['date'].isin(val_dates)].copy()
print(f"  Train: {train_df['date'].nunique()} days, Val: {val_df['date'].nunique()} days")


def objective(trial):
    params = {
        "objective": "lambdarank",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "label_gain": [i for i in range(10)],
        "verbose": -1,
        "random_state": 42,
    }

    # 标准化
    fm = train_df[feat_cols].mean()
    fs = train_df[feat_cols].std().replace(0, 1)
    X_tr = (train_df[feat_cols].values - fm.values) / fs.values
    y_tr = train_df['rank_label'].values.astype(int)
    grp = train_df.groupby('date').size().values

    model = lgb.LGBMRanker(**params)
    model.fit(X_tr, y_tr, group=grp,
              eval_metric=['ndcg'], callbacks=[lgb.log_evaluation(0)])

    # 验证集 RankIC
    X_val = (val_df[feat_cols].values - fm.values) / fs.values
    pred = model.predict(X_val)
    val_df_copy = val_df.copy()
    val_df_copy['lgb_score'] = pred
    rankic = compute_rankic(val_df_copy, 'lgb_score', 'target')

    return rankic


print("[2/3] Running Optuna search...")
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n[3/3] Results")
print("=" * 60)
print(f"Best RankIC: {study.best_value:.4f}")
print(f"Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# 按 RankIC 排列 Top-5
print(f"\nTop-5 trials:")
for i, t in enumerate(sorted(study.trials, key=lambda x: x.value or 0, reverse=True)[:5]):
    print(f"  {i+1}. RankIC={t.value:.4f}: {t.params}")

# 保存最佳参数
import json
(Path("output") / "lgbm_best_params.json").write_text(json.dumps({
    "rankic": study.best_value,
    "params": study.best_params,
}, indent=2))
print(f"\nBest params saved to output/lgbm_best_params.json")
