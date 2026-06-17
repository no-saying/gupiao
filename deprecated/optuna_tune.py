"""
Optuna 自动调参 — 优化 LGBM Huber 在验证集上的 RankIC。

使用方法:
  python optuna_tune.py              # 默认 50 trials
  python optuna_tune.py --trials 100 # 自定义 trial 数
  python optuna_tune.py --xgb        # 调 XGBoost 而非 LGBM
"""

import argparse, warnings, time, json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from config import (MODEL_DIR, LGBM_VAL_N, LGBM_PURGE_DAYS, LGBM_N_FOLDS,
                    LGBM_MIN_TREES)
warnings.filterwarnings("ignore")


def compute_rankic_per_date(df, pred_col, target_col='target'):
    """截面 RankIC：按日计算 Spearman 后取均值。"""
    ics = []
    for _, day in df.groupby('date'):
        if day[pred_col].nunique() <= 1 or day[target_col].nunique() <= 1:
            continue
        ic, _ = spearmanr(day[pred_col], day[target_col])
        if pd.notna(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def _get_fold_dates(all_dates, n_folds=4, purge=2, val_n=60):
    folds = []
    n = len(all_dates)
    for i in range(n_folds):
        val_end = n - i * val_n
        val_start = val_end - val_n
        if val_start < 30:
            break
        train_end = val_start - purge
        folds.append({'train': all_dates[:train_end], 'val': all_dates[val_start:val_end]})
    return folds


def tune_lgbm(df, feature_cols, n_trials=50):
    """Optuna 调 LGBM Huber — 直接优化 Top-15 平均真实收益（非 RankIC）。"""
    import lightgbm as lgb
    import optuna

    dates = sorted(df['date'].unique())
    folds = _get_fold_dates(dates, n_folds=3, val_n=40)

    if len(folds) < 2:
        print("Not enough data for CV folds")
        return None

    fm = df[feature_cols].mean()
    fs = df[feature_cols].std().replace(0, 1)

    def objective(trial):
        params = {
            'objective': 'huber',
            'boosting_type': 'gbdt',
            'alpha': trial.suggest_float('alpha', 1.0, 1.5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 60),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),  # ← 上限 1.0
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'n_estimators': 300,
            'num_threads': -1,
            'verbose': -1,
            'random_state': 42,
        }

        fold_top_returns = []
        for fold in folds:
            tr = df[df['date'].isin(fold['train'])]
            va = df[df['date'].isin(fold['val'])]
            X_tr = (tr[feature_cols].values - fm.values) / fs.values
            y_tr = tr['target'].values.astype(np.float32)
            X_va = (va[feature_cols].values - fm.values) / fs.values
            y_va = va['target'].values.astype(np.float32)

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr,
                      eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            va_pred = model.predict(X_va)

            # 【目标】Top-15 平均真实收益
            top_15_idx = np.argsort(va_pred)[-15:]
            fold_top_return = float(y_va[top_15_idx].mean())
            fold_top_returns.append(fold_top_return)

        return float(np.mean(fold_top_returns))

    print(f"Optuna tuning LGBM — top15 return ({len(folds)} folds × val_n=40, {n_trials} trials)...")
    study = optuna.create_study(direction='maximize',
                                study_name='lgbm_top15_tune',
                                storage=None)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,
                   n_jobs=1)

    print(f"\nBest Top-15 Return: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    return study.best_params, study.best_value


def tune_xgb(df, feature_cols, n_trials=50):
    """Optuna 调 XGBoost 参数 — Purged K-Fold CV 优化 RankIC。"""
    import xgboost as xgb
    import optuna

    dates = sorted(df['date'].unique())
    folds = _get_fold_dates(dates, n_folds=3, val_n=40)

    if len(folds) < 2:
        print("Not enough data for CV folds")
        return None

    fm = df[feature_cols].mean()
    fs = df[feature_cols].std().replace(0, 1)

    def objective(trial):
        params = {
            'objective': 'reg:pseudohubererror',
            'huber_slope': trial.suggest_float('huber_slope', 1.0, 2.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 1e-4, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'n_estimators': 500,
            'early_stopping_rounds': 30,
            'random_state': 42, 'verbosity': 0, 'n_jobs': -1,
        }

        fold_ics = []
        for fold in folds:
            tr = df[df['date'].isin(fold['train'])]
            va = df[df['date'].isin(fold['val'])]
            X_tr = (tr[feature_cols].values - fm.values) / fs.values
            y_tr = tr['target'].values.astype(np.float32)
            X_va = (va[feature_cols].values - fm.values) / fs.values
            y_va = va['target'].values.astype(np.float32)

            model = xgb.XGBRegressor(**params)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            va_pred = model.predict(X_va)
            va_df = va.copy()
            va_df['pred'] = va_pred
            ic = compute_rankic_per_date(va_df, 'pred')
            fold_ics.append(ic)

        return float(np.mean(fold_ics))

    print(f"Optuna tuning XGBoost ({len(folds)} folds × val_n=40, {n_trials} trials)...")
    study = optuna.create_study(direction='maximize',
                                study_name='xgb_huber_tune',
                                storage=None)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True,
                   n_jobs=1)

    print(f"\nBest RankIC: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    return study.best_params, study.best_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Tuning")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    parser.add_argument("--xgb", action="store_true", help="Tune XGBoost")
    parser.add_argument("--output", type=str, default=None, help="Save best params to JSON")
    args = parser.parse_args()

    from controller import build_lgbm_data
    from tushare_loader import fetch_csi300_stocks, build_tushare_only_panel
    from features import engineer_features
    from features_alpha158 import add_alpha158_features

    print("Loading data...")
    stock_ids = fetch_csi300_stocks()
    panel = build_tushare_only_panel(stock_ids)
    panel = engineer_features(panel, stock_ids)
    panel = add_alpha158_features(panel)

    df, feat_cols = build_lgbm_data(panel)
    print(f"Data: {len(df)} rows, {len(feat_cols)} features\n")

    t0 = time.time()
    if args.xgb:
        best_params, best_ic = tune_xgb(df, feat_cols, args.trials)
    else:
        best_params, best_ic = tune_lgbm(df, feat_cols, args.trials)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump({'best_params': best_params, 'best_rankic': best_ic,
                       'n_trials': args.trials, 'model': 'xgb' if args.xgb else 'lgbm'}, f, indent=2)
        print(f"Saved to {args.output}")
