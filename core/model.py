"""
core/model.py — LGBM 训练与预测
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import (MODEL_DIR, LGBM_ALPHA, LGBM_LEAVES, LGBM_LR,
                    LGBM_MAX_DEPTH, LGBM_MIN_CHILD, LGBM_REG_LAMBDA,
                    LGBM_REG_ALPHA, LGBM_SUBSAMPLE, LGBM_COLSAMPLE,
                    LGBM_BLIND_TREES_FALLBACK, LGBM_VAL_N)


def compute_rankic(pred_df: pd.DataFrame, pred_col='lgb_score',
                   target_col='target') -> float:
    """截面 RankIC：Spearman 相关系数按日平均。"""
    ics = []
    for _, day in pred_df.groupby('date'):
        if day[pred_col].nunique() <= 1 or day[target_col].nunique() <= 1:
            continue
        ic = day[pred_col].corr(day[target_col], method='spearman')
        if pd.notna(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def train_lgbm(df: pd.DataFrame, feature_cols: list[str],
               model_path: Path | None = None) -> tuple:
    """训练 LGBM Huber 回归。

    切分: train = 除最后 25 天外的全部, val = 最后 25 天

    Returns: (model, feat_mean, feat_std, rankic)
    """
    import lightgbm as lgb

    if model_path and model_path.exists():
        model = lgb.Booster(model_file=str(model_path))
        print(f"  Loaded cached LGBM model")
        fm = df[feature_cols].mean()
        fs = df[feature_cols].std().replace(0, 1)
        return model, fm, fs, None

    dates = sorted(df['date'].unique())
    val_dates = dates[-LGBM_VAL_N:]
    train_dates = dates[:-LGBM_VAL_N]

    tr = df[df['date'].isin(train_dates)]
    va = df[df['date'].isin(val_dates)]

    fm = tr[feature_cols].mean()
    fs = tr[feature_cols].std().replace(0, 1)

    X_tr = (tr[feature_cols].values - fm.values) / fs.values
    y_tr = tr['target'].values.astype(np.float32)

    params = dict(
        objective='huber', boosting_type='gbdt',
        alpha=LGBM_ALPHA,
        n_estimators=LGBM_BLIND_TREES_FALLBACK,
        num_leaves=LGBM_LEAVES,
        max_depth=LGBM_MAX_DEPTH,
        learning_rate=LGBM_LR,
        min_child_samples=LGBM_MIN_CHILD,
        reg_lambda=LGBM_REG_LAMBDA,
        reg_alpha=LGBM_REG_ALPHA,
        subsample=LGBM_SUBSAMPLE,
        colsample_bytree=LGBM_COLSAMPLE,
        verbose=-1, random_state=42,
    )

    print(f"  Training LGBM [{len(tr)} rows, {len(feature_cols)} features]...")
    model = lgb.LGBMRegressor(**params)

    if len(va) > 50:
        X_va = (va[feature_cols].values - fm.values) / fs.values
        y_va = va['target'].values.astype(np.float32)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    else:
        model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(0)])

    # Val RankIC
    rankic = None
    if len(va) > 0:
        X_va = (va[feature_cols].values - fm.values) / fs.values
        va_pred = model.predict(X_va)
        va_df = va.copy(); va_df['lgb_score'] = va_pred
        rankic = compute_rankic(va_df)
        print(f"  Val RankIC: {rankic:.4f}")

    if model_path:
        model.booster_.save_model(str(model_path))

    return model, fm, fs, rankic


def predict_lgbm(model, feat_mean, feat_std,
                 df: pd.DataFrame, feature_cols: list[str],
                 date) -> pd.DataFrame | None:
    """单日预测。返回含 lgb_score 列的 DataFrame。"""
    day = df[df['date'] == date].copy()
    if len(day) == 0:
        return None
    X = (day[feature_cols].values - feat_mean.values) / feat_std.values
    day['lgb_score'] = model.predict(X)
    return day
