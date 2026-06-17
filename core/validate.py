"""
core/validate.py — Walk-forward 滚动验证
"""

import numpy as np
import pandas as pd
from core.model import train_lgbm, predict_lgbm, compute_rankic
from core.selection import select_top_stocks


def walk_forward_validate(lgbm_df: pd.DataFrame, feature_cols: list[str],
                          panel, n_weeks: int = 12) -> dict:
    """12 周 walk-forward 验证。

    每周：用历史数据训练 → 预测当天 → 选股 → 计算真实收益。

    Args:
        lgbm_df: LGBM 数据（build_lgbm_data 产出）
        feature_cols: 特征列名
        panel: 原始特征面板（用于反查价格计算收益）
        n_weeks: 回测周数

    Returns:
        dict with mean, sharpe, win_rate, min_score, scores list
    """
    import lightgbm as lgb
    from config import (LGBM_ALPHA, LGBM_LEAVES, LGBM_LR, LGBM_MAX_DEPTH,
                        LGBM_MIN_CHILD, LGBM_REG_LAMBDA, LGBM_REG_ALPHA,
                        LGBM_SUBSAMPLE, LGBM_COLSAMPLE,
                        LGBM_BLIND_TREES_FALLBACK, LGBM_VAL_N)

    all_dates = sorted(lgbm_df['date'].unique())
    panel_dates = sorted(panel.index.get_level_values('date').unique())

    # 有效预测日期：需要有 T+1~T+5 评估数据
    valid_pred_dates = []
    for d in all_dates:
        eval_dates = [ed for ed in panel_dates if ed > d][:5]
        if len(eval_dates) >= 3:
            valid_pred_dates.append(d)

    # 按周采样：每 5 个交易日取一个，持仓窗口不重叠
    weekly_dates = []
    last_used = None
    for d in valid_pred_dates:
        if last_used is None or (d - last_used).days >= 5:
            weekly_dates.append(d)
            last_used = d

    if len(weekly_dates) < n_weeks:
        n_weeks = max(4, len(weekly_dates))

    pred_dates = weekly_dates[-n_weeks:]

    print("=" * 60)
    print(f"  Walk-Forward Validation — {n_weeks} weeks")
    print(f"  Period: {pred_dates[0].date()} → {pred_dates[-1].date()}")
    print("=" * 60)

    params = dict(
        objective='huber', boosting_type='gbdt',
        alpha=LGBM_ALPHA, n_estimators=LGBM_BLIND_TREES_FALLBACK,
        num_leaves=LGBM_LEAVES, max_depth=LGBM_MAX_DEPTH,
        learning_rate=LGBM_LR, min_child_samples=LGBM_MIN_CHILD,
        reg_lambda=LGBM_REG_LAMBDA, reg_alpha=LGBM_REG_ALPHA,
        subsample=LGBM_SUBSAMPLE, colsample_bytree=LGBM_COLSAMPLE,
        verbose=-1, random_state=42,
    )

    weekly_scores = []

    for wi, pred_date in enumerate(pred_dates):
        print(f"[Week {wi+1}/{n_weeks}] pred_date={pred_date.date()}")

        # 只用 pred_date 之前的数据训练
        df_week = lgbm_df[lgbm_df['date'] <= pred_date].copy()
        if len(df_week) < 500:
            print(f"  Not enough data, skipping")
            continue

        dates_w = sorted(df_week['date'].unique())
        train_cutoff = max(0, len(dates_w) - LGBM_VAL_N)
        tr = df_week[df_week['date'].isin(dates_w[:train_cutoff])]
        va = df_week[df_week['date'].isin(dates_w[train_cutoff:])]

        fm = tr[feature_cols].mean()
        fs = tr[feature_cols].std().replace(0, 1)
        X_tr = (tr[feature_cols].values - fm.values) / fs.values
        y_tr = tr['target'].values.astype(np.float32)

        model = lgb.LGBMRegressor(**params)
        if len(va) > 50:
            X_va = (va[feature_cols].values - fm.values) / fs.values
            y_va = va['target'].values.astype(np.float32)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        else:
            model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(0)])

        # 预测
        pred_day = df_week[df_week['date'] == pred_date]
        if len(pred_day) == 0:
            available = sorted(df_week['date'].unique())
            pred_date_actual = available[-1]
            pred_day = df_week[df_week['date'] == pred_date_actual]
        else:
            pred_date_actual = pred_date

        X_pred = (pred_day[feature_cols].values - fm.values) / fs.values
        pred_day = pred_day.copy()
        pred_day['lgb_score'] = model.predict(X_pred)
        scores = pred_day.set_index('stock_id')['lgb_score'].to_dict()

        # 选股
        _, psc = select_top_stocks(scores, panel, top_k=5)
        sel_ids, _ = select_top_stocks(scores, panel, top_k=5)

        # 计算真实收益
        eval_dates = [d for d in panel_dates if d > pred_date_actual][:5]
        sc = _compute_weekly_return(sel_ids, panel, eval_dates)
        sc_str = f"{sc:.4f}" if sc is not None else "N/A"
        print(f"  SCORE={sc_str}  [{len(sel_ids)} stocks]")
        if sc is not None:
            weekly_scores.append(sc)

    return _summarize(weekly_scores, n_weeks)


def _compute_weekly_return(sel_ids: list, panel, eval_dates: list) -> float | None:
    """计算 T+1 开盘 → T+5 开盘的加权收益。"""
    if not sel_ids or len(eval_dates) < 3:
        return None

    panel_flat = panel.reset_index()
    eval_panel = panel_flat[panel_flat['date'].isin(eval_dates)][
        ['date', 'stock_id', 'open']].copy()
    if len(eval_panel) == 0:
        return None

    returns = []
    for sid in sel_ids:
        try:
            sd = eval_panel[eval_panel['stock_id'] == sid].sort_values('date')
            if len(sd) < 3:
                continue
            ret = (float(sd['open'].iloc[-1]) / float(sd['open'].iloc[0]) - 1)
            returns.append(ret)
        except Exception:
            continue

    if not returns:
        return None
    return float(np.mean(returns))


def _summarize(scores: list, n_weeks: int) -> dict:
    """汇总统计。"""
    if not scores:
        return {}

    arr = np.array(scores)
    mean_v = float(np.mean(arr))
    std_v = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0
    sharpe = mean_v / std_v * np.sqrt(52) if std_v > 1e-8 else 0
    win_rate = float((arr > 0).mean())

    print("\n" + "=" * 60)
    print(f"  Validation Results ({len(scores)} weeks)")
    print("=" * 60)
    print(f"  Mean:     {mean_v:+.4f}")
    print(f"  Std:      {std_v:.4f}")
    print(f"  Min:      {np.min(arr):+.4f}")
    print(f"  Max:      {np.max(arr):+.4f}")
    print(f"  Sharpe:   {sharpe:.2f}")
    print(f"  WinRate:  {win_rate:.0%}")

    return {
        'mean': mean_v, 'sharpe': sharpe, 'win_rate': win_rate,
        'min': float(np.min(arr)), 'max': float(np.max(arr)),
        'scores': scores,
    }
