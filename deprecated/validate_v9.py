"""
=============================================================================
  滚动窗口验证模块 — 模拟历史N周的真实预测，评估模型泛化能力
=============================================================================

用法:
  from validate import walk_forward_validate
  walk_forward_validate(lgbm_df, feature_cols, n_weeks=52)

评估指标: Mean, Sharpe(年化), WinRate, MaxDD, Calmar, 牛熊分治
=============================================================================
"""

import numpy as np
import lightgbm as lgb

from config import LGBM_GAP, LGBM_VAL_N


def walk_forward_validate(df, feature_cols, n_weeks=52):
    """
    滚动窗口验证：模拟过去 N 周的真实预测，汇总 out-of-sample 表现。

    每轮：
      - 预测日设为历史某个时点，之前数据全部用于训练
      - 严格 GAP 防止标签泄漏
      - 记录该周的选股和收益
    汇总：Mean, Sharpe, WinRate, 最差周, MaxDD, Calmar
    """
    dates = sorted(df['date'].unique())
    scores = []   # 每周的组合收益
    details = []  # 每周详情
    GAP = LGBM_GAP
    VAL_N = LGBM_VAL_N

    for w in range(n_weeks, 0, -1):
        pred_idx = len(dates) - w * 5
        if pred_idx < 200:
            break
        pred_date = dates[pred_idx]

        train_end_idx = pred_idx - GAP - VAL_N
        if train_end_idx < 100:
            continue
        train_dates = dates[:train_end_idx]
        val_dates = dates[train_end_idx:train_end_idx + VAL_N]

        train_df = df[df['date'].isin(train_dates)].copy()
        val_df = df[df['date'].isin(val_dates)].copy()
        pred_df = df[df['date'] == pred_date].copy()

        if len(train_df) < 1000 or len(pred_df) < 50:
            continue

        # 训练
        fm = train_df[feature_cols].mean()
        fs = train_df[feature_cols].std().replace(0, 1)
        X_tr = (train_df[feature_cols].values - fm.values) / fs.values
        y_tr = train_df['rank_label'].values.astype(int)
        grp = train_df.groupby('date').size().values
        X_es = ((val_df[feature_cols].values - fm.values) / fs.values).astype(np.float32)
        y_es = val_df['rank_label'].values.astype(int)
        grp_es = val_df.groupby('date').size().values

        model = lgb.LGBMRanker(
            objective='lambdarank', boosting_type='gbdt',
            n_estimators=300, num_leaves=64, learning_rate=0.02,
            min_child_samples=42, subsample=0.74, colsample_bytree=0.94,
            label_gain=[i for i in range(10)], verbose=-1, random_state=42)
        model.fit(X_tr, y_tr, group=grp,
                  eval_set=[(X_es, y_es)], eval_group=[grp_es],
                  eval_metric=['ndcg'], eval_at=[5],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

        # 预测
        X_pred = ((pred_df[feature_cols].values - fm.values) / fs.values).astype(np.float32)
        pred_df = pred_df.copy()
        pred_df['lgb_score'] = model.predict(X_pred)

        # 选 top 5 等权
        top5 = pred_df.nlargest(5, 'lgb_score')
        week_return = top5['target'].mean()
        scores.append(week_return)
        details.append({
            'pred_date': str(pred_date.date()),
            'return': round(week_return, 4),
            'top5': list(top5['stock_id'].values),
        })

    if not scores:
        print("  [Walk-forward] Not enough data for validation")
        return None

    scores_arr = np.array(scores)
    mean_ret = np.mean(scores_arr)
    std_ret = np.std(scores_arr)
    sharpe = mean_ret / (std_ret + 1e-12) * np.sqrt(52)
    win_rate = (scores_arr > 0).mean()
    worst = np.min(scores_arr)
    best = np.max(scores_arr)

    # Max Drawdown
    cumsum = np.cumsum(scores_arr)
    peak = np.maximum.accumulate(cumsum)
    drawdown = cumsum - peak
    max_dd = np.min(drawdown)
    calmar_val = mean_ret * 52 / (abs(max_dd) + 1e-12)
    calmar_str = f"{calmar_val:.2f}" if calmar_val < 1000 else "∞ (无回撤周)"

    # 输出
    print(f"\n{'='*60}")
    print(f"  Walk-Forward Validation ({len(scores)} weeks)")
    print(f"  Mean: {mean_ret:.4f}  |  Sharpe(年化): {sharpe:.2f}  |  WinRate: {win_rate:.0%}")
    print(f"  Best: {best:.4f}  |  Worst: {worst:.4f}  |  Std: {std_ret:.4f}")
    print(f"  MaxDD: {max_dd:.4f}  |  Calmar: {calmar_str}")
    print(f"  Weekly returns: {[round(s, 4) for s in scores]}")
    for d in details[-3:]:
        print(f"    {d['pred_date']}: ret={d['return']:.4f}  stocks={d['top5']}")
    print(f"{'='*60}\n")

    return {'mean': mean_ret, 'sharpe': sharpe, 'win_rate': win_rate,
            'worst': worst, 'max_dd': max_dd, 'calmar': calmar_val,
            'scores': scores, 'details': details}


# =============================================================================
# 消融实验：逐项开关保护机制，测量边际贡献
# =============================================================================

ABLATION_MECHANISMS = [
    # (key, description, category)
    ("tail_detect",     "极端周检测 (tail risk)",         "risk"),
    ("beta_penalty",    "高Beta惩罚",                     "risk"),
    ("crowd_penalty",   "拥挤行业惩罚",                   "risk"),
    ("gap_penalty",     "开盘跳空惩罚",                   "scoring"),
    ("turnover_bonus",  "换手惩罚/惯性奖励",              "scoring"),
    ("soft_cluster",    "隐式软聚类 (Embedding去重)",      "scoring"),
    ("hrp_weights",     "HRP隐式权重",                    "scoring"),
    ("nash_equilibrium","纳什均衡选股",                   "scoring"),
    ("industry_diverse","行业分散过滤",                   "scoring"),
    ("vol_filter",      "波动率过滤",                     "scoring"),
    ("nn_progressive",  "NN渐进融合",                     "ensemble"),
    ("sw_expert",       "SW60短窗专家",                   "ensemble"),
    ("catboost",        "CatBoost融合",                   "ensemble"),
    ("hmm_smoothing",   "HMM平滑曲面",                    "risk"),
    ("reversal_bonus",  "反转候选股加分",                 "scoring"),
]


def ablation_study(get_scores_fn, stock_ids, panel, args):
    """
    消融实验：逐个关闭保护机制，观察 top-5 组合变化。

    get_scores_fn(disabled_mechanisms: set) → (scores_array, selected_stocks, weights)
    """
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY — 保护机制消融实验")
    print(f"{'='*70}")

    # 1. Baseline: all mechanisms enabled
    base_scores, base_stocks, base_weights = get_scores_fn(set())
    print(f"\n  Baseline: {len(base_stocks)} stocks, weights={[f'{w:.3f}' for w in base_weights]}")
    print(f"  Baseline stocks: {base_stocks}")

    results = []
    print(f"\n  {'Mechanism':<30s} {'Category':<10s} {'Jaccard':>8s} {'Score Δ':>10s} {'Verdict':>10s}")
    print(f"  {'-'*68}")

    for key, desc, category in ABLATION_MECHANISMS:
        try:
            disabled = {key}
            ab_scores, ab_stocks, ab_weights = get_scores_fn(disabled)

            # Jaccard similarity of top-5
            base_set = set(base_stocks[:5])
            ab_set = set(ab_stocks[:5])
            jaccard = len(base_set & ab_set) / max(len(base_set | ab_set), 1)

            # Score shift (normalized)
            if base_scores is not None and ab_scores is not None:
                score_corr = np.corrcoef(base_scores, ab_scores)[0, 1] if len(base_scores) > 1 else 1.0
            else:
                score_corr = 1.0

            delta = 1.0 - score_corr  # Higher delta = bigger impact

            if jaccard == 1.0:
                verdict = "NEGLIGIBLE"
            elif jaccard >= 0.8:
                verdict = "MINOR"
            elif jaccard >= 0.5:
                verdict = "MODERATE"
            else:
                verdict = "SIGNIFICANT"

            results.append({
                'mechanism': desc, 'category': category,
                'jaccard': jaccard, 'score_delta': delta, 'verdict': verdict,
            })
            print(f"  {desc:<30s} {category:<10s} {jaccard:>8.3f} {delta:>10.4f} {verdict:>10s}")

        except Exception as e:
            print(f"  {desc:<30s} {'ERROR':<10s} {'——':>8s} {'——':>10s} {str(e)[:20]:>10s}")

    # Summary by category
    print(f"\n  {'='*40}")
    print(f"  Summary by Category")
    print(f"  {'='*40}")
    for cat in ['risk', 'scoring', 'ensemble']:
        cat_results = [r for r in results if r['category'] == cat]
        if not cat_results:
            continue
        avg_j = np.mean([r['jaccard'] for r in cat_results])
        sig_count = sum(1 for r in cat_results if r['verdict'] in ('SIGNIFICANT', 'MODERATE'))
        print(f"  {cat:>12s}: avg Jaccard={avg_j:.3f}, {sig_count}/{len(cat_results)} significant")

    print(f"\n  ✓ Ablation complete. {'='*40}\n")
    return results
