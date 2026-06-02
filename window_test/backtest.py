#!/usr/bin/env python3
"""
window_test/backtest.py — 跨窗口滚动后验回测

流程（每个窗口）:
  1. 候选池: 取模型评分 top-40
  2. 【纳什均衡】: 候选>NASH_MIN 时，在 top-40 上求解纳什 → 多样化候选池
  3. 精度门控: 近5日动量>0, 回撤>-8%
  4. 波动率过滤: 剔除波动率 > median×1.2
  5. Top-5: 取剩余候选中的最高分 5 只
  6. 加权: equal / softmax / inv_vol
  7. 评分: 使用 calculate_predict_weight_score (与 score_self.py 一致)

用法:
  python window_test/backtest.py --n-windows 14
  python window_test/backtest.py --fast
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import MODEL_DIR, DEVICE, MAX_STOCKS
from data_loader import get_official_stock_ids, build_panel_from_official
from features import engineer_features, make_window_samples

from controller import (
    build_lgbm_data, load_predict, nash_equilibrium_selection,
    GP_WEIGHTS, NEW_FEATS, EXCLUDE,
)
from score_self import calculate_predict_weight_score

OUT_DIR = Path(__file__).resolve().parent.parent / "output" / "window_test"


def norm(s):
    """Min-max 归一化 [0,1]"""
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-9:
        return np.zeros_like(s)
    return (s - mn) / (mx - mn)


def compute_score_self(sel_ids, weights, panel, test_date_ts, n_days=5):
    """
    模拟 score_self.py 的 calculate_predict_weight_score:
    取 test_date 为起点的连续 n_days 个交易日，计算每只股票
    (末开盘-首开盘)/首开盘 的加权和。
    """
    df = panel.reset_index()[['date', 'stock_id', 'open']].copy()
    # 取从 test_date 开始的 n_days 个交易日
    all_dates = sorted(df['date'].unique())
    try:
        start_pos = all_dates.index(test_date_ts)
    except ValueError:
        return 0.0
    end_pos = min(start_pos + n_days, len(all_dates))
    period_dates = all_dates[start_pos:end_pos]

    period = df[df['date'].isin(period_dates)]
    period = period[period['stock_id'].isin(sel_ids)]

    stock_returns = {}
    for sid in sel_ids:
        sd = period[period['stock_id'] == sid].sort_values('date')
        if len(sd) >= 2:
            ret = (sd['open'].iloc[-1] - sd['open'].iloc[0]) / sd['open'].iloc[0]
        else:
            ret = 0.0
        stock_returns[sid] = ret

    return sum(stock_returns.get(sid, 0) * w for sid, w in zip(sel_ids, weights))


def run_selection(
    stock_ids, stock_idx_map, panel,
    scores, mask, test_ts,
    use_nash=False, nash_lam=0.25, nash_min=12, n_features=57, X_lt=None,
    no_overbought=False, diverse_industry=None, market_state=False,
):
    """
    选股流水线：纳什均衡 → 精度门控 → 波动率过滤 → Top-5
    """
    scores_arr = np.array([scores.get(s, -999) for s in stock_ids])
    valid = np.where(mask > 0.5)[0]

    if len(valid) < 5:
        return [], []

    # ═══ Step 1: 候选池 top-40 ═══
    pool = valid[np.argsort(scores_arr[valid])[-40:]]
    pool_ids = [stock_ids[i] for i in pool]
    pool_sc = [scores_arr[i] for i in pool]

    # ═══ Step 2: 纳什均衡（在 top-40 上，门控前） ═══
    if use_nash and len(pool_ids) > nash_min:
        try:
            pool_ids, pool_sc = nash_equilibrium_selection(
                pool_ids, pool_sc, stock_idx_map, panel, X_lt, nash_lam,
                n_features=n_features,
            )
        except Exception:
            pass

    # ═══ Step 3: 精度门控 + 超买过滤 ═══
    precision_ids, psc = [], []
    for sid, scv in zip(pool_ids, pool_sc):
        try:
            sd = panel.xs(sid, level="stock_id")
            sd_hist = sd[sd.index <= test_ts]
            if len(sd_hist) < 5:
                ret_5d, dd = 0, -0.01
            else:
                ret_5d = sd_hist['pctChg'].iloc[-5:].mean() / 100.0
                dd_series = (sd_hist['close'] / sd_hist['close'].rolling(20).max() - 1.0).iloc[-20:]
                dd = dd_series.min() if len(dd_series) > 0 else -0.01
        except Exception:
            ret_5d, dd = 0, -0.01

        # 超买过滤: RSI>75 或 10日涨幅>20%
        if no_overbought:
            try:
                rsi_val = float(sd_hist['rsi_14'].iloc[-1]) if 'rsi_14' in sd_hist.columns else 50
                ret_10d = sd_hist['pctChg'].iloc[-10:].sum() / 100.0
                if rsi_val > 75 or ret_10d > 0.20:
                    continue
            except:
                pass

        if ret_5d > 0 and dd > -0.08:
            precision_ids.append(sid)
            psc.append(scv)

    if len(precision_ids) < 2:
        return pool_ids[:MAX_STOCKS], pool_sc[:MAX_STOCKS]

    # ═══ Step 4: 波动率过滤 ═══
    pool_vols = []
    for sid in pool_ids:
        try:
            sd = panel.xs(sid, level="stock_id")
            sd_hist = sd[sd.index <= test_ts]
            pool_vols.append(sd_hist['pctChg'].iloc[-20:].std() / 100.0)
        except Exception:
            pool_vols.append(0.03)
    vol_med = np.median(pool_vols) if pool_vols else 0.03

    final_ids, fsc = [], []
    for sid, scv in zip(precision_ids, psc):
        try:
            sd = panel.xs(sid, level="stock_id")
            sd_hist = sd[sd.index <= test_ts]
            v = sd_hist['pctChg'].iloc[-20:].std() / 100.0
        except Exception:
            v = 0.03
        if v <= vol_med * 1.2:
            final_ids.append(sid)
            fsc.append(scv)

    # ═══ Step 5: 市场状态门控（20日均线向下时切防御） ═══
    if market_state and len(final_ids) >= 2:
        try:
            first_stock = panel.index.get_level_values('stock_id').unique()[0]
            sd_ref = panel.xs(first_stock, level='stock_id')
            sd_ref = sd_ref[sd_ref.index <= test_ts]
            market_20d = sd_ref['pctChg'].iloc[-20:].sum() / 100.0 if len(sd_ref) >= 20 else 0
            market_5d = sd_ref['pctChg'].iloc[-5:].mean() / 100.0 if len(sd_ref) >= 5 else 0
            defensive_kw = ['煤炭', '石油', '公用', '银行', '电力', '交运', '钢铁', '建筑']
            if market_20d < -0.03 and market_5d < 0:
                def_filtered = [(s, c) for (s, c) in zip(final_ids, fsc)
                                if any(kw in str(panel.xs(s, level='stock_id')['industry'].iloc[-1] if 'industry' in panel.columns else '') for kw in defensive_kw)]
                if len(def_filtered) >= 2:
                    final_ids = [s for s, c in def_filtered]
                    fsc = [c for s, c in def_filtered]
        except:
            pass

    # ═══ Step 6: 行业分散（同行业最多 N 只） ═══
    if diverse_industry is not None and diverse_industry > 0 and 'industry' in panel.columns:
        sel_ordered = sorted(zip(final_ids, fsc), key=lambda x: -x[1])
        ind_count = {}
        new_ids, new_sc = [], []
        for sid, scv in sel_ordered:
            try:
                ind = str(panel.xs(sid, level='stock_id')['industry'].iloc[-1])
            except:
                ind = 'Unknown'
            if ind_count.get(ind, 0) < diverse_industry:
                ind_count[ind] = ind_count.get(ind, 0) + 1
                new_ids.append(sid)
                new_sc.append(scv)
        if len(new_ids) >= max(2, MAX_STOCKS // 2):
            final_ids, fsc = new_ids, new_sc

    # ═══ Step 7: Top-K ═══
    if len(final_ids) >= 1:
        topk = min(MAX_STOCKS, len(final_ids))
        order = np.argsort(fsc)[-topk:]
        return [final_ids[i] for i in order], [fsc[i] for i in order]
    else:
        ti = np.argsort(scores_arr[valid])[-MAX_STOCKS:]
        return [stock_ids[i] for i in ti], [scores_arr[i] for i in ti]


def allocate_weights(sel_ids, sel_sc, wtype, panel, test_ts,
                     cash_buffer: float | None = None):
    sc_arr = np.array(sel_sc)
    n = len(sel_ids)
    if n == 0:
        return np.array([])
    if wtype == "equal":
        return np.ones(n) / n
    elif wtype == "softmax":
        x = sc_arr - sc_arr.max()
        exp_x = np.exp(x / 0.3)
        return exp_x / exp_x.sum()
    elif wtype == "inv_vol":
        vols = []
        for sid in sel_ids:
            try:
                sd = panel.xs(sid, level="stock_id")
                v = float(sd[sd.index <= test_ts]['pctChg'].std()) / 100.0
            except Exception:
                v = 0.03
            vols.append(max(v, 0.005))
        inv = 1.0 / np.array(vols)
        return inv / inv.sum()
    elif wtype == "bdc":
        # BDC 风格：预测比例加权 + 单票权重上限 0.5
        raw = np.clip(np.array(sel_sc, dtype=float), 0, None)
        if raw.sum() <= 1e-12:
            raw = np.ones_like(raw)
        invested = n / min(MAX_STOCKS, max(n, 1))
        w = invested * raw / raw.sum()
        cap = 0.5
        if cap < 1.0:
            capped = np.clip(w, None, cap)
            excess = max(0.0, float(w.sum() - capped.sum()))
            if excess > 1e-12:
                room = np.clip(cap - capped, 0, None)
                total_room = float(room.sum())
                if total_room > 1e-12:
                    capped += room / total_room * min(excess, total_room)
                    capped = np.clip(capped, None, cap)
            w = capped
        return w
    w = np.ones(n) / n

    # ── 自适应现金仓位 ──
    if cash_buffer is not None and cash_buffer > 0:
        top_scores = np.sort(sc_arr)
        signal_strength = top_scores[-1] - top_scores[-min(10, n)]
        buffer_threshold = cash_buffer * float(np.std(sc_arr)) if np.std(sc_arr) > 1e-9 else 0.05
        cash_pct = 0.0
        if signal_strength < buffer_threshold:
            cash_pct = 0.15
        if signal_strength < buffer_threshold * 0.5:
            cash_pct = 0.30
        if cash_pct > 0:
            w = w * (1.0 - cash_pct)
    return w


def main():
    parser = argparse.ArgumentParser(description="滚动后验回测")
    parser.add_argument("--n-windows", type=int, default=14,
                        help="测试窗口数")
    parser.add_argument("--fast", action="store_true",
                        help="快速: 5 窗口")
    parser.add_argument("--nash-min", type=int, default=12,
                        help="纳什触发最小候选数（默认12）")
    parser.add_argument("--nash-lam", type=float, default=0.25,
                        help="纳什 λ")
    parser.add_argument("--cash-buffer", type=float, default=None,
                        help="自适应现金仓位阈值")
    parser.add_argument("--no-overbought", action="store_true",
                        help="超买过滤: RSI>75 或 10日涨幅>20%")
    parser.add_argument("--diverse-industry", type=int, default=None,
                        help="行业分散: 同行业最多N只")
    parser.add_argument("--market-state", action="store_true",
                        help="市场状态门控")
    args = parser.parse_args()

    n_windows = 5 if args.fast else args.n_windows
    NASH_MIN = args.nash_min
    NASH_LAM = args.nash_lam
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    strategies = [
        ("LGBM_Only",       "lgbm",    False),
        ("LGBM_Nash",       "lgbm",    True),
        ("NN_GP",           "gp",      False),
        ("NN_GP_Nash",      "gp",      True),
        ("LGBM_NN_Blend",   "lgbm-nn", False),
        ("LGBM_NN_BlendNash", "lgbm-nn", True),
        ("Nash_Blend_Fusion","fusion", True),
    ]
    weight_types = ["equal", "softmax", "inv_vol", "bdc"]
    t0 = time.time()

    # ═══ 1. 加载数据 ═══
    print("=" * 60)
    print("滚动后验回测 — 纳什在门控前 + score_self 评分")
    print("=" * 60)

    print("\n[1/4] 加载数据 & 特征...")
    stock_ids = get_official_stock_ids()
    panel = build_panel_from_official(stock_ids)
    panel = engineer_features(panel, stock_ids)
    stock_idx_map = {s: i for i, s in enumerate(stock_ids)}

    print("\n[2/4] 构建样本 & LGBM 数据...")
    X_all, y_all, mask_all, dates_all = make_window_samples(
        panel, stock_ids, normalize=False)
    n_total = len(X_all)
    n_features = X_all.shape[-1]
    print(f"  样本: {n_total}, 特征: {n_features}, 日期: {dates_all[0]} ~ {dates_all[-1]}")

    print("  预构建 LGBM 数据...")
    lgbm_df, feat_cols = build_lgbm_data(panel)
    lgbm_all_dates = sorted(lgbm_df['date'].unique())
    print(f"  LGBM: {len(lgbm_df)} 行, {len(lgbm_all_dates)} 交易日")

    # ═══ 3. 测试窗口 ═══
    gap = 20
    min_train = 250
    step = max(50 if args.fast else 15, (n_total - min_train - gap) // n_windows)
    test_indices = list(range(min_train + gap, n_total - 1, step))[:n_windows]
    print(f"\n[3/4] 窗口: {len(test_indices)} 个, 步长={step}")
    print(f"  日期: {dates_all[test_indices[0]]} ~ {dates_all[test_indices[-1]]}")

    # ═══ 4. 回测循环 ═══
    print(f"\n[4/4] 执行回测...")

    all_records = []

    for wi, w_idx in enumerate(test_indices):
        test_date = dates_all[w_idx]
        test_ts = pd.Timestamp(test_date)

        X_win = X_all[w_idx:w_idx + 1]
        m_win = mask_all[w_idx]

        print(f"\n  [窗口 {wi+1}/{len(test_indices)}] idx={w_idx} date={test_date}")

        # ── a) LightGBM ──
        t_lgbm = time.time()
        trainable_dates = [d for d in lgbm_all_dates if d <= test_ts]
        if len(trainable_dates) >= 30:
            train_dates_set = set(trainable_dates[:-5])
            train_df = lgbm_df[lgbm_df['date'].isin(train_dates_set)].copy()
            fm = train_df[feat_cols].mean()
            fs = train_df[feat_cols].std().replace(0, 1)
            X_tr = (train_df[feat_cols].values - fm.values) / fs.values
            y_tr = train_df['rank_label'].values.astype(int)
            grp = train_df.groupby('date').size().values

            import lightgbm as lgb
            lgbm_model = lgb.LGBMRanker(
                objective='lambdarank', boosting_type='gbdt',
                n_estimators=350, num_leaves=68, learning_rate=0.0196,
                min_child_samples=42, reg_lambda=0.0022, reg_alpha=0.0054,
                subsample=0.738, colsample_bytree=0.939,
                label_gain=[i for i in range(10)], verbose=-1, random_state=42)
            lgbm_model.fit(X_tr, y_tr, group=grp,
                           eval_metric=['ndcg'], callbacks=[lgb.log_evaluation(0)])

            test_df = lgbm_df[lgbm_df['date'] == test_ts].copy()
            if len(test_df) > 0:
                X_te = (test_df[feat_cols].values - fm.values) / fs.values
                test_df['lgb_score'] = lgbm_model.predict(X_te)
                lgbm_scores = test_df.set_index('stock_id')['lgb_score'].to_dict()
            else:
                lgbm_scores = {s: 0.0 for s in stock_ids}
        else:
            lgbm_scores = {s: 0.0 for s in stock_ids}
        t_lgbm = time.time() - t_lgbm

        # ── b) NN GP ──
        t_nn = time.time()
        gp_scores = np.zeros(len(stock_ids))
        try:
            for name, w in GP_WEIGHTS.items():
                if w > 0:
                    seed, fold = name.split("_f")
                    gp_scores += w * load_predict(
                        MODEL_DIR / f"portfolio_model_{seed}_v2_fold{fold}.pt",
                        X_win, n_features)
        except Exception as e:
            print(f"    [WARN] NN: {e}")
        gp_dict = {s: float(gp_scores[i]) for i, s in enumerate(stock_ids)}
        t_nn = time.time() - t_nn

        # ── c) Blend ──
        lgbm_arr = np.array([lgbm_scores.get(s, 0) for s in stock_ids])
        if np.abs(lgbm_arr).max() < 1e-9:
            blend_dict = gp_dict
        else:
            blend_arr = 0.7 * norm(lgbm_arr) + 0.3 * norm(gp_scores)
            blend_dict = {s: float(blend_arr[i]) for i, s in enumerate(stock_ids)}

        # ── e) Fusion: LGBM_Nash(0.6) + BlendNash(0.4) 分数融合 ──
        fs_arr = np.zeros(len(stock_ids))
        lgbm_nash_arr = lgbm_arr if np.abs(lgbm_arr).max() >= 1e-9 else np.zeros(len(stock_ids))
        for i, s in enumerate(stock_ids):
            ls = norm(lgbm_nash_arr)[i]
            bs = norm(gp_scores)[i]
            fs_arr[i] = 0.6 * ls + 0.4 * bs
        fusion_scores = {s: float(fs_arr[i]) for i, s in enumerate(stock_ids)}

        sources = {"lgbm": lgbm_scores, "gp": gp_dict, "lgbm-nn": blend_dict, "fusion": fusion_scores}

        # ── d) 执行选股 → 评分 ──
        for sname, ensemble, use_nash in strategies:
            sd = sources.get(ensemble, sources["lgbm"])
            if ensemble == "lgbm" and np.abs(lgbm_arr).max() < 1e-9:
                continue

            sel_ids, sel_sc = run_selection(
                stock_ids, stock_idx_map, panel, sd, m_win, test_ts,
                use_nash=use_nash, nash_lam=NASH_LAM, nash_min=NASH_MIN,
                n_features=n_features, X_lt=X_win,
                no_overbought=args.no_overbought,
                diverse_industry=args.diverse_industry,
                market_state=args.market_state,
            )
            if len(sel_ids) == 0:
                continue

            for wtype in weight_types:
                weights = allocate_weights(
                    sel_ids, sel_sc, wtype, panel, test_ts,
                    cash_buffer=args.cash_buffer)

                # ── 评分: 使用 score_self 风格（5天open-to-open） ──
                score = compute_score_self(sel_ids, weights, panel, test_ts, n_days=5)

                cb_label = f"CB{args.cash_buffer}" if args.cash_buffer else "NoCB"
                all_records.append({
                    "window":       wi + 1,
                    "test_date":    str(test_date),
                    "strategy":     f"{sname}_{cb_label}",
                    "ensemble":     ensemble,
                    "use_nash":     int(use_nash),
                    "cash_buffer":  args.cash_buffer or 0,
                    "weight_type":  wtype,
                    "n_stocks":     len(sel_ids),
                    "score":        round(score, 8),
                })

        print(f"    LGBM={t_lgbm:.1f}s NN={t_nn:.1f}s")

    # ═══ 5. 统计 ═══
    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"汇总统计 (耗时 {elapsed:.0f}s)")
    print(f"{'=' * 60}")

    df = pd.DataFrame(all_records)
    df.to_csv(OUT_DIR / "per_window_results.csv", index=False)

    # 详细统计
    stats = []
    for sname in df["strategy"].unique():
        for wtype in weight_types:
            sub = df[(df["strategy"] == sname) & (df["weight_type"] == wtype)]
            if len(sub) < 3:
                continue
            sc = sub["score"].values
            nash_lbl = "Nash" if sub["use_nash"].iloc[0] else "NoNash"
            stats.append({
                "Strategy":  sname, "Weight": wtype, "Nash": nash_lbl,
                "N_Win":     len(sub),
                "Mean":      np.mean(sc),
                "Std":       np.std(sc),
                "Min":       np.min(sc),
                "Max":       np.max(sc),
                "Median":    np.median(sc),
                "Q25":       np.percentile(sc, 25),
                "Q75":       np.percentile(sc, 75),
                "Variance":  np.var(sc),
                "Sharpe":    np.mean(sc) / (np.std(sc) + 1e-9),
                "Win_Rate":  (sc > 0).mean(),
            })

    stats_df = pd.DataFrame(stats).sort_values("Mean", ascending=False)
    stats_df.to_csv(OUT_DIR / "strategy_statistics.csv", index=False)

    # 聚合
    agg = []
    for sname in df["strategy"].unique():
        sub = df[df["strategy"] == sname]
        if len(sub) < 3:
            continue
        sc_all = sub["score"].values
        best_wt = sub.groupby("weight_type")["score"].mean()
        agg.append({
            "Strategy": sname,
            "CashBuf": sub["cash_buffer"].iloc[0],
            "N_Total":  len(sub),
            "Mean":     np.mean(sc_all),
            "Std":      np.std(sc_all),
            "Sharpe":   np.mean(sc_all) / (np.std(sc_all) + 1e-9),
            "WinRate":  (sc_all > 0).mean(),
            "BestWt":   best_wt.idxmax(),
            "BestMean": best_wt.max(),
        })

    agg_df = pd.DataFrame(agg).sort_values("Sharpe", ascending=False)
    agg_df.to_csv(OUT_DIR / "strategy_aggregate.csv", index=False)

    # 打印
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 160)
    print("\n策略详细统计排名 (Score, 按 Mean 降序)")
    print(stats_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    print("\n策略聚合排名 (按 Sharpe 降序)")
    print(agg_df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))

    best = stats_df.iloc[0]
    print(f"\n{'=' * 60}")
    print(f"🏆 最佳: {best['Strategy']} (权重={best['Weight']}, {best['Nash']})")
    print(f"   Mean Score={best['Mean']:.6f}, Sharpe={best['Sharpe']:.4f}, WinRate={best['Win_Rate']:.2%}")

    # 报告
    with open(OUT_DIR / "report.txt", "w") as f:
        f.write("滚动后验回测报告\n")
        f.write(f"生成: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"窗口: {len(test_indices)} 个, 日期: {dates_all[test_indices[0]]} ~ {dates_all[test_indices[-1]]}\n")
        f.write(f"纳什: 阈值>{NASH_MIN}, λ={NASH_LAM}, 在门控前执行\n")
        f.write(f"评分: 5日open-to-open (对齐 score_self)\n\n")
        f.write("详细统计排名 (按 Mean 降序):\n")
        f.write(stats_df.to_string(index=False))
        f.write("\n\n聚合排名 (按 Sharpe 降序):\n")
        f.write(agg_df.to_string(index=False))
        f.write(f"\n\n🏆 {best['Strategy']} (权重={best['Weight']}, {best['Nash']})")

    print(f"\n结果: {OUT_DIR}/")
    print(f"  明细: per_window_results.csv")
    print(f"  统计: strategy_statistics.csv")
    print(f"  聚合: strategy_aggregate.csv")
    print(f"  报告: report.txt")


if __name__ == "__main__":
    main()
