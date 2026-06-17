"""
core/data.py — LGBM 训练数据构建

标签: open[t+4]/open[t] - 1
Target: 行业内 L2 Z-score
特征: 截面 rank 化 (groupby date → rank pct)
"""

import numpy as np
import pandas as pd

EXCLUDE_COLS = {
    'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount',
    'turn', 'pctChg', 'peTTM', 'pbMRQ', 'stock_id',
    'vol_ma5', 'vol_ma20', 'turn_ma5', 'turn_ma20',
    'industry', 'amplitude', 'change', 'date', 'index_name',
    'adj_factor', 'tradestatus', 'adjustflag',
    'roe_ttm', 'np_margin', 'gp_margin', 'eps_ttm',
    'current_ratio', 'debt_to_asset', 'profit_yoy', 'equity_yoy',
    'buy_sm_vol', 'buy_sm_amount', 'sell_sm_vol', 'sell_sm_amount',
    'buy_md_vol', 'buy_md_amount', 'sell_md_vol', 'sell_md_amount',
    'buy_lg_vol', 'buy_lg_amount', 'sell_lg_vol', 'sell_lg_amount',
    'buy_elg_vol', 'buy_elg_amount', 'sell_elg_vol', 'sell_elg_amount',
    'net_mf_vol', 'net_mf_amount',
    'rzye', 'rqye', 'rzmre', 'rqyl', 'rzche', 'rqchl', 'rqmcl', 'rzrqye',
    'north_hold_vol', 'north_hold_ratio',
    'total_share', 'float_share', 'free_share', 'total_mv', 'circ_mv',
    'turnover_rate', 'turnover_rate_f', 'volume_ratio',
    'cost_5pct', 'cost_15pct', 'cost_50pct', 'cost_85pct', 'cost_95pct',
    'weight_avg', 'winner_rate', 'holder_num',
    'limit_up_times', 'limit_down_times', 'broken_board_times',
    'pe', 'pb', 'ps', 'dv_ratio',
    'l1_code', 'l1_name', 'l2_code', 'l3_code', 'l3_name',
    'market_ret', 'nn_score',
}


def build_lgbm_data(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """构建 LGBM 训练数据。"""

    feature_cols = [c for c in panel.columns
                    if c not in EXCLUDE_COLS and c not in ('date', 'stock_id', 'index_name', 'target_raw')]

    keep_cols = ['date', 'stock_id', 'open'] + feature_cols
    for extra in ['l2_name', 'industry']:
        if extra in panel.columns:
            keep_cols.append(extra)

    df = panel.reset_index()[keep_cols].copy()
    df = df.sort_values(['date', 'stock_id']).reset_index(drop=True)

    # ── 1. 标签 ──
    open_pivot = df.pivot(index='date', columns='stock_id', values='open')
    open_future = open_pivot.shift(-4).values
    target_matrix = open_future / open_pivot.values - 1.0
    target_df = pd.DataFrame(target_matrix, index=open_pivot.index, columns=open_pivot.columns)
    target_long = target_df.stack().reset_index()
    target_long.columns = ['date', 'stock_id', 'target_raw']
    df = df.merge(target_long, on=['date', 'stock_id'], how='left')
    df = df.dropna(subset=['target_raw']).reset_index(drop=True)
    df['target_raw'] = df['target_raw'].replace([np.inf, -np.inf], np.nan).clip(-0.15, 0.20)

    # ── 2. 行业 Z-score ──
    ind_col = 'l2_name' if 'l2_name' in df.columns else (
        'industry' if 'industry' in df.columns else None)
    if ind_col:
        target_pivot = df.pivot(index='date', columns='stock_id', values='target_raw')
        industry_map = df.drop_duplicates('stock_id').set_index('stock_id')[ind_col].to_dict()
        cols = list(target_pivot.columns)
        stock_to_col = {s: i for i, s in enumerate(cols)}
        ind_groups = {}
        for s in cols:
            ind = industry_map.get(s, 'unknown')
            ind_groups.setdefault(ind, []).append(stock_to_col[s])
        mat = target_pivot.values
        result = np.zeros_like(mat, dtype=np.float64)
        n_rows = mat.shape[0]
        for ci in ind_groups.values():
            if len(ci) < 3:
                continue
            idx_arr = np.array(ci, dtype=int)
            for t in range(n_rows):
                row = mat[t, idx_arr]
                valid = ~np.isnan(row)
                if valid.sum() < 3:
                    continue
                vals = row[valid]
                m, s = np.nanmean(vals), np.nanstd(vals)
                if s > 1e-8:
                    result[t, idx_arr[valid]] = np.clip((vals - m) / s, -5, 5)
        target_z = pd.DataFrame(result, index=target_pivot.index, columns=target_pivot.columns)
        target_z_long = target_z.stack().reset_index()
        target_z_long.columns = ['date', 'stock_id', 'target']
        df = df.merge(target_z_long, on=['date', 'stock_id'], how='left')
        df['target'] = df['target'].fillna(0).clip(-5, 5)
    else:
        df['target'] = df['target_raw']

    # ── 3. 截面 rank — 分批 rank 全部数值列一起做 ──
    num_cols = [c for c in feature_cols
                if pd.api.types.is_numeric_dtype(df[c])
                and c not in ('open', 'adj_factor', 'target', 'target_raw')]

    # 用 transform rank 对所有列做（一次 groupby）
    df[num_cols] = df.groupby('date')[num_cols].rank(pct=True)
    df[num_cols] = df[num_cols].fillna(0.5)

    for col in [c for c in feature_cols if c not in num_cols]:
        if col in df.columns:
            df[col] = df[col].fillna('')

    df = df.drop(columns=['open', 'target_raw'], errors='ignore')

    return df, num_cols
