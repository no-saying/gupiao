"""
Alpha158 因子子集 — 纯 pandas/numpy 实现，无 TA-Lib 依赖。

从 Qlib Alpha158 中选取用户 57 因子尚未覆盖的高价值因子。
新增约 80 个因子，兼容现有管线（NN 继续用 57 因子，LGBM 额外使用这些）。
"""
import numpy as np
import pandas as pd


ALPHA158_NEW_COLS = [
    # 标准偏差 (std/close)
    'STD5', 'STD10', 'STD20', 'STD30', 'STD60',
    # 分位数
    'QTLU20', 'QTLU60', 'QTLD20', 'QTLD60',
    # 极值位置
    'IMAX5', 'IMAX10', 'IMAX20', 'IMIN5', 'IMIN10', 'IMIN20', 'IMXD10', 'IMXD20',
    # 相关系数
    'CORR10', 'CORR20', 'CORD10', 'CORD20',
    # 计数因子
    'CNTP5', 'CNTP10', 'CNTP20', 'CNTN5', 'CNTN10', 'CNTN20',
    'CNTD5', 'CNTD10', 'CNTD20',
    # 涨跌比例
    'SUMP5', 'SUMP10', 'SUMP20', 'SUMN5', 'SUMN10', 'SUMN20',
    'SUMD5', 'SUMD10', 'SUMD20',
    # Beta 和 R²
    'BETA5', 'BETA10', 'BETA20', 'RSQR10', 'RSQR20',
    # 残差
    'RESI5', 'RESI10', 'RESI20',
    # 加权波动
    'WVMA5', 'WVMA10', 'WVMA20',
    # VWAP 比
    'VWAP0',
    # 极值比
    'MAX5', 'MAX10', 'MAX20', 'MIN5', 'MIN10', 'MIN20',
    # 排名
    'RANK5', 'RANK10', 'RANK20',
]


def compute_rolling_slope(series: pd.Series, w: int) -> pd.Series:
    """滚动线性回归斜率（相当于 talib.LINEARREG_SLOPE）。"""
    def _slope(arr):
        if len(arr) < w or np.isnan(arr).any():
            return np.nan
        x = np.arange(w, dtype=float)
        y = arr.astype(float)
        cov = np.cov(x, y, ddof=0)[0, 1]
        var = np.var(x, ddof=0)
        return cov / var if var > 1e-12 else 0.0
    return series.rolling(w).apply(_slope, raw=False)


def compute_rolling_rsqr(series: pd.Series, w: int) -> pd.Series:
    """滚动 R²（基于斜率推算）。"""
    slope = compute_rolling_slope(series, w)
    std_c = series.rolling(w).std(ddof=0)
    std_t = np.sqrt((w * w - 1) / 12.0)
    corr = (slope * std_t / (std_c + 1e-12)).clip(-1.0, 1.0)
    return corr ** 2


def compute_rolling_residual(series: pd.Series, w: int) -> pd.Series:
    """滚动回归残差（实际值 - 拟合值）。"""
    def _resid(arr):
        if len(arr) < w or np.isnan(arr).any():
            return np.nan
        x = np.arange(w, dtype=float)
        y = arr.astype(float)
        A = np.vstack([x, np.ones(w)]).T
        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        return y[-1] - (beta[0] * (w - 1) + beta[1])
    return series.rolling(w).apply(_resid, raw=False)


def add_alpha158_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    对 panel（MultiIndex: date, stock_id）逐股计算 Alpha158 子集因子。
    返回增加列后的 panel（不修改原始数据）。
    """
    result = panel.copy()
    stock_ids = result.index.get_level_values("stock_id").unique()
    total = len(stock_ids)

    close = result["close"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    volume = result["volume"].astype(float)
    amount = result["amount"].astype(float)
    vwap = amount / (volume + 1e-12)

    windows = [5, 10, 20, 30, 60]
    short_w = [5, 10, 20]

    print(f"[alpha158] Computing factors for {total} stocks...")

    # 标准偏差
    for w in windows:
        result[f'STD{w}'] = close.groupby(level="stock_id").transform(
            lambda s: s.rolling(w).std(ddof=0) / (s + 1e-12))

    # 分位数 (rolling quantile)
    for w in [20, 60]:
        result[f'QTLU{w}'] = close.groupby(level="stock_id").transform(
            lambda s: s.rolling(w).quantile(0.8) / (s + 1e-12))
        result[f'QTLD{w}'] = close.groupby(level="stock_id").transform(
            lambda s: s.rolling(w).quantile(0.2) / (s + 1e-12))

    # 极值位置 (argmax, argmin)
    for w in short_w:
        result[f'IMAX{w}'] = high.groupby(level="stock_id").transform(
            lambda s: s.rolling(w).apply(lambda x: np.argmax(x) / w if len(x) == w and not np.isnan(x).any() else np.nan, raw=False))
        result[f'IMIN{w}'] = low.groupby(level="stock_id").transform(
            lambda s: s.rolling(w).apply(lambda x: np.argmin(x) / w if len(x) == w and not np.isnan(x).any() else np.nan, raw=False))
    for w in [10, 20]:
        imax = result[f'IMAX{w}']
        imin = result[f'IMIN{w}']
        result[f'IMXD{w}'] = imax - imin

    # 相关系数 (close vs log_volume, close_return vs volume_return)
    log_vol = np.log(volume + 1)
    close_ret = close / close.shift(1)
    vol_log_ret = np.log((volume / volume.shift(1)).clip(lower=1e-9))
    for w in [10, 20]:
        result[f'CORR{w}'] = close.groupby(level="stock_id").transform(
            lambda s: s.rolling(w).corr(log_vol.loc[s.index]) if len(log_vol.loc[s.index]) == len(s) else np.nan)
        # Hmm, this per-group correlation needs a different approach
        # Let me compute it differently
    # Better approach for correlation - compute on the full panel
    for w in [10, 20]:
        corr_vals = close.rolling(w).corr(log_vol)
        result[f'CORR{w}'] = corr_vals
        cord_vals = close_ret.rolling(w).corr(vol_log_ret)
        result[f'CORD{w}'] = cord_vals

    # 计数因子
    pos = (close > close.shift(1)).astype(float)
    neg = (close < close.shift(1)).astype(float)
    for w in short_w:
        result[f'CNTP{w}'] = pos.groupby(level="stock_id").transform(lambda s: s.rolling(w).mean())
        result[f'CNTN{w}'] = neg.groupby(level="stock_id").transform(lambda s: s.rolling(w).mean())
        result[f'CNTD{w}'] = result[f'CNTP{w}'] - result[f'CNTN{w}']

    # 涨跌比例
    diff_abs = (close - close.shift(1)).abs()
    diff_up = (close - close.shift(1)).clip(lower=0)
    diff_down = (-(close - close.shift(1))).clip(lower=0)
    for w in short_w:
        sum_abs = diff_abs.groupby(level="stock_id").transform(lambda s: s.rolling(w).sum())
        sum_up = diff_up.groupby(level="stock_id").transform(lambda s: s.rolling(w).sum())
        sum_down = diff_down.groupby(level="stock_id").transform(lambda s: s.rolling(w).sum())
        result[f'SUMP{w}'] = sum_up / (sum_abs + 1e-12)
        result[f'SUMN{w}'] = sum_down / (sum_abs + 1e-12)
        result[f'SUMD{w}'] = (sum_up - sum_down) / (sum_abs + 1e-12)

    # Beta, R², Residual (按每只股票计算，太慢只算短窗口)
    for i, sid in enumerate(stock_ids):
        if (i + 1) % 50 == 0:
            print(f"  [alpha158] regression {i+1}/{total}")
        try:
            sd = result.xs(sid, level="stock_id").sort_index()
            c = sd["close"].astype(float)
            for w in short_w:
                if len(c) >= w:
                    sl = compute_rolling_slope(c, w)
                    result.loc[(slice(None), sid), f'BETA{w}'] = sl.values / (c.values + 1e-12)
            for w in [10, 20]:
                if len(c) >= w:
                    result.loc[(slice(None), sid), f'RSQR{w}'] = compute_rolling_rsqr(c, w).values
                    result.loc[(slice(None), sid), f'RESI{w}'] = (compute_rolling_residual(c, w) / (c + 1e-12)).values
        except Exception:
            continue

    # 加权波动
    vol_weighted_ret = (close / close.shift(1) - 1).abs() * volume
    for w in short_w:
        mean_vwr = vol_weighted_ret.groupby(level="stock_id").transform(lambda s: s.rolling(w).mean())
        std_vwr = vol_weighted_ret.groupby(level="stock_id").transform(lambda s: s.rolling(w).std(ddof=0))
        result[f'WVMA{w}'] = std_vwr / (mean_vwr + 1e-12)

    # VWAP ratio
    result['VWAP0'] = vwap / (close + 1e-12)

    # 极值比
    for w in [5, 10, 20]:
        result[f'MAX{w}'] = high.groupby(level="stock_id").transform(lambda s: s.rolling(w).max()) / (close + 1e-12)
        result[f'MIN{w}'] = low.groupby(level="stock_id").transform(lambda s: s.rolling(w).min()) / (close + 1e-12)

    # 排名
    for w in short_w:
        result[f'RANK{w}'] = close.groupby(level="stock_id").transform(lambda s: s.rolling(w).rank(pct=True))

    # 统一填充
    for col in ALPHA158_NEW_COLS:
        if col in result.columns:
            result[col] = result[col].fillna(0).replace([np.inf, -np.inf], 0).astype(float)

    n_added = len([c for c in ALPHA158_NEW_COLS if c in result.columns])
    print(f"[alpha158] Added {n_added} factors")
    return result
