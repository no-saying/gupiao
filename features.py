"""
=============================================================================
  特征工程模块 —— 从原始行情数据中提取预测因子
=============================================================================

因子分类（共 50+ 个特征）：

  一、动量因子（4 个）         ret_5d, ret_10d, ret_20d, ret_60d
  二、波动率因子（3 个）       vol_5d, vol_10d, vol_20d
  三、均线偏离因子（4 个）     ma5_dev, ma10_dev, ma20_dev, ma60_dev
  四、量价因子（4 个）         volume_ratio_5, volume_ratio_20, turn_change, amplitude
  五、技术指标因子（4 个）     rsi_14, macd, macd_signal, macd_hist
  六、风险与位置因子（2 个）   max_dd_20d, price_position
  七、估值因子（2 个）         peTTM, pbMRQ
  八、市场因子（1 个）         beta_60d
  九、布林带（2 个）           bb_dev, bb_width
  十、ATR（1 个）              atr_14
  十一、KDJ（3 个）            kdj_k, kdj_d, kdj_j
  十二、OBV（1 个）            obv_norm
  十三、Williams %R（1 个）    wr_14
  十四、截面排名（4 个）       ret_5d_rank, ret_20d_rank, vol_5d_rank, rsi_rank
  十五、行业特征（6 个）       ind_ret_5d, ind_ret_20d, alpha_ret_5d/20d, ind_vol_5d, industry_size
  十六、指数特征（12 个）      4 指数 × 3 窗口
=============================================================================
"""

import numpy as np
import pandas as pd

from config import (
    LOOKBACK_DAYS, PREDICT_HORIZON, STEP_DAYS,
    MOMENTUM_WINDOWS, VOLATILITY_WINDOWS, MA_WINDOWS,
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BETA_WINDOW,
    EXTRA_INDEX_RET_WINDOWS, EFFECTIVE_START, RAW_DIR,
)
from data_loader import fetch_index_data, build_event_mask, fetch_stock_industries, fetch_fundamental_data
from sklearn.linear_model import LinearRegression

# 特征标准化参数（由 make_window_samples 设置，供 train.py/predict.py 加载）
_NORM_STATS = None  # (feat_mean, feat_std)

def get_norm_stats():
    """返回特征标准化参数 (feat_mean, feat_std)。"""
    return _NORM_STATS


# =============================================================================
# 单只股票的因子计算
# =============================================================================

def _per_stock(group: pd.DataFrame) -> pd.DataFrame:
    """
    对单只股票的时间序列计算所有技术因子。
    使用 pandas 向量化操作，不含 Python 循环。
    """
    df = group.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    preclose = df["preclose"]
    volume = df["volume"]

    # 一、动量因子
    ret_1d = close.pct_change()
    for w in MOMENTUM_WINDOWS:
        df[f"ret_{w}d"] = close.pct_change(w)
        df[f"ret_{w}d"] = df[f"ret_{w}d"].replace([np.inf, -np.inf], np.nan)

    # 二、波动率因子
    for w in VOLATILITY_WINDOWS:
        df[f"vol_{w}d"] = ret_1d.rolling(w).std()

    # 三、均线偏离
    for w in MA_WINDOWS:
        ma = close.rolling(w).mean()
        df[f"ma{w}_dev"] = (close - ma) / (ma + 1e-9)

    # 四、量价因子
    df["vol_ma5"] = volume.rolling(5).mean()
    df["vol_ma20"] = volume.rolling(20).mean()
    df["volume_ratio_5"] = volume / df["vol_ma5"] - 1.0
    df["volume_ratio_20"] = volume / df["vol_ma20"] - 1.0
    if "turn" in df.columns:
        df["turn_ma5"] = df["turn"].rolling(5).mean()
        df["turn_ma20"] = df["turn"].rolling(20).mean()
        df["turn_change"] = df["turn"] / df["turn_ma5"] - 1.0

    # 五、振幅（THU-BDC 标准：振幅% = (最高-最低)/昨收 × 100）
    df["amplitude"] = (high - low) / preclose.replace(0, np.nan) * 100.0

    # 六、RSI
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(RSI_PERIOD).mean()
    avg_loss = loss.rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

    # 七、MACD
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # 八、最大回撤
    roll_max = close.rolling(20).max()
    dd = close / roll_max - 1.0
    df["max_dd_20d"] = dd.rolling(20).min()

    # 九、价格位置
    df["price_position"] = (close - low.rolling(20).min()) / (
        high.rolling(20).max() - low.rolling(20).min() + 1e-9)

    # 十、布林带
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df["bb_dev"] = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)
    df["bb_width"] = (bb_upper - bb_lower) / (bb_mid + 1e-9)

    # 十一、ATR
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean() / (close + 1e-9)

    # 十二、KDJ
    low_9 = low.rolling(9).min()
    high_9 = high.rolling(9).max()
    rsv = (close - low_9) / (high_9 - low_9 + 1e-9) * 100
    df["kdj_k"] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(alpha=1/3, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # 十三、OBV
    obv = (volume * (delta / delta.abs()).fillna(0)).cumsum()
    obv_ma20 = obv.rolling(20).mean()
    df["obv_norm"] = (obv - obv_ma20) / (obv_ma20 + 1e-9)

    # 十四、Williams %R
    high_14 = high.rolling(14).max()
    low_14 = low.rolling(14).min()
    df["wr_14"] = -100 * (high_14 - close) / (high_14 - low_14 + 1e-9)

    # ── 十五、缺口比率（隔夜跳空）───────────────────────────────
    # (开盘-昨收)/昨收，衡量隔夜市场情绪变化
    df["gap_ratio"] = (df["open"].fillna(preclose) - preclose) / preclose.replace(0, np.nan)
    df["gap_ratio"] = df["gap_ratio"].replace([np.inf, -np.inf], np.nan)

    # ── 十六、收盘位置（日内强弱）───────────────────────────────
    # (收盘-最低)/(最高-最低)，数值越高表示收盘越强
    daily_range = high - low
    df["close_pos"] = (close - low) / daily_range.replace(0, np.nan)
    df["close_pos"] = df["close_pos"].replace([np.inf, -np.inf], np.nan)

    # ── 十七、日内振幅相对值 ─────────────────────────────────
    df["intraday_range"] = daily_range / preclose.replace(0, np.nan)
    df["intraday_range"] = df["intraday_range"].replace([np.inf, -np.inf], np.nan)

    # ── 十八、连涨/连跌天数 ─────────────────────────────────
    pos_ret = (delta > 0).astype(int)
    neg_ret = (delta < 0).astype(int)
    df["streak_up"] = pos_ret * (pos_ret.groupby((pos_ret != pos_ret.shift()).cumsum()).cumsum())
    df["streak_down"] = neg_ret * (neg_ret.groupby((neg_ret != neg_ret.shift()).cumsum()).cumsum())

    # ── 十九、量价相关性（10日滚动）────────────────────────
    # 上涨放量、下跌缩量 → 正相关（健康上涨）
    # 上涨缩量、下跌放量 → 负相关（出货信号）
    vol_pct = volume.pct_change()
    df["vol_ret_corr_10d"] = ret_1d.rolling(10).corr(vol_pct)
    df["vol_ret_corr_10d"] = df["vol_ret_corr_10d"].replace([np.inf, -np.inf], np.nan)

    # 估值因子
    for col in ["peTTM", "pbMRQ"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    return df


# =============================================================================
# 市场层面因子
# =============================================================================

def add_market_index_features(panel: pd.DataFrame) -> pd.DataFrame:
    """计算等权市场收益率和个股 Beta。"""
    daily_ret = panel.groupby("date")["pctChg"].transform(lambda x: x.mean()) / 100.0
    panel["market_ret"] = daily_ret
    result = panel.copy()
    result["beta_60d"] = np.nan
    for sid in result.index.get_level_values("stock_id").unique():
        stock_data = result.xs(sid, level="stock_id").copy()
        stock_ret = stock_data["pctChg"] / 100.0
        market_ret = stock_data["market_ret"]
        cov = stock_ret.rolling(BETA_WINDOW).cov(market_ret)
        var = market_ret.rolling(BETA_WINDOW).var()
        beta = cov / var.replace(0, np.nan)
        result.loc[(slice(None), sid), "beta_60d"] = beta.values
    return result


# =============================================================================
# 关联市场指数特征
# =============================================================================

def add_extra_index_features(panel: pd.DataFrame) -> pd.DataFrame:
    """合并上证50、中证500、创业板指、上证综指的特征。"""
    index_panel = fetch_index_data()
    if index_panel.empty:
        return panel
    index_names = index_panel.index.get_level_values("index_name").unique()
    result = panel.copy()
    for idx_name in index_names:
        idx_data = index_panel.xs(idx_name, level="index_name").copy()
        idx_data = idx_data.sort_index()
        for w in EXTRA_INDEX_RET_WINDOWS:
            col_name = f"{idx_name}_ret_{w}d"
            idx_data[col_name] = idx_data["close"].pct_change(w)
            idx_data[col_name] = idx_data[col_name].replace([np.inf, -np.inf], np.nan)
            date_series = idx_data[col_name]
            result[col_name] = result.groupby("date").apply(
                lambda g: pd.Series(date_series.get(g.name, np.nan), index=g.index)
            ).droplevel(0)
    print(f"[features] Added {len(index_names)} extra indices, "
          f"{len(EXTRA_INDEX_RET_WINDOWS)} windows each")
    return result


# =============================================================================
# 截面排名特征
# =============================================================================

def add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """计算个股在全市场中的排名分位数和截面特征。"""
    # 原有截面排名
    cross_cols = {
        "ret_5d": "ret_5d_rank", "ret_20d": "ret_20d_rank",
        "vol_5d": "vol_5d_rank", "rsi_14": "rsi_rank",
    }
    for src, dst in cross_cols.items():
        if src in panel.columns:
            panel[dst] = panel.groupby("date")[src].rank(pct=True)

    # 超额收益 (excess_return_1d): 个股日收益 - 等权市场平均
    if "pctChg" in panel.columns:
        market_avg = panel.groupby("date")["pctChg"].transform("mean")
        panel["excess_return_1d"] = (panel["pctChg"] - market_avg) / 100.0

    # 截面收盘价和成交量百分位
    if "close" in panel.columns:
        panel["cs_rank_close"] = panel.groupby("date")["close"].rank(pct=True)
    if "volume" in panel.columns:
        panel["cs_rank_volume"] = panel.groupby("date")["volume"].rank(pct=True)

    # 量价背离信号: 价格跌3天但成交量持续萎缩 → 下跌动能衰减(潜在反转)
    if all(c in panel.columns for c in ("close", "volume")):
        ret_3d = panel.groupby("stock_id")["close"].transform(lambda x: x.pct_change(3))
        vol_ma5 = panel.groupby("stock_id")["volume"].transform(lambda x: x.rolling(5).mean())
        vol_ratio = panel["volume"] / vol_ma5.replace(0, np.nan)
        # 背离 = price下跌(负) AND volume萎缩(vol_ratio<0.8)
        panel["divergence_bull"] = ((ret_3d < -0.02) & (vol_ratio < 0.8)).astype(np.float32)
        panel["divergence_bear"] = ((ret_3d > 0.02) & (vol_ratio > 1.5)).astype(np.float32)

    return panel


# =============================================================================
# 微观结构特征 (Micro-structure Proxies)
# =============================================================================

def add_micro_structure_features(panel: pd.DataFrame) -> pd.DataFrame:
    """添加微观结构代理因子：流动性、偏度、峰度、小波分解。

    仅使用 OHLCV，不依赖外部数据。
    """
    # 1. Amihud 缺乏流动性指标: |Return| / Volume (×1e9 缩放)
    if all(c in panel.columns for c in ("pctChg", "volume")):
        ret_abs = panel["pctChg"].abs() / 100.0  # 转小数
        vol_safe = panel["volume"].replace(0, np.nan)
        panel["amihud_illiq"] = (ret_abs / vol_safe).fillna(0).astype(np.float32)

    # 2. 偏度 (Skewness): 过去20日收益率的偏度，反映"彩票型"偏好
    if "pctChg" in panel.columns:
        skew_20d = panel.groupby("stock_id")["pctChg"].transform(
            lambda x: x.rolling(20).skew())
        panel["ret_skew_20d"] = skew_20d.fillna(0).astype(np.float32)

    # 3. 峰度 (Kurtosis): 过去20日收益率的峰度，反映厚尾风险
    if "pctChg" in panel.columns:
        kurt_20d = panel.groupby("stock_id")["pctChg"].transform(
            lambda x: x.rolling(20).kurt())
        panel["ret_kurt_20d"] = kurt_20d.fillna(0).astype(np.float32)

    # 4. 小波分解特征 (Wavelet): 价格序列的趋势/噪声比
    if "close" in panel.columns:
        try:
            import pywt
            def _wavelet_features(series):
                x = series.values[-60:].astype(np.float64)
                x = np.nan_to_num(x, nan=0.0)
                if len(x) < 16:
                    return 0.0, 0.0
                try:
                    coeffs = pywt.dwt(x, 'db4')
                    cA, cD = coeffs
                    energy_total = np.sum(x ** 2) + 1e-12
                    energy_trend = np.sum(cA ** 2)
                    energy_noise = np.sum(cD ** 2)
                    trend_ratio = energy_trend / energy_total
                    noise_level = energy_noise / max(energy_trend, 1e-12)
                    return trend_ratio, noise_level
                except:
                    return 0.0, 0.0
            wavelet_feats = panel.groupby("stock_id")["close"].transform(
                lambda x: pd.Series([_wavelet_features(x) for _ in range(len(x))], index=x.index))
            # 拆分为两列
            if isinstance(wavelet_feats, pd.DataFrame) and wavelet_feats.shape[1] == 2:
                panel["wavelet_trend"] = wavelet_feats.iloc[:, 0].astype(np.float32)
                panel["wavelet_noise"] = wavelet_feats.iloc[:, 1].astype(np.float32)
            else:
                panel["wavelet_trend"] = 0.0
                panel["wavelet_noise"] = 0.0
        except:
            pass

    # 5. 高低价差比 (Corwin-Schultz 简化版): (High - Low) / Close
    if all(c in panel.columns for c in ("high", "low", "close")):
        spread_hl = (panel["high"] - panel["low"]) / panel["close"].replace(0, np.nan)
        panel["hl_spread"] = spread_hl.fillna(0).astype(np.float32)
        # 20日均值
        spread_ma20 = panel.groupby("stock_id")["hl_spread"].transform(
            lambda x: x.rolling(20).mean())
        panel["hl_spread_20d"] = spread_ma20.fillna(0).astype(np.float32)

    return panel


# =============================================================================
# 日历特征（A股日历效应）
# =============================================================================

def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    """添加日历特征：周几、月份周期、月末、春节前后。

    参考 Game-BDC2026 features_extra.py 的 add_calendar_features
    """
    df = panel.reset_index()
    dates = df["date"]

    # 周几 one-hot (Mon=0, Thu=3, Fri=4为baseline)
    dow = dates.dt.dayofweek
    for i in range(4):
        df[f"wday_{i}"] = (dow == i).astype(np.float32)

    # 月份正弦/余弦编码 (捕获周期性)
    month = dates.dt.month.astype(float)
    df["month_sin"] = np.sin(2 * np.pi * month / 12.0).astype(np.float32)
    df["month_cos"] = np.cos(2 * np.pi * month / 12.0).astype(np.float32)

    # 月末标记: 当月最后交易日
    df["is_month_end"] = dates.dt.is_month_end.astype(np.float32)

    # 春节前后窗口 (~5个交易日)
    # 2021-2027年春节日期
    cny_dates = [
        "2021-02-12", "2022-02-01", "2023-01-22",
        "2024-02-10", "2025-01-29", "2026-02-17",
        "2027-02-06",
    ]
    cny_dates = pd.to_datetime(cny_dates)
    cny_window = pd.Timedelta(days=10)  # 春节前后约5个交易日

    df["is_cny_before"] = 0.0
    df["is_cny_after"] = 0.0
    for cny in cny_dates:
        before_mask = (dates >= cny - cny_window) & (dates < cny)
        after_mask = (dates > cny) & (dates <= cny + cny_window)
        df.loc[before_mask, "is_cny_before"] = 1.0
        df.loc[after_mask, "is_cny_after"] = 1.0

    # ── 沪深300成分股调整日历（6月/12月） ──
    # 调整生效前2周=预期炒作窗口，生效前后=被动资金调仓
    import datetime
    rebalance_dates = [
        "2021-06-11", "2021-12-10", "2022-06-10", "2022-12-09",
        "2023-06-09", "2023-12-08", "2024-06-14", "2024-12-13",
        "2025-06-13", "2025-12-12", "2026-06-12", "2026-12-11",
    ]
    rebalance_dates = pd.to_datetime(rebalance_dates)
    reb_window = pd.Timedelta(days=14)  # 调整前2周
    df["is_rebalance_soon"] = 0.0
    for rd in rebalance_dates:
        mask = (dates >= rd - reb_window) & (dates < rd)
        df.loc[mask, "is_rebalance_soon"] = 1.0

    return df.set_index(["date", "stock_id"])


# =============================================================================
# 行业分类特征
# =============================================================================

def add_industry_features(panel: pd.DataFrame, stock_ids: list[str]) -> pd.DataFrame:
    """计算行业平均收益率和个股 Alpha。"""
    industries = fetch_stock_industries(stock_ids)
    stock_id_level = panel.index.get_level_values("stock_id")
    panel["industry"] = stock_id_level.map(industries).fillna("Unknown")
    for w in [5, 20]:
        col = f"ret_{w}d"
        if col not in panel.columns:
            continue
        ind_mean = panel.groupby(["date", "industry"])[col].transform("mean")
        panel[f"ind_{col}"] = ind_mean
        panel[f"alpha_{col}"] = panel[col] - ind_mean
    if "vol_5d" in panel.columns:
        panel["ind_vol_5d"] = panel.groupby(["date", "industry"])["vol_5d"].transform("mean")
    panel["industry_size"] = panel.groupby(["date", "industry"])["industry"].transform("count")
    # 行业内收益百分位排名
    if "pctChg" in panel.columns:
        panel["industry_rank_return"] = panel.groupby(["date", "industry"])["pctChg"].rank(pct=True)
    return panel


# =============================================================================
# 特征工程主入口
# =============================================================================

# =============================================================================
# 基本面因子（季度财报数据）
# =============================================================================

def add_fundamental_features(panel: pd.DataFrame, stock_ids: list[str]) -> pd.DataFrame:
    """
    合并季度财报数据到日线 Panel。

    实现：
      1. 下载所有股票的季度财报
      2. 以财报发布日期为准，向前填充到每个交易日
      3. 新增因子：ROE、毛利率、净利润率、EPS、资产负债率、利润增速等
    """
    fundamental = fetch_fundamental_data(stock_ids)
    if fundamental.empty:
        print("[features] WARNING: No fundamental data available")
        return panel

    result = panel.reset_index()
    result = result.sort_values(["stock_id", "date"])

    # 按 stock_id 合并：对每个股票，用财报发布日期向前填充
    fund_cols = ["roe_ttm", "np_margin", "gp_margin", "eps_ttm",
                 "current_ratio", "debt_to_asset", "profit_yoy",
                 "equity_yoy"]

    result = result.merge(
        fundamental, on=["stock_id", "date"], how="left"
    )

    # 向前填充（每个股票独立，填充到下一个财报发布前）
    for col in fund_cols:
        if col in result.columns:
            result[col] = result.groupby("stock_id")[col].transform(
                lambda s: s.ffill()
            )

    # 填充缺失值
    for col in fund_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0).replace([np.inf, -np.inf], 0)

    result = result.set_index(["date", "stock_id"])
    panel = result[panel.columns.tolist() + [c for c in fund_cols if c in result.columns]]
    return panel


# =============================================================================
# 行业中性化：对每个截面因子做行业+市值回归取残差
# =============================================================================

def neutralize_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    对因子做行业中性化：每个交易日，对每个数值因子做行业哑变量回归，取残差。

    为什么？
      很多因子（如动量、波动率）在不同行业分布不均匀。
      如果某段时间银行股集体上涨，动量因子会偏向银行股。
      行业中性化去掉行业共同部分，保留个股特异信号。

    实现：
      对每个交易日，每个数值因子：
        factor = β₀ + Σβᵢ·industry_ⱼ + ε
      返回 ε 作为中性化后的因子值。
    """
    df = panel.reset_index()
    dates = df["date"].unique()

    # 识别数值因子列（排除索引、行业标签和原始价格列）
    EXCLUDE = {"open", "high", "low", "close", "preclose", "volume",
               "amount", "turn", "pctChg", "amplitude", "change",
               "stock_id", "date", "index_name", "industry",
               "vol_ma5", "vol_ma20", "turn_ma5", "turn_ma20"}
    feat_cols = [c for c in df.columns if c not in EXCLUDE and c not in ("date", "stock_id", "industry")]

    # 确保 industry 列存在
    if "industry" not in df.columns:
        print("[features] No industry data for neutralization")
        return panel

    print(f"[features] Neutralizing {len(feat_cols)} features across {len(dates)} dates ...")

    df_neut = df.copy()
    for d in dates:
        mask = df_neut["date"] == d
        sub = df_neut.loc[mask]
        if len(sub) < 10:
            continue

        # 行业哑变量
        industries = sub["industry"].values
        unique_ind = list(set(industries))
        if len(unique_ind) < 2:
            continue

        # 手动构造 one-hot（避免 sklearn 的 sparse）
        ind_dummies = np.zeros((len(sub), len(unique_ind)))
        for j, ind in enumerate(unique_ind):
            ind_dummies[:, j] = (industries == ind).astype(float)

        # 去掉第一列（防止多重共线性）
        ind_dummies = ind_dummies[:, 1:]

        for col in feat_cols:
            if col not in sub.columns:
                continue
            y = sub[col].values
            valid = ~np.isnan(y) & ~np.isnan(y) & (np.abs(y) < 1e8)
            if valid.sum() < len(unique_ind) + 5:
                continue
            try:
                lr = LinearRegression()
                lr.fit(ind_dummies[valid], y[valid])
                residual = y - lr.predict(ind_dummies)
                df_neut.loc[mask, col] = residual
            except Exception:
                continue

    panel_neut = df_neut.set_index(["date", "stock_id"])
    # 保留 industry 列
    panel_neut = panel_neut[panel.columns.tolist()]
    print(f"[features] Neutralization done")
    return panel_neut


def engineer_features(panel: pd.DataFrame, stock_ids: list[str] | None = None) -> pd.DataFrame:
    """完整特征工程流水线（结果缓存到 raw/features_panel.parquet）。"""
    cache_path = RAW_DIR / "features_panel.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "stock_id"])
        print(f"[features] Loaded cached features: {df.shape}")
        return df

    print("[features] Computing per-stock features ...")
    panel = panel.groupby("stock_id", group_keys=False).apply(_per_stock)

    print("[features] Computing market features ...")
    panel = add_market_index_features(panel)

    print("[features] Adding cross-market index features ...")
    panel = add_extra_index_features(panel)

    print("[features] Adding cross-sectional rank features ...")
    panel = add_cross_sectional_features(panel)

    print("[features] Adding micro-structure features ...")
    panel = add_micro_structure_features(panel)

    print("[features] Adding calendar features ...")
    panel = add_calendar_features(panel)

    print("[features] Adding industry features ...")
    if stock_ids is None:
        stock_ids = list(panel.index.get_level_values("stock_id").unique())
    panel = add_industry_features(panel, stock_ids)

    # 注意：基本面因子（ROE/毛利率等）通过 fetch_fundamental_data() 单独获取，
    # 当前训练流程中未使用（数据量太少，提升有限）

    # 缓存
    panel.reset_index().to_parquet(cache_path, index=False)
    print(f"[features] Cached to {cache_path}")

    print(f"[features] Done. Panel shape: {panel.shape}, columns: {len(panel.columns)}")
    return panel


# =============================================================================
# 向量化预计算
# =============================================================================

def _precompute_stock_arrays(panel, stock_ids, feature_cols, dates):
    """
    全向量化预计算：将 MultiIndex Panel 转换为 3D numpy 数组。
    无 Python 逐行循环，单次 numpy advanced indexing 填充。
    """
    n_stocks = len(stock_ids)
    n_dates = len(dates)
    n_features = len(feature_cols)

    stock_to_idx = {s: i for i, s in enumerate(stock_ids)}
    date_to_idx  = {d: i for i, d in enumerate(dates)}

    df = panel.reset_index()[["date", "stock_id", "open"] + feature_cols]
    df["stock_idx"] = df["stock_id"].map(stock_to_idx).astype(np.int32)
    df["date_idx"]  = df["date"].map(date_to_idx).astype(np.int32)
    df = df.dropna(subset=["stock_idx", "date_idx"])

    s_idx = df["stock_idx"].values
    d_idx = df["date_idx"].values

    feats_arr = np.zeros((n_stocks, n_dates, n_features), dtype=np.float32)
    open_arr  = np.zeros((n_stocks, n_dates), dtype=np.float32)
    has_data  = np.zeros((n_stocks, n_dates), dtype=bool)

    feats_arr[s_idx, d_idx] = df[feature_cols].values.astype(np.float32)
    open_arr[s_idx, d_idx]  = df["open"].values.astype(np.float32)
    has_data[s_idx, d_idx]  = True

    np.nan_to_num(feats_arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(open_arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    return feats_arr, open_arr, has_data


# =============================================================================
# 窗口采样（向量化加速版）
# =============================================================================

def make_window_samples(
    panel: pd.DataFrame,
    stock_ids: list[str],
    lookback: int = LOOKBACK_DAYS,
    horizon: int = PREDICT_HORIZON,
    step: int = STEP_DAYS,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    向量化窗口采样：将 Panel 切分为 (输入窗口, 未来收益) 样本对。
    """
    global _NORM_STATS

    dates = sorted(panel.index.get_level_values("date").unique())
    n_stocks = len(stock_ids)
    n_dates = len(dates)

    from features_alpha158 import ALPHA158_NEW_COLS
    EXCLUDE_COLS = {"open", "high", "low", "close", "preclose", "volume",
                    "amount", "adjustflag", "turn", "tradestatus",
                    "pctChg", "peTTM", "pbMRQ", "market_ret",
                    "stock_id", "index_name", "vol_ma5", "vol_ma20",
                    "turn_ma5", "turn_ma20", "industry",
                    "amplitude", "change",
                    "roe_ttm", "np_margin", "gp_margin", "eps_ttm",
                    "current_ratio", "debt_to_asset", "profit_yoy", "equity_yoy",
                    "nn_score",
                    *ALPHA158_NEW_COLS}
    feature_cols = [c for c in panel.columns if c not in EXCLUDE_COLS]
    n_features = len(feature_cols)

    # 预计算 3D 数组
    feats_arr, open_arr, has_data = _precompute_stock_arrays(
        panel, stock_ids, feature_cols, dates,
    )

    X_list, y_list, mask_list, date_labels = [], [], [], []

    for idx in range(lookback + horizon, n_dates, step):
        target_end_date = dates[idx]
        if target_end_date < pd.Timestamp(EFFECTIVE_START):
            continue

        hist_start = idx - horizon - lookback + 1
        hist_end   = idx - horizon + 1
        X = feats_arr[:, hist_start:hist_end, :].copy()
        if X.shape[1] < lookback:
            pad_w = lookback - X.shape[1]
            X = np.pad(X, ((0, 0), (pad_w, 0), (0, 0)), mode="constant")

        open_t1 = open_arr[:, idx - horizon + 1]
        open_t5 = open_arr[:, idx]
        valid_open = open_t1 > 1e-8
        y = np.divide(open_t5 - open_t1, open_t1, out=np.zeros_like(open_t1), where=valid_open)

        hist_ok = has_data[:, hist_start:hist_end].sum(axis=1) >= lookback * 0.7
        target_ok = valid_open & (open_t5 > 1e-8)
        mask = (hist_ok & target_ok).astype(np.float32)

        if mask.sum() < 5:
            continue

        X_list.append(X)
        y_list.append(y)
        mask_list.append(mask)
        date_labels.append(str(target_end_date.date()))

    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    m_arr = np.array(mask_list)

    # 事件过滤
    date_timestamps = [pd.Timestamp(d) for d in date_labels]
    event_mask = build_event_mask(date_timestamps)
    X_arr = X_arr[event_mask]
    y_arr = y_arr[event_mask]
    m_arr = m_arr[event_mask]
    date_labels = [d for d, keep in zip(date_labels, event_mask) if keep]

    # z-score 标准化
    if normalize:
        feat_mean = X_arr.mean(axis=(0, 1, 2), keepdims=True)
        feat_std  = X_arr.std(axis=(0, 1, 2), keepdims=True)
        feat_std  = np.where(feat_std < 1e-8, 1.0, feat_std)
        X_arr = (X_arr - feat_mean) / feat_std
        _NORM_STATS = (feat_mean, feat_std)

        X_valid = X_arr[m_arr > 0.5]
        print(f"[features] Normalized: mean={X_valid.mean():.4f}, std={X_valid.std():.4f}")

    print(f"[features] Built {len(X_arr)} samples: "
          f"X={X_arr.shape}, features={n_features}, "
          f"valid stocks/sample ≈ {m_arr.mean(axis=1).mean():.0f}")
    return X_arr, y_arr, m_arr, date_labels
