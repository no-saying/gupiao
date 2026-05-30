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
    EXTRA_INDEX_RET_WINDOWS, EFFECTIVE_START,
)
from data_loader import fetch_index_data, build_event_mask, fetch_stock_industries

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

    # 五、振幅
    df["amplitude"] = (high - low) / close

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
    """计算个股在全市场中的排名分位数。"""
    cross_cols = {
        "ret_5d": "ret_5d_rank", "ret_20d": "ret_20d_rank",
        "vol_5d": "vol_5d_rank", "rsi_14": "rsi_rank",
    }
    for src, dst in cross_cols.items():
        if src in panel.columns:
            panel[dst] = panel.groupby("date")[src].rank(pct=True)
    return panel


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
    return panel


# =============================================================================
# 特征工程主入口
# =============================================================================

def engineer_features(panel: pd.DataFrame, stock_ids: list[str] | None = None) -> pd.DataFrame:
    """完整特征工程流水线。"""
    print("[features] Computing per-stock features ...")
    panel = panel.groupby("stock_id", group_keys=False).apply(_per_stock)

    print("[features] Computing market features ...")
    panel = add_market_index_features(panel)

    print("[features] Adding cross-market index features ...")
    panel = add_extra_index_features(panel)

    print("[features] Adding cross-sectional rank features ...")
    panel = add_cross_sectional_features(panel)

    print("[features] Adding industry features ...")
    if stock_ids is None:
        stock_ids = list(panel.index.get_level_values("stock_id").unique())
    panel = add_industry_features(panel, stock_ids)

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

    EXCLUDE_COLS = {"open", "high", "low", "close", "preclose", "volume",
                    "amount", "adjustflag", "turn", "tradestatus",
                    "pctChg", "peTTM", "pbMRQ", "market_ret",
                    "stock_id", "index_name", "vol_ma5", "vol_ma20",
                    "turn_ma5", "turn_ma20", "industry"}
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
