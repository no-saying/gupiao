"""
=============================================================================
  特征工程模块 —— 从原始行情数据中提取预测因子
=============================================================================

因子分类（共 22 个特征）：

  一、动量因子（4 个）
      ret_5d, ret_10d, ret_20d, ret_60d
      不同时间窗口的累计收益率，捕捉趋势强度
      5d = 周动量，20d = 月动量，60d = 季动量

  二、波动率因子（3 个）
      vol_5d, vol_10d, vol_20d
      日收益率在不同窗口的标准差，衡量股票的风险特征
      低波动股票通常更稳定，高波动股票弹性更大

  三、均线偏离因子（4 个）
      ma5_dev, ma10_dev, ma20_dev, ma60_dev
      当前价格相对于移动均线的偏离百分比
      正值 = 短期强势（突破均线），负值 = 短期弱势

  四、量价因子（4 个）
      volume_ratio_5, volume_ratio_20, turn_change, amplitude
      成交量和换手率的变化，反映市场关注度变化
      放量上涨通常是积极信号

  五、技术指标因子（4 个）
      rsi_14, macd, macd_signal, macd_hist
      经典技术指标，捕捉超买超卖和趋势变化

  六、风险与位置因子（2 个）
      max_dd_20d, price_position
      最大回撤和价格在 20 日区间内的相对位置

  七、估值因子（2 个）
      peTTM（滚动市盈率）, pbMRQ（市净率）
      低估值股票通常有更高的安全边际

  八、市场因子（1 个）
      beta_60d：个股相对于等权市场组合的敏感度
      Beta > 1 = 比市场波动更大（进攻型）
      Beta < 1 = 比市场波动更小（防御型）

特征选择的原则：
  - 所有特征都可以从 baostock 的免费数据中计算得到
  - 覆盖动量、波动、量价、估值四个维度
  - 避免高度冗余的特征（如 OBV 和成交量本身就是强相关的）

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
from data_loader import fetch_index_data, build_event_mask


# =============================================================================
# 单只股票的因子计算
# =============================================================================
# 这个函数会被 groupby("stock_id").apply() 调用，
# 对每只股票的时间序列独立计算所有因子


def _per_stock(group: pd.DataFrame) -> pd.DataFrame:
    """
    对单只股票的时间序列计算所有技术因子。

    输入：一只股票的日线数据，按日期排序，以 date 为 index
    输出：增加了因子列的 DataFrame

    所有计算都是使用 pandas 的向量化操作（rolling/ewm），
    不会出现 Python 循环，保证计算效率。
    """
    df = group.copy()
    close  = df["close"]        # 收盘价序列
    volume = df["volume"]       # 成交量序列
    high   = df["high"]         # 最高价序列
    low    = df["low"]          # 最低价序列

    # ==================================================================
    # 一、动量因子 —— "这只股票最近涨了多少？"
    # ==================================================================
    # pct_change(N) 计算 N 日累计收益率：(close[t] - close[t-N]) / close[t-N]
    # 不同窗口捕捉不同周期的趋势：
    #   短期（5d）→ 捕捉最近一周的动量
    #   中期（20d）→ 捕捉月度趋势
    #   长期（60d）→ 捕捉季度趋势，也是反转效应的判断依据
    ret_1d = close.pct_change()                           # 日收益率（用于后续波动率计算）
    for w in MOMENTUM_WINDOWS:
        df[f"ret_{w}d"] = close.pct_change(w)             # N 日累计收益率
        df[f"ret_{w}d"] = df[f"ret_{w}d"].replace([np.inf, -np.inf], np.nan)

    # ==================================================================
    # 二、波动率因子 —— "这只股票的风险有多大？"
    # ==================================================================
    # rolling(N).std() 计算过去 N 个交易日收益率的标准差
    # 波动率越高 → 不确定性越大 → 需要更高的预期收益补偿
    for w in VOLATILITY_WINDOWS:
        df[f"vol_{w}d"] = ret_1d.rolling(w).std()

    # ==================================================================
    # 三、均线偏离因子 —— "当前价格在趋势通道的什么位置？"
    # ==================================================================
    # 收盘价 / N日均价 - 1 → 偏离百分比
    # 正值 = 价格突破均线（动能向上），负值 = 价格跌破均线（动能向下）
    # MA60 偏离对于判断中长期趋势特别重要（如年线战法）
    for w in MA_WINDOWS:
        ma = close.rolling(w).mean()                      # N 日简单移动平均
        df[f"ma{w}_dev"] = (close / ma) - 1.0             # 偏离度

    # ==================================================================
    # 四、量价因子 —— "这股票被关注的程度有变化吗？"
    # ==================================================================
    # 成交量放大通常是价格变动的前兆
    # volume_ratio_5  > 0 → 近 5 日平均量 > 近 20 日平均量（放量）
    # volume_ratio_5  < 0 → 近 5 日平均量 < 近 20 日平均量（缩量）
    df["vol_ma5"] = volume.rolling(5).mean()              # 5 日均量
    df["vol_ma20"] = volume.rolling(20).mean()             # 20 日均量
    df["volume_ratio_5"] = volume / df["vol_ma5"] - 1.0    # 当日量 / 5日均量 - 1
    df["volume_ratio_20"] = volume / df["vol_ma20"] - 1.0  # 当日量 / 20日均量 - 1

    # 换手率变化：反映市场参与度的变化
    # baostock 的 turn 字段是换手率（%），表示流通股本中有多少比例被交易
    if "turn" in df.columns:
        df["turn_ma5"] = df["turn"].rolling(5).mean()
        df["turn_ma20"] = df["turn"].rolling(20).mean()
        df["turn_change"] = df["turn"] / df["turn_ma5"] - 1.0  # 换手率相对于 5 日平均的变化

    # ==================================================================
    # 五、日内振幅 —— "当天波动大吗？"
    # ==================================================================
    # (最高价 - 最低价) / 收盘价
    # 振幅大可能意味着多空分歧大，也可能是流动性差
    df["amplitude"] = (high - low) / close

    # ==================================================================
    # 六、RSI（相对强弱指标）—— "这股票超买还是超卖了？"
    # ==================================================================
    # RSI = 100 - 100/(1 + RS)，其中 RS = N日内平均涨幅 / N日内平均跌幅
    # RSI > 70 → 超买（可能回调）
    # RSI < 30 → 超卖（可能反弹）
    delta = close.diff()                                  # 每日价格变化
    gain = delta.clip(lower=0)                            # 涨幅（正值保留，负值变 0）
    loss = (-delta).clip(lower=0)                         # 跌幅（取正）
    avg_gain = gain.rolling(RSI_PERIOD).mean()            # N 日平均涨幅
    avg_loss = loss.rolling(RSI_PERIOD).mean()            # N 日平均跌幅
    rs = avg_gain / avg_loss.replace(0, np.nan)           # 相对强弱
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))          # 转换为 0-100 的 RSI 值

    # ==================================================================
    # 七、MACD（指数平滑异同移动平均线）—— "趋势在加速还是减速？"
    # ==================================================================
    # MACD 线  = EMA(12) - EMA(26)      → 短期趋势强度
    # 信号线   = EMA(MACD, 9)          → 平滑后的趋势
    # 柱状图   = MACD - 信号线          → 趋势加速度
    #   柱状图 > 0 且扩大 → 上涨加速
    #   柱状图 < 0 且扩大 → 下跌加速
    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow                      # MACD 线
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()  # 信号线
    df["macd_hist"] = df["macd"] - df["macd_signal"]      # 柱状图（MACD 与信号的差值）

    # ==================================================================
    # 八、最大回撤 —— "最近出现过恐慌性下跌吗？"
    # ==================================================================
    # 20 日最大回撤：过去 20 天内，从最高点到之后最低点的最大跌幅
    roll_max = close.rolling(20).max()
    dd = close / roll_max - 1.0                           # 每一日距离 20 日高点的回撤
    df["max_dd_20d"] = dd.rolling(20).min()               # 取最小值 = 最大回撤

    # ==================================================================
    # 九、价格位置 —— "当前价格在近期的相对位置？"
    # ==================================================================
    # (收盘价 - 20日最低) / (20日最高 - 20日最低)
    # 范围 [0, 1]，越接近 1 表示价格越接近近期高点
    df["price_position"] = (close - low.rolling(20).min()) / (
        high.rolling(20).max() - low.rolling(20).min() + 1e-9  # +1e-9 防止除零
    )

    # ==================================================================
    # 十、估值因子 —— 不做额外计算，直接使用 baostock 提供的值
    # ==================================================================
    # peTTM = 滚动市盈率（Trailing Twelve Months）
    # pbMRQ = 最新季报的市净率（Most Recent Quarter）
    # 把值为 0 的替换为 NaN（市盈率/市净率为 0 通常意味着数据异常）
    for col in ["peTTM", "pbMRQ"]:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    return df


# =============================================================================
# 市场层面的因子
# =============================================================================

def add_market_index_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    计算市场层面的特征：等权市场收益率和个股 Beta。

    为什么用等权而不是市值加权？
      - 简单：不需要获取总市值数据
      - 稳健：市值加权会被大市值股票（如茅台、平安）主导
      - 等权的 Beta 更能反映个股相对于"平均水平"的偏离

    Beta 的含义：
      Beta = Cov(个股收益, 市场收益) / Var(市场收益)
      衡量个股对市场整体波动的敏感度
    """
    # ---- 等权市场日收益率 ----
    # transform + mean: 每个交易日所有股票涨跌幅的平均值
    daily_ret = panel.groupby("date")["pctChg"].transform(lambda x: x.mean()) / 100.0
    panel["market_ret"] = daily_ret

    # ---- 滚动 Beta ----
    result = panel.copy()
    result["beta_60d"] = np.nan

    # 对于每只股票，计算其 60 日滚动 Beta
    for sid in result.index.get_level_values("stock_id").unique():
        stock_data = result.xs(sid, level="stock_id").copy()
        stock_ret = stock_data["pctChg"] / 100.0       # 转换为小数形式的收益率
        market_ret = stock_data["market_ret"]
        # Beta = 协方差(个股, 市场) / 方差(市场)
        cov = stock_ret.rolling(BETA_WINDOW).cov(market_ret)
        var = market_ret.rolling(BETA_WINDOW).var()
        beta = cov / var.replace(0, np.nan)
        result.loc[(slice(None), sid), "beta_60d"] = beta.values

    return result


# =============================================================================
# 关联市场指数特征
# =============================================================================

def add_extra_index_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    将上证50、中证500、创业板指、上证综指的收益率序列合并到 Panel 中，
    作为宏观/跨市场特征。

    为什么加这些指数？
      - 沪深 300 主要代表大盘蓝筹，但市场风格经常在大盘/中小盘间轮动
      - 上证 50：超大盘代表（银行、保险、白酒）
      - 中证 500：中盘成长代表
      - 创业板指：科技成长代表
      - 上证综指：全市场情绪基准

    提取的特征：
      对每个指数计算 5/10/20 日收益率（pctChg 的滚动和），
      这样模型可以看到：
        - "过去一周中证500是不是比沪深300强" → 风格轮动信号
        - "创业板最近一个月是不是暴跌" → 风险偏好信号

    这些特征会被广播到每只个股上（同一天的个股共享同样的指数特征），
    让模型在做截面注意力时能感知到宏观环境。
    """
    # 获取指数数据
    index_panel = fetch_index_data()
    if index_panel.empty:
        print("[features] WARNING: No index data available, skipping cross-market features")
        return panel

    index_names = index_panel.index.get_level_values("index_name").unique()
    result = panel.copy()

    # 对每个指数，计算不同窗口的收益率并合并到主 Panel
    for idx_name in index_names:
        idx_data = index_panel.xs(idx_name, level="index_name").copy()
        # 确保 idx_data 按日期排序
        idx_data = idx_data.sort_index()

        for w in EXTRA_INDEX_RET_WINDOWS:
            col_name = f"{idx_name}_ret_{w}d"
            # N 日收益率 = 当日收盘 / N日前收盘 - 1
            idx_data[col_name] = idx_data["close"].pct_change(w)
            idx_data[col_name] = idx_data[col_name].replace([np.inf, -np.inf], np.nan)

            # 广播到每只个股：按日期合并
            # idx_data 的 index 是 date，需要对齐到 panel 的 (date, stock_id)
            date_series = idx_data[col_name]
            # 通过 index 的第一层（date）映射
            result[col_name] = result.groupby("date").apply(
                lambda g: pd.Series(date_series.get(g.name, np.nan), index=g.index)
            ).droplevel(0)

    print(f"[features] Added {len(index_names)} extra indices, "
          f"{len(index_names) * len(EXTRA_INDEX_RET_WINDOWS)} new features")
    return result


# =============================================================================
# 主入口：特征工程
# =============================================================================

def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    """
    对原始 Panel 数据执行特征工程。

    流程：
      1. 对每只股票独立计算技术指标因子（22 个）
      2. 计算沪深 300 内部的市场因子（等权指数、个股 Beta）
      3. 合并外部关联市场指数特征（上证50、中证500、创业板指、上证综指）
    """
    print("[features] Computing per-stock features ...")
    panel = panel.groupby("stock_id", group_keys=False).apply(_per_stock)

    print("[features] Computing market features ...")
    panel = add_market_index_features(panel)

    print("[features] Adding cross-market index features ...")
    panel = add_extra_index_features(panel)

    print(f"[features] Done. Panel shape: {panel.shape}, columns: {len(panel.columns)}")
    return panel


# =============================================================================
# 因子列表 —— 训练时使用的全部特征列
# =============================================================================
# 定义在这里（而不是 config.py）是为了保持与特征计算函数在同一个文件中，
# 方便维护——添加新因子时只需要改这一个文件和 _per_stock 函数

FEATURE_COLS = [
    # 动量
    "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    # 波动率
    "vol_5d", "vol_10d", "vol_20d",
    # 均线偏离
    "ma5_dev", "ma10_dev", "ma20_dev", "ma60_dev",
    # 量价
    "volume_ratio_5", "volume_ratio_20",
    "turn_change",
    "amplitude",
    # 技术指标
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    # 风险与位置
    "max_dd_20d", "price_position",
    # 估值
    "peTTM", "pbMRQ",
    # 市场
    "beta_60d",
    # ── 跨市场指数特征（动态生成） ──
    # SSE50_ret_5d, SSE50_ret_10d, SSE50_ret_20d,
    # CSI500_ret_5d, CSI500_ret_10d, CSI500_ret_20d,
    # ChiNext_ret_5d, ChiNext_ret_10d, ChiNext_ret_20d,
    # SSE_Composite_ret_5d, SSE_Composite_ret_10d, SSE_Composite_ret_20d,
]


# =============================================================================
# 构建训练窗口（滑窗采样）
# =============================================================================

def make_window_samples(
    panel: pd.DataFrame,
    stock_ids: list[str],
    lookback: int = LOOKBACK_DAYS,
    horizon: int = PREDICT_HORIZON,
    step: int = STEP_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    将 Panel 数据切分成 (输入窗口, 未来收益) 的样本对。

    与之前版本的区别：
      1. 动态发现特征列（不再硬编码 FEATURE_COLS，而是从 panel 中自动获取）
         这样跨市场指数特征会被自动纳入，无需手动维护列名列表
      2. 过滤 EFFECTIVE_START 之前的样本（只保留近 5 年数据）
      3. 过滤重大事件窗口内的样本（label 窗口与事件重叠的剔除）

    Label 构造：
      收益 = (T+5 开盘价 - T+1 开盘价) / T+1 开盘价
    """
    dates = sorted(panel.index.get_level_values("date").unique())
    n_stocks = len(stock_ids)

    # ---- 动态发现特征列 ----
    # 排除原始行情字段和非特征列，保留所有计算出的因子
    EXCLUDE_COLS = {"open", "high", "low", "close", "preclose", "volume",
                    "amount", "adjustflag", "turn", "tradestatus",
                    "pctChg", "peTTM", "pbMRQ", "market_ret",
                    "stock_id", "index_name", "vol_ma5", "vol_ma20",
                    "turn_ma5", "turn_ma20"}
    feature_cols = [c for c in panel.columns if c not in EXCLUDE_COLS]
    n_features = len(feature_cols)

    X_list, y_list, mask_list, date_labels = [], [], [], []

    # 从 lookback+horizon 处开始
    for idx in range(lookback + horizon, len(dates), step):
        target_start_date = dates[idx - horizon + 1]  # T+1
        target_end_date   = dates[idx]                # T+5
        hist_start_date   = dates[idx - horizon - lookback + 1]
        hist_end_date     = dates[idx - horizon]

        # ---- 日期范围过滤 ----
        # 只保留 EFFECTIVE_START 之后的样本
        if target_end_date < pd.Timestamp(EFFECTIVE_START):
            continue

        # ---- 样本初始化 ----
        X = np.zeros((n_stocks, lookback, n_features), dtype=np.float32)
        y = np.zeros(n_stocks, dtype=np.float32)
        mask = np.zeros(n_stocks, dtype=np.float32)

        for i, sid in enumerate(stock_ids):
            try:
                stock_data = panel.xs(sid, level="stock_id")
            except KeyError:
                continue

            hist = stock_data.loc[hist_start_date:hist_end_date]
            if len(hist) < lookback * 0.7:
                continue

            hist = hist.iloc[-lookback:]
            feats = hist[feature_cols].values

            if feats.shape[0] < lookback:
                pad = np.zeros((lookback - feats.shape[0], n_features), dtype=np.float32)
                feats = np.concatenate([pad, feats], axis=0)

            # NaN → 0 (模型处理不了 NaN)
            feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
            X[i] = feats.astype(np.float32)

            # 目标收益
            try:
                target_data = stock_data.loc[target_start_date:target_end_date]
                if len(target_data) >= 1:
                    open_t1 = target_data.iloc[0]["open"]
                    open_t5 = target_data.iloc[-1]["open"]
                    y[i] = (open_t5 - open_t1) / open_t1 if open_t1 > 0 else 0.0
                    mask[i] = 1.0
            except (KeyError, IndexError):
                pass

        if mask.sum() < 5:
            continue

        X_list.append(X)
        y_list.append(y)
        mask_list.append(mask)
        date_labels.append(str(target_end_date.date()))

    X_arr = np.array(X_list)
    y_arr = np.array(y_list)
    m_arr = np.array(mask_list)

    # ---- 事件过滤 ----
    # 构建事件掩码（基于 sample 的目标日期）
    date_timestamps = [pd.Timestamp(d) for d in date_labels]
    event_mask = build_event_mask(date_timestamps)

    X_arr = X_arr[event_mask]
    y_arr = y_arr[event_mask]
    m_arr = m_arr[event_mask]
    date_labels = [d for d, keep in zip(date_labels, event_mask) if keep]

    print(f"[features] Built {len(X_arr)} samples: "
          f"X={X_arr.shape}, "
          f"features={n_features}, "
          f"valid stocks/sample ≈ {m_arr.mean(axis=1).mean():.0f}")
    return X_arr, y_arr, m_arr, date_labels
