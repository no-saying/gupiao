"""
=============================================================================
  数据加载模块 —— 从 baostock 获取数据并缓存到本地
=============================================================================

数据获取策略：
  1. baostock 是免费的 A 股数据源，提供开高低收、成交量、换手率、
     市盈率、市净率等字段，完全满足赛题"公开免费可获取"的要求
  2. 下载后的数据以 parquet 格式缓存到 data/raw/ 目录，下次运行直接读取
  3. CSI 300 成分股列表同样会被缓存，避免每次查询

关键设计决策：
  - 使用前复权（adjustflag="2"）数据，这样价格序列是连续的
  - 只保留交易日数据（tradestatus == "1"），过滤休市日
  - 最终产出是一个 MultiIndex Panel DataFrame，方便后续特征工程

=============================================================================
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import baostock as bs

from config import RAW_DIR, OFFICIAL_DIR, START_DATE, END_DATE, EXTRA_INDICES, EVENT_WINDOWS


# =============================================================================
# 辅助函数：股票代码格式转换
# =============================================================================
# baostock 内部使用 "sh.600000" 或 "sz.000001" 格式
# 但赛题提交要求的是纯数字 "600000" 或 "000001"
# 所以需要两种格式之间的转换


def _baostock_code(stock_id: str) -> str:
    """
    将纯数字代码转换为 baostock 格式。

    规则：6 开头 → 上海交易所 (sh)，其余 → 深圳交易所 (sz)

    示例：
      '000001' → 'sz.000001'  （平安银行，深交所）
      '600036' → 'sh.600036'  （招商银行，上交所）
    """
    code = int(stock_id)
    prefix = "sh" if code >= 600000 else "sz"
    return f"{prefix}.{stock_id}"


def _strip_code(baostock_code: str) -> str:
    """
    将 baostock 格式转换为纯数字代码。

    示例：
      'sh.600036' → '600036'
      'sz.000001' → '000001'
    """
    return baostock_code.split(".")[1]


# =============================================================================
# 获取沪深 300 成分股列表
# =============================================================================

def fetch_csi300_stocks() -> list[str]:
    """
    从 baostock 获取当天沪深 300 成分股列表。

    注意：沪深 300 成分股会每半年调整一次（6 月和 12 月）。
    这里获取的是当前生效的成分股列表。

    对于历史回测来说，理想情况下应该使用每个时点实际的成分股列表，
    但为了简化，这里假设成分股相对稳定。

    Returns:
        排序后的股票代码列表，如 ['000001', '000002', ..., '688981']
    """
    # ---- 检查缓存 ----
    cache_path = RAW_DIR / "csi300_stocks.pkl"
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    # ---- 从 baostock 获取 ----
    bs.login()                                          # baostock 必须先登录
    rs = bs.query_hs300_stocks()                        # 查询沪深 300 成分股
    stocks = []
    while rs.next():                                    # 遍历结果集
        row = rs.get_row_data()
        stocks.append(row[1])                           # 第 2 列是股票代码
    bs.logout()                                         # 用完后登出

    # 转换为纯数字代码并排序
    processed = sorted(set(_strip_code(s) for s in stocks))

    # 写入缓存（pickle 格式，读取速度比 CSV 快）
    cache_path.write_bytes(pickle.dumps(processed))
    print(f"[data] Fetched {len(processed)} CSI 300 stocks")
    return processed


# =============================================================================
# 获取单只股票的日线行情
# =============================================================================

def fetch_daily_kline(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    从 baostock 获取单只股票的日线行情数据（独立使用，带 login/logout）。

    批量下载请使用 build_panel()，它会合并为单次登录，效率高很多。
    """
    # ---- 检查缓存 ----
    cache_path = RAW_DIR / f"{stock_id}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        return df[mask].sort_values("date").reset_index(drop=True) if mask.any() else pd.DataFrame()

    bs.login()
    df = _download_single_stock(stock_id, start_date, end_date)
    bs.logout()
    return df


def _download_single_stock(stock_id: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    下载单只股票数据（假设已登录，不做 login/logout）。
    用于 build_panel 批量下载，避免每次登录开销。
    """
    cache_path = RAW_DIR / f"{stock_id}.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        return df[mask].sort_values("date").reset_index(drop=True) if mask.any() else pd.DataFrame()

    try:
        code = _baostock_code(stock_id)
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,preclose,volume,amount,turn,pctChg",
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="1",
        )

        # 使用 rs.data 直接获取全部数据
        if rs.error_code != "0":
            return pd.DataFrame()
        rows = rs.data
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=[
            "date", "code", "open", "high", "low", "close",
            "preclose", "volume", "amount", "turn", "pctChg",
        ])

        numeric_cols = ["open", "high", "low", "close", "preclose",
                        "volume", "amount", "turn", "pctChg"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 计算振幅和涨跌额（THU-BDC 官方标准）
        df["amplitude"] = ((df["high"] - df["low"]) / df["preclose"].replace(0, np.nan) * 100).round(2)
        df["change"] = (df["close"] - df["preclose"]).round(2)

        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_parquet(cache_path, index=False)
        return df
    except Exception as e:
        print(f"  [WARN] {stock_id}: {e}")
        return pd.DataFrame()


# =============================================================================
# 构建全市场 Panel 数据
# =============================================================================

def build_panel(
    stock_ids: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    将所有股票的日线数据合并为一个 Panel DataFrame。

    输出格式：
      MultiIndex: (date, stock_id)
      列：open, high, low, close, preclose, volume, amount, turn, pctChg, peTTM, pbMRQ

    为什么用 MultiIndex 而不是 (dates × stocks) 的宽表？
      - 稀疏性：不同股票上市/退市时间不同，宽表有很多 NaN
      - 灵活性：MultiIndex 天然支持不同股票的不同时间范围
      - 效率：后续 groupby("stock_id") 操作很方便

    Returns:
        MultiIndex DataFrame，index 为 (date, stock_id)
    """
    # ---- 检查整体缓存 ----
    cache_path = RAW_DIR / "panel.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        # Handle both MultiIndex and flat-column formats
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df = df[mask]
        df = df.set_index(["date", "stock_id"])
        print(f"[data] Loaded cached panel: {df.shape[0]} rows")
        return df

    # ---- 批量下载（单次登录，大幅提速）----
    bs.login()
    frames = []
    n_total = len(stock_ids)
    for i, sid in enumerate(stock_ids):
        if (i + 1) % 50 == 0:
            print(f"[data] Downloading... {i+1}/{n_total}")
        df = _download_single_stock(sid, start_date, end_date)
        if df.empty:
            continue
        df["stock_id"] = sid
        frames.append(df[["date", "stock_id",
                          "open", "high", "low", "close", "preclose",
                          "volume", "amount", "turn", "pctChg"]])
    bs.logout()

    # ---- 合并并排序 ----
    panel = pd.concat(frames, ignore_index=True)
    panel.sort_values(["date", "stock_id"], inplace=True)
    panel.set_index(["date", "stock_id"], inplace=True)

    # ---- 缓存 ----
    panel.to_parquet(cache_path)
    print(f"[data] Built panel: {panel.shape[0]} rows, "
          f"{panel.index.get_level_values('stock_id').nunique()} stocks, "
          f"{panel.index.get_level_values('date').nunique()} trading days")
    return panel


# =============================================================================
# 从官方 THU-BDC CSV 文件加载数据（超期部分用 baostock 补充）
# =============================================================================

def _extend_with_baostock(panel: pd.DataFrame, stock_ids: list[str]) -> pd.DataFrame:
    """
    从 baostock 补充官方数据未覆盖的日期（到 END_DATE）。

    官方 train.csv+test.csv 只到 2026-03-13，
    用 baostock 补充到 END_DATE（2026-05-30），
    让训练数据多 50+ 个交易日。
    """
    from config import END_DATE
    last_date = panel.index.get_level_values("date").max()
    end = pd.Timestamp(END_DATE)
    if last_date >= end:
        return panel  # 已经够了

    print(f"[data] Extending data from {last_date.date()} to {end.date()} via baostock ...")

    bs.login()
    frames = [panel.reset_index()]
    n_stocks = len(stock_ids)
    for i, sid in enumerate(stock_ids):
        if (i + 1) % 50 == 0:
            print(f"[data] Baostock extension... {i+1}/{n_stocks}")
        try:
            code = _baostock_code(sid)
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,preclose,volume,amount,turn,pctChg",
                start_date=last_date.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
                frequency="d", adjustflag="1",
            )
            if rs.error_code != "0" or not rs.data:
                continue
            rows = rs.data
            # 过滤掉已有的日期
            new_rows = []
            for r in rows:
                d = pd.Timestamp(r[0])
                if d <= last_date:
                    continue
                if d > end:
                    continue
                new_rows.append(r)
            if not new_rows:
                continue

            df = pd.DataFrame(new_rows, columns=[
                "date", "open", "high", "low", "close", "preclose",
                "volume", "amount", "turn", "pctChg",
            ])
            for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["date"] = pd.to_datetime(df["date"])
            df["stock_id"] = sid
            df["amplitude"] = ((df["high"] - df["low"]) / df["preclose"].replace(0, np.nan) * 100).round(2)
            df["change"] = (df["close"] - df["preclose"]).round(2)
            frames.append(df[["date", "stock_id", "open", "high", "low", "close",
                               "preclose", "volume", "amount", "turn", "pctChg",
                               "amplitude", "change"]])
        except Exception as e:
            print(f"  [WARN] {sid}: {e}")
            continue
    bs.logout()

    extended = pd.concat(frames, ignore_index=True)
    extended = extended.sort_values(["date", "stock_id"]).reset_index(drop=True)

    # 重新计算 preclose（extended 的 preclose 来自 baostock，但拼接处可能断裂）
    extended["preclose"] = extended.groupby("stock_id")["close"].shift(1)
    extended["preclose"] = extended["preclose"].fillna(extended["open"])

    extended = extended.set_index(["date", "stock_id"])
    print(f"[data] Extended: {extended.shape[0]} rows, "
          f"{extended.index.get_level_values('date').nunique()} trading days")
    return extended


def get_official_stock_ids() -> list[str]:
    """从官方数据中获取所有股票代码（按数值排序，避免 '100' 排 '2' 前面）"""
    train = pd.read_csv(OFFICIAL_DIR / "train.csv", dtype={"股票代码": str})
    codes = sorted(train["股票代码"].unique(), key=lambda x: int(x))
    return [str(c).zfill(6) for c in codes]


def build_panel_from_official(
    stock_ids: list[str] | None = None,
    train_path: str | None = None,
    test_path: str | None = None,
) -> pd.DataFrame:
    """
    从 THU-BDC 官方提供的 train.csv + test.csv 加载数据。

    官方数据格式：
      股票代码(1-300), 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额,
      振幅(%), 涨跌额, 换手率(%), 涨跌幅(%)

    Notes:
      - 股票代码是 1-300 的整数，对应 CSI 300 成分股
      - 所有价格均为后复权（adjustflag=1）
      - preclose 由 close.shift(1) 按股票分组计算
    """
    # 使用默认项目本地路径
    if train_path is None:
        train_path = str(OFFICIAL_DIR / "train.csv")
    if test_path is None:
        test_path = str(OFFICIAL_DIR / "test.csv")

    cache_path = RAW_DIR / "panel_official.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "stock_id"])
        print(f"[data] Loaded cached official panel: {df.shape[0]} rows")
        return df

    # 读取官方 CSV
    train = pd.read_csv(train_path, dtype={"股票代码": str})
    test = pd.read_csv(test_path, dtype={"股票代码": str})

    df = pd.concat([train, test], ignore_index=True)
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])

    # 统一的列名映射
    rename_map = {
        "股票代码": "stock_id",
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turn",
        "涨跌幅": "pctChg",
        "振幅": "amplitude",
        "涨跌额": "change",
    }
    df = df.rename(columns=rename_map)
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # 计算 preclose（按股票分组，close.shift(1)）
    df["preclose"] = df.groupby("stock_id")["close"].shift(1)
    df["preclose"] = df["preclose"].fillna(df["open"])  # 缺失用开盘价代替

    # 数值列
    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 筛选股票
    if stock_ids is not None:
        df = df[df["stock_id"].isin(stock_ids)]

    panel = df.set_index(["date", "stock_id"])

    # 从 baostock 补充官方数据覆盖不到的日期（到 END_DATE）
    panel = _extend_with_baostock(panel, stock_ids)

    panel.to_parquet(cache_path)
    print(f"[data] Built official panel: {panel.shape[0]} rows, "
          f"{panel.index.get_level_values('stock_id').nunique()} stocks, "
          f"{panel.index.get_level_values('date').nunique()} trading days")
    return panel


# =============================================================================
# 获取中证 500 成分股
# =============================================================================

def fetch_zz500_stocks() -> list[str]:
    """
    从 baostock 获取中证 500 成分股列表，与沪深 300 互补覆盖中盘股。
    """
    cache_path = RAW_DIR / "zz500_stocks.pkl"
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())

    bs.login()
    rs = bs.query_zz500_stocks()
    stocks = []
    while rs.next():
        row = rs.get_row_data()
        stocks.append(row[1])
    bs.logout()

    processed = sorted(set(_strip_code(s) for s in stocks))
    cache_path.write_bytes(pickle.dumps(processed))
    print(f"[data] Fetched {len(processed)} CSI 500 stocks")
    return processed


def fetch_extended_stocks() -> list[str]:
    """
    合并沪深 300 + 中证 500 成分股，去重后作为扩展股票池。
    覆盖约 800 只大中盘股票，提供更丰富的截面信息。
    """
    hs300 = fetch_csi300_stocks()
    zz500 = fetch_zz500_stocks()
    combined = sorted(set(hs300) | set(zz500))
    print(f"[data] Extended stock pool: {len(hs300)} CSI300 + "
          f"{len(zz500)} CSI500 = {len(combined)} unique stocks")
    return combined


# =============================================================================
# 获取股票行业分类
# =============================================================================

# =============================================================================
# 获取股票基本面数据（季度财务报表）
# =============================================================================

def fetch_fundamental_data(
    stock_ids: list[str],
    start_year: int = 2022,
    end_year: int = 2026,
) -> pd.DataFrame:
    """
    从 baostock 批量下载季度财务报表数据。

    获取的因子：
      - 盈利能力: ROE, 净利润率, 毛利率, EPS(TTM)
      - 偿债能力: 流动比率, 资产负债率
      - 成长能力: 净利润同比增速, 营收同比增速
      - 营运能力: 总资产周转率

    返回:
        DataFrame with columns: stock_id, date, roe_ttm, np_margin, gp_margin,
                                eps_ttm, current_ratio, debt_to_asset,
                                profit_yoy, equity_yoy, asset_turnover
        date = 财报发布日期 (pubDate)，表示该信息可用的日期
    """
    cache_path = RAW_DIR / "fundamental_data.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"[data] Loaded cached fundamental data: {df.shape[0]} rows")
        return df

    # ---- 批量下载（单次登录，大幅提速）----
    print("[data] Downloading fundamental data (~1-2 min for 300 stocks) ...")
    all_rows = []
    total = len(stock_ids)

    try:
        bs.login()
        for i, sid in enumerate(stock_ids):
            if (i + 1) % 30 == 0:
                print(f"[data] Fundamental data... {i+1}/{total}")
            try:
                code = _baostock_code(sid)
                # 只用最近3年（2023-2025）
                for year in [2023, 2024, 2025]:
                    for quarter in [4, 2]:  # 年报和半年报就够了
                        # 利润表
                        rs = bs.query_profit_data(code, year, quarter)
                        if rs.error_code != "0" or not rs.data:
                            continue
                        profit = rs.data[0]
                        # 资产负债表
                        rs2 = bs.query_balance_data(code, year, quarter)
                        balance = rs2.data[0] if rs2.error_code == "0" and rs2.data else None
                        # 成长能力
                        rs3 = bs.query_growth_data(code, year, quarter)
                        growth = rs3.data[0] if rs3.error_code == "0" and rs3.data else None

                        all_rows.append({
                            "stock_id": sid,
                            "date": profit[1],  # pubDate
                            "roe_ttm": float(profit[3]) if profit[3] else None,
                            "np_margin": float(profit[4]) if profit[4] else None,
                            "gp_margin": float(profit[5]) if profit[5] else None,
                            "eps_ttm": float(profit[7]) if profit[7] else None,
                            "current_ratio": float(balance[3]) if balance and balance[3] else None,
                            "debt_to_asset": float(balance[7]) if balance and balance[7] else None,
                            "profit_yoy": float(growth[5]) if growth and growth[5] else None,
                            "equity_yoy": float(growth[3]) if growth and growth[3] else None,
                        })
            except Exception:
                continue
    finally:
        bs.logout()

    if not all_rows:
        print("[data] WARNING: No fundamental data fetched")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # 去重
    df = df.drop_duplicates(subset=["stock_id", "date"], keep="last")

    df.to_parquet(cache_path, index=False)
    print(f"[data] Fetched fundamental data: {df.shape[0]} rows, {df['stock_id'].nunique()} stocks")
    return df
    print(f"[data] Fetched fundamental data: {df.shape[0]} rows, "
          f"{df['stock_id'].nunique()} stocks")
    return df


def fetch_stock_industries(stock_ids: list[str] | None = None) -> dict[str, str]:
    """
    从 baostock 获取股票行业分类（证监会行业分类标准）。
    """
    cache_path = RAW_DIR / "stock_industries.pkl"
    if cache_path.exists():
        industries = pickle.loads(cache_path.read_bytes())
        if stock_ids is None:
            return industries
        return {s: industries.get(s, "Unknown") for s in stock_ids}

    bs.login()
    rs = bs.query_stock_industry()
    industries = {}
    while rs.next():
        row = rs.get_row_data()
        code = _strip_code(row[1])
        industry = row[3] if row[3] else "Unknown"
        industries[code] = industry
    bs.logout()

    cache_path.write_bytes(pickle.dumps(industries))
    print(f"[data] Fetched {len(industries)} stock industries")
    return industries


# =============================================================================
# 获取关联市场指数数据
# =============================================================================

def fetch_index_data(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    获取上证50、中证500、创业板指、上证综指的日线行情。

    这些指数的收益率序列将被用作宏观特征，让模型感知：
      - 大/中/小盘风格轮动（SSE50 vs CSI500 vs ChiNext）
      - 全市场情绪（SSE Composite）
      - 跨市场联动效应

    baostock 的指数数据包含和股票类似的 OHLCV 字段，
    但没有 PE/PB/turn/tradestatus 等个股专属字段。

    Returns:
        DataFrame with columns: date, index_name, open, high, low, close,
                                preclose, volume, amount, pctChg
        Index: (date, index_name)  MultiIndex
    """
    cache_path = RAW_DIR / "extra_indices.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df = df[mask]
        df = df.set_index(["date", "index_name"])
        print(f"[data] Loaded cached indices: {df.shape[0]} rows")
        return df

    bs.login()
    frames = []
    for baostock_code, index_name in EXTRA_INDICES.items():
        try:
            rs = bs.query_history_k_data_plus(
                baostock_code,
                "date,open,high,low,close,preclose,volume,amount,pctChg",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="2",
            )
            if rs.error_code != "0":
                print(f"[data] WARNING: Query failed for {index_name}: {rs.error_msg}")
                continue
            rows = rs.data
            if not rows:
                print(f"[data] WARNING: No data for index {index_name} ({baostock_code})")
                continue

            df = pd.DataFrame(rows, columns=[
                "date", "open", "high", "low", "close", "preclose",
                "volume", "amount", "pctChg",
            ])
            for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "pctChg"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            df["date"] = pd.to_datetime(df["date"])
            df["index_name"] = index_name
            df.sort_values("date", inplace=True)
            frames.append(df)
            print(f"[data] Fetched {index_name}: {len(df)} trading days")
        except Exception as e:
            print(f"[data] ERROR fetching {index_name}: {e}")

    bs.logout()

    panel = pd.concat(frames, ignore_index=True)
    panel.sort_values(["date", "index_name"], inplace=True)
    panel.set_index(["date", "index_name"], inplace=True)
    panel.to_parquet(cache_path)

    print(f"[data] Built index panel: {panel.shape[0]} rows, "
          f"{panel.index.get_level_values('index_name').nunique()} indices")
    return panel


# =============================================================================
# 事件窗口过滤
# =============================================================================

def build_event_mask(
    dates: list[pd.Timestamp],
) -> np.ndarray:
    """
    构建事件过滤掩码 —— 标记哪些训练样本的 label 窗口与重大事件重叠。

    原理：
      每笔训练样本的 label 是 [T+1, T+5] 这 4 个交易日的收益率。
      如果这 4 天刚好跨越了重大事件（如俄乌冲突、上海封城），
      那这期间的收益率是非正常的（恐慌暴跌 / 政策暴力拉升 / 流动性枯竭）。

    为什么剔除而不是保留？
      - 事件期间的定价逻辑与正常市场完全不同
      - 从这些样本学习可能导致模型学到错误模式
        （如"俄乌开战后军工股涨"——这不是可复现的选股规律）
      - 剔除后模型在正常市场环境下表现更稳健

    检测逻辑：
      对每个样本目标日期 d（=T+5），检查事件区间 [event_start-5, event_end+5]
      （前后各扩展5个自然日，约3-4个交易日，覆盖 pre-leak 和 post-recovery）。
      如果 d 落在这个扩展区间内，标记为剔除。

    Args:
        dates: 所有样本的目标日期列表（T+5，按时间排序的 Timestamp list）

    Returns:
        mask: (n_samples,) bool array, True = 保留, False = 剔除
    """
    from config import EVENT_FILTER_MODE

    mask = np.ones(len(dates), dtype=bool)
    event_hits = {e[0]: 0 for e in EVENT_WINDOWS}

    # 对每个事件区间，检查哪些样本与之重叠
    for name, start_str, end_str, reason in EVENT_WINDOWS:
        # 事件核心区间 ± padding（覆盖 label 窗口和事件前后的过渡期）
        ev_start = pd.Timestamp(start_str) - pd.Timedelta(days=5)
        ev_end   = pd.Timestamp(end_str)   + pd.Timedelta(days=5)

        for i, d in enumerate(dates):
            if not mask[i]:
                continue  # 已被之前的事件标记过
            if ev_start <= d <= ev_end:
                mask[i] = False
                event_hits[name] += 1

    total_excluded = (~mask).sum()
    total = len(dates)
    print(f"[data] Event filtering ({EVENT_FILTER_MODE} mode):")
    print(f"  Total samples: {total}, Excluded: {total_excluded} "
          f"({total_excluded/max(total,1)*100:.1f}%)")
    for name, count in event_hits.items():
        if count > 0:
            print(f"  - {name}: {count} samples excluded")
    print(f"  Retained: {mask.sum()} samples")

    return mask
