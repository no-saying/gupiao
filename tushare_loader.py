"""
=============================================================================
  Tushare 数据加载模块 —— 替代 baostock，全面接入 Tushare Pro
=============================================================================

数据源（按优先级）：
  1. THU-BDC 官方 CSV     — 基础 OHLCV（保留，赛题基准）
  2. Tushare daily        — 补充/验证 OHLCV + 扩展到 END_DATE
  3. Tushare daily_basic  — PE_TTM/PB/PS/市值/换手率(自由流通)/股息率
  4. Tushare moneyflow    — 大单/中单/小单/超大单资金流向
  5. Tushare margin_detail — 融资融券余额
  6. Tushare hk_hold       — 北向资金持股
  7. Tushare stk_factor_pro — 158个技术因子（替代手算）
  8. Tushare cyq_perf      — 筹码分布+胜率
  9. Tushare stk_holdernumber — 股东人数（筹码集中度）
  10. Tushare index_member_all — 申万行业分类（SW2021，3级）
  11. Tushare index_daily   — 市场指数行情
  12. Tushare adj_factor    — 复权因子
  13. Tushare trade_cal     — 交易日历
  14. Tushare limit_list_d  — 涨跌停数据
  15. Tushare fut_daily     — 股指期货（升贴水）

缓存策略：所有数据缓存为 parquet，分目录存放。
=============================================================================
"""

import time
import pickle
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import tushare as ts

from config import (
    RAW_DIR, OFFICIAL_DIR, START_DATE, END_DATE, EFFECTIVE_START,
    EXTRA_INDICES, EXTRA_INDEX_RET_WINDOWS,
)

# =============================================================================
# Tushare 配置
# =============================================================================

TUSHARE_TOKEN = "c517f0f062c5202be84c818bf38afc6372970aaa240cff5aa1df5923"
TUSHARE_DIR = RAW_DIR / "tushare"
TUSHARE_DIR.mkdir(parents=True, exist_ok=True)

# 初始化 tushare pro
_pro = None

def _get_pro():
    """获取 tushare pro 实例（懒加载 + 重试）。"""
    global _pro
    if _pro is None:
        _pro = ts.pro_api(TUSHARE_TOKEN)
    return _pro


def _ts_code(stock_id: str) -> str:
    """纯数字代码 → Tushare 格式: 600000 → 600000.SH, 000001 → 000001.SZ"""
    code = int(stock_id)
    if code >= 600000:
        return f"{stock_id}.SH"
    if code >= 200000:
        return f"{stock_id}.SZ"  # 20开头是深交所B股
    if code >= 900000:
        return f"{stock_id}.SH"  # 90开头是上交所B股
    if code >= 800000:
        return f"{stock_id}.BJ"  # 8开头是北交所
    if code >= 400000:
        return f"{stock_id}.BJ"  # 4开头是北交所/三板
    return f"{stock_id}.SZ"  # 0/3开头 → 深交所


def _strip_ts_code(ts_code: str) -> str:
    """600000.SH → 600000"""
    return ts_code.split(".")[0]


def _with_retry(fn, max_retries=3, sleep_sec=1.5):
    """带重试的 API 调用包装器。"""
    for attempt in range(max_retries):
        try:
            result = fn()
            if isinstance(result, pd.DataFrame) and result.empty and attempt < max_retries - 1:
                time.sleep(sleep_sec * (attempt + 1))
                continue
            return result
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"  [retry {attempt+1}/{max_retries}] {e}")
            time.sleep(sleep_sec * (attempt + 1))
    return pd.DataFrame()


# =============================================================================
# 交易日历
# =============================================================================

def fetch_trade_cal(start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
    """获取交易日历（上交所）。"""
    cache_path = TUSHARE_DIR / "trade_cal.parquet"
    if cache_path.exists():
        cal = pd.read_parquet(cache_path)
        cal["cal_date"] = pd.to_datetime(cal["cal_date"])
        return cal

    print("[tushare] Fetching trade calendar...")
    pro = _get_pro()
    cal = _with_retry(lambda: pro.trade_cal(
        exchange="SSE",
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
    ))
    if cal.empty:
        raise RuntimeError("Failed to fetch trade calendar")

    cal["cal_date"] = pd.to_datetime(cal["cal_date"])
    cal = cal.sort_values("cal_date").reset_index(drop=True)
    cal.to_parquet(cache_path, index=False)
    print(f"[tushare] Trade calendar: {len(cal)} days, "
          f"{cal['is_open'].sum()} trading days")
    return cal


def get_trading_days(start_date: str = START_DATE, end_date: str = END_DATE) -> list[pd.Timestamp]:
    """获取交易日列表。"""
    cal = fetch_trade_cal(start_date, end_date)
    open_days = cal[cal["is_open"] == 1]
    mask = (open_days["cal_date"] >= start_date) & (open_days["cal_date"] <= end_date)
    return sorted(open_days.loc[mask, "cal_date"].tolist())


# =============================================================================
# 股票列表
# =============================================================================

def fetch_csi300_stocks() -> list[str]:
    """获取沪深300成分股（从官方CSV，保持与赛题一致）。"""
    from config import OFFICIAL_DIR
    train = pd.read_csv(OFFICIAL_DIR / "train.csv", dtype={"股票代码": str})
    codes = sorted(train["股票代码"].unique(), key=lambda x: int(x))
    return [str(c).zfill(6) for c in codes]


def fetch_csi500_stocks() -> list[str]:
    """获取中证500成分股（最新一期）。"""
    cache_path = TUSHARE_DIR / "csi500_stocks.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)["stock_id"].tolist()

    print("[tushare] Fetching CSI 500 constituents...")
    pro = _get_pro()
    df = _with_retry(lambda: pro.index_weight(
        index_code="000905.SH",
        fields="index_code,con_code,weight",
    ))
    if df.empty:
        print("[tushare] CSI 500: empty, using backup")
        return []

    stocks = sorted(df["con_code"].str.replace(".SH", "").str.replace(".SZ", "").unique(),
                    key=lambda x: int(x) if x.isdigit() else 0)
    stocks = [s.zfill(6) for s in stocks]
    pd.DataFrame({"stock_id": stocks}).to_parquet(cache_path)
    print(f"[tushare] CSI 500: {len(stocks)} stocks")
    return stocks


def fetch_stock_basic() -> pd.DataFrame:
    """获取全市场股票基础信息（代码、名称、行业、上市日期）。"""
    cache_path = TUSHARE_DIR / "stock_basic.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    print("[tushare] Fetching stock basic info...")
    pro = _get_pro()
    df = _with_retry(lambda: pro.stock_basic(
        exchange="", list_status="L",
        fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,curr_type",
    ))
    if df.empty:
        raise RuntimeError("Failed to fetch stock basic")

    df["list_date"] = pd.to_datetime(df["list_date"])
    df.to_parquet(cache_path, index=False)
    print(f"[tushare] Stock basic: {len(df)} listed stocks")
    return df


def _fetch_stock_data(
    api_method_name: str,
    stock_ids: list[str],
    start_date: str,
    end_date: str,
    cache_name: str,
    extra_fields: list[str] | None = None,
) -> pd.DataFrame:
    """
    通用单股票数据下载器。逐只股票请求（Tushare 大部分 API 不支持批量 ts_code）。

    Args:
        api_method_name: pro.xxx 的方法名，如 "daily_basic", "moneyflow"
        stock_ids: 纯数字股票代码列表
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
        cache_name: parquet 缓存文件名（不含路径）
        extra_fields: 额外请求字段（除 ts_code, trade_date 外）
    """
    cache_path = TUSHARE_DIR / cache_name
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    print(f"[tushare] Downloading {cache_name} for {len(stock_ids)} stocks...")
    pro = _get_pro()
    api_fn = getattr(pro, api_method_name)
    s_date = start_date.replace("-", "")
    e_date = end_date.replace("-", "")

    all_frames = []
    for i, sid in enumerate(stock_ids):
        if (i + 1) % 50 == 0:
            print(f"  [tushare] {cache_name}: {i+1}/{len(stock_ids)}")
        ts_code = _ts_code(sid)
        try:
            df = _with_retry(lambda: api_fn(
                ts_code=ts_code,
                start_date=s_date,
                end_date=e_date,
            ), max_retries=2, sleep_sec=1.0)
        except Exception:
            continue
        if df is not None and not df.empty:
            all_frames.append(df)
        time.sleep(0.10)  # 频率控制: 500次/分钟, 每0.12秒一次

    if not all_frames:
        print(f"[tushare] WARNING: No {cache_name} data")
        return pd.DataFrame()

    result = pd.concat(all_frames, ignore_index=True)
    if "trade_date" in result.columns:
        result["trade_date"] = pd.to_datetime(result["trade_date"])
    result["stock_id"] = result["ts_code"].apply(_strip_ts_code)
    result.to_parquet(cache_path, index=False)
    print(f"[tushare] {cache_name}: {result.shape[0]} rows, {result['stock_id'].nunique()} stocks")
    return result

def fetch_daily_panel(
    stock_ids: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    从 Tushare 获取 OHLCV 日线数据，构建 Panel。

    合并策略：
      1. 优先使用已缓存的 per-stock parquet
      2. 增量下载缺失的日期范围
      3. 最终合并为 MultiIndex Panel
    """
    cache_path = TUSHARE_DIR / "daily_panel.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df = df[mask]
        df = df.set_index(["date", "stock_id"])
        print(f"[tushare] Loaded cached daily panel: {df.shape[0]} rows")
        return df

    print(f"[tushare] Downloading daily data for {len(stock_ids)} stocks...")
    pro = _get_pro()
    ts_codes = ",".join(_ts_code(s) for s in stock_ids)

    all_frames = []
    # 按年分批下载（Tushare 单次最大 6000 条，按年分保证不超）
    for year in range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1):
        y_start = f"{year}0101"
        y_end = f"{year}1231"
        if year == pd.Timestamp(start_date).year:
            y_start = start_date.replace("-", "")
        if year == pd.Timestamp(end_date).year:
            y_end = end_date.replace("-", "")

        print(f"  [tushare] Daily {year}...")
        # 分批请求（每次最多100只股票，避免URL过长）
        batch_size = 100
        for i in range(0, len(stock_ids), batch_size):
            batch_codes = ",".join(_ts_code(s) for s in stock_ids[i:i+batch_size])
            df = _with_retry(lambda: pro.daily(
                ts_code=batch_codes,
                start_date=y_start,
                end_date=y_end,
            ))
            if not df.empty:
                all_frames.append(df)
            time.sleep(0.15)  # 频率控制

    if not all_frames:
        raise RuntimeError("Failed to fetch any daily data")

    daily = pd.concat(all_frames, ignore_index=True)
    daily = daily.rename(columns={
        "trade_date": "date",
        "pct_chg": "pctChg",
        "vol": "volume",
    })
    daily["date"] = pd.to_datetime(daily["date"])
    daily["stock_id"] = daily["ts_code"].apply(_strip_ts_code)
    daily = daily.sort_values(["date", "stock_id"]).reset_index(drop=True)

    # 计算 preclose（Tushare 日线不含昨收）
    daily["preclose"] = daily.groupby("stock_id")["close"].shift(1)
    daily["preclose"] = daily["preclose"].fillna(daily["open"])

    # 振幅
    daily["amplitude"] = (
        (daily["high"] - daily["low"]) / daily["preclose"].replace(0, np.nan) * 100
    ).round(2)

    # 涨跌额
    daily["change"] = (daily["close"] - daily["preclose"]).round(2)

    panel = daily.set_index(["date", "stock_id"])
    panel.to_parquet(cache_path)
    print(f"[tushare] Daily panel built: {panel.shape[0]} rows, "
          f"{panel.index.get_level_values('stock_id').nunique()} stocks, "
          f"{panel.index.get_level_values('date').nunique()} days")
    return panel


# =============================================================================
# 每日基本面指标（PE/PB/PS/市值/换手率/股息率）
# =============================================================================

def fetch_daily_basic(
    stock_ids: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """获取每日基本面指标（PE/PB/PS/市值/换手率/股息率）。"""
    return _fetch_stock_data("daily_basic", stock_ids, start_date, end_date,
                             "daily_basic.parquet")


# =============================================================================
# 资金流向
# =============================================================================

def fetch_moneyflow(
    stock_ids: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """获取个股资金流向（大单/中单/小单/超大单）。"""
    return _fetch_stock_data("moneyflow", stock_ids, start_date, end_date,
                             "moneyflow.parquet")


# =============================================================================
# 融资融券
# =============================================================================

def fetch_margin_detail(
    stock_ids: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """获取融资融券明细。"""
    return _fetch_stock_data("margin_detail", stock_ids, start_date, end_date,
                             "margin_detail.parquet")


# =============================================================================
# 北向资金（沪深港通持股）
# =============================================================================

def fetch_hk_hold(
    stock_ids: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """获取北向资金持股明细。"""
    return _fetch_stock_data("hk_hold", stock_ids, start_date, end_date,
                             "hk_hold.parquet")


# =============================================================================
# 技术因子（Tushare 预计算 158 因子）
# =============================================================================

# 精选因子子集（避免维度爆炸，取信息量最大的 ~60 个）
STK_FACTOR_FIELDS = [
    # 均线类
    "ma_bfq_5", "ma_bfq_10", "ma_bfq_20", "ma_bfq_60",
    "ema_bfq_10", "ema_bfq_20", "ema_bfq_60",
    # 布林带
    "boll_upper_bfq", "boll_mid_bfq", "boll_lower_bfq",
    # MACD
    "macd_bfq", "macd_dif_bfq", "macd_dea_bfq",
    # RSI
    "rsi_bfq_6", "rsi_bfq_12", "rsi_bfq_24",
    # KDJ
    "kdj_k_bfq", "kdj_d_bfq", "kdj_bfq",
    # DMI/ADX（趋势强度，手算里没有）
    "dmi_pdi_bfq", "dmi_mdi_bfq", "dmi_adx_bfq", "dmi_adxr_bfq",
    # CCI（手算里没有）
    "cci_bfq",
    # BIAS（乖离率）
    "bias1_bfq", "bias2_bfq", "bias3_bfq",
    # BBI
    "bbi_bfq",
    # ATR
    "atr_bfq",
    # WR
    "wr_bfq", "wr1_bfq",
    # OBV 能量潮
    "obv_bfq",
    # 量价
    "vr_bfq", "mfi_bfq",
    # PSY 心理线（手算里没有）
    "psy_bfq", "psyma_bfq",
    # ROC 变动率
    "roc_bfq", "maroc_bfq",
    # MTM 动量
    "mtm_bfq", "mtmma_bfq",
    # TRIX 三重指数平滑
    "trix_bfq", "trma_bfq",
    # 肯特纳通道（手算里没有）
    "ktn_upper_bfq", "ktn_mid_bfq", "ktn_down_bfq",
    # 唐奇安通道（手算里没有）
    "taq_up_bfq", "taq_mid_bfq", "taq_down_bfq",
    # 薛斯通道
    "xsii_td1_bfq", "xsii_td2_bfq", "xsii_td3_bfq", "xsii_td4_bfq",
    # 简易波动
    "emv_bfq", "maemv_bfq",
    # 人气/意愿
    "brar_ar_bfq", "brar_br_bfq",
    # 其他
    "mass_bfq", "ma_mass_bfq",
    "updays", "downdays", "topdays", "lowdays",
    "asi_bfq", "asit_bfq",
    "dpo_bfq", "madpo_bfq",
    "expma_12_bfq", "expma_50_bfq",
    "cr_bfq",
]


def fetch_stk_factor_pro(
    stock_ids: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """获取 Tushare 技术因子（精选子集，~60个）。"""
    cache_path = TUSHARE_DIR / "stk_factor_pro.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    print(f"[tushare] Downloading stk_factor_pro ({len(STK_FACTOR_FIELDS)} factors × {len(stock_ids)} stocks)...")
    pro = _get_pro()
    fields_str = "ts_code,trade_date," + ",".join(STK_FACTOR_FIELDS)
    s_date = start_date.replace("-", "")
    e_date = end_date.replace("-", "")

    all_frames = []
    for i, sid in enumerate(stock_ids):
        if (i + 1) % 30 == 0:
            print(f"  [tushare] stk_factor: {i+1}/{len(stock_ids)}")
        ts_code = _ts_code(sid)
        df = _with_retry(lambda: pro.stk_factor_pro(
            ts_code=ts_code,
            start_date=s_date,
            end_date=e_date,
            fields=fields_str,
        ), max_retries=2, sleep_sec=1.0)
        if df is not None and not df.empty:
            all_frames.append(df)
        time.sleep(0.10)

    if not all_frames:
        print("[tushare] WARNING: No stk_factor_pro data")
        return pd.DataFrame()

    factor = pd.concat(all_frames, ignore_index=True)
    factor["trade_date"] = pd.to_datetime(factor["trade_date"])
    factor["stock_id"] = factor["ts_code"].apply(_strip_ts_code)

    # 保留关键列
    keep_cols = ["ts_code", "trade_date", "stock_id"] + [
        c for c in STK_FACTOR_FIELDS if c in factor.columns
    ]
    factor = factor[[c for c in keep_cols if c in factor.columns]]

    factor.to_parquet(cache_path, index=False)
    print(f"[tushare] stk_factor_pro: {factor.shape[0]} rows, {factor.shape[1]-3} factors")
    return factor


# =============================================================================
# 筹码分布 + 胜率
# =============================================================================

def fetch_cyq_perf(
    stock_ids: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """获取每日筹码平均成本和胜率。"""
    cache_path = TUSHARE_DIR / "cyq_perf.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "trade_date" in df.columns:
            df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    print(f"[tushare] Downloading cyq_perf for {len(stock_ids)} stocks...")
    pro = _get_pro()
    s_date = start_date.replace("-", "")
    e_date = end_date.replace("-", "")

    all_frames = []
    for i, sid in enumerate(stock_ids):
        if (i + 1) % 30 == 0:
            print(f"  [tushare] cyq_perf: {i+1}/{len(stock_ids)}")
        ts_code = _ts_code(sid)
        df = _with_retry(lambda: pro.cyq_perf(
            ts_code=ts_code,
            start_date=s_date,
            end_date=e_date,
        ), max_retries=2, sleep_sec=1.0)
        if df is not None and not df.empty:
            all_frames.append(df)
        time.sleep(0.10)

    if not all_frames:
        print("[tushare] WARNING: No cyq_perf data")
        return pd.DataFrame()

    cyq = pd.concat(all_frames, ignore_index=True)
    cyq["trade_date"] = pd.to_datetime(cyq["trade_date"])
    cyq["stock_id"] = cyq["ts_code"].apply(_strip_ts_code)
    cyq.to_parquet(cache_path, index=False)
    print(f"[tushare] cyq_perf: {cyq.shape[0]} rows")
    return cyq


# =============================================================================
# 股东人数（筹码集中度）
# =============================================================================

def fetch_stk_holdernumber(stock_ids: list[str],
                           start_date: str = "2021-01-01",
                           end_date: str = "2026-12-31") -> pd.DataFrame:
    """获取股东人数变化数据（每只股票一次API调用覆盖全历史）。"""
    cache_path = TUSHARE_DIR / "stk_holdernumber.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "ann_date" in df.columns:
            df["ann_date"] = pd.to_datetime(df["ann_date"])
        if "end_date" in df.columns:
            df["end_date"] = pd.to_datetime(df["end_date"])
        return df

    return _fetch_stock_data("stk_holdernumber", stock_ids,
                             "2021-01-01", "2026-12-31",
                             "stk_holdernumber.parquet")


# =============================================================================
# 申万行业分类（SW2021，3级）
# =============================================================================

def fetch_sw_industry() -> pd.DataFrame:
    """
    获取申万2021版行业分类成分股。
    返回：l1_code, l1_name, l2_code, l2_name, l3_code, l3_name, ts_code, name
    """
    cache_path = TUSHARE_DIR / "sw_industry.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    print("[tushare] Fetching SW2021 industry classification...")
    pro = _get_pro()

    df = _with_retry(lambda: pro.index_member_all(
        is_new="Y",
    ))
    if df.empty:
        print("[tushare] WARNING: No SW industry data")
        return pd.DataFrame()

    df["stock_id"] = df["ts_code"].apply(_strip_ts_code)
    df.to_parquet(cache_path, index=False)
    print(f"[tushare] SW industry: {df.shape[0]} rows, "
          f"{df['l1_name'].nunique()} L1, {df['l2_name'].nunique()} L2, "
          f"{df['l3_name'].nunique()} L3")
    return df


# =============================================================================
# 市场指数行情
# =============================================================================

def fetch_index_daily(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    获取市场指数日线行情。
    指数列表：上证50(000016), 中证500(000905), 创业板指(399006),
             上证综指(000001), 沪深300(000300), 中证1000(000852)
    """
    cache_path = TUSHARE_DIR / "index_daily.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    print("[tushare] Fetching index daily...")
    pro = _get_pro()

    # 6个关键指数
    index_codes = {
        "000016.SH": "SSE50",
        "000905.SH": "CSI500",
        "399006.SZ": "ChiNext",
        "000001.SH": "SSE_Composite",
        "000300.SH": "CSI300",
        "000852.SH": "CSI1000",
    }

    all_frames = []
    for code, name in index_codes.items():
        df = _with_retry(lambda: pro.index_daily(
            ts_code=code,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        ))
        if not df.empty:
            df["index_name"] = name
            all_frames.append(df)
            print(f"  [tushare] {name}: {len(df)} days")
        time.sleep(0.15)

    if not all_frames:
        print("[tushare] WARNING: No index data")
        return pd.DataFrame()

    idx = pd.concat(all_frames, ignore_index=True)
    idx["trade_date"] = pd.to_datetime(idx["trade_date"])
    # 统一列名
    idx = idx.rename(columns={
        "trade_date": "date",
        "pct_chg": "pctChg",
        "vol": "volume",
    })
    idx["preclose"] = idx.groupby("index_name")["close"].shift(1)
    idx.to_parquet(cache_path, index=False)
    print(f"[tushare] index_daily: {idx.shape[0]} rows, {idx['index_name'].nunique()} indices")
    return idx


# =============================================================================
# 涨跌停数据
# =============================================================================

def fetch_limit_list_d(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    获取每日涨跌停/炸板数据。
    字段：trade_date, ts_code, name, close, pct_chg, amount,
          limit_amount, float_mv, total_mv, turnover_ratio,
          fd_amount(封单金额), first_time, last_time, open_times,
          up_stat(涨停统计), limit_times(连板数), limit(U/D/Z)
    """
    cache_path = TUSHARE_DIR / "limit_list_d.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    print("[tushare] Fetching limit list...")
    pro = _get_pro()

    all_frames = []
    for year in range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1):
        y_start = f"{year}0101"
        y_end = f"{year}1231"
        if year == pd.Timestamp(start_date).year:
            y_start = start_date.replace("-", "")
        if year == pd.Timestamp(end_date).year:
            y_end = end_date.replace("-", "")

        print(f"  [tushare] limit_list {year}...")
        # 分涨停/跌停/炸板三种
        for lt in ["U", "D", "Z"]:
            df = _with_retry(lambda: pro.limit_list_d(
                limit_type=lt,
                start_date=y_start,
                end_date=y_end,
            ), max_retries=2, sleep_sec=1.0)
            if not df.empty:
                all_frames.append(df)
            time.sleep(0.15)

    if not all_frames:
        print("[tushare] WARNING: No limit_list_d data")
        return pd.DataFrame()

    limits = pd.concat(all_frames, ignore_index=True)
    limits["trade_date"] = pd.to_datetime(limits["trade_date"])
    limits["stock_id"] = limits["ts_code"].apply(_strip_ts_code)

    limits.to_parquet(cache_path, index=False)
    print(f"[tushare] limit_list_d: {limits.shape[0]} rows")
    return limits


# =============================================================================
# 股指期货（判断升贴水）
# =============================================================================

def fetch_futures_basis(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    获取股指期货主力合约日线，计算升贴水。
    升贴水 = (期货 - 现货) / 现货，正值=升水(看涨)，负值=贴水(看跌)
    """
    cache_path = TUSHARE_DIR / "futures_basis.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    print("[tushare] Fetching futures basis...")
    pro = _get_pro()

    # IF(沪深300), IC(中证500), IM(中证1000), IH(上证50)
    fut_codes = ["IF", "IC", "IM", "IH"]
    all_frames = []

    for fc in fut_codes:
        # 获取主力连续合约
        mapping = _with_retry(lambda: pro.fut_mapping(
            ts_code=f"{fc}L.CFX",
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", ""),
        ))
        if mapping.empty:
            print(f"  [tushare] No mapping for {fc}")
            continue

        # 取每日对应的主力合约，获取行情
        for _, row in mapping.iterrows():
            td = row["trade_date"]
            mts = row["mapping_ts_code"]
            df = _with_retry(lambda: pro.fut_daily(
                ts_code=mts,
                trade_date=td,
            ), max_retries=1, sleep_sec=0.5)
            if not df.empty:
                df["fut_code"] = fc
                all_frames.append(df)
            time.sleep(0.1)

    if not all_frames:
        print("[tushare] WARNING: No futures data")
        return pd.DataFrame()

    fut = pd.concat(all_frames, ignore_index=True)
    fut["trade_date"] = pd.to_datetime(fut["trade_date"])
    fut.to_parquet(cache_path, index=False)
    print(f"[tushare] futures: {fut.shape[0]} rows")
    return fut


# =============================================================================
# 沪深300指数权重
# =============================================================================

def fetch_csi300_weight() -> pd.DataFrame:
    """获取沪深300成分股权重（月度）。"""
    cache_path = TUSHARE_DIR / "csi300_weight.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df

    print("[tushare] Fetching CSI300 weights...")
    pro = _get_pro()

    # 按月获取
    all_frames = []
    for year in range(2021, 2027):
        for month in range(1, 13):
            # 每月第一天
            start = f"{year}{month:02d}01"
            end = f"{year}{month:02d}{28 if month == 2 else 30}"
            if year == 2021 and month < 6:
                continue
            if year == 2026 and month > 6:
                break

            df = _with_retry(lambda: pro.index_weight(
                index_code="000300.SH",
                start_date=start,
                end_date=end,
            ), max_retries=1, sleep_sec=0.3)
            if not df.empty:
                all_frames.append(df)
            time.sleep(0.12)

    if not all_frames:
        print("[tushare] WARNING: No CSI300 weight data")
        return pd.DataFrame()

    w = pd.concat(all_frames, ignore_index=True)
    w["trade_date"] = pd.to_datetime(w["trade_date"])
    w["stock_id"] = w["con_code"].apply(_strip_ts_code)
    w.to_parquet(cache_path, index=False)
    print(f"[tushare] CSI300 weights: {w.shape[0]} rows")
    return w


# =============================================================================
# 综合 Panel 构建（主入口）
# =============================================================================

def _load_official_csv(stock_ids: list[str]) -> pd.DataFrame:
    """直接读取官方 CSV（不经过 baostock 扩展）。"""
    from config import OFFICIAL_DIR

    train = pd.read_csv(OFFICIAL_DIR / "train.csv", dtype={"股票代码": str})
    test = pd.read_csv(OFFICIAL_DIR / "test.csv", dtype={"股票代码": str})
    df = pd.concat([train, test], ignore_index=True)
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])

    rename_map = {
        "股票代码": "stock_id", "日期": "date",
        "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
        "成交量": "volume", "成交额": "amount", "换手率": "turn",
        "涨跌幅": "pctChg", "振幅": "amplitude", "涨跌额": "change",
    }
    df = df.rename(columns=rename_map)
    df = df.sort_values(["stock_id", "date"]).reset_index(drop=True)

    # preclose
    df["preclose"] = df.groupby("stock_id")["close"].shift(1)
    df["preclose"] = df["preclose"].fillna(df["open"])

    for col in ["open", "high", "low", "close", "preclose", "volume", "amount", "turn", "pctChg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if stock_ids is not None:
        df = df[df["stock_id"].isin(stock_ids)]
    return df


def _fetch_daily_ohlcv(stock_ids: list[str], start_dt: str, end_dt: str) -> pd.DataFrame:
    """用 Tushare stk_factor_pro 的后复权(_hfq)价格拉取 OHLCV。

    为什么用 stk_factor_pro 而不是 daily:
      - Tushare daily 返回不复权价格，官方 CSV 是后复权
      - stk_factor_pro 提供 open_hfq/high_hfq/low_hfq/close_hfq/pre_close_hfq
      - 后复权价格和官方 CSV 完全一致，不会产生拼接断裂
    """
    pro = _get_pro()
    frames = []
    s = start_dt.replace("-", "")
    e = end_dt.replace("-", "")
    fields = "ts_code,trade_date,open_hfq,high_hfq,low_hfq,close_hfq,pre_close_hfq,vol,amount"

    for i, sid in enumerate(stock_ids):
        if (i + 1) % 50 == 0:
            print(f"  [tushare] hfq {start_dt}→{end_dt}: {i+1}/{len(stock_ids)}")
        ts_code = _ts_code(sid)
        df = _with_retry(lambda: pro.stk_factor_pro(
            ts_code=ts_code, start_date=s, end_date=e, fields=fields,
        ), max_retries=2, sleep_sec=0.5)
        if df is None or df.empty:
            continue
        df["date"] = pd.to_datetime(df["trade_date"])
        df = df.rename(columns={
            "open_hfq": "open", "high_hfq": "high",
            "low_hfq": "low", "close_hfq": "close",
            "vol": "volume",
        })
        df = df.sort_values("date")
        df["stock_id"] = sid
        # stk_factor_pro 没有 pre_close_hfq，从 close.shift(1) 算
        df["preclose"] = df["close"].shift(1)
        df["preclose"] = df["preclose"].fillna(df["open"])
        df["amplitude"] = ((df["high"] - df["low"]) / df["preclose"].replace(0, np.nan) * 100).round(2)
        df["change"] = (df["close"] - df["preclose"]).round(2)
        # pctChg 从 close/preclose 重算
        df["pctChg"] = ((df["close"] - df["preclose"]) / df["preclose"].replace(0, np.nan) * 100).round(2)
        # turn 缺失（stk_factor_pro 无换手率），填0
        df["turn"] = 0.0
        keep = ["date", "stock_id", "open", "high", "low", "close", "preclose",
                "volume", "amount", "turn", "pctChg", "amplitude", "change"]
        frames.append(df[[c for c in keep if c in df.columns]])
        time.sleep(0.10)

    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["stock_id", "date"]).reset_index(drop=True)
    result["preclose"] = result.groupby("stock_id")["close"].shift(1)
    result["preclose"] = result["preclose"].fillna(result["open"])
    return result


def _extend_with_tushare(panel: pd.DataFrame, stock_ids: list[str]) -> pd.DataFrame:
    """用 Tushare 后复权(hfq)价格扩展官方 CSV 之后的日期。

    只往后扩展，不往前补：
      - 官方 CSV 的复权基准与 Tushare hfq 不同，往前补会产生价格断裂
      - 往后扩展(2026-03→2026-06)时间短，且近期分红少，断裂可忽略
    """
    last_date = panel["date"].max()
    end = pd.Timestamp(END_DATE)
    if last_date >= end:
        return panel

    print(f"[tushare] Extending OHLCV (hfq): {last_date.date()} → {end.date()} ...")
    post = _fetch_daily_ohlcv(stock_ids,
                              (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                              end.strftime("%Y-%m-%d"))
    if not post.empty:
        post = post[post["date"] > last_date]
        # 拼接处对齐：用拼接日前一天的官方价格做基准，缩放Tushare价格
        boundary_date = last_date
        for sid in stock_ids:
            off_row = panel[(panel['stock_id'] == sid) & (panel['date'] == boundary_date)]
            ts_rows = post[post['stock_id'] == sid]
            if off_row.empty or ts_rows.empty:
                continue
            off_close = off_row['close'].values[0]
            ts_first_close = ts_rows['close'].iloc[0]
            if ts_first_close > 0:
                ratio = off_close / ts_first_close
                if 0.5 < ratio < 2.0:  # 合理性检查
                    for col in ['open', 'high', 'low', 'close', 'preclose']:
                        post.loc[post['stock_id'] == sid, col] *= ratio
        panel = pd.concat([panel, post], ignore_index=True)

    panel = panel.sort_values(["stock_id", "date"]).reset_index(drop=True)
    panel["preclose"] = panel.groupby("stock_id")["close"].shift(1)
    panel["preclose"] = panel["preclose"].fillna(panel["open"])
    n_dates = panel["date"].nunique()
    print(f"[tushare] OHLCV ready: {panel.shape[0]} rows, {n_dates} trading days, "
          f"{panel['date'].min().date()} ~ {panel['date'].max().date()}")
    return panel


def build_enriched_panel(
    stock_ids: list[str],
    start_date: str = START_DATE,
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    构建增强版 Panel：官方 CSV 为底，Tushare 扩展 + 增强。

    OHLCV 来源：
      1. 官方 CSV (train.csv + test.csv) — 赛题基准
      2. Tushare daily — 扩展官方数据之后的日期

    增强来源（全部 Tushare）：
      3. daily_basic → PE/PB/PS/市值/换手率
      4. moneyflow → 资金流向
      5. margin_detail → 融资融券
      6. hk_hold → 北向资金
      7-13. stk_factor_pro / cyq_perf / 股东人数 / SW行业 / 指数 / 涨跌停 / 权重
    """
    cache_path = TUSHARE_DIR / "enriched_panel.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df = df[mask]
        df = df.set_index(["date", "stock_id"])
        print(f"[tushare] Loaded cached enriched panel: {df.shape}")
        return df

    # ---- Step 1: 官方 CSV OHLCV + Tushare 扩展 ----
    print("[tushare] Step 1/10: Loading official CSV + Tushare extension...")
    panel = _load_official_csv(stock_ids)
    panel = _extend_with_tushare(panel, stock_ids)

    # ---- Step 2: 合并 daily_basic ----
    print("[tushare] Step 2/10: Merging daily_basic (PE/PB/PS/mktcap)...")
    basic = fetch_daily_basic(stock_ids, start_date, end_date)
    if not basic.empty:
        basic_cols = ["trade_date", "stock_id",
                      "pe", "pe_ttm", "pb", "ps", "ps_ttm",
                      "dv_ratio", "dv_ttm",
                      "total_share", "float_share", "free_share",
                      "total_mv", "circ_mv",
                      "turnover_rate", "turnover_rate_f", "volume_ratio"]
        basic_cols = [c for c in basic_cols if c in basic.columns]
        # 前缀区分 Tushare 的 pe/pb（比 baostock 更可靠）
        rename = {}
        for c in ["pe", "pe_ttm", "pb", "ps", "ps_ttm"]:
            if c in basic.columns:
                rename[c] = f"ts_{c}"
        basic = basic.rename(columns=rename)
        basic_cols_renamed = [rename.get(c, c) for c in basic_cols]
        panel = panel.merge(
            basic[basic_cols_renamed],
            left_on=["date", "stock_id"],
            right_on=["trade_date", "stock_id"],
            how="left",
        )
        # 去重 trade_date 列
        if "trade_date" in panel.columns and "date" in panel.columns:
            panel = panel.drop(columns=["trade_date"])

    # ---- Step 3: 合并 moneyflow ----
    print("[tushare] Step 3/10: Merging moneyflow...")
    mf = fetch_moneyflow(stock_ids, start_date, end_date)
    if not mf.empty:
        mf_cols = ["trade_date", "stock_id",
                   "buy_sm_vol", "buy_sm_amount", "sell_sm_vol", "sell_sm_amount",
                   "buy_md_vol", "buy_md_amount", "sell_md_vol", "sell_md_amount",
                   "buy_lg_vol", "buy_lg_amount", "sell_lg_vol", "sell_lg_amount",
                   "buy_elg_vol", "buy_elg_amount", "sell_elg_vol", "sell_elg_amount",
                   "net_mf_vol", "net_mf_amount"]
        mf_cols = [c for c in mf_cols if c in mf.columns]
        panel = panel.merge(
            mf[mf_cols],
            left_on=["date", "stock_id"],
            right_on=["trade_date", "stock_id"],
            how="left",
        )
        if "trade_date" in panel.columns:
            panel = panel.drop(columns=["trade_date"])

    # ---- Step 4: 合并 margin ----
    print("[tushare] Step 4/10: Merging margin_detail...")
    mg = fetch_margin_detail(stock_ids, start_date, end_date)
    if not mg.empty:
        mg_cols = ["trade_date", "stock_id",
                   "rzye", "rqye", "rzmre", "rqyl", "rzche", "rqchl", "rqmcl", "rzrqye"]
        mg_cols = [c for c in mg_cols if c in mg.columns]
        panel = panel.merge(
            mg[mg_cols],
            left_on=["date", "stock_id"],
            right_on=["trade_date", "stock_id"],
            how="left",
        )
        if "trade_date" in panel.columns:
            panel = panel.drop(columns=["trade_date"])

    # ---- Step 5: 合并 hk_hold ----
    print("[tushare] Step 5/10: Merging north-bound holdings...")
    hk = fetch_hk_hold(stock_ids, start_date, end_date)
    if not hk.empty:
        hk_cols = ["trade_date", "stock_id", "vol", "ratio"]
        hk_cols = [c for c in hk_cols if c in hk.columns]
        hk_use = hk[hk_cols].rename(columns={
            "vol": "north_hold_vol",
            "ratio": "north_hold_ratio",
        })
        panel = panel.merge(
            hk_use,
            left_on=["date", "stock_id"],
            right_on=["trade_date", "stock_id"],
            how="left",
        )
        if "trade_date" in panel.columns:
            panel = panel.drop(columns=["trade_date"])

    # ---- Step 6: 合并 stk_factor_pro ----
    print("[tushare] Step 6/10: Merging technical factors...")
    factor = fetch_stk_factor_pro(stock_ids, start_date, end_date)
    if not factor.empty:
        factor = factor.drop(columns=["ts_code"], errors="ignore")
        panel = panel.merge(
            factor,
            left_on=["date", "stock_id"],
            right_on=["trade_date", "stock_id"],
            how="left",
            suffixes=("", "_factor"),
        )
        if "trade_date" in panel.columns:
            panel = panel.drop(columns=["trade_date"])

    # ---- Step 7: 合并 cyq_perf ----
    print("[tushare] Step 7/10: Merging chip distribution...")
    cyq = fetch_cyq_perf(stock_ids, start_date, end_date)
    if not cyq.empty:
        cyq = cyq.drop(columns=["ts_code"], errors="ignore")
        panel = panel.merge(
            cyq,
            left_on=["date", "stock_id"],
            right_on=["trade_date", "stock_id"],
            how="left",
            suffixes=("", "_cyq"),
        )
        if "trade_date" in panel.columns:
            panel = panel.drop(columns=["trade_date"])

    # ---- Step 8: 合并股东人数 ----
    print("[tushare] Step 8/10: Merging shareholder numbers...")
    holders = fetch_stk_holdernumber(stock_ids)
    if not holders.empty:
        holders = holders.drop(columns=["ts_code"], errors="ignore")
        # 股东人数是季度数据，用 ann_date 向前填充
        holders_use = holders[["ann_date", "stock_id", "holder_num"]].rename(
            columns={"ann_date": "date"})
        holders_use["date"] = pd.to_datetime(holders_use["date"])
        holders_use = holders_use.sort_values(["stock_id", "date"])
        panel = panel.merge(
            holders_use,
            on=["date", "stock_id"],
            how="left",
        )
        # 向前填充
        panel["holder_num"] = panel.groupby("stock_id")["holder_num"].ffill()

    # ---- Step 9: 合并申万行业 ----
    print("[tushare] Step 9/10: Merging SW industry...")
    sw = fetch_sw_industry()
    if not sw.empty:
        sw_use = sw[["stock_id", "l1_code", "l1_name", "l2_code", "l2_name",
                      "l3_code", "l3_name"]].drop_duplicates(subset="stock_id", keep="first")
        panel = panel.merge(sw_use, on="stock_id", how="left")

    # ---- Step 10b: 合并复权因子 ----
    # 注意：stk_factor_pro 自带 adj_factor 列，merge 会产生 _x/_y 后缀
    # 直接用 adj_factor 接口的数据覆盖
    print("[tushare] Step 10b/10: Merging adj_factor...")
    adj = _fetch_stock_data("adj_factor", stock_ids, start_date, end_date, "adj_factor.parquet")
    if not adj.empty:
        # 先删除 panel 中可能存在的 adj_factor（来自 stk_factor_pro 的旧列）
        if "adj_factor" in panel.columns:
            panel = panel.drop(columns=["adj_factor"])
        adj_use = adj[["trade_date", "stock_id", "adj_factor"]].rename(
            columns={"trade_date": "date"})
        panel = panel.merge(adj_use, on=["date", "stock_id"], how="left")
        panel["adj_factor"] = panel["adj_factor"].fillna(1.0).clip(0.01, 1000)

    # ---- Step 10: 合并涨跌停 ----
    print("[tushare] Step 10/10: Merging limit_list_d...")
    limits = fetch_limit_list_d(start_date, end_date)
    if not limits.empty:
        # 对每个 stock_id + trade_date，统计涨停/跌停/炸板次数
        limit_pivot = limits.pivot_table(
            index=["trade_date", "stock_id"],
            columns="limit",
            values="limit_times",
            aggfunc="max",
        ).fillna(0).reset_index()
        limit_pivot = limit_pivot.rename(columns={
            "U": "limit_up_times",
            "D": "limit_down_times",
            "Z": "broken_board_times",
        })
        for c in ["limit_up_times", "limit_down_times", "broken_board_times"]:
            if c not in limit_pivot.columns:
                limit_pivot[c] = 0
        panel = panel.merge(
            limit_pivot[["trade_date", "stock_id",
                         "limit_up_times", "limit_down_times", "broken_board_times"]],
            left_on=["date", "stock_id"],
            right_on=["trade_date", "stock_id"],
            how="left",
        )
        if "trade_date" in panel.columns:
            panel = panel.drop(columns=["trade_date"])

    # ---- 清理 ----
    panel = panel.set_index(["date", "stock_id"])
    panel = panel.sort_index()

    # 去重（merge 可能产生重复行）
    n_before = panel.shape[0]
    panel = panel[~panel.index.duplicated(keep='first')]
    if panel.shape[0] < n_before:
        print(f"  Dedup: removed {n_before - panel.shape[0]} duplicate rows")

    # 填充合并产生的 NaN
    panel = panel.groupby("stock_id").ffill()
    # 剩下的 NaN 填 0
    panel = panel.fillna(0)

    # 修复混合类型列（避免 Parquet 写入报错）
    for col in panel.columns:
        if panel[col].dtype == object:
            try:
                panel[col] = panel[col].astype(str)
            except Exception:
                pass

    # 缓存
    panel.to_parquet(cache_path)
    print(f"[tushare] Enriched panel built: {panel.shape[0]} rows × {panel.shape[1]} cols")
    return panel


# =============================================================================
# 事件过滤（保留，从 data_loader 迁出）
# =============================================================================

def build_event_mask(dates: list[pd.Timestamp]) -> np.ndarray:
    """
    构建事件过滤掩码 — 软降权替代硬删除，平滑断裂处。

    返回: mask (bool array), sample_weights (float array)
      - mask[i]=False: 样本 label 窗口完全落在极端事件内 → 剔除
      - weight[i]=0.3~0.5: 样本在事件边缘 (±10天) → 保留但降权
      - weight[i]=1.0: 正常样本
    """
    from config import EVENT_WINDOWS

    n = len(dates)
    mask = np.ones(n, dtype=bool)
    weights = np.ones(n, dtype=np.float32)
    event_hits = {e[0]: 0 for e in EVENT_WINDOWS}

    for name, start_str, end_str, reason in EVENT_WINDOWS:
        ev_start = pd.Timestamp(start_str)
        ev_end = pd.Timestamp(end_str)
        # 硬剔除窗口：事件核心 ± 3天（label窗口完全覆盖时必定异常）
        hard_start = ev_start - pd.Timedelta(days=0)
        hard_end = ev_end + pd.Timedelta(days=4)   # label覆盖T+1~T+5
        # 软降权窗口：事件边缘 ± 10天（过渡期，保留但降权）
        soft_start = ev_start - pd.Timedelta(days=10)
        soft_end = ev_end + pd.Timedelta(days=10)

        for i, d in enumerate(dates):
            if not mask[i]:
                continue
            # 硬剔除：预测日在事件核心内 → 未来一周必然被污染
            if hard_start <= d <= hard_end:
                mask[i] = False
                event_hits[name] += 1
            # 软降权：事件边缘 → 三角权重 (距事件越近权重越低)
            elif soft_start <= d <= soft_end:
                # 计算到事件核心的最短距离 (0=紧邻, 10=边缘)
                dist = min(abs((d - ev_start).days), abs((d - ev_end).days))
                dist = max(0, min(10, dist))
                w = 0.3 + 0.7 * (dist / 10.0)  # 紧邻→0.3, 边缘→1.0
                weights[i] = min(weights[i], w)

    total_excluded = (~mask).sum()
    n_downweighted = (weights < 1.0).sum() - total_excluded
    print(f"[tushare] Event filter: {total_excluded}/{n} excluded, "
          f"{n_downweighted} soft-downweighted (edge blend), "
          f"{mask.sum()} retained")
    return mask, weights


# =============================================================================
# 批量下载（一键下载所有数据，供首次使用）
# =============================================================================

def download_all(stock_ids: list[str] | None = None):
    """一键下载所有 Tushare 数据并缓存。"""
    if stock_ids is None:
        stock_ids = fetch_csi300_stocks()

    print("=" * 60)
    print("[tushare] Downloading ALL data types...")
    print(f"[tushare] {len(stock_ids)} stocks, {START_DATE} → {END_DATE}")
    print("=" * 60)

    # 交易日历（先下载，后续可能需要）
    fetch_trade_cal()

    # 基本面
    fetch_daily_basic(stock_ids)

    # 资金流向
    fetch_moneyflow(stock_ids)

    # 融资融券
    fetch_margin_detail(stock_ids)

    # 北向资金
    fetch_hk_hold(stock_ids)

    # 技术因子（耗时最长）
    fetch_stk_factor_pro(stock_ids)

    # 筹码分布
    fetch_cyq_perf(stock_ids)

    # 股东人数
    fetch_stk_holdernumber(stock_ids)

    # 申万行业
    fetch_sw_industry()

    # 指数行情
    fetch_index_daily()

    # 涨跌停
    fetch_limit_list_d()

    # 沪深300权重
    fetch_csi300_weight()

    # 构建综合 Panel
    print("\n[tushare] Building enriched panel...")
    build_enriched_panel(stock_ids)

    print("\n[tushare] All done! Data cached in", TUSHARE_DIR)


def build_tushare_only_panel(
    stock_ids: list[str],
    start_date: str = "2021-06-01",
    end_date: str = END_DATE,
) -> pd.DataFrame:
    """
    纯 Tushare hfq Panel：用于跨周期验证（覆盖2021-2026含熊市）。
    不依赖官方 CSV，全部使用 Tushare stk_factor_pro 后复权价格。
    与竞赛 Panel 分开，仅用于模型评估。
    """
    cache_path = TUSHARE_DIR / "tushare_only_panel.parquet"
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df = df[mask]
        df = df.set_index(["date", "stock_id"])
        print(f"[tushare] Loaded Tushare-only panel: {df.shape}")
        return df

    print(f"[tushare] Building Tushare-only panel ({start_date}~{end_date})...")
    # Step 1: 全周期 hfq OHLCV
    ohlcv = _fetch_daily_ohlcv(stock_ids, start_date, end_date)
    if ohlcv.empty:
        raise RuntimeError("Failed to fetch Tushare OHLCV")
    panel = ohlcv
    print(f"  OHLCV: {panel.shape[0]} rows, {panel.date.nunique()} days")

    # Step 2-10: 合并所有 Tushare 增强数据（与 build_enriched_panel 相同）
    for step_name, fetch_fn, cache_fn in [
        ("daily_basic", fetch_daily_basic, "daily_basic.parquet"),
        ("moneyflow", fetch_moneyflow, "moneyflow.parquet"),
        ("margin_detail", fetch_margin_detail, "margin_detail.parquet"),
        ("hk_hold", fetch_hk_hold, "hk_hold.parquet"),
        ("stk_factor_pro", fetch_stk_factor_pro, "stk_factor_pro.parquet"),
        ("cyq_perf", fetch_cyq_perf, "cyq_perf.parquet"),
        ("holder_number", fetch_stk_holdernumber, "stk_holdernumber.parquet"),
    ]:
        print(f"  Merging {step_name}...")
        df = fetch_fn(stock_ids, start_date, end_date)
        if df is None or df.empty:
            continue
        if "adj_factor" in df.columns and "adj_factor" in panel.columns:
            df = df.drop(columns=["adj_factor"])
        if "ts_code" in df.columns:
            df = df.drop(columns=["ts_code"], errors="ignore")
        # 确定日期列名（不同API返回不同列名）
        date_col = "trade_date"
        if "trade_date" not in df.columns:
            if "ann_date" in df.columns:
                date_col = "ann_date"
            elif "date" in df.columns:
                date_col = "date"
        panel = panel.merge(
            df, left_on=["date", "stock_id"], right_on=[date_col, "stock_id"],
            how="left", suffixes=("", "_dup"))
        # 清理重复列
        for c in panel.columns:
            if c.endswith("_dup"):
                panel = panel.drop(columns=[c])

    # SW 行业
    sw = fetch_sw_industry()
    if not sw.empty:
        sw_use = sw[["stock_id", "l1_code", "l1_name", "l2_code", "l2_name",
                      "l3_code", "l3_name"]].drop_duplicates(subset="stock_id", keep="first")
        panel = panel.merge(sw_use, on="stock_id", how="left")

    # adj_factor
    adj = _fetch_stock_data("adj_factor", stock_ids, start_date, end_date, "adj_factor_t.parquet")
    if not adj.empty:
        adj_use = adj[["trade_date", "stock_id", "adj_factor"]].rename(
            columns={"trade_date": "date"})
        panel = panel.merge(adj_use, on=["date", "stock_id"], how="left")
        panel["adj_factor"] = panel["adj_factor"].fillna(1.0).clip(0.01, 1000)

    # 清理
    panel = panel.set_index(["date", "stock_id"]).sort_index()
    panel = panel[~panel.index.duplicated(keep='first')]
    panel = panel.groupby("stock_id").ffill()
    panel = panel.fillna(0)

    # 修复混合类型列
    for col in panel.columns:
        if panel[col].dtype == object:
            try:
                panel[col] = panel[col].astype(str)
            except Exception:
                pass

    panel.to_parquet(cache_path)
    print(f"[tushare] Tushare-only panel: {panel.shape[0]} rows × {panel.shape[1]} cols, "
          f"{panel.index.get_level_values('date').nunique()} trading days")
    return panel


if __name__ == "__main__":
    download_all()
