"""Data loading from CSV files (offline, no baostock)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR, EVENT_WINDOWS

# CSV column mapping (Chinese -> English)
_COL_MAP = {
    "股票代码": "stock_id", "日期": "date",
    "开盘": "open", "最高": "high", "最低": "low", "收盘": "close",
    "前收盘": "preclose", "成交量": "volume", "成交额": "amount",
    "换手率": "turn", "涨跌幅": "pctChg", "市盈率": "peTTM", "市净率": "pbMRQ",
}


# ---------------------------------------------------------------------------
# Load CSV data
# ---------------------------------------------------------------------------

def load_panel_from_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load daily kline data from CSV, return MultiIndex (date, stock_id) DataFrame.

    CSV columns (Chinese): 股票代码, 日期, 开盘, 最高, 最低, 收盘, 前收盘,
                           成交量, 成交额, 换手率, 涨跌幅, 市盈率, 市净率
    """
    csv_path = Path(csv_path)
    cache_path = DATA_DIR / "panel.pkl"

    if cache_path.exists():
        df = pd.read_pickle(cache_path)
        print(f"[data] Loaded cached panel: {df.shape[0]} rows")
        return df

    df = pd.read_csv(csv_path, dtype={"股票代码": str})

    # Rename Chinese columns to English
    df.rename(columns=_COL_MAP, inplace=True)

    # Parse
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "preclose", "volume",
                "amount", "turn", "pctChg", "peTTM", "pbMRQ"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["stock_id"] = df["stock_id"].astype(str).str.zfill(6)
    df.sort_values(["date", "stock_id"], inplace=True)
    df.set_index(["date", "stock_id"], inplace=True)

    df.to_pickle(cache_path)
    print(f"[data] Built panel: {df.shape[0]} rows, "
          f"{df.index.get_level_values('stock_id').nunique()} stocks")
    return df


def load_test_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load test.csv: 股票代码, 日期, 开盘, 收盘."""
    df = pd.read_csv(csv_path, dtype={"股票代码": str})
    df.rename(columns={"股票代码": "stock_id", "日期": "date",
                       "开盘": "open", "收盘": "close"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df["stock_id"] = df["stock_id"].astype(str).str.zfill(6)
    return df


# ---------------------------------------------------------------------------
# Event mask
# ---------------------------------------------------------------------------

def build_event_mask(dates: list[pd.Timestamp]) -> np.ndarray:
    """Filter samples whose label window overlaps with extreme events."""
    mask = np.ones(len(dates), dtype=bool)
    hits: dict[str, int] = {}

    for name, start_str, end_str, _ in EVENT_WINDOWS:
        ev_start = pd.Timestamp(start_str) - pd.Timedelta(days=5)
        ev_end = pd.Timestamp(end_str) + pd.Timedelta(days=5)
        hits[name] = 0
        for i, d in enumerate(dates):
            if mask[i] and ev_start <= d <= ev_end:
                mask[i] = False
                hits[name] += 1

    excluded = (~mask).sum()
    total = len(dates)
    print(f"[data] Event filter: {excluded}/{total} excluded ({excluded/max(total,1)*100:.1f}%)")
    for name, c in hits.items():
        if c > 0:
            print(f"  - {name}: {c}")
    return mask
