"""
Alpha158 因子 — 使用 ultimate_ols 引擎（Numba 12核 + O(1) 滑动窗口 DP）。

用法:
  from core.alpha158 import add_alpha158_features
  panel = add_alpha158_features(panel)   # panel: MultiIndex(date, stock_id) DataFrame

缓存到 data/raw/alpha158_panel.parquet，重跑秒级。
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd

from config import RAW_DIR

# ── 因子输出列名（用于缓存校验和清理） ──
ALPHA158_COLS = [
    # 标准差
    'STD5', 'STD10', 'STD20', 'STD30', 'STD60',
    # 均值
    'MEAN5', 'MEAN10', 'MEAN20', 'MEAN30', 'MEAN60',
    # 分位数
    'QTLU20', 'QTLU60', 'QTLD20', 'QTLD60',
    # 极值比值
    'MAX5', 'MAX10', 'MAX20', 'MIN5', 'MIN10', 'MIN20',
    # 极值位置
    'IMAX5', 'IMAX10', 'IMAX20',
    'IMIN5', 'IMIN10', 'IMIN20',
    'IMXD10', 'IMXD20',
    # 计数
    'CNTP5', 'CNTP10', 'CNTP20',
    'CNTN5', 'CNTN10', 'CNTN20',
    'CNTD5', 'CNTD10', 'CNTD20',
    # 涨跌比例
    'SUMP5', 'SUMP10', 'SUMP20',
    'SUMN5', 'SUMN10', 'SUMN20',
    'SUMD5', 'SUMD10', 'SUMD20',
    # OLS 回归
    'BETA5', 'BETA10', 'BETA20',
    'RSQR10', 'RSQR20',
    'RESI5', 'RESI10', 'RESI20',
    # 加权波动
    'WVMA5', 'WVMA10', 'WVMA20',
    # VWAP
    'VWAP0',
    # 相关系数
    'CORR10', 'CORR20',
    # 排名
    'RANK5', 'RANK10', 'RANK20',
]


def add_alpha158_features(panel: pd.DataFrame, force_recompute: bool = False) -> pd.DataFrame:
    """
    使用 ultimate_ols 引擎计算 Alpha158 因子（12核 Numba 并行）。

    Parameters
    ----------
    panel : pd.DataFrame
        MultiIndex(date, stock_id) 或普通列 date/stock_id
    force_recompute : bool
        强制重新计算（忽略缓存）

    Returns
    -------
    pd.DataFrame — 原 panel + 所有 alpha158 列（MultiIndex 格式）
    """
    cache_path = RAW_DIR / "alpha158_panel.parquet"

    # ── 缓存命中 ──
    if cache_path.exists() and not force_recompute:
        df = pd.read_parquet(cache_path)
        if "date" not in df.columns:
            df = df.reset_index()
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["date", "stock_id"])
        n_cols = len([c for c in ALPHA158_COLS if c in df.columns])
        print(f"[alpha158] Loaded cached: {df.shape} ({n_cols} factors)")
        return df

    t0 = time.time()

    # ── 转换为 flat 格式 ──
    if isinstance(panel.index, pd.MultiIndex):
        df = panel.reset_index()
    else:
        df = panel.copy()

    n_stocks = df['stock_id'].nunique()
    n_days = df['date'].nunique()
    print(f"[alpha158] Computing factors for {n_stocks} stocks × {n_days} days...")

    # ── 使用 ultimate_ols 引擎 ──
    from core.ultimate_ols import RollingFeatureEngine
    engine = RollingFeatureEngine()
    df = engine.compute_all(df)

    # ── 恢复 MultiIndex ──
    result = df.set_index(["date", "stock_id"])

    n_added = len([c for c in ALPHA158_COLS if c in result.columns])
    elapsed = time.time() - t0
    print(f"[alpha158] Added {n_added} factors in {elapsed:.1f}s ({elapsed/n_stocks:.3f}s/stock)")

    # ── 缓存 ──
    result.reset_index().to_parquet(cache_path, index=False)
    print(f"[alpha158] Cached to {cache_path}")

    return result
