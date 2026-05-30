"""Feature engineering — compute 50+ factors from raw OHLCV data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    LOOKBACK_DAYS, PREDICT_HORIZON, STEP_DAYS,
    MOMENTUM_WINDOWS, VOLATILITY_WINDOWS, MA_WINDOWS,
    RSI_PERIOD, MACD_FAST, MACD_SLOW, MACD_SIGNAL, BETA_WINDOW,
)

_NORM_STATS: tuple | None = None


def get_norm_stats() -> tuple | None:
    return _NORM_STATS


# ---------------------------------------------------------------------------
# Per-stock factor computation
# ---------------------------------------------------------------------------

def _compute_stock_factors(group: pd.DataFrame) -> pd.DataFrame:
    """Compute all technical factors for a single stock's time-series."""
    df = group.copy()
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    # Momentum
    ret_1d = close.pct_change()
    for w in MOMENTUM_WINDOWS:
        df[f"ret_{w}d"] = close.pct_change(w).replace([np.inf, -np.inf], np.nan)

    # Volatility
    for w in VOLATILITY_WINDOWS:
        df[f"vol_{w}d"] = ret_1d.rolling(w).std()

    # MA deviation
    for w in MA_WINDOWS:
        ma = close.rolling(w).mean()
        df[f"ma{w}_dev"] = (close - ma) / (ma + 1e-9)

    # Volume
    vma5 = volume.rolling(5).mean()
    vma20 = volume.rolling(20).mean()
    df["volume_ratio_5"] = volume / vma5 - 1.0
    df["volume_ratio_20"] = volume / vma20 - 1.0
    if "turn" in df.columns:
        tma5 = df["turn"].rolling(5).mean()
        df["turn_change"] = df["turn"] / tma5 - 1.0

    # Amplitude
    df["amplitude"] = (high - low) / close

    # RSI
    delta = close.diff()
    avg_gain = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    avg_loss = (-delta).clip(lower=0).rolling(RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)

    # MACD
    ema_f = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_s = close.ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"] = ema_f - ema_s
    df["macd_signal"] = df["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Max drawdown (20d)
    rm = close.rolling(20).max()
    df["max_dd_20d"] = (close / rm - 1.0).rolling(20).min()

    # Price position
    h20, l20 = high.rolling(20).max(), low.rolling(20).min()
    df["price_position"] = (close - l20) / (h20 - l20 + 1e-9)

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_u = bb_mid + 2 * bb_std
    bb_l = bb_mid - 2 * bb_std
    df["bb_dev"] = (close - bb_l) / (bb_u - bb_l + 1e-9)
    df["bb_width"] = (bb_u - bb_l) / (bb_mid + 1e-9)

    # ATR
    tr = pd.concat([high - low, (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean() / (close + 1e-9)

    # KDJ
    h9, l9 = high.rolling(9).max(), low.rolling(9).min()
    rsv = (close - l9) / (h9 - l9 + 1e-9) * 100
    df["kdj_k"] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(alpha=1/3, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]

    # OBV
    obv = (volume * (delta / delta.abs()).fillna(0)).cumsum()
    obv_ma20 = obv.rolling(20).mean()
    df["obv_norm"] = (obv - obv_ma20) / (obv_ma20 + 1e-9)

    # Williams %R
    h14, l14 = high.rolling(14).max(), low.rolling(14).min()
    df["wr_14"] = -100 * (h14 - close) / (h14 - l14 + 1e-9)

    # Valuation
    for c in ["peTTM", "pbMRQ"]:
        if c in df.columns:
            df[c] = df[c].replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# Market-level features
# ---------------------------------------------------------------------------

def _add_market_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight market return + per-stock rolling beta."""
    panel["market_ret"] = panel.groupby("date")["pctChg"].transform("mean") / 100.0

    for sid in panel.index.get_level_values("stock_id").unique():
        sd = panel.xs(sid, level="stock_id")
        sr = sd["pctChg"] / 100.0
        mr = sd["market_ret"]
        cov = sr.rolling(BETA_WINDOW).cov(mr)
        var = mr.rolling(BETA_WINDOW).var()
        beta = cov / var.replace(0, np.nan)
        panel.loc[(slice(None), sid), "beta_60d"] = beta.values
    return panel


def _add_cross_sectional_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Rank percentiles across stocks on each date."""
    for src, dst in [("ret_5d", "ret_5d_rank"), ("ret_20d", "ret_20d_rank"),
                      ("vol_5d", "vol_5d_rank"), ("rsi_14", "rsi_rank")]:
        if src in panel.columns:
            panel[dst] = panel.groupby("date")[src].rank(pct=True)
    return panel


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def engineer_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    print("[features] Per-stock factors ...")
    panel = panel.groupby("stock_id", group_keys=False).apply(_compute_stock_factors)

    print("[features] Market features ...")
    panel = _add_market_features(panel)

    print("[features] Cross-sectional features ...")
    panel = _add_cross_sectional_features(panel)

    n_cols = len(panel.columns)
    print(f"[features] Done: {panel.shape[0]} rows, {n_cols} columns")
    return panel


# ---------------------------------------------------------------------------
# Window sampling (vectorized)
# ---------------------------------------------------------------------------

def _precompute_arrays(panel, stock_ids, feature_cols, dates):
    """Convert MultiIndex panel to 3D numpy arrays via vectorized indexing."""
    n_stocks, n_dates, nf = len(stock_ids), len(dates), len(feature_cols)
    sidx = {s: i for i, s in enumerate(stock_ids)}
    didx = {d: i for i, d in enumerate(dates)}

    df = panel.reset_index()[["date", "stock_id", "open"] + feature_cols]
    df["si"] = df["stock_id"].map(sidx).astype(np.int64)
    df["di"] = df["date"].map(didx).astype(np.int64)
    df = df.dropna(subset=["si", "di"])

    feats = np.zeros((n_stocks, n_dates, nf), dtype=np.float32)
    opens = np.zeros((n_stocks, n_dates), dtype=np.float32)
    valid = np.zeros((n_stocks, n_dates), dtype=bool)

    si, di = df["si"].values, df["di"].values
    feats[si, di] = df[feature_cols].values.astype(np.float32)
    opens[si, di] = df["open"].values.astype(np.float32)
    valid[si, di] = True

    np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    np.nan_to_num(opens, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return feats, opens, valid


def make_window_samples(
    panel: pd.DataFrame,
    stock_ids: list[str],
    lookback: int = LOOKBACK_DAYS,
    horizon: int = PREDICT_HORIZON,
    step: int = STEP_DAYS,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Vectorized sliding-window sample construction."""
    global _NORM_STATS

    dates = sorted(panel.index.get_level_values("date").unique())
    exclude = {"open", "high", "low", "close", "preclose", "volume", "amount",
               "adjustflag", "turn", "tradestatus", "pctChg", "peTTM", "pbMRQ",
               "market_ret", "stock_id", "index_name", "vol_ma5", "vol_ma20",
               "turn_ma5", "turn_ma20", "industry"}
    fcols = [c for c in panel.columns if c not in exclude]
    nf = len(fcols)
    n_dates = len(dates)

    feats_arr, open_arr, has_data = _precompute_arrays(panel, stock_ids, fcols, dates)

    X_list, y_list, m_list, d_list = [], [], [], []
    for idx in range(lookback + horizon, n_dates, step):
        hs = idx - horizon - lookback + 1
        he = idx - horizon + 1
        X = feats_arr[:, hs:he, :].copy()
        if X.shape[1] < lookback:
            X = np.pad(X, ((0, 0), (lookback - X.shape[1], 0), (0, 0)))

        o1 = open_arr[:, idx - horizon + 1]
        o5 = open_arr[:, idx]
        vo = o1 > 1e-8
        y = np.divide(o5 - o1, o1, out=np.zeros_like(o1), where=vo)

        h_ok = has_data[:, hs:he].sum(axis=1) >= lookback * 0.7
        t_ok = vo & (o5 > 1e-8)
        mask = (h_ok & t_ok).astype(np.float32)

        if mask.sum() < 5:
            continue
        X_list.append(X)
        y_list.append(y)
        m_list.append(mask)
        d_list.append(str(dates[idx].date()))

    Xa = np.array(X_list)
    ya = np.array(y_list)
    ma = np.array(m_list)

    if normalize:
        mu = Xa.mean(axis=(0, 1, 2), keepdims=True)
        sg = Xa.std(axis=(0, 1, 2), keepdims=True)
        sg = np.where(sg < 1e-8, 1.0, sg)
        Xa = (Xa - mu) / sg
        _NORM_STATS = (mu, sg)

    print(f"[features] {len(Xa)} samples, X={Xa.shape}, features={nf}")
    return Xa, ya, ma, d_list
