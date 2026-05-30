import numpy as np
import pandas as pd

def add_v8_score(df):
    df = df.copy()
    df = df.sort_values(["股票代码","日期"]).reset_index(drop=True)
    g = df.groupby("股票代码")
    day_ret = df["涨跌幅"].astype(float)
    turnover = df["换手率"].astype(float)
    vol_5d = (g["涨跌幅"].rolling(5,min_periods=1).std()
              .reset_index(level=0,drop=True).astype(float))
    ret_20d = (g["涨跌幅"].rolling(20,min_periods=1).mean()
               .reset_index(level=0,drop=True).astype(float))
    df["v8_score"] = day_ret * turnover / (1.0 + vol_5d.abs() + ret_20d.abs())
    df["v8_score"] = df["v8_score"].fillna(0).replace([np.inf,-np.inf],0)
    return df

def build_gate_features(df):
    df = df.copy()
    df = df.sort_values(["股票代码","日期"]).reset_index(drop=True)
    g = df.groupby("股票代码")
    close = df["收盘"].astype(float)
    volume = df["成交量"].astype(float)
    turnover = df["换手率"].astype(float)
    day_ret = df["涨跌幅"].astype(float)
    amplitude = df["振幅"].astype(float)
    high = df["最高"].astype(float)
    low = df["最低"].astype(float)
    feats = pd.DataFrame(index=df.index)
    feats["股票代码"] = df["股票代码"].values
    feats["日期"] = df["日期"].values
    for w in [1,3,5,10,20]:
        feats[f"ret_{w}d"] = (g["涨跌幅"].rolling(w,min_periods=1).mean()
                               .reset_index(level=0,drop=True).astype(float))
    for w in [5,10,20]:
        feats[f"vol_{w}d"] = (g["涨跌幅"].rolling(w,min_periods=2).std()
                               .reset_index(level=0,drop=True).astype(float))
    for w in [5,20]:
        feats[f"volume_chg_{w}d"] = g["成交量"].transform(lambda x: x.astype(float).pct_change(w))
        feats[f"turnover_chg_{w}d"] = g["换手率"].transform(lambda x: x.astype(float).pct_change(w))
    for w in [5,10,20,60]:
        ma = (g["收盘"].rolling(w,min_periods=1).mean()
              .reset_index(level=0,drop=True).astype(float))
        feats[f"close_vs_ma{w}"] = (close - ma) / (ma + 1e-12)
    for w in [5,20]:
        hmax = (g["最高"].rolling(w,min_periods=1).max()
                .reset_index(level=0,drop=True).astype(float))
        lmin = (g["最低"].rolling(w,min_periods=1).min()
                .reset_index(level=0,drop=True).astype(float))
        feats[f"hl_spread_{w}d"] = (hmax - lmin) / (close + 1e-12)
    peak = (g["收盘"].rolling(20,min_periods=1).max()
            .reset_index(level=0,drop=True).astype(float))
    feats["max_dd_20d"] = (close - peak) / (peak + 1e-12)
    feats["day_return"] = day_ret
    feats["day_turnover"] = turnover
    feats["day_volume"] = volume
    feats["day_amplitude"] = amplitude
    dg = feats.groupby("日期")
    feats["cs_rank_return"] = dg["day_return"].rank(pct=True)
    feats["cs_rank_turnover"] = dg["day_turnover"].rank(pct=True)
    feats["cs_rank_ret5d"] = dg["ret_5d"].rank(pct=True)
    feats["cs_excess_return"] = feats["day_return"] - dg["day_return"].transform("mean")
    feats["cs_excess_ret5d"] = feats["ret_5d"] - dg["ret_5d"].transform("mean")
    feats["mkt_return"] = dg["day_return"].transform("mean")
    feats["mkt_vol"] = dg["day_return"].transform("std")
    feats["mkt_up_ratio"] = dg["day_return"].transform(lambda x: (x>0).mean())
    return feats.replace([np.inf,-np.inf],np.nan).fillna(0)

def get_gate_feature_names():
    cols = []
    for w in [1,3,5,10,20]: cols.append(f"ret_{w}d")
    for w in [5,10,20]: cols.append(f"vol_{w}d")
    for w in [5,20]: cols.append(f"volume_chg_{w}d"); cols.append(f"turnover_chg_{w}d")
    for w in [5,10,20,60]: cols.append(f"close_vs_ma{w}")
    for w in [5,20]: cols.append(f"hl_spread_{w}d")
    cols.append("max_dd_20d")
    cols += ["day_return","day_turnover","day_volume","day_amplitude",
             "cs_rank_return","cs_rank_turnover","cs_rank_ret5d",
             "cs_excess_return","cs_excess_ret5d",
             "mkt_return","mkt_vol","mkt_up_ratio"]
    return cols

def cross_sectional_rank(feats, feature_cols):
    feats = feats.copy()
    dates = feats["日期"].values
    out = pd.DataFrame(index=feats.index)
    for col in feature_cols:
        if col not in feats.columns:
            out[col] = 0.5
            continue
        out[col] = feats.groupby(dates)[col].rank(pct=True)
    return out.fillna(0.5)
