import numpy as np
import pandas as pd

def load_data(path):
    df = pd.read_csv(path, dtype={"股票代码": str})
    df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
    df["日期"] = pd.to_datetime(df["日期"])
    numeric_cols = ["开盘","收盘","最高","最低","成交量","成交额","振幅","涨跌额","换手率","涨跌幅"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.sort_values(["股票代码","日期"]).reset_index(drop=True)
    return df

def add_label(df):
    df = df.copy()
    df = df.sort_values(["股票代码","日期"]).reset_index(drop=True)
    g = df.groupby("股票代码", group_keys=False)
    df["open_t1"] = g["开盘"].shift(-1)
    df["open_t5"] = g["开盘"].shift(-5)
    df["label"] = (df["open_t5"] - df["open_t1"]) / df["open_t1"]
    df.loc[df["open_t1"] <= 1e-8, "label"] = np.nan
    return df
