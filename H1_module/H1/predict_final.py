"""
predict_final.py — B1-P15-Top3 final submission.
Reads data/train.csv, outputs output/result.csv.
V8 Top15 candidate pool -> LogisticRegression p_positive -> Top3 [0.4,0.3,0.3]
"""
import os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from H1_dataset import load_data, add_label
from H1_features import add_v8_score, build_gate_features, get_gate_feature_names, cross_sectional_rank

TRAIN_PATH = os.path.join(os.path.dirname(BASE), 'data', 'train.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(BASE), 'output')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'result.csv')
POOL = 15; TOP_N = 3; WEIGHTS = [0.4, 0.3, 0.3]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load data
    df = load_data(TRAIN_PATH)
    df = add_label(df)
    df = add_v8_score(df)

    # 2. Gate features
    gf = build_gate_features(df)
    fn = get_gate_feature_names()
    X_all = cross_sectional_rank(gf, fn)

    # 3. Train gate on labeled rows
    train_mask = df["label"].notna().values
    Xt = X_all.loc[train_mask, fn].values.astype(np.float64)
    yt = (df.loc[train_mask, "label"].values > 0).astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(Xt, yt)

    # 4. Predict at latest date
    latest_date = df["日期"].max()
    pred_mask = (df["日期"] == latest_date).values
    today = df.loc[pred_mask, ["股票代码", "v8_score"]].copy()
    today["p_positive"] = model.predict_proba(
        X_all.loc[pred_mask, fn].values.astype(np.float64))[:, 1]

    # 5. V8 Top15 pool -> p_positive rerank -> Top3
    pool = today.nlargest(POOL, "v8_score")
    final = pool.nlargest(TOP_N, "p_positive")
    stocks = final["股票代码"].tolist()

    # 6. Output
    # 6. Monitoring (informational only, does not affect stock selection)
    today_data = df.loc[pred_mask]
    ret_disp_5d = today_data['涨跌幅'].astype(float).std()
    mkt_vol_5d = today_data['涨跌幅'].astype(float).rolling(5, min_periods=1).std().mean()
    mkt_up_ratio_5d = (today_data['涨跌幅'].astype(float) > 0).mean()
    v8_avg = pool['v8_score'].mean()
    b1_avg_pp = final['p_positive'].mean()

    print(f"Predict date: {latest_date.date()}")
    print(f"Top3: {stocks}")
    print(f"Monitoring: ret_disp_5d={ret_disp_5d:.4f} mkt_vol_5d={mkt_vol_5d:.4f} "
          f"mkt_up_ratio_5d={mkt_up_ratio_5d:.3f} v8_pool_avg={v8_avg:.4f} b1_avg_pp={b1_avg_pp:.3f}")

    # 7. Output
    result = pd.DataFrame({"stock_id": stocks, "weight": WEIGHTS})
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
