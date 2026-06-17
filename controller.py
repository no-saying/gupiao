"""
CSI 300 Stock Prediction — LGBM + Tushare

Usage:
  python controller.py                    # Train + predict + score
  python controller.py --validate         # 12-week walk-forward
  python controller.py --no-cache         # Force retrain
  python controller.py --topk 3           # Select top 3
"""

import argparse, warnings, sys, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from config import MODEL_DIR, SUBMISSION_PATH, OUTPUT_DIR
from core import (build_lgbm_data,
                  engineer_features,
                  add_alpha158_features,
                  train_lgbm, predict_lgbm, compute_rankic,
                  select_top_stocks, compute_weights,
                  walk_forward_validate)

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser(description="LGBM + Tushare 股票预测")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output", type=str, default=str(SUBMISSION_PATH))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-weeks", type=int, default=12)
    args = parser.parse_args()

    # 1. 数据
    from tushare_loader import fetch_csi300_stocks, build_tushare_only_panel

    stock_ids = fetch_csi300_stocks()
    panel = build_tushare_only_panel(stock_ids)

    # 2. 特征
    panel = engineer_features(panel, stock_ids)
    panel = add_alpha158_features(panel)

    # 3. LGBM 数据
    lgbm_df, feat_cols = build_lgbm_data(panel)

    # 4. 验证（可选）
    if args.validate:
        walk_forward_validate(lgbm_df, feat_cols, panel, args.validate_weeks)
        return

    # 5. 训练
    model_path = MODEL_DIR / "lgbm_model.txt"
    if args.no_cache and model_path.exists():
        model_path.unlink()
    model, fm, fs, rankic = train_lgbm(lgbm_df, feat_cols, model_path)

    # 6. 预测最新日
    latest_date = sorted(lgbm_df['date'].unique())[-1]
    pred_day = predict_lgbm(model, fm, fs, lgbm_df, feat_cols, latest_date)
    scores = pred_day.set_index('stock_id')['lgb_score'].to_dict()

    # 7. 选股 + 权重
    sel_ids, sel_sc = select_top_stocks(scores, panel, top_k=args.topk)
    weights = compute_weights(sel_ids, sel_sc, method="equal")

    # 8. 输出
    df_out = pd.DataFrame({"stock_id": [str(s).zfill(6) for s in sel_ids],
                           "weight": weights})
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(args.output, index=False)

    print(f"\n  Selected {len(sel_ids)} stocks:")
    for sid, w in zip(sel_ids, weights):
        print(f"    {sid}: w={w:.3f}")

    # 9. 自评
    r = subprocess.run(["python", "score_self.py"], capture_output=True, text=True)
    for line in r.stdout.strip().split("\n"):
        print(f"  {line}")


if __name__ == "__main__":
    main()
