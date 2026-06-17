"""
本脚本用于将预测出的五支股票与实际的股票数据进行对比，计算加权收益，形成最终得分。
"""
import pandas as pd
import sys
from pathlib import Path
from config import SUBMISSION_PATH, OUTPUT_DIR
output_path = str(SUBMISSION_PATH)


def write_failed_score() -> None:
    result = pd.DataFrame(
        {
            "Team Name": ["中州奶龙"],
            "Final Score": [-999],
        }
    )
    result.to_csv(str(OUTPUT_DIR / "tmp_score.csv"), index=False)


def is_valid_prediction(prediction_data):
    """验证输出是否合法：最多5只，权重和0~1."""
    id_col = 'stock_id' if 'stock_id' in prediction_data.columns else '股票代码' if '股票代码' in prediction_data.columns else None
    weight_col = 'weight' if 'weight' in prediction_data.columns else '权重' if '权重' in prediction_data.columns else None
    if id_col is None or weight_col is None:
        raise ValueError('预测结果缺少必要字段，必须包含 stock_id/股票代码 和 weight/权重。')
    if len(prediction_data) > 5:
        raise ValueError('预测结果不合法：最多只能包含五支股票。')
    weight_sum = prediction_data[weight_col].sum()
    if not (0 <= float(weight_sum) <= 1.0):
        raise ValueError(f"预测结果不合法：权重之和必须为0到1之间. 当前权重之和为 {weight_sum}.")


def calculate_return(group):
    start = group.iloc[0]
    end = group.iloc[-1]
    return (end['open'] - start['open']) / start['open']


def calculate_predict_weight_score(output_data, test_data):
    """计算加权收益率。"""
    # 统一股票代码格式
    output_data['股票代码'] = output_data['股票代码'].astype(str).str.zfill(6)
    test_data = test_data[test_data['股票代码'].isin(output_data['股票代码'])]
    test_data = test_data.groupby('股票代码').tail(5)
    result = test_data.groupby('股票代码').apply(calculate_return).reset_index().rename(columns={0: '收益率'})
    result = result.merge(output_data, on='股票代码')
    final_score = (result['收益率'] * result['权重']).sum()
    return final_score


if __name__ == "__main__":
    # 使用扩展 Panel 的最后5个交易日作为测试数据
    try:
        from tushare_loader import fetch_csi300_stocks, build_tushare_only_panel
        stock_ids = fetch_csi300_stocks()
        panel = build_tushare_only_panel(stock_ids)
        df = panel.reset_index()[['date', 'stock_id', 'open']]
        # 取最后5个交易日
        last_dates = sorted(df['date'].unique())[-5:]
        test_data = df[df['date'].isin(last_dates)].copy()
        test_data = test_data.rename(columns={'stock_id': '股票代码'})
        test_data['股票代码'] = test_data['股票代码'].astype(str).str.zfill(6)
        test_data = test_data.sort_values(['股票代码', 'date'])
        print(f"[score] Using panel data: {last_dates[0].date()} ~ {last_dates[-1].date()} for scoring")

        raw_output_data = pd.read_csv(output_path)
        is_valid_prediction(raw_output_data)
    except Exception as e:
        print(f"Error reading test data or validating prediction: {e}")
        write_failed_score()
        sys.exit(0)

    # 读取输出数据
    output_data = raw_output_data.rename(columns={'stock_id': '股票代码', 'weight': '权重'})

    required_columns = {'股票代码', '权重'}
    if not required_columns.issubset(output_data.columns):
        print('错误：输出结果缺少股票代码或权重字段。')
        write_failed_score()
        sys.exit(0)

    # 计算加权收益率
    predict_weight_score = calculate_predict_weight_score(output_data, test_data)


    # 保存结果到 CSV 文件
    result = pd.DataFrame(
        {
            "Team Name": ["中州奶龙"],
            "Final Score": [predict_weight_score],
        }
    )
    result.to_csv(str(OUTPUT_DIR / "tmp_score.csv"), index=False)
    print(f"预测股票的加权收益率得分: {predict_weight_score}")
