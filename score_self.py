"""Score prediction against test data."""
import pandas as pd
import sys

output_path = 'output/result.csv'
test_data_path = 'app/data/test.csv'


def is_valid_prediction(pred):
    id_col = 'stock_id' if 'stock_id' in pred.columns else '股票代码'
    weight_col = 'weight' if 'weight' in pred.columns else '权重'
    if len(pred) > 5:
        raise ValueError(f'最多5只股票，当前{len(pred)}只')
    ws = pred[weight_col].sum()
    if not (0 <= float(ws) <= 1.0):
        raise ValueError(f'权重和={ws:.4f}，需在[0,1]之间')


def calculate_return(group):
    start, end = group.iloc[0], group.iloc[-1]
    return (end['开盘'] - start['开盘']) / start['开盘']


def calculate_predict_weight_score(output_data, test_data):
    test_data = test_data[test_data['股票代码'].isin(output_data['股票代码'])]
    test_data = test_data.groupby('股票代码').tail(5)
    result = test_data.groupby('股票代码').apply(
        calculate_return, include_groups=False
    ).reset_index().rename(columns={0: '收益率'})
    result = result.merge(output_data, on='股票代码')
    return (result['收益率'] * result['权重']).sum()


try:
    test_data = pd.read_csv(test_data_path)
    raw = pd.read_csv(output_path)
    is_valid_prediction(raw)
except Exception as e:
    print(f"Error: {e}")
    pd.DataFrame({"Team Name": ["team_name"], "Final Score": [-999]}).to_csv("./temp/tmp.csv", index=False)
    sys.exit(0)

test_data = test_data[['股票代码', '日期', '开盘', '收盘']]
output_data = raw.rename(columns={'stock_id': '股票代码', 'weight': '权重'})
score = calculate_predict_weight_score(output_data, test_data)

pd.DataFrame({"Team Name": ["中州奶龙"], "Final Score": [score]}).to_csv("./temp/tmp.csv", index=False)
print(f"预测股票的加权收益率得分: {score}")
