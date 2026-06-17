# 股票组合预测 — 快速开始

> 赛题：基于沪深 300 成分股历史数据，预测未来一周收益最高的 ≤5 只股票组合。
> 队伍：中州奶龙

## 环境

```bash
pip install tushare lightgbm catboost torch numpy pandas scipy scikit-learn hmmlearn pywt
```

Tushare Token 设为环境变量：
```bash
export TUSHARE_TOKEN="your_token"
```

## 运行

```bash
# 完整预测（训练 + 选股）
python controller.py

# 滚动验证
python controller.py --validate

# 对比所有模型
python controller.py --ensemble compare

# 启用 isotonic 校准
python controller.py --calibrate

# 保护机制消融实验
python controller.py --ablation

# NN 模型训练
python train.py --loss listnet --epochs 300 --batch-size 32 --augment --seed 791
```

## 输出

- `output/result.csv` — 最终选股结果
- `output/run_log.csv` — 历史运行记录
- `models/` — 模型文件

## 数据

首次运行自动从 Tushare 拉取 2021-2026 全量数据，缓存于 `data/raw/tushare/`。
后续运行直接从缓存读取。
