# 股票组合预测 — 项目手册

## 项目目标
基于沪深300成分股数据，预测未来一周（T+1~T+5）收益最高的 ≤5 只股票组合。

## 数据源
- **THU-BDC 官方**: `data/official/train.csv` + `test.csv`
- 时间: 2024-01-02 ~ 2026-05-30，300只股票，580交易日
- 所有价格为**后复权**
- baostock 备选更新

## 项目结构

### 核心模块
| 文件 | 功能 |
|------|------|
| `controller.py` | **主入口**: LGBM+NN融合 + 多管线选股 + 纳什多样化 |
| `train.py` | NN训练: TSCV/decay/augment/AMP |
| `model.py` | GRU + Pre-LN Transformer + ScoreHead (可选 Bahdanau Attention) |
| `features.py` | 特征工程: 57因子 |
| `data_loader.py` | 数据加载 |
| `predict.py` | 单模型预测 |
| `score_self.py` | 评分脚本 |
| `config.py` | 超参数 (含 USE_ATTENTION) |
| `run_log.csv` | 自动记录每次运行的参数和分数 |
| `window_test/backtest.py` | 跨窗口滚动回测 |
| `test_window.py` | 数据泄漏检测 |

### 模型文件
```
models/
├── portfolio_model_g791_fold[1~4].pt       51特征 v1
├── portfolio_model_seed42_fold[1~4].pt
├── portfolio_model_g787_fold[1~4].pt
├── portfolio_model_g791_v2_fold[1~4].pt    57特征 v2
├── portfolio_model_seed42_v2_fold[1~4].pt
├── portfolio_model_g787_v2_fold[1~4].pt
├── lgbm_ranker.txt                         LightGBM缓存
└── backup_*/                               备份(自动)
```

## 运行命令

### 推理（预测+选股）
```bash
# 默认: LGBM+NN + BDC权重 (最优)
python controller.py

# LightGBM单独
python controller.py --ensemble lgbm

# 对比所有策略
python controller.py --ensemble compare

# BDC 权重 (旧选股 + 预测比例加权 + 单票上限0.5)
python controller.py --weight bdc

# + 纳什均衡多样化 (门控后)
python controller.py --game 0.3

# + 频谱纳什 (FFT频谱相似度替代NN embedding)
python controller.py --game 0.3 --nash-mode spectral

# + 多日 Rolling 预测
python controller.py --multi-day 3

# BDC 纯管线 (替换旧选股逻辑)
python controller.py --bdc pure

# BDC 混合管线 (精度门控 + BDC排序 + BDC权重)
python controller.py --bdc hybrid
```

### 训练
```bash
# NN模型训练 (总时间 < 2小时)
python train.py --seed 791 --loss topk_listnet --decay 2.0 --tscv --batch-size 32
python train.py --seed 42  --loss topk_listnet --decay 2.0 --tscv --batch-size 32
python train.py --seed 787 --loss lambdarank    --tscv --batch-size 32

# 含 Bahdanau 注意力 (需先设置 USE_ATTENTION=True in config.py)
python train.py --seed 791 --loss topk_listnet --decay 2.0 --batch-size 32
```

### 回测与验证
```bash
# 14周滚动回测
python window_test/backtest.py --n-windows 14

# 数据泄漏检测
python test_window.py --n-windows 5 --epochs 15
```

## 命令行参数速查

| 参数 | 默认值 | 说明 |
|------|:------:|------|
| `--ensemble` | `lgbm-nn` | lgbm / lgbm-nn / gp / fold1 / compare |
| `--weight` | `bdc` | equal / softmax / inv_vol / bdc |
| `--game` | — | 纳什多样化 lambda (门控后触发) |
| `--nash-mode` | `embedding` | embedding / spectral |
| `--multi-day` | `1` | 1~5，多日 rolling 预测天数 |
| `--bdc` | — | pure / hybrid，切换 BDC 管线 |
| `--risk-penalty` | `0.30` | BDC管线风险惩罚权重 |
| `--topk` | 5 | 选股数量 |
| `--no-cache` | — | 强制重训 LGBM |
| `--output` | `output/result.csv` | 输出路径 |

## 管线架构

```
原始数据 → 特征工程 (57因子) → LGBM Ranker + NN Ensemble → 融合分数
                                                              ↓
候选池 (Top40) → 精度门控(ret_5d>0, dd>-8%) → 约10只动量候选股
                    └ (可选 --game) → 纳什均衡多样化 (embedding/spectral)
                    → 波动率过滤 → Top5
                    → 权重分配 (bdc/softmax/equal)
                    → output/result.csv
```

## 最终成绩

### 最新 Score (2026-06-01)

| 策略 | Score |
|------|:-----:|
| **LGBM+NN + BDC权重** | **0.148063** |
| LGBM+NN + softmax | 0.146988 |
| LightGBM Ranker | 0.129519 |
| NN GP v2 | 0.059295 |

### 14 周滚动回测 (2025-06 ~ 2026-05)

**按 Sharpe 排名（含纳什策略）：**

| 策略 | 权重 | Mean | Std | Sharpe | WinRate | 最差周 |
|------|:----:|:----:|:---:|:------:|:-------:|:------:|
| **LGBM_NN_BlendNash** | **softmax** | **0.0725** | **0.050** | **1.44** | **92.9%** | -0.052 |
| LGBM_NN_BlendNash | inv_vol | 0.0703 | 0.050 | 1.42 | 92.9% | -0.033 |
| LGBM_NN_BlendNash | equal | 0.0697 | 0.050 | 1.39 | 92.9% | -0.046 |
| **LGBM_NN_BlendNash** | **bdc** | **0.0664** | **0.048** | **1.39** | **92.9%** | -0.048 |
| LGBM_Nash | equal | 0.0807 | 0.064 | 1.27 | 92.9% | -0.046 |
| LGBM_NN_Blend (无纳什) | softmax | 0.0550 | 0.058 | 0.95 | 85.7% | -0.009 |
| LGBM_Only | softmax | 0.0718 | 0.106 | 0.68 | 92.9% | -0.005 |

### 冠军组合决定因素
| 对比 | Sharpe Delta | WinRate Delta | 结论 |
|:----|:-----------:|:-------------:|------|
| Blend+Nash vs Blend 无纳什 | +0.49 | +7.1% | **纳什价值巨大** |
| Blend+Nash vs LGBM Alone | +0.76 | 0% | **NN 平滑 Sharpe** |
| BDC vs softmax (Blend+Nash) | -0.05 | 0% | **BDC 约等于 softmax** |

### 最佳 / 最差窗口
| 策略 | 最佳 | Score | 最差 | Score |
|------|------|:-----:|------|:-----:|
| LGBM_NN_BlendNash | W3 2025-08-08 | **+0.153** | W4 2025-09-01 | -0.052 |
| LGBM_Only | W10 2026-01-23 | **+0.409** | W12 2026-03-18 | -0.023 |

## 改动记录 (2026-06-01)

| 改动 | 状态 | 效果 |
|------|:----:|:----:|
| BDC 权重 (pred+capped) | 已集成 | Sharpe 1.39, 与 softmax 相当 |
| 纳什移动到门控后 | 已集成 | 三档自适应阈值 |
| 频谱纳什多样化 | 已集成 | FFT 频谱相似度替代 NN embedding |
| 多日 Rolling 预测 | 已集成 | `--multi-day 2~5` |
| Bahdanau Attention | 代码已集成 | 需重训全量模型 |
| 日志记录 | 已集成 | 自动写入 `run_log.csv` |
| stock_id 格式 bug | 已修复 | `int()` -> zfilled string |
| 默认权重同步 | 已修复 | `softmax` -> `bdc` |
| **LGBM 日期错位 bug** | **已修复** | **多日 Rolling 中 LGBM 用了无 target 的未来日导致贡献为 0** |
| RankIC 滤波集成 | 已放弃 | 不互补反而弱 |
| 频谱特征加入模型 | 已放弃 | 57因子已饱和 |

## 关键发现

- **LGBM+NN+纳什 = 最优组合**: Sharpe 1.44, WinRate 92.9%
- **NN + LGBM 互补性极强**: LGBM Alone Sharpe=0.68 -> 加 NN 后 1.44（翻倍）
- **纳什均衡提升巨大**: 同一策略不加纳什 Sharpe=0.95 -> 加纳什后 1.44（+0.49）
- **BDC 权重约等于 softmax**: Sharpe 1.39 vs 1.44，差距<4%，bdc 在困难周更稳健
- **精度门控是核心**: ret_5d > 0 + dd > -8% 简单但有效
- **纳什在门控后更合理**: 池子 7~10 只时 lambda 自动稀释
- **频谱纳什给出不同 signal**: 与 embedding 纳什选股结果不一致
- **57 特征饱和**: 加频谱特征无增量
- **数据无泄漏**: 信号逐年增强
最后备份: /root/gupiao_final_20260601_162504
