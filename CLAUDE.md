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

# + 自适应现金仓位 (信号弱时保留现金)
python controller.py --cash-buffer 0.3
```

### 训练
```bash
# NN模型训练 (总时间 < 2小时)
python train.py --seed 791 --loss topk_listnet --decay 2.0 --tscv --batch-size 32
python train.py --seed 42  --loss topk_listnet --decay 2.0 --tscv --batch-size 32
python train.py --seed 787 --loss lambdarank    --tscv --batch-size 32

# PCC Loss (直接优化RankIC, 从AAAI 2024 MASTER借鉴)
python train.py --seed 791 --loss pcc --tscv --batch-size 32

# 含 Bahdanau 注意力 (需先设置 USE_ATTENTION=True in config.py)
python train.py --seed 791 --loss topk_listnet --decay 2.0 --batch-size 32

# 含市场门控 (需先设置 USE_MARKET_GATE=True in config.py)
python train.py --seed 791 --loss pcc --tscv --batch-size 32
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
| `--loss` | `lambdarank` | 损失: lambdarank/pairwise/listnet/topk_listnet/**pcc** |
| `--cash-buffer` | — | 自适应现金仓位阈值 (参考Game-BDC2026 T7) |

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

### 最新 Score (2026-06-02)

| 策略 | Score |
|------|:-----:|
| **LGBM+NN + BDC权重 (70特征)** | **0.147637** |
| LGBM+NN + softmax (70特征) | 0.147637 |
| LightGBM Ranker (旧 57特征) | 0.129519 |

### 14 周滚动回测 (2025-06 ~ 2026-05) — 70特征模型

**按 Sharpe 排名（含纳什策略）：**

| 策略 | 权重 | Mean | Std | Sharpe | WinRate |
|------|:----:|:----:|:---:|:------:|:-------:|
| **LGBM_NN_BlendNash** | **softmax** | **0.0933** | **0.060** | **1.56** | **85.7%** |
| LGBM_NN_BlendNash | equal | 0.0931 | 0.062 | 1.50 | 92.9% |
| LGBM_NN_BlendNash | bdc | 0.0888 | 0.061 | 1.45 | 92.9% |
| LGBM_NN_BlendNash | inv_vol | 0.0968 | 0.067 | 1.44 | 100% |
| LGBM_Nash | bdc | 0.0816 | 0.056 | 1.46 | 100% |
| LGBM_Nash | equal | 0.0794 | 0.055 | 1.44 | 92.9% |
| **LGBM_Nash** | **softmax** | **0.1010** | **0.073** | **1.37** | **100%** |
| LGBM_NN_Blend (无纳什) | softmax | 0.0530 | 0.057 | 0.93 | 85.7% |
| LGBM_Only | softmax | 0.0745 | 0.106 | 0.70 | 78.6% |

### 新旧对比 (BlendNash softmax)

| 版本 | Mean | Std | Sharpe | WinRate | 最差周 | 最佳周 |
|:----|:---:|:---:|:-----:|:-------:|:------:|:------:|
| **70特征 (新)** | **0.0933** | **0.060** | **1.56** | 85.7% | **-0.0012** | +0.203 |
| 57特征 (旧) | 0.0725 | 0.050 | 1.44 | 92.9% | -0.052 | +0.153 |

**70特征提升：均值 +29%，Sharpe +0.12，最差周从 -5.2% 缩至 -0.12%**

### 冠军组合决定因素
| 对比 | Sharpe Delta | WinRate Delta | 结论 |
|:----|:-----------:|:-------------:|------|
| Blend+Nash vs Blend 无纳什 | +0.63 | 0% | **纳什价值持续巨大** |
| Blend+Nash vs LGBM Alone | +0.86 | +7.1% | **NN 平滑效果更明显** |
| 70特征 vs 57特征 (Blend+Nash) | +0.12 | -7.2% | **均值大幅提升，方差略增** |

### 最佳 / 最差窗口
| 策略 | 最佳 | Score | 最差 | Score |
|------|------|:-----:|------|:-----:|
| LGBM_NN_BlendNash | W3 2025-08-08 | **+0.203** | W4 2025-09-01 | **-0.001** |
| LGBM_Nash | W9 2025-12-30 | **+0.284** | W10 2026-01-23 | +0.006 |

## 改动记录 (2026-06-02)

| 改动 | 状态 | 效果 |
|------|:----:|:----:|
| BDC 权重 (pred+capped) | 已集成 | Sharpe 1.39, 与 softmax 相当 |
| 纳什移动到门控后 | 已集成 | 三档自适应阈值 |
| 频谱纳什多样化 | 已集成 | FFT 频谱相似度替代 NN embedding |
| 多日 Rolling 预测 | 已集成 | `--multi-day 2~5` |
| Bahdanau Attention | 代码已集成 | 需重训全量模型 |
| 日志记录 | 已集成 | 自动写入 `output/run_log.csv` |
| stock_id 格式 bug | 已修复 | `int()` -> zfilled string |
| 默认权重同步 | 已修复 | `softmax` -> `bdc` |
| **LGBM 日期错位 bug** | **已修复** | **多日 Rolling 中 LGBM 用了无 target 的未来日导致贡献为 0** |
| RankIC 滤波集成 | 已放弃 | 不互补反而弱 |
| 频谱特征加入模型 | 已放弃 | 57因子已饱和 |
| **PCC Loss** | **已集成** | **Pearson相关系数损失，直接优化RankIC** |
| **市场门控 (Market Gate)** | **已集成** | **从embedding均值推导市场状态调制个股** |
| **自适应现金仓位** | **已集成** | **`--cash-buffer` 信号弱时保留现金** |
| **日历特征(9维)** | **已集成** | **wday/month_sin/cos/月末/春节** |
| **截面特征(4维)** | **已集成** | **excess_return/cs_rank/industry_rank** |
| **零均值居中** | **已集成** | **`--center` 选股前消除偏差** |
| **DoubleEnsemble LGBM** | **已集成** | **`--double-ensemble` 6模型特征子采样+误差加权** |
| **特征升级 57→70** | **已重训** | **BlendNash Sharpe 1.44→1.56, Mean +29%** |
| **load_predict n_features** | **已修复** | **从checkpoint config读取特征数，兼容新旧模型** |

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
- **PCC Loss > 排序损失**: PCC直接优化RankIC，与LambdaRank互补
- **现金仓位合规**: 权重和[0,1.0]允许留现金，参考Game-BDC2026 T7
- **市场门控无需额外数据**: 从个股embedding均值推导市场状态
- **70特征显著优于57特征**: BlendNash Mean +29%, 最差周 -5.2%→-0.12%
- **日历+截面特征贡献大**: 9维日历+4维截面，A股日历效应明显
- **LGBM Nash 100%胜率**: 14周全部正收益，Softmax Sharpe 1.37
最后备份: /root/gupiao_final_20260601_221132
