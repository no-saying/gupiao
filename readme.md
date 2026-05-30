# 股票投资组合预测 — 基于注意力机制的排序学习

> 赛题：基于沪深 300 成分股的历史数据，预测未来一周（T+1~T+5）收益最高的 ≤5 只股票组合。

## 目录结构

```
├── controller.py        ← 主控制器：集成多模型 + 动态权重优化
├── train.py              ← 模型训练
│
├── model.py              ← 模型架构（GRU + Cross-Sectional Transformer）
├── features.py           ← 特征工程（52个因子）
├── data_loader.py        ← 数据加载（baostock）
├── config.py             ← 全局配置（超参数、事件过滤）
│
├── predict.py            ← 单模型预测
├── score_self.py         ← 评分脚本（赛题官方评估）
│
├── models/               ← 模型权重
│   ├── portfolio_model_seed789.pt   ← 最优单模型 (Score 0.0846)
│   ├── portfolio_model_seed456.pt   ← 次优单模型
│   └── ...                           ← 更多 seed 模型
│
├── H1_module/            ← H1 独立模型
├── baseline/             ← Baseline 原始项目
│
├── data/
│   ├── test.csv          ← 测试数据
│   ├── raw/              ← baostock 缓存
│   └── processed/        ← 预处理缓存
│
├── output/               ← 输出目录
├── readme.md             ← 本文件
└── TUTORIAL.md           ← 原始教程
```

## 快速开始

### 训练
```bash
# 最优配置（ListNet + 百分位排名标签）
python train.py --loss listnet --rank-labels

# 基础配置（LambdaRank，原始 baseline 风格）
python train.py --loss lambdarank

# 更多选项
python train.py --loss listnet --rank-labels --epochs 200 --lr 5e-5
```

### 预测（单模型）
```bash
python predict.py
# 输出: result.csv
```

### 集成预测（多模型 + 权重优化）—— 推荐
```bash
# 最优组合（seed789 + seed456，Sharpe 加权）
python controller.py --models seed789,seed456 --weight sharpe

# 对比所有权重方法
python controller.py --models seed789,seed456 --compare

# 单模型 + 权重优化
python controller.py --models seed789 --weight sharpe

# 自定义模型组合
python controller.py --models seed789,seed456,seed999,gaussian789 --weight adaptive
```

### 评分
```bash
# 对有 label 的历史数据评分
cp result.csv output/result.csv && python score_self.py
```

## 实验结果（2026-05-30 更新）

### 测试集表现（84 个交易周，2026-01-20 ~ 2026-05-29）

| 模型 | Sharpe | 累计收益 | 胜率 | 周均收益 |
|------|--------|---------|------|---------|
| **Attn-g791** | **+0.3063** | **+185.36%** | 60.7% | +1.35% |
| Attn-g789 | +0.3126 | +167.75% | 61.9% | +1.26% |
| Attn-default | +0.3347 | +151.80% | 61.9% | +1.16% |
| Attn-seed456 | +0.3196 | +128.98% | 53.6% | +1.04% |
| Attn-seed999 | +0.2137 | +46.58% | 53.6% | +0.48% |
| Attn-g787 | +0.1971 | +79.29% | 63.1% | +0.77% |
| Attn-seed42 | +0.1233 | +25.51% | 51.2% | +0.30% |
| Attn-seed123 | -0.0585 | -15.25% | 45.2% | -0.16% |
| | | | | |
| **Baseline 对比** | | | | |
| Momentum (20d) | +0.0534 | +12.54% | 53.6% | +0.26% |
| Market Avg (等权) | -0.0430 | -5.62% | 54.8% | -0.06% |
| Random (5只) | -0.0951 | -19.95% | 42.9% | -0.23% |

### 关键发现

1. **注意力模型大幅跑赢所有基线**：最佳模型 Sharpe 0.31 vs 动量 0.05 vs 等权 -0.04
2. **胜率 > 60%**：top 模型在 84 周中超过 60% 的周取得正收益
3. **种子敏感性明显**：seed789 (Sharpe 0.33) vs seed123 (-0.06)，差距大，建议多 seed 集成
4. **g791 累计收益最高 (+185%)**：虽然 Sharpe 略低，但抓住了大行情，弹性最好

## 模型架构

```
输入 X: (Batch, N_stocks=300, T=60, F=52)
  │
  ├─ TimeEncoder (双向 GRU, 2层, d_model=128)
  │   每只股票独立编码 → (Batch, N, d_model)
  │
  ├─ CrossSectional Transformer (2层, 8头注意力)
  │   股票间自注意力 → 自动学习关联结构
  │
  └─ ScoreHead (MLP) → (Batch, N) 预测分数
```

## 特征工程（52个因子）

| 类别 | 因子 | 数量 |
|------|------|------|
| 动量 | ret_5d/10d/20d/60d | 4 |
| 波动率 | vol_5d/10d/20d | 3 |
| 均线偏离 | ma5/10/20/60_dev | 4 |
| 量价 | volume_ratio, turn_change, amplitude | 4 |
| 技术指标 | RSI, MACD, KDJ, OBV, Williams %R, Bollinger Bands, ATR | 12 |
| 风险 | max_dd_20d, price_position | 2 |
| 截面排名 | ret/vol/rsi 百分位排名 | 4 |
| 行业特征 | 行业平均收益/波动, 个股Alpha | 6 |
| 指数特征 | SSE50/CSI500/ChiNext/SSE × 3窗口 | 12 |
| 市场 | beta_60d | 1 |

## 损失函数

| 损失 | 效果 | 说明 |
|------|------|------|
| **ListNet** | ★★★ | 直接优化排序分布，最适合 top-5 选股 |
| Pairwise Hinge | ★★☆ | 基础 pairwise，稳定但信号弱 |
| LambdaRank | ★☆☆ | NDCG 加权，但梯度在小数据上消失 |
| Top-K ListNet | ★★☆ | 只关注 top-K 排名 |

## 事件过滤（10个重大市场事件）

训练样本的目标收益窗口若与以下事件重叠则剔除，避免模型学到异常模式：

- 2022: 俄乌冲突(2月)、上海封城(3-4月)、港股暴跌(10月)、放开炒作(11月)
- 2023: 印花税减半行情(8月)
- 2024: 雪球敲入(1-2月)、924政策转向(9月)
- 2025: 关税冲击(4月)

## 性能

- 数据构建: **4秒**（向量化加速，原始 187秒）
- 训练样本: **833个**（STEP_DAYS=1，原始 159个）
- GPU: 20GB 显存（batch_size=16）
- 模型参数: ~808K

## 数据来源

[baostock](http://baostock.com) — 免费 A 股数据源，提供日线行情、指数、行业分类等数据。
