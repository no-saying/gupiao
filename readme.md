# 股票投资组合预测 — LightGBM + NN 混合集成

> 赛题：基于沪深 300 成分股的历史数据，预测未来一周（T+1~T+5）收益最高的 ≤5 只股票组合。

## 最终 Score: 0.149 (LGBM+NN+门控+softmax)

---

## 快速开始

```bash
# 预测（默认: LGBM+NN+门控）
python controller.py --weight softmax

# LightGBM 单独（长期最稳）
python controller.py --ensemble lgbm --weight softmax

# 对比所有策略
python controller.py --ensemble compare
```

## 提交文件

参见 `output/README.md`

| 文件 | 策略 | 本周 Score | 长期 Sharpe |
|------|------|:----------:|:-----------:|
| `result_lgbm_nn.csv` | LGBM+NN+门控 | **0.149** | — |
| `result_lgbm.csv` | LightGBM 单独 | 0.130 | **1.19** |
| `result_nash.csv` | +纳什均衡 | 0.149 | — |
| `result_gp.csv` | NN GP v2 | 0.059 | 0.31 |
| `result_fold1.csv` | old fold1 | 0.127 | 0.06 |
| `result_3model.csv` | old 3model | 0.107 | 0.12 |

## 模型架构

```
特征 Panel (date, stock, 57 features)
    │
    ├─ LightGBM Ranker ─── 截面排序 (lambdarank)
    │    500 trees, num_leaves=63, 训练10秒
    │
    ├─ NN GP Ensemble ──── 时序编码 (GRU+Transformer)
    │    12个子模型 (3种子×4折) GP优化加权
    │
    ├─ 精度门控 ── 动量>0 + 回撤<8% + 波动率过滤
    │
    └─ softmax加权 ── 高分多配, 低分少配
```

## 目录结构

```
├── controller.py       ← 主入口
├── train.py            ← NN 训练
├── model.py            ← GRU + Transformer
├── features.py         ← 57因子
├── data_loader.py      ← 数据加载
├── score_self.py       ← 评分
├── config.py           ← 超参数
│
├── output/README.md    ← 提交文件说明
├── HANDOVER.md         ← 详细文档
├── CLAUDE.md           ← 项目手册
│
├── models/             ← 模型权重 (27个模型)
│   ├── *_v2_*.pt       ← 57特征 v2 模型
│   ├── *.pt            ← 51特征 旧模型
│   └── lgbm_ranker.txt ← LightGBM 缓存
│
└── data/               ← 数据缓存
```

## 特征工程（57个因子）

| 类别 | 因子 | 数量 |
|------|------|:----:|
| 动量 | ret_5d/10d/20d/60d | 4 |
| 波动率 | vol_5d/10d/20d | 3 |
| 均线偏离 | ma5/10/20/60_dev | 4 |
| 量价 | volume_ratio, turn_change, gap_ratio | 5 |
| 技术指标 | RSI, MACD, KDJ, OBV, Williams %R, BB, ATR | 12 |
| 风险 | max_dd_20d, price_position, close_pos, intraday_range | 4 |
| 走势 | streak_up/down, vol_ret_corr_10d | 3 |
| 截面排名 | ret/vol/rsi 百分位排名 | 4 |
| 行业 | 行业平均收益/波动, Alpha, 拥挤度 | 6 |
| 指数 | SSE50/CSI500/ChiNext/SSE × 3窗口 | 12 |
| 市场 | beta_60d | 1 |

## 运行命令

```bash
# 预测
python controller.py --weight softmax

# 单模型预测
python predict.py --model models/portfolio_model_g791.pt

# NN 训练 (约1小时)
python train.py --seed 791 --loss topk_listnet --decay 2.0 --tscv --batch-size 32

# LightGBM (10秒, 自动缓存)
python controller.py --no-cache

# 评分
python score_self.py
```

## 关键训练参数
- `D_MODEL=128`, `BATCH_SIZE=32`（57特征）
- `N_EPOCHS=300`, `EARLY_STOP_PATIENCE=50`
- `LR=1e-4`, OneCycleLR + FP16 AMP
- LightGBM: 500 trees, num_leaves=63, lambdarank
