# 股票投资组合预测 — LightGBM + NN + 纳什均衡

> 赛题：基于沪深 300 成分股历史数据，预测未来一周（T+1~T+5）收益最高的 ≤5 只股票组合。
> 队伍：中州奶龙 | **最终 14周 Sharpe 1.98, 100%胜率**

---

## 快速开始

```bash
# 🏆 极限分数模式 (0.143)
python controller.py --max-score --game 0.25 --weight softmax

# 🛡️ 稳健模式 (Sharpe 1.98, 100%胜率)
python controller.py --ensemble lgbm --game 0.25 --no-overbought --diverse-industry 2 --weight inv_vol

# ⚖️ 均衡模式 (Sharpe 1.52)
python controller.py --no-overbought --diverse-industry 2 --game 0.25 --weight inv_vol

# 📊 对比所有策略
python controller.py --ensemble compare
```

## 最终成绩

| 策略 | Mean | Sharpe | WinRate | 最差周 |
|:-----|:----:|:------:|:-------:|:------:|
| **Nash + inv_vol** 🏆 | **0.075** | **1.98** | **100%** | **+0.021** |
| Nash + softmax | **0.106** | 1.71 | 100% | +0.010 |
| BlendNash + inv_vol | 0.081 | 1.52 | 93% | -0.033 |
| BlendNash + softmax | 0.082 | 1.44 | 93% | -0.049 |

## 核心创新

| 技术 | 效果 |
|:-----|:------|
| **纳什均衡选股** | Sharpe +0.49 — embedding余弦相似度+博弈论 |
| **精度门控** | 动量>0 + 回撤>-8% — 最简但最有效的过滤器 |
| **73维特征** | 量价+技术+日历+截面+成分股调整+量价背离 |
| **LGBM+NN融合** | LGBM截面排序 + NN时序编码 = 互补 |
| **超买过滤** | RSI>75或10d>20%排除 — 防追高 |
| **行业分散** | 同行业最多N只 — 降集中度 |
| **市场分析** | 宏观周期+赛道拥挤+博弈升级 |

## 模型架构

```
输入 73维特征(300股×60天)
    │
    ├─ GRU时序编码 → Cross-Sectional Transformer ×2 → ScoreHead
    │   12个子模型(3种子×4折TSCV) GP加权融合
    │
    ├─ LightGBM Ranker (lambdarank, 500 trees)
    │
    ├─ 融合: 0.7×LGBM + 0.3×NN → 候选Top40
    │
    ├─ 精度门控 → 纳什均衡 → 波动率过滤 → Top-5
    │
    └─ 权重分配 (softmax/inv_vol/bdc/equal)
```

## 特征工程 (73维)

| 类别 | 数量 | 因子 |
|:-----|:----:|:------|
| 动量+波动率 | 7 | ret_5/10/20/60d, vol_5/10/20d |
| 均线偏离 | 4 | ma5/10/20/60_dev |
| 量价 | 5 | volume_ratio, turn_change, gap_ratio, amplitude |
| 技术指标 | 12 | RSI, MACD, KDJ, OBV, WR, BB, ATR |
| 风险走势 | 7 | max_dd, price_position, streak, corr |
| 截面排名 | 7 | ret/vol/rsi_rank, cs_rank_close/vol, excess_return |
| 行业 | 7 | ind_ret/alpha/vol, industry_rank_return |
| 指数 | 12 | SSE50/CSI500/ChiNext/SSE × 3窗口 |
| 市场 | 1 | beta_60d |
| 日历 | 9 | wday, month_sin/cos, month_end, CNY, **rebalance_soon** |
| 量价背离 | 2 | **divergence_bull/bear** |

## 详细文档

参见 [CLAUDE.md](CLAUDE.md) — 完整版项目手册（含回测明细、博弈论推导、风险分析）
