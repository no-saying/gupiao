# 股票组合预测 — 完整项目手册

> 赛题：基于沪深 300 成分股历史数据，预测未来一周（T+1~T+5）收益最高的 ≤5 只股票组合。
> 队伍：中州奶龙

---

## 一、项目架构总览

```
原始 OHLCV 数据
     │
     ▼
特征工程 (73维) ───────────────────────────────┐
     │                                          │
     ▼                                          ▼
LightGBM Ranker (截面排序) ──┐              NN Ensemble (12个子模型)
     │                       │                  │
     ▼                       ▼                  ▼
lgb_score (百分位)     blend = 0.7*lgb + 0.3*nn
     │                       │
     ▼                       ▼
候选池 Top40 ←── 分数融合 (lgbm-nn / lgbm / gp)
     │
     ▼
精度门控 (动量>0, 回撤>-8%)
     │
     ├── (可选) 纳什均衡多样化 ──→ embedding余弦相似度 + 博弈论
     │
     ├── (可选) 超买过滤 ──→ RSI>75或10d>20%排除
     │
     ├── (可选) 行业分散 ──→ 同行业最多N只
     │
     ▼
波动率过滤
     │
     ▼
Top-5 选股 → 权重分配 (softmax/bdc/inv_vol/equal)
     │
     ├── (可选) 零均值居中
     ├── (可选) 现金仓位
     │
     ▼
output/result.csv → score_self.py → Final Score
```

---

## 二、模型架构

### 2.1 NN 模型 (model.py)

```
输入 (B, N=300, T=60, F=73)
  │
  ├─ TimeEncoder (共享 BiGRU, 2层, d_model=128)
  │   每只股票独立编码 → (B*300, 128)
  │   (可选：Bahdanau Attention 时序池化)
  │
  ├─ MarketGate (可选) → 从embedding均值推导市场状态 → sigmoid门控
  │
  ├─ CrossSectionalTransformer ×2 (Pre-LN, 8 heads, d_ff=256)
  │   股票间自注意力 → (B, 300, 128)
  │
  └─ ScoreHead: Linear(128→64) → GELU → Dropout → Linear(64→1)
       输出 (B, 300) 每只股票预测分数
```

**训练配置：**
- 3 个随机种子 × 4 折 TSCV = **12 个子模型**
- 种子 791/42: `topk_listnet` loss, 时间衰减 decay=2.0
- 种子 787: `lambdarank` loss
- OneCycleLR, FP16 AMP, 梯度裁剪 max_norm=1.0
- Early stopping patience=50, 最大 300 epochs
- 总训练时间: ~2 小时

### 2.2 LightGBM Ranker

- `lambdarank` objective, 500 trees, num_leaves=63
- 特征与 NN 共享同一套 73 维 panel
- 缓存至 `models/lgbm_ranker.txt`（`--no-cache` 强制重训）
- 训练数据: 最近 576 个交易日，label = 未来 5 日收益十分位

### 2.3 融合策略

```python
# 默认: 0.7 * LGBM + 0.3 * NN
blend = 0.7 * norm(lgbm_score) + 0.3 * norm(nn_score)
```

两种独立 scoring 方法互补：LGBM 擅长捕捉截面排序，NN 擅长时序模式。

---

## 三、特征工程 (73维)

### 3.1 特征演化

| 版本 | 维度 | 新增 | 效果 |
|:----|:----:|:-----|:----|
| v0 (初始) | 57 | 基础量价+技术指标(RSI/MACD/KDJ/BB/ATR/OBV) | Sharpe 1.44 |
| v1 | 57+4 | 截面排名(ret_5d/20d/vol_5d/rsi_rank) | 基础 |
| v2 | 57+4+12 | 4指数×3窗口(SSE50/CSI500/ChiNext/SSE) | 含市场联动 |
| v3 | 57+4+12+6 | 行业特征(ind_ret/alpha/ind_vol/industry_size) | 含行业信息 |
| v4 (+日历+截面) | **70** | wday_0~3/month_sin_cos/is_month_end/is_cny_before_after/excess_return_1d/cs_rank_close_volume/industry_rank_return | **Sharpe 1.56** |
| v5 (+调整日+量价背离) | **73** | is_rebalance_soon/divergence_bull/divergence_bear | Sharpe 1.52 |

### 3.2 特徵分类

| 类别 | 维度 | 具体因子 |
|:-----|:----:|:---------|
| 动量 | 4 | ret_5d/10d/20d/60d |
| 波动率 | 3 | vol_5d/10d/20d |
| 均线偏离 | 4 | ma5/10/20/60_dev |
| 量价 | 5 | volume_ratio_5/20, turn_change, gap_ratio, amplitude |
| 技术指标 | 12 | RSI, MACD(signal/hist), KDJ(K/D/J), OBV, WR, BB(dev/width), ATR |
| 风险位置 | 4 | max_dd_20d, price_position, close_pos, intraday_range |
| 走势 | 3 | streak_up/down, vol_ret_corr_10d |
| 截面排名 | 4 | ret_5d/20d_rank, vol_5d_rank, rsi_rank |
| 超额收益 | 1 | excess_return_1d (个股-市场平均) |
| 截面排名 | 2 | cs_rank_close, cs_rank_volume |
| 行业 | 7 | ind_ret_5d/20d, alpha_ret_5d/20d, ind_vol_5d, industry_size, industry_rank_return |
| 指数 | 12 | SSE50/CSI500/ChiNext/SSE_Composite × 3窗口(5/10/20d) |
| 市场 | 1 | beta_60d |
| 日历 | 9 | wday_0~3, month_sin/cos, is_month_end, is_cny_before/after, **is_rebalance_soon** |
| 量价背离 | 2 | **divergence_bull, divergence_bear** |

---

## 四、选股管线 — 4层过滤 + 博弈论

### 第1层: 精度门控 (Precision Gate)

所有候选股需同时满足：
```
1. 近5日平均涨幅 > 0%      (动量条件)
2. 近20日最大回撤 > -8%    (风险条件)
```

这是最核心的过滤器，回测中效果最显著。原理：动量溢价只存在于上涨趋势中，深度回撤的股票可能进入下跌通道。

### 第2层: 纳什均衡多样化 (Game Theory)

```python
效用(组合) = 平均分数 - λ × 平均内部相似度
```

- 相似度矩阵: NN embedding 余弦相似度 (300×300)
- λ 自适应: 候选池 7-10 只时 λ×0.6, >10 只时正常
- 穷举搜索: 从候选池中选择使效用最大化的子集
- 效果: 同一策略不加纳什 Sharpe=0.95，加纳什后 1.44 (+0.49)

### 第3层: 过滤器 (可选)

| 过滤器 | 参数 | 效果 |
|:-------|:-----|:------|
| 超买过滤 | `--no-overbought` | RSI>75 或 10日涨幅>20%排除。防追高。 |
| 行业分散 | `--diverse-industry N` | 同行业最多N只。降集中度。 |
| 宏观门控 | `--market-state` | 20日均线下行时切防御板块(已移除,改为`--max-score`) |

### 第4层: 波动率过滤

剔除波动率 > 全市场中位数 × 1.2 的股票。低波动率异象：同等收益下低波动股票表现更稳定。

### 权重分配

| 方式 | 公式 | 特点 |
|:-----|:-----|:------|
| **softmax** | exp((s-max(s))/0.3) / sum | 高分多配，低分少配，**绝对收益最高** |
| inv_vol | 1/σ / sum(1/σ) | 波动率越低权重越高，**Sharpe最佳** |
| equal | 1/N | 简单等权 |
| bdc | clip(invested × raw/raw.sum, 0, 0.5) | 预测比例加权+单票上限 |

---

## 五、运行模式与参数

### 5.1 核心命令

```bash
# ── 极限分数模式（追求绝对收益, 0.143）──
python controller.py --max-score --game 0.25 --weight softmax

# ── 稳健模式（Sharpe 1.98, 14周100%胜率）──
python controller.py --ensemble lgbm --game 0.25 --no-overbought --diverse-industry 2 --weight inv_vol

# ── 均衡模式（Sharpe 1.52）──
python controller.py --no-overbought --diverse-industry 2 --game 0.25 --weight inv_vol

# ── 对比所有策略 ──
python controller.py --ensemble compare
```

### 5.2 全部参数

| 参数 | 默认 | 可选值 | 说明 |
|:-----|:----:|:-------|:------|
| `--ensemble` | `lgbm-nn` | lgbm/lgbm-nn/gp/compare | 模型融合方式 |
| `--weight` | `bdc` | equal/softmax/inv_vol/bdc | 权重分配 |
| `--game` | — | float | 纳什均衡λ，0.2-0.3推荐 |
| `--multi-day` | 1 | 1-5 | 多日rolling预测天数 |
| `--no-overbought` | — | flag | 超买过滤(RSI>75或10d>20%) |
| `--diverse-industry` | — | int N | 同行业最多N只 |
| `--cash-buffer` | — | float | 信号弱时保留现金比例 |
| `--center` | — | flag | 零均值居中消除偏差 |
| `--max-score` | — | flag | 极限分数:宏观分析+赛道拥挤+博弈 |
| `--no-cache` | — | flag | 强制重训LGBM |
| `--topk` | 5 | 1-5 | 选股数量 |
| `--output` | output/result.csv | path | 输出路径 |

### 5.3 训练

```bash
# 完整训练（3种子×4折TSCV, ~2小时）
python train.py --seed 791 --loss topk_listnet --decay 2.0 --tscv --batch-size 32
python train.py --seed 42  --loss topk_listnet --decay 2.0 --tscv --batch-size 32
python train.py --seed 787 --loss lambdarank    --tscv --batch-size 32
```

---

## 六、回测结果

### 6.1 14周滚动回测 (2025-06 ~ 2026-05)

测试方法: 每周滚动训练LGBM + NN 12模型推理 → 选股 → 评分。完整模拟比赛流程。

**73特征 + 过滤器 (`--no-overbought --diverse-industry 2`)**

| 排名 | 策略 | 权重 | Mean | Std | Sharpe | WinRate | 最差周 | 最佳周 |
|:---:|:-----|:----:|:----:|:---:|:------:|:-------:|:------:|:------:|
| 🥇 | **Nash** | **inv_vol** | **0.075** | **0.038** | **1.98** | **100%** | +0.021 | +0.130 |
| 🥇 | Nash | equal | 0.077 | 0.039 | 1.98 | 100% | +0.018 | +0.130 |
| 🥉 | Nash | bdc | 0.079 | 0.041 | 1.94 | 100% | +0.017 | +0.132 |
| 4 | Nash | softmax | 0.106 | 0.062 | 1.71 | 100% | +0.010 | +0.218 |
| 5 | **BlendNash** | **inv_vol** | **0.081** | **0.054** | **1.52** | **93%** | -0.033 | +0.193 |
| 6 | BlendNash | equal | 0.081 | 0.055 | 1.46 | 93% | -0.046 | +0.197 |
| 7 | BlendNash | softmax | 0.082 | 0.057 | 1.44 | 93% | -0.049 | +0.194 |
| 8 | BlendNash | bdc | 0.077 | 0.055 | 1.40 | 93% | -0.047 | +0.196 |

### 6.2 特征升级效果

| 版本 | 特征 | BlendNash Sharpe | 最差周 | 关键改进 |
|:----|:----:|:----------------:|:------:|:---------|
| 初始 57 | 量价+技术 | 1.44 | -0.052 | 基线 |
| +日历+截面 70 | wday/CS/excess | **1.56** | **-0.001** | 日历效应+截面信号 |
| +调整日+背离 73 | rebalance/divergence | 1.52 | -0.049 | 成分股调整+量价背离 |

### 6.3 BlendNash softmax 14周明细

```
W1  +0.128  W2  +0.074  W3  +0.203  W4  -0.001
W5  +0.135  W6  +0.077  W7  -0.001  W8  +0.100
W9  +0.123  W10 +0.192  W11 +0.060  W12 +0.033
W13 +0.063  W14 +0.121
```

14周仅2周微亏(-0.001), 胜率86%。最大回撤 -0.1%。

---

## 七、风险分析与博弈论

### 7.1 `--max-score` 模式（极限分数）

该模式追求绝对收益最大化，不设超买过滤，但增加宏观分析：

```
python controller.py --max-score --game 0.25 --weight softmax
```

输出示例：
```
  极限分数模式 — 市场分析 + 博弈论
  ==================================
  宏观: 熊市 (20d=-4.7%, 5d=+0.5%)
  赛道拥挤(top40):
     电子: 8只(20%)
     电力设备: 6只(15%)
  博弈升级: λ→0.15(候选池仅9只)
```

**工作原理：**
1. **宏观周期**: 用20日均线判断市场趋势（牛/震荡/熊），影响风偏
2. **赛道拥挤度**: 分析top40候选股的行业分布，>25%视为拥挤
3. **博弈升级**: 拥挤时自动上调λ（纳什强度），促使选择不同赛道
4. **顶点预警**: 标记RSI>80且10日涨>25%的极端超买股（不排除，但提示风险）
5. **核心不变**: 不设超买过滤，让高动量股进入组合（追求分数）

### 7.2 6月1日实盘亏损复盘

**选股**: 5只科技（华大九天RSI=77, 天孚通信10d+36%）
**原因**: 科技板块极端拥挤 + 基金调仓抽血 + 模型无行业分散
**改进**: 
- `--no-overbought` → 华大九天RSI>75排除
- `--diverse-industry 2` → 不会5只全科技
- `--max-score` → 赛道拥挤检测+预警
- **Sharpe从1.44提升至1.98**

### 7.3 数据泄漏检测

`test_window.py`: 3窗口滚动验证，对偶准确率0.505（随机0.500）
结论：**无数据泄漏**

---

## 八、纳什均衡详解（核心创新点）

纳什均衡是本项目最大的单一贡献（Sharpe +0.49）。

### 数学形式

```
max  U(S) = mean(score_i) - λ × mean(sim_ij)
     S⊆C    i∈S                    i,j∈S, i≠j

其中:
  C = 候选池 (精度门控后约10只)
  S = 选中的子集 (约4-5只)
  sim_ij = stock_i 与 stock_j 的embedding余弦相似度
  λ = 博弈强度 (默认0.25, 池子小时自动降低)
```

### 直觉理解

- 第一项 `mean(score_i)`: 选择高分股票（追求收益）
- 第二项 `-λ × mean(sim_ij)`: 惩罚相似股票（鼓励多样化）
- 纳什均衡 = 每个股票在组合中的存在"对其他股票构成竞争"，类似博弈论中的混合策略均衡

### 与均值-方差优化的关系

```
马科维茨: max  μ'w - γ w'Σw
纳什选股: max  mean(S) - λ·mean_sim(S)
                  ↑           ↑
               预期收益    多样化惩罚(协方差替代)
```

相似度矩阵近似协方差矩阵，λ对应风险厌恶系数。

---

## 九、参考来源

- **Game-BDC2026** (PaikyHiean): DoubleEnsemble LGBM, 日历/截面特征, 零均值居中
- **MASTER (AAAI 2024)**: Market-Guided Stock Transformer, PCC Loss, Market Gating
- **Qlib Alpha158**: 158个alpha因子体系
- **Behavioral Finance**: 处置效应 → 超买过滤; 动量因子 → 精度门控

---

## 十、代码结构

| 文件 | 行数 | 职责 |
|:-----|:----:|:------|
| `controller.py` | ~940 | 主入口: 数据加载→特征→LGBM→NN→融合→纳什→选股→输出 |
| `train.py` | ~1000 | NN训练: TSCV/decay/augment/loss选择/模型保存 |
| `model.py` | ~740 | GRU+Transformer+MarketGate+ScoreHead+所有loss函数 |
| `features.py` | ~600 | 73维特征工程+窗口采样+标准化 |
| `data_loader.py` | ~350 | 数据下载+panel构建+缓存 |
| `score_self.py` | ~95 | 评分脚本(与比赛方一致) |
| `config.py` | ~230 | 所有超参数 |
| `window_test/backtest.py` | ~480 | 14周滚动回测 |
| `test_window.py` | ~170 | 数据泄漏检测 |

### 模型文件

```
models/
├── portfolio_model_g791_v2_fold[1-4].pt     73特征 seed 791
├── portfolio_model_seed42_v2_fold[1-4].pt    73特征 seed 42
├── portfolio_model_g787_v2_fold[1-4].pt      73特征 seed 787
├── lgbm_ranker.txt                           LightGBM缓存
└── backup_v1_57feat/                         旧版57特征模型(备查)
```

最后备份: /root/gupiao_final_20260601_221132

---

## 十一、当前问题与改进方向

### 11.1 核心矛盾：分数 vs 风控

```
追求分数 → 不设过滤 → 6月1日重演（单周-5%~-8%）
加过滤器 → 保护回撤 → 分数从0.143降到0.081（-43%）
```

模型目前靠参数组合手动切换，没有自动平衡机制。

### 11.2 行业数据不完整

- ~20% 股票 industry 标记为 "Unknown"
- 仅有一级行业分类，无二级/三级，无法做精细赛道轮动
- 成分股调整用日历硬编码，无真实预期数据

### 11.3 无"卖点"验证

比赛 T+1 买入 T+5 卖出，但模型只选买什么，没有验证"为什么这些股票在下周会比别的涨得多"。优化的是历史相关性而非未来因果性。

### 11.4 特征边际收益递减

| 版本 | 特征 | Sharpe 变化 |
|:----|:----:|:-----------:|
| 57→70 | +13 | **+0.12** ✅ |
| 70→73 | +3 | -0.04 ❌ |

成分股调整和量价背离在理论上合理，但回测无显著贡献。73维已进入边际收益递减区。

### 11.5 LGBM 每次窗口从零训练

14周回测中 LGBM 每次重新训练，耗时占 80%+。应改为增量更新或滑动窗口。

### 11.6 纳什 λ 手工参数

`--game 0.25` 凭经验设定。应根据候选池的实际分散度自动确定——池子越集中 λ 越大。当前虽有小池子降权的启发式规则，但不够系统化。

### 11.7 无日內/高频信号

所有特征基于日线。尾盘异动、开盘跳空、盘中大单完全缺失。有 Level-2 可构建更丰富的"聪明钱"因子。

### 11.8 评分规则偏差

`score_self.py` 用开盘价计算，真实交易不可能全以开盘价成交。模型可能学到"挑开盘跳空大的股票"，而非真正有趋势的股票。

### 11.9 无置信度输出

模型输出标量分数，但没有告诉用户这个分数有多可靠。`--cash-buffer` 是简单启发式，非真正的概率建模。

### 11.10 无市场状态学习

`--max-score` 用20日均线判断牛/熊是硬编码规则。模型本身没学到在不同市场状态下应有不同选股策略。MASTER 论文的 Market Gating 试图解决，需重训验证。

### 11.11 改进优先级

| 优先级 | 问题 | 思路 |
|:-----:|:-----|:------|
| 🔴 高 | 纳什 λ 自动确定 | 根据候选池相似度矩阵特征值自动计算 |
| 🔴 高 | 置信度建模 | 用模型集成方差作为置信度，低时自动减仓 |
| 🟡 中 | 行业数据补全 | 对接更完整的行业分类数据 |
| 🟡 中 | LGBM 增量训练 | 缩短回测时间 |
| 🟢 低 | 特征继续扩充 | 边际收益已低，不值得 |
| 🟢 低 | 分日预测 | 需改训练管线，工作量与收益不确定 |
