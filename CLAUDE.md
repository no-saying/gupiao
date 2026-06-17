# 股票组合预测 — 项目手册

> **赛题**：沪深 300 成分股，预测 T+1→T+5 收益最高的 ≤5 只组合
> **队伍**：中州奶龙
> **更新**：2026-06-17

---

## 零、快速开始

```bash
# LGBM 基线
python controller.py                      # LGBM 预测
python controller.py --validate           # 12 周回测

# NN 训练
python train_nn.py --pretrain --epochs 200 # MAE 预训练
python train_nn.py --finetune --epochs 100 # CVXPY 微调

# DANN 域自适应（886 只股票 → 只在 CSI300 选股）
python train_dann.py                        # MSE 损失 (默认, 最优)
python train_dann.py --loss margin          # Margin Ranking (实验性)
python train_dann.py --loss lambdarank      # LambdaRank (实验性)
python backtest_dann.py                     # 回测 dann_mse.pt
python backtest_dann.py --model dann_margin.pt

# 特征工程 — 删除缓存强制重算
rm data/raw/alpha158_panel.parquet && python controller.py
```

---

## 一、当前最优基线（2026-06-17）

### DANN 12 周真实回测（按周采样，持仓不重叠）🟢 最优

| Mean | Min | Max | Sharpe | WinRate |
|:---|:---|:---|:---|:---|
| **+0.0329** | -0.0523 | +0.1201 | **4.54** | **75%** |

逐周：W1 -0.3% → W2 +2.4% → W3 +0.9% → W4 +11.2% → W5 +5.4% → W6 +1.0% → W7 +12.0% → W8 +1.3% → W9 +8.4% → W10 -5.2% → W11 -2.1% → W12 +4.5%

### DANN vs LGBM 同期对比

| 指标 | DANN | LGBM |
|:---|:---|:---|
| **Mean** | **+0.0329** | +0.0140 |
| **Sharpe** | **4.54** | 3.84 |
| **WinRate** | **75%** | 58% |
| Std | 0.0523 | 0.0264 |
| Min | -0.0523 | -0.0211 |
| Max | +0.1201 | +0.0643 |

**DANN 全面碾压 LGBM**：均值 2.35×，Sharpe +0.7，WinRate +17pp。
886 只训练→300 只选股策略验证有效。DANN 波动更大但收益碾压，Sharpe 4.54 达到实战水平。

### 旧基线（已淘汰）

| 模型 | Mean | Sharpe | WinRate | 结论 |
|:---|:---|:---|:---|:---|
| LGBM (老版) | +0.0003 | 0.17 | 50% | 抛硬币 |
| NN (MAE+GAT) | -0.0004 | — | 50% | =LGBM 平手 |

---

## 二、代码结构

```
gupiao/
├── controller.py          # LGBM 主入口
├── train_nn.py            # MAE 预训练 + 微调
├── backtest_nn.py         # NN walk-forward 回测
├── predict_nn.py          # NN 单次预测
├── config.py              # 全局配置
├── tushare_loader.py      # Tushare 数据管线
├── score_self.py          # 自评脚本
├── optuna_tune.py         # 调参工具
├── core/
│   ├── data.py            # build_lgbm_data()
│   ├── features.py        # engineer_features()
│   ├── alpha158.py        # Alpha158 因子（调用 ultimate_ols）
│   ├── ultimate_ols.py    # Numba 12核 O(1) 滚动特征引擎 🚀
│   ├── model.py           # LGBM train/predict
│   ├── selection.py       # 选股
│   ├── validate.py        # LGBM walk-forward
│   ├── nn_model.py        # Mamba/GAT/MAE 架构
│   ├── cvxpy_layer.py     # 可微组合层
│   └── dann_model.py      # DANN + 多损失 (MSE/Margin/LambdaRank)
├── train_dann.py          # DANN 训练入口（886只→300只）
├── backtest_dann.py       # DANN walk-forward 回测
└── deprecated/            # V6~V10 废弃代码
```

---

## 三、踩坑记录

### 核心教训

| # | 教训 |
|:---|:---|
| 1 | **NN 不适合此赛题** — 300只×232窗口，信噪比太低 |
| 2 | **简单 > 复杂（除 DANN）** — LGBM 碾压融合方案，但 DANN 域自适应打破此规律 |
| 3 | **行业中性化是核心** — RankIC 0.009→0.146 |
| 4 | **RankIC ≠ SCORE** — 全市场排序 ≠ Top-5 选股 |
| 5 | **Optuna 代理指标调参均失败** — RankIC/Top-15 Z 都不能替代 SCORE |
| 6 | **反中性化有害** — 行业趋势均值回归，不可预测 |
| 7 | **Attack 选股无效** — 门控后池太小 |
| 8 | **CVaR 压死信号** — λ>0.5 模型坍塌为常数输出 |
| 9 | **逐日回测虚假高估** — 窗口重叠导致 Sharpe 2.10，真实按周只有 0.17 |
| 10 | **sliding_window_view 维度翻转** — 5x 显存爆炸 |
| 11 | **CUDA 版本不匹配** — PyTorch 13.0 vs 驱动 12.4，需装 cu124 版 |
| 12 | **MAE+GAT 打不过 LGBM** — 200 epoch 预训练 + 100 epoch 微调 = 50% 胜率平手 |
| 13 | **Pairwise 排名损失导致分数坍缩** — Margin Ranking + alpha_mse=0.1→std 0.008, Sharpe 1.26 |
| 14 | **LambdaRank 不适合神经网络** — 梯度结构为 GBDT 设计，NN 上 Sharpe 仅 0.60 |
| 15 | **MSE 仍是最好选择** — DANN 的域自适应已做主要贡献，损失函数改变无增量 |

### SCORE 演进史

| 日期 | 版本 | 单次 | 12周 Mean | Sharpe | WinRate | 结局 |
|:---|:---|:---|:---|:---|:---|:---|
| 06-02 | v2~v5 | 0.12~0.15 | — | — | — | baostock+旧口径 |
| 06-05 | v9.1 LGBM | 0.0443 | +0.0012 | 0.40 | 50% | 唯一正期望基线 |
| 06-15 | v9.3 反中性化 | 0.0212 | -0.0283 | -8.18 | 8% | ❌ 溃败 |
| 06-16 | V9.4 RankIC调参 | — | -0.0025 | -0.70 | 25% | ❌ |
| 06-16 | V10.2 Attack | — | +0.0091 | 1.31 | 50% | ❌ 池太小 |
| 06-16 | V10.2 Top15调参 | — | -0.0014 | -0.45 | 58% | ❌ |
| 06-16 | **重构版 LGBM** | — | +0.0056 | 2.10* | 67%* | 🟡 逐日回测有偏 |
| **06-17** | **LGBM 按周回测** | 0.0234 | **+0.0003** | **0.17** | **50%** | 🟡 真实水平 |
| 06-17 | NN(MAE+GAT) | -0.0042 | -0.0004(4周) | — | 50% | = LGBM 平手 |
| **06-17** | **LGBM 重构+alpha158Numba** | 0.0643 | **+0.0242(4周)** | **4.25** | **75%** | 🟢 信号显著增强 |
| **06-17** | **🏆 DANN 域自适应** | — | **+0.0329** | **4.54** | **75%** | 🟢🏆 **当前最优**：886训练→300选股 |
| **06-17** | **LGBM (alpha158新缓存)** | — | **+0.0140** | **3.84** | **58%** | 🟢 配新缓存后提升明显 |
| 06-17 | DANN+Margin+MSE | — | +0.0046 | 1.26 | 50% | ❌ 分数坍缩 std=0.008 |
| 06-17 | DANN+LambdaRank+MSE | — | +0.0025 | 0.60 | 58% | ❌ LambdaRank 不适合 NN |

> *逐日回测（窗口重叠）虚高，按周才是真实水平。

---

## 四、下一步方向

### 根本问题

沪深 300 定价极度充分，信噪比极低。300 只大盘股的信息量不足以驱动任何深度学习模型。

### 🔴 当前方向：非对称样本空间

**用 886 只股票训练（沪深 300 + 中证 500），只在沪深 300 里选股。**

小盘股量价规律更明显，信噪比更高。让模型在 886 只里"学会抓 Alpha"，然后到 300 里执行。

### 四种前沿架构

| # | 方案 | 核心思路 | 状态 |
|:---|:---|:---|:---|
| 1 | **DANN 域自适应** | 对抗训练剥离"市值/波动率偏见"，学到通用量价逻辑 | ✅ **已验证** — Sharpe 4.54, WinRate 75%, 碾压 LGBM |
| 2 | **异构图注意力** | 500 只股票为 300 只提供"邻居异动信号" | 待实现 |
| 3 | **知识蒸馏** | 886 只上的 Teacher → Soft Labels → 300 只上的 Student | 待实现 |
| 4 | **宏观门控** | 用 886 只数据合成市场宽度/风格强弱，HMM 切换攻防 | 待实现 |

### 当前状态 (2026-06-17 19:00)

- ✅ 代码重构完成 — 旧文件移入 `deprecated/`
- ✅ `core/ultimate_ols.py` — Numba 12核引擎 (886×1220 < 72s)
- ✅ `core/alpha158.py` — 已重写 + 缓存重生成功
- ✅ **DANN+MSE Sharpe 4.54, WinRate 75% — 当前最优模型**
- ✅ LGBM 基线更新 (Sharpe 3.84)
- ❌ 排名损失实验失败 — Margin(Sharpe 1.26) / LambdaRank(Sharpe 0.60)
- 🔄 下一步方向：DANN+LGBM 集成 / CDAN / 风险约束

---

## 五、前沿技术路线图（优先级排序）

### 🥇 P0 — 立即执行（改动小、胜率高）

| # | 方案 | 原理 | 预期收益 |
|:---|:---|:---|:---|
| 1 | **LambdaRank 目标** | LGBM 原生 `objective='lambdarank'`，NDCG@5 评估，直接优化排序而非回归 | 选股质量跃升 |
| 2 | **非对称损失** | 自定义 Gradient/Hessian：y<0 且 ŷ>0（误判亏损）给予指数级惩罚 | 根治单周暴跌 |
| 3 | **预测置信度加权** | 10 个不同种子的 LGBM 集成，高分+低方差的股票给更高权重（≠ 等权） | 提升 Sharpe |

### 🥈 P1 — 核心杀器（需 3-5 天开发）

| # | 方案 | 原理 | 预期收益 |
|:---|:---|:---|:---|
| 4 | **GP 遗传规划挖因子** | gplearn 进化算法，将 OHLCV+算子作为节点，自动进化出高 IC 非线性因子 | 因子数量翻倍 |
| 5 | **Barra 纯 Alpha 正交化** | 个股收益对 10 风格因子+31 行业做多元回归，取残差作为训练目标 | 学到特异收益 |
| 6 | **全市场预训练+微调** | 5000 只全 A 股预训练 Transformer，冻结底层，只微调沪深 300 最后几层 | 数据量碾压 |

### 🥉 P2 — 降维打击（需 1-2 周）

| # | 方案 | 原理 |
|:---|:---|:---|
| 7 | **可微夏普比率** | Batch 内构建模拟组合收益序列，直接对 Sharpe 求导做 Loss |
| 8 | **时空图注意力 (HIST/Trax)** | 股票作为图节点，产业链+概念重合度作为边，Attention 上下游异动 |
| 9 | **端到端 RL 组合管理** | PPO/DDPG，状态=全市场特征，动作=权重向量，奖励=组合收益-λ·回撤 |

### 实施建议

**不要同时铺开。** 按 P0 → P1 → P2 顺序，每步验证 SCORE 提升后再前进。

---

## 六、特征工程 — 性能基准

### Numba 引擎 vs 传统方案

| 方案 | 886只×1100天 | 原理 |
|:---|:---|:---|
| ~~groupby + rolling.apply~~ | 数小时 | Python 逐股循环 + GIL |
| ~~joblib.Parallel~~ | 10-15 分钟 | 多进程，进程间通信开销大 |
| **Numba 12核 O(1)** | **< 60 秒** | Pivot 矩阵 + prange 并行 + 递推累加器 |

### 关键实现 (`core/ultimate_ols.py`)

- **矩阵透视**: 长表 → [T×S] 矩阵，消除 groupby 哈希开销
- **O(1) 递推**: `sum += in - out`，窗口从 10 到 252 天速度相同
- **混合精度**: float32 存储（省内存+高缓存命中），float64 累加（杜绝精度漂移）
- **12核 prange**: 截面并行，12 线程同时跑不同股票
- **JIT 预热**: 模块导入时用小数据预编译，正式运行零延迟
