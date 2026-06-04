# 股票组合预测 — 完整项目手册

> 赛题：基于沪深 300 成分股历史数据，预测未来一周（T+1~T+5）收益最高的 ≤5 只股票组合。
> 队伍：中州奶龙
> 最后更新：2026-06-05

---

## 一、当前版本状态

### 1.1 版本来源

基于 `/root/gupiao_backup_20260603_234410`（v2 生产版），在此之上做了三项改进。

### 1.2 运行

```bash
python controller.py
# SCORE ≈ 0.125（当前周，与旧版持平但多一层防御）
```

### 1.3 核心参数

| 参数 | 值 |
|:-----|:---|
| LGBM 标签 | `open[t+4] / open[t] - 1` |
| LGBM 超参 | Optuna: lr=0.0196, leaves=68, trees=350 |
| NN 模型 | 12 v2 (3种子×4折 TSCV) |
| **辅助模型** | **3 个 (pinball/top1/focal)** |
| 融合方式 | 0.75×(0.8 LGBM + 0.2 NN) + 0.25×aux |
| 权重 | GP固定 + bdc |
| 选股 | 精度门控 → 纳什 → 波动率过滤 |
| 门控 | 通过数决定仓位（FULL/CAUTIOUS/DEFENSIVE/MINIMAL/CASH） |

---

## 二、2026-06-05 改进记录

### 2.1 三项改进

| # | 改进 | 状态 | 效果 |
|:--|:-----|:----|:-----|
| 1 | OOF Stacking 元模型 | ✅ 代码就绪 | 函数 `train_stacking_meta()` 已添加 |
| 2 | 辅助预测模型 | ✅ 已训练 | pinball(Sharpe0.37) + top1(0.54) + focal(0.27) |
| 3 | 紧门控+空仓 | ✅ 已集成 | 门控通过数→仓位决策，lam bug修复 |

### 2.2 新增损失函数 (model.py)

| 损失 | 用途 | 标签 |
|:-----|:-----|:-----|
| `pinball_loss(q=0.90)` | 极端涨幅预测 | 连续收益率 |
| `top1_loss` | 当日最强股预测 | 二元 (最强=1) |
| `focal_loss` | top 5% 概率预测 | 二元 (top5%=1) |

### 2.3 辅助模型

```bash
python train.py --loss pinball --seed 791 --epochs 100 --batch-size 32
python train.py --loss top1    --seed 791 --epochs 100 --batch-size 32
python train.py --loss focal   --seed 791 --epochs 100 --batch-size 16
```

### 2.4 已知问题

1. `--game` 未指定时 lam=None 导致崩溃 → 默认改为 0.25
2. LGBM 缓存污染 → `rm models/lgbm_ranker.txt` 清理

---

## 三、参考代码分析

### 3.1 Game-BDC2026（/tmp/Game-BDC2026）

**架构**：三 Slot 集成 —— Slot1 DoubleEnsemble LGBM (272维) + Slot2 MASTER (Market-Gated Transformer) + Slot3 ALSTM (LSTM+Attention)

**关键发现**：

| # | 发现 | 我们是否已有 | 优先级 |
|:--|:-----|:----------|:------|
| 1 | **三异构模型 Rank 均值集成** | 有 LGBM+NN，缺第三异构 | 🟡 中 |
| 2 | **正确标签**: `(open.shift(-5)-open.shift(-1))/open.shift(-1)` | ❌ LGBM 用的是 shift(-4) | 🔴 待讨论 |
| 3 | DoubleEnsemble: M0全特征 + M1-M5 70%子集误差加权 | ✅ `--de` 已实现 | 🟢 已有 |
| 4 | 5日 Rolling [0.4,0.25,0.15,0.12,0.08] | ✅ `--multi-day 5` | 🟢 已有 |
| 5 | 零均值居中增强截面排序 | ✅ `--center` | 🟢 已有 |
| 6 | PurgedKFold (embargo=10天) | ⚠️ TSCV gap=5天，可加 | 🟢 低 |
| 7 | **PCC Loss 天花板**: 优化全部排序忽略 top-5 爆发力 | ⚠️ 我们的 NN 用 topk_listnet 部分缓解 | 🟡 中 |
| 8 | MASTER-Q90 (Pinball Loss, q=0.90): 专门预测极端涨幅 | ❌ | 🔴 高 |
| 9 | MASTER-Cls (Focal Loss): 二分类"会不会进 top 5%" | ❌ | 🔴 高 |

### 3.2 gupiao_mirror3（/tmp/gupiao_mirror3）

**架构**：多专家模型集成 —— 6 种专家 (Transformer/卷积/季节性/激进/布朗噪声/Neural GARCH) + Meta Aggregator + MC Dropout 推理

**关键发现**：

| # | 发现 | 耗时 | 优先级 |
|:--|:-----|:----|:------|
| 1 | MC Dropout: 推理时保持 dropout, 20 次前向取平均 | 推理 ×20 | 🟡 中 |
| 2 | Stochastic Depth: 训练时随机跳过层, 推理时亮度缩放 | 训练+20% | 🟢 低 |
| 3 | Neural GARCH: 神经网络估计时变波动率 | 重训 | 🟢 低 |
| 4 | Meta Aggregator: 学习专家权重 (Attention over experts) | 需重训 | 🟡 中 |
| 5 | MonthSeasonalExpert: 月份+股票季节性 embedding | 重训 | 🟢 低 |
| 6 | BrownianNoiseExpert: 注入布朗噪声训练抗扰动 | 重训 | 🟢 低 |

### 3.3 bdc2026（/tmp/bdc2026）

官方教程仓库，基础参考。已采纳的：Stock Embedding、并行 Transformer（实验失败）、混合标签（回退了）。

---

## 四、可采纳改进（优先级排序）

### 🔴 第一优先：低成本高收益

**1. MASTER-Q90 预测模型**
- 问题：当前模型系统性低估极端涨幅
- 方案：复制 NN 模型，改用 Pinball Loss (q=0.90)，专门预测"未来一周可能暴涨的股票"
- 工作量：新增 `pinball_loss()` 到 model.py + 训练 1 个种子
- 预期效果：抓到 +15%+ 的爆发股，突破 R_total 天花板
- 参考：TODO_3.md § 0.2

**2. MASTER-Cls 二分类模型**
- 问题：不知道哪些股票"大概率进 top 5%"
- 方案：用 Focal Loss 训练二分类器，label = (未来5日收益 > 截面 95% 分位)
- 工作量：新增 `focal_loss()` + 训练 1 个种子
- 参考：TODO_3.md Phase 2

**3. 标签公式审查**
- Game-BDC2026 明确使用 `(open.shift(-5)-open.shift(-1))/open.shift(-1)`（T+1→T+5）
- 我们 LGBM 用 `open.shift(-4)/open - 1`（T→T+4）
- NN 用 `(open_t5 - open_t1)/open_t1`（T+1→T+5，正确）
- **待决定**：LGBM 标签是否需要对齐？14 周回测证明旧标签有效（Sharpe 1.53）

### 🟡 第二优先：中期改进

**4. 第三异构模型（ALSTM）**
- LSTM + Temporal Attention，轻量（~150K 参数），训练 1 小时
- 与 LGBM + NN 形成三足鼎立，Rank 平均集成降方差
- 参考：TODO_1.md Phase 5

**5. 日历 + 截面特征扩充**
- 周几 one-hot (4维)、月份 cyclical (2维)、月末/季末 (2维)、春节窗口 (2维)
- 截面排名: excess_return_1d、cs_rank_close、cs_rank_volume、industry_rank_return
- 预计新增 ~15 维
- 参考：TODO_1.md P2-T2, P2-T3

### 🟢 第三优先：长期储备

**6. MC Dropout 推理**（已完成原型，效果不显著）
**7. 对抗学习时间不变特征**（训练成本高）
**8. DoubleEnsemble 升级**（已有 --de，未充分使用）
**9. 市场情绪特征**（涨跌比、新高新低比）

---

## 五、竞赛分享对比分析

来源：THU-BDC2026 参赛队伍经验分享。

### 5.1 特征工程对比

| 维度 | 他们 | 我们 | 差距 |
|:-----|:-----|:-----|:-----|
| 窗口覆盖 | 3,5,10,20,40 天 | 5,10,20,60 天 | 缺 3 天超短期 |
| 截面特征 | 日排名、超额收益、市场情绪、涨跌比 | cs_rank_close/volume, excess_return_1d | 缺涨跌比 |
| 外部数据 | 无 | 无 | 一致 |

### 5.2 模型对比

| 维度 | 他们 | 我们 | 差距 |
|:-----|:-----|:-----|:-----|
| 基线模型 | LGBM + HistGB + RF | LGBM only | 缺 HistGB/RF |
| **核心融合** | **OOF Stacking（二层元模型）** | LGBM+NN 线性加权 | **最大差距** |
| 辅助模型 | Top1 预测 + 短期爆发预测 | 无 | **缺辅助方向** |
| 建模方式 | 收益预测和涨跌概率分开 | 统一排序模型 | 可以借鉴 |

### 5.3 训练对比

| 维度 | 他们 | 我们 | 差距 |
|:-----|:-----|:-----|:-----|
| TSCV | 4折, gap=5天, val=20天, train≥120天 | 4折, gap=5天, val=50天 | val更大 |
| 种子 | 固定 20260416 | 3 个种子 (791/42/787) | 更丰富 |
| 超参 | 默认值为主，调树数量 | Optuna 搜索 | 更精细 |
| 重点 | 特征和标签设计 | 模型架构 | 方向不同 |

### 5.4 组合策略对比

| 维度 | 他们 | 我们 | 差距 |
|:-----|:-----|:-----|:-----|
| 筛选流程 | 候选池→精排→风险惩罚 | 候选池→精度门控→纳什→波动率 | 类似 |
| **精度门控** | 收益>0 + **排名前25%** + **DD<3%** | 收益>0 + DD<8% | **他们严得多** |
| 仓位 | 市场不好时空仓/少选 | 默认满仓5只 | **缺防御** |
| 树数量 | 200-300棵 | 350棵(Optuna) | 类似 |

### 5.5 关键差距总结

1. **OOF Stacking** — 他们用多个基模型的 OOF 预测训练二层元模型，我们只是简单线性加权
2. **辅助预测方向** — Top1 预测 + 短期爆发预测，两个独立子模型提供额外信号
3. **涨跌概率分离** — 收益预测和涨跌方向分开建模
4. **门控更严** — DD<3% + 排名前25%，比我们的 DD<8% 严格得多
5. **空仓机制** — 市场不好主动空仓，不硬凑

---

## 六、14周回测记录

| 排名 | 策略 | 权重 | Mean | Sharpe | WinRate |
|:---:|:-----|:----:|:----:|:------:|:-------:|
| 1 | **LGBM_Nash** | softmax | 0.0865 | **1.53** | **100%** |
| 2 | BlendNash | inv_vol | 0.0828 | 1.36 | 92.9% |

14周明细（LGBM_Nash softmax）:
```
W1 +0.2041  W2 +0.1223  W3 +0.0631  W4 +0.0015  W5 +0.1420
W6 +0.0380  W7 +0.0145  W8 +0.1011  W9 +0.1072  W10 +0.1202
W11 +0.0297 W12 +0.0304 W13 +0.0959 W14 +0.1416
```

---

## 七、代码结构

| 文件 | 职责 |
|:-----|:------|
| `controller.py` | 主入口: LGBM训练→NN推理→融合→纳什→选股 |
| `train.py` | NN训练: TSCV |
| `model.py` | BiGRU+Transformer+7种loss |
| `features.py` | 80维特征+窗口采样 |
| `data_loader.py` | 数据下载+panel构建 |
| `config.py` | 超参数 |
| `score_self.py` | 评分脚本 |

---

## 八、备份

- 当前版本: `/root/gupiao_backup_20260603_234410`
- 最新实验: `/root/gupiao_backup_20260604_030346`
- 早期稳定: `/root/gupiao_final_20260602_233555`

## 九、参考仓库

- `/tmp/Game-BDC2026` — 三 Slot 集成方案（最全面，含详细诊断和 TODO）
- `/tmp/gupiao_mirror3` — 多专家 MC Dropout 集成（架构创新最多）
- `/tmp/bdc2026` — 官方教程基线

---

