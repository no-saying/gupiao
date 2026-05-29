# 基于注意力机制的股票组合预测 —— 完整教程

## 目录

1. [赛题理解](#1-赛题理解)
2. [核心思路](#2-核心思路)
3. [为什么用注意力机制](#3-为什么用注意力机制)
4. [项目结构](#4-项目结构)
5. [代码详解](#5-代码详解)
6. [运行指南](#6-运行指南)
7. [结果解读与调优](#7-结果解读与调优)
8. [扩展方向](#8-扩展方向)

---

## 1. 赛题理解

### 问题定义

给定沪深 300 指数成分股的**历史股价数据**，预测**未来一周**收益最大的股票组合（不超过 5 只），并分配权重。

### 收益计算

```
单只股票收益 = (T+5 开盘价 - T+1 开盘价) / T+1 开盘价
组合总收益  = Σ(每只股票权重 × 单只股票收益) + 现金权重 × 0
```

所以是预测**4 个交易日**（约一周）的股价变动。

### 关键约束

- 最多选 5 只股票
- 权重总和 ≤ 1，剩余为现金
- 只能使用公开免费数据（如 baostock）
- 组合收益率必须高于基准程序

---

## 2. 核心思路

### 从"预测收益"到"排序学习"

一个问题：**我们真的需要精确预测每只股票的未来收益率吗？**

不需要。我们只需要知道"哪 5 只股票未来一周涨得最多"。这是排序问题，不是回归问题。

这也是为什么我们选择 **LambdaRank 排序损失** 而不是传统的 MSE 回归损失：
- MSE 会平等地惩罚所有预测误差
- LambdaRank 更关注**排名靠前的股票**是否排对了位置

### 从"独立预测"到"关联预测"

另一个问题：**每只股票应该独立预测吗？**

如果独立预测每只股票，模型很可能给同一个行业的 5 只股票都打高分。因为：
- 这些股票的特征相似（同样的行业 Beta、相似的技术指标）
- 模型会倾向给出相似的预测分数
- 结果：选了 5 只高度相关的银行股/白酒股

所以我们引入了**截面自注意力机制**，让模型在打分时"看到"所有其他股票，自动学习分散化。

### 整体流程

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│ 数据获取  │ → │ 特征工程  │ → │ 注意力模型排序 │ → │ 组合输出  │
│ (baostock)│   │ (22因子)  │   │ (GRU+Attn)   │   │ (result.csv)│
└──────────┘    └──────────┘    └──────────────┘    └──────────┘
```

---

## 3. 为什么用注意力机制

### 传统方法的问题

```
传统 pipeline:
  特征 → 独立预测每只股票的收益 → 选 Top-5 → 手动分散化（协方差/行业约束）
                ↑                              ↑
          每只股票孤立处理              预测完了再考虑相关性
```

**问题**：预测阶段完全不知道股票间的关联，容易把高分集中给同一板块。

### 注意力方法的优势

```
注意力 pipeline:
  特征 → GRU(时序) → Self-Attention(截面) → 同时输出300只股票分数 → Top-5
                          ↑
                   300只股票互相"看"
                   自动学习谁和谁有关
```

**优势**：
- 预测分数已经包含了"分散化"的信息
- 茅台和五粮液的 embedding 相似 → 模型会自动压低其中一只的分数
- 不需要手动定义行业分类或协方差矩阵

### 具体架构

```
输入: (B, 300, 60, 22)
  B = batch_size, 300 = 股票数, 60 = 交易日, 22 = 特征数
      │
      ▼
┌─ TimeEncoder (共享双向GRU) ───────────────────┐
│  每只股票的 60 天特征 → 128 维向量              │
│  300 只股票共享同一套 GRU 参数                  │
│  输出: (B, 300, 128)                           │
└────────────────────────────────────────────────┘
      │
      ▼
┌─ CrossSectionalTransformer × 2 ───────────────┐
│  Layer 1: 股票之间做多头自注意力                │
│  Layer 2: 学习更深层的间接关联                   │
│  每只股票的 128 维向量融合了其他股票的信息       │
│  输出: (B, 300, 128) + 注意力权重 (300×300)    │
└────────────────────────────────────────────────┘
      │
      ▼
┌─ ScoreHead (MLP) ─────────────────────────────┐
│  128 → 64 → 1 标量分数                         │
│  输出: (B, 300) — 每只股票的买入选股分数        │
└────────────────────────────────────────────────┘
```

---

## 4. 项目结构

```
gupiao/
├── config.py           # 所有超参数和配置
├── data_loader.py      # 从 baostock 获取数据 + 本地缓存
├── features.py         # 22 个量化因子的计算 + 滑窗样本构造
├── model.py            # 注意力模型 + 排序损失函数
├── train.py            # 训练脚本 + 回测验证
├── predict.py          # 预测脚本 → 生成 result.csv
├── requirements.txt    # Python 依赖
├── TUTORIAL.md         # 本文档
├── readme.md           # 赛题说明
│
├── data/               # 数据目录（自动创建）
│   ├── raw/            #   原始数据缓存 (parquet)
│   └── processed/      #   特征样本缓存 (pickle)
│
├── models/             # 模型权重目录（自动创建）
│   └── portfolio_model.pt
│
└── result.csv          # 提交文件（由 predict.py 生成）
```

---

## 5. 代码详解

### 5.1 config.py —— 配置中心

所有超参数集中管理，方便调参。关键参数：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `LOOKBACK_DAYS` | 60 | 输入窗口的交易天数（约3个月）|
| `PREDICT_HORIZON` | 4 | 预测周期（T+1到T+5）|
| `D_MODEL` | 128 | 隐藏层维度 |
| `N_HEADS` | 8 | 注意力头数 |
| `N_TRANSFORMER_LAYERS` | 2 | 截面Transformer层数 |

### 5.2 data_loader.py —— 数据获取

**设计要点**：
- **baostock 登录/登出**：baostock 是 request-response 模式，每次查询前要 `bs.login()`，完成后 `bs.logout()`
- **本地缓存**：下载的数据存成 parquet 文件，重复运行不用重新下载
- **前复权数据**：`adjustflag="2"` 表示前复权，这样历史价格对分红除权做了调整
- **MultiIndex Panel**：用 `(date, stock_id)` 作为双层索引，方便按日期切片和按股票分组

### 5.3 features.py —— 特征工程

**22 个因子的设计思路**：

```
动量因子（4个）：ret_5d, ret_10d, ret_20d, ret_60d
  短/中/长期趋势强度。学术研究表明动量因子在A股有显著效果。

波动率因子（3个）：vol_5d, vol_10d, vol_20d
  风险度量。低波动异象（低波动股票长期收益高于高波动）是著名市场异象。

均线偏离（4个）：ma5_dev ~ ma60_dev
  价格相对于趋势线的位置。突破均线通常是趋势延续的信号。

量价因子（4个）：volume_ratio, turn_change, amplitude
  成交量和换手率反映市场关注度。放量上涨说明资金在进场。

技术指标（4个）：rsi_14, macd, macd_signal, macd_hist
  经典技术指标。虽然简单，但作为基础特征对模型有帮助。

风险因子（2个）：max_dd_20d, price_position
  回撤幅度和价格位置。大涨后的股票可能面临获利回吐。

估值因子（2个）：peTTM, pbMRQ
  市盈率和市净率。低估值股票通常安全边际更高。

市场因子（1个）：beta_60d
  个股相对于市场的敏感度。高Beta = 进攻型，低Beta = 防御型。
```

**滑窗采样**：

```
时间轴:  ├──── 60天特征 ────┤  ├4天┤
         T-64            T-5  T-1  T
         输入 X 窗口         目标 y
```

每 5 个交易日生成一个样本，避免相邻窗口过度冗余。

### 5.4 model.py —— 注意力模型

**TimeEncoder（时序编码器）**：

双向 GRU 处理 60 天的时序数据：
- 前向 GRU：从过去到现在的趋势
- 后向 GRU：从后往前看（识别反转模式）
- 拼接两个方向的最终隐态得到更丰富的表示

**CrossSectionalTransformer（截面自注意力）**：

模型的核心创新。300 只股票互相做注意力：
- 每只股票的 Query 查询 "哪些股票和我相关"
- Key/Value 提供所有股票的信息
- 输出融合了相关股票的信息
- 注意力权重矩阵 (300×300) 可以解读为股票关联图谱

**损失函数**：

| 损失 | 特点 | 适用场景 |
|------|------|---------|
| Pairwise Hinge | 简单快速，所有pair一视同仁 | 快速验证、baseline |
| LambdaRank | NDCG delta 加权，关注top排序 | **推荐**，最终训练 |
| Diversity Penalty | 惩罚top-K embedding过于相似 | 配合LambdaRank使用 |

LambdaRank 的核心公式：
```
loss_ij = |ΔNDCG_ij| × log(1 + exp(-σ × (s_i - s_j)))
```
如果交换排序第1和第2的股票，NDCG变化很大（|ΔNDCG|大）→ 这个pair的loss被放大。
如果交换排序第100和第101的股票，NDCG变化很小 → 这个pair的loss被压缩。

### 5.5 train.py —— 训练脚本

**关键设计**：

1. **时间序列切分**：训练/验证/测试按时间顺序切分（不能shuffle split），避免用未来的数据预测过去
2. **梯度裁剪**：`clip_grad_norm_` 防止GRU+Transformer长序列上的梯度爆炸
3. **ReduceLROnPlateau**：验证损失不降时自动将学习率减半
4. **Early Stopping**：连续25轮不改善就停止
5. **回测验证**：在测试集上模拟真实交易，计算均收益率、夏普比率、胜率

### 5.6 predict.py —— 预测输出

**多样化的贪心选股算法**：

```python
1. 取模型打分最高的 30 只股票进入候选池
2. 第 1 只: 选分数最高的
3. 第 2~5 只:
   for 每个候选:
       计算它与"已选股票"的最大 embedding 余弦相似度
       调整分数 = 原始分 - 相似度 × 原始分 × 0.5
   选调整后分数最高的
4. 等权分配
```

举例说明：
- 已选：贵州茅台（白酒）
- 候选A：五粮液（白酒），embedding相似度 0.95 → 调整后分数大幅降低
- 候选B：宁德时代（新能源），embedding相似度 0.10 → 调整后分数几乎不变
- 结果：模型会选宁德时代而不是五粮液，因为"已经有一个白酒了"

---

## 6. 运行指南

### 环境准备

```bash
# 安装依赖
pip install -r requirements.txt
```

依赖列表：
- `torch` — 深度学习框架
- `baostock` — A股数据API
- `numpy`, `pandas` — 数据处理
- `scikit-learn` — 辅助工具
- `tqdm` — 进度条

### 训练模型

```bash
# 默认配置训练（推荐）
python train.py

# 使用 Pairwise Hinge Loss（更快但效果可能稍差）
python train.py --loss pairwise

# 自定义超参数
python train.py --epochs 150 --lr 5e-5

# 强制重新下载数据（清缓存）
python train.py --download
```

训练时间估计（NVIDIA GPU）：
- 数据下载（首次）：~5 分钟
- 特征工程：~1 分钟
- 模型训练（100 epochs）：~10 分钟

没有 GPU 也可以用 CPU 训练，只是速度慢一些。

### 生成提交

```bash
# 默认生成 result.csv
python predict.py

# 指定模型和输出路径
python predict.py --model models/portfolio_model.pt --output submission.csv
```

输出格式：
```csv
stock_id,weight
000408,0.2
000975,0.2
002028,0.2
600372,0.2
600036,0.2
```

---

## 7. 结果解读与调优

### 训练输出解读

```
Step 4: Training
  Model params: 803,713
  Device: cuda
  Loss: lambdarank

Epoch   1: loss=0.0432
Epoch   2: loss=0.0381
  ...

Step 5: Evaluation
  Test loss:        0.0123          ← 越低越好
  Test pair acc:    0.587           ← 比0.5高说明模型有效

  Backtest (top-5 equal-weight):
  Mean weekly return:  0.003456     ← 平均每周收益 0.35%
  Std weekly return:   0.025678
  Weekly Sharpe:       0.1345       ← >0 说明有正的风险调整收益
  Win rate:            0.534        ← 53.4% 的周收益为正
```

### 调优方向

| 方向 | 方法 | config.py 中对应参数 |
|------|------|---------------------|
| 加因子 | 在 features.py `_per_stock` 加新特征，并添加到 FEATURE_COLS | FEATURE_COLS |
| 加数据 | 增大 LOOKBACK_DAYS 或调小 STEP_DAYS | LOOKBACK_DAYS, STEP_DAYS |
| 调模型 | 调整 d_model, n_heads, transformer layers | D_MODEL, N_HEADS, N_TRANSFORMER_LAYERS |
| 防过拟合 | 增大 dropout, weight_decay | DROPOUT, WEIGHT_DECAY |
| 调损失 | 调整 lambda_diverse 让模型更分散 | 在 model.py 的 combined_loss 中改 |
| 更长的时序 | 用 Transformer Encoder 替代 GRU | model.py TimeEncoder |

### 常见问题

**Q: 训练集 loss 下降但测试集 loss 不降？**
过拟合。增大 DROPOUT（如 0.1→0.2），或减小 D_MODEL（如 128→64）。

**Q: 回测收益波动太大？**
调大 lambda_diverse 让组合更分散，或增加 TOP_K_CANDIDATES。

**Q: baostock 查询失败？**
检查网络连接。baostock 有时候不稳定，可以过几分钟重试。数据有缓存，重试会跳过已下载的。

---

## 8. 扩展方向

### 短期改进

1. **增加基本面因子**：ROE、毛利率、营收增速等（需要 baostock 的季节数据接口）
2. **行业中性化**：对因子做行业+市值回归取残差，消除行业偏差
3. **多模型集成**：训练 LightGBM + Attention 两个模型，分数加权取平均
4. **交易成本**：考虑手续费和滑点，加入换手率约束

### 中期改进

1. **图神经网络（GAT）**：用行业/供应链关系构建邻接矩阵，替代全连接自注意力
2. **时序 Transformer**：用 Transformer Encoder 替代 GRU 做时序编码
3. **多任务学习**：同时预测收益率和涨跌方向，共享底层表示
4. **宏观变量**：加入利率、汇率、PMI 等宏观因子

### 长期改进

1. **强化学习**：把选股视为 sequential decision making，用 PPO/DQN 优化
2. **预训练+微调**：在更大数据集上预训练时序编码器，在沪深300上微调
3. **不确定性估计**：用 Monte Carlo Dropout 或 Ensemble 估计预测的不确定性

---

## 附录：关键参考文献

- Burges et al. "Learning to Rank using Gradient Descent." ICML 2005.（LambdaRank 原始论文）
- Vaswani et al. "Attention is All You Need." NeurIPS 2017.（Transformer 架构）
- Qin et al. "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction." IJCAI 2017.（DA-RNN：时序+注意力的先驱工作）
- Feng et al. "Temporal Relational Ranking for Stock Prediction." ACM Transactions on Information Systems, 2019.（股票排序预测）
