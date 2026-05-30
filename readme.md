# 代码说明

**队伍名称**: 中州奶龙

## 环境配置

- Python 3.10
- PyTorch 2.0.1
- CUDA 11.7
- numpy 1.24.4, pandas 2.0.3, scikit-learn 1.3.2, tqdm 4.66.1

Docker 镜像: `bdc2026:latest`

## 数据

使用 baostock (http://baostock.com) 公开免费 A 股数据，获取沪深 300 成分股日线行情：
- 时间范围: 2021-06-01 ~ 2026-05-30
- 字段: 开高低收、成交量、成交额、换手率、涨跌幅、市盈率、市净率
- 前复权处理（adjustflag=2）
- 数据下载后保存为 train.csv（含完整历史）和 test.csv（竞赛评估用）

## 预训练模型

未使用任何预训练模型。全部模型参数从零（随机初始化）训练。

## 算法

### 整体思路

将股票组合选择建模为**排序学习（Learning to Rank）**问题：
1. 对每只股票计算 50+ 个技术因子（动量、波动率、量价、MACD、RSI、KDJ、布林带等）
2. 用时序编码器（双向 GRU）提取每只股票的时序特征
3. 用截面自注意力（Cross-Sectional Transformer）让 300 只股票互相 attend，自动学习分散化
4. 用 ListNet 排序损失直接优化 Top-K 股票的排序质量
5. 多模型集成 + Sharpe 加权分配最终组合权重

### 创新点

1. **截面自注意力机制**：传统方法独立预测每只股票后手动分散化；本方法在打分时就让股票互相"看到"，模型内生地避免选中高度相关标的
2. **向量化数据构建**：将 Panel 预计算为 3D numpy 数组，窗口采样 4 秒完成（原始逐行循环需 187 秒）
3. **事件过滤**：剔除 9 个重大市场事件窗口（俄乌冲突、上海封城、雪球敲入等），避免模型学到异常模式

### 网络结构

```
输入 (300 stocks, 60 days, 52 features)
  |
  ├─ TimeEncoder: BiGRU(2 layers, d_model=128)
  |   每只股票独立编码 -> (300, 128)
  |
  ├─ CrossSectionalTransformer x2: MultiHeadAttention(8 heads)
  |   股票间自注意力 -> (300, 128)
  |
  └─ ScoreHead: Linear(128->64) + GELU + Linear(64->1)
       输出 (300,) 排序分数
```

总参数: ~808K

### 损失函数

| 损失 | 说明 |
|------|------|
| **ListNet** (主) | KL 散度匹配预测分数分布与真实收益分布，最适合 Top-K 排序 |
| LambdaRank | NDCG delta 加权的 pairwise logistic loss |
| Pairwise Hinge | 基础 pairwise 排序损失 |

### 数据扩增

未使用数据扩增。通过 STEP_DAYS=1 最大程度利用时间窗口样本。

### 模型集成

- 训练多个随机种子（seed=789, 456, 42 等）的模型
- 集成时对预测分数取平均，embedding 取平均做多样化选股
- 权重分配: Sharpe 加权（历史夏普比率高的股票配更高权重）

### 算法的其他细节

- 梯度裁剪 (max_norm=1.0) 防止 GRU+Transformer 梯度爆炸
- ReduceLROnPlateau 学习率调度，patience=10
- Early Stopping patience=25
- 事件过滤: strict 模式，label 窗口与事件区间有重叠即剔除
- 因子 z-score 标准化

## 训练流程

```
app/train.sh
  |
  ├─ 1. 读取 /app/data/train.csv (OHLCV 日线数据)
  ├─ 2. 特征工程 (featurework.py: 50+ 因子计算)
  ├─ 3. 窗口采样 (60天输入, 4天预测, 步长1天)
  ├─ 4. 时间序列切分 (75%/15%/10%)
  ├─ 5. 模型训练 (ListNet loss, AdamW, 100 epochs)
  ├─ 6. 回测评估 (测试集 Top-5 等权组合收益)
  └─ 7. 保存模型到 /app/model/portfolio_model.pt
```

训练时间: 约 4-6 小时 (i7-13650H + RTX 4060 8GB)

## 推理流程

```
test.sh
  |
  ├─ 1. 加载训练好的模型 (/app/model/portfolio_model.pt)
  ├─ 2. 读取训练数据计算特征 (需要历史数据做特征工程)
  ├─ 3. 构建最新时间窗口
  ├─ 4. 模型前向传播得到排序分数
  ├─ 5. 贪心多样化选股 (Top-5, embedding cosine distance)
  ├─ 6. 等权分配权重
  └─ 7. 输出 /app/output/result.csv
```

预测时间: < 2 分钟

## 其他注意事项

- 所有随机种子固定 (--seed 789)，确保可复现
- 训练和推理过程完全离线，不依赖网络
- 数据按时间顺序切分，避免未来信息泄露
- Docker 镜像需命名为 `bdc2026`，导出为 `队伍名称.tar`
- 使用的开源数据 (baostock) 已通过邮件报备
