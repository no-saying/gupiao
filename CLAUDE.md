# CLAUDE.md — 股票组合预测项目

## 项目概述

基于注意力机制的沪深 300 股票排序预测。输入 300 只股票 60 天 × 52 因子，输出 Top-5 组合 + 权重。

## 环境

```bash
# 依赖
pip install numpy pandas torch scikit-learn tqdm pyarrow baostock openpyxl

# 或用 Docker（推荐）
docker build -t bdc2026 .
```

## 常用命令

```bash
# 训练（主流程）
python train.py --seed 789 --loss listnet --epochs 100

# 预测
python predict.py --model models/portfolio_model_seed789.pt

# 集成预测（多模型）
python controller.py --models seed789,seed456 --weight sharpe --compare

# 评分
mkdir -p output temp
cp result.csv output/ && python score_self.py

# Docker 部署测试
docker build -t bdc2026 . && docker run --rm bdc2026 /app/init.sh
docker run --rm -v $(pwd)/app/data:/app/data -v $(pwd)/app/model:/app/model bdc2026 /app/train.sh
docker run --rm -v $(pwd)/app/data:/app/data -v $(pwd)/app/model:/app/model -v $(pwd)/app/output:/app/output bdc2026 /test.sh
```

## 关键文件

| 文件 | 作用 |
|------|------|
| `train.py` | 训练入口，`--seed` 控制种子，`--loss` 选 listnet/lambdarank/pairwise |
| `predict.py` | 单模型预测，输出 result.csv |
| `controller.py` | 多模型集成 + 权重优化，`--compare` 自动找最佳权重方法 |
| `model.py` | BiGRU + CrossAttention + ScoreHead，三个 loss 函数 |
| `features.py` | 50+ 因子（动量/波动/均线/技术/行业/指数） |
| `config.py` | 所有超参数 |
| `score_self.py` | 赛题评分脚本 |
| `a.sh` | AutoDL Claude Code + DeepSeek 一键部署 |

## 最优配置

- **种子**: 789 + 456 集成
- **Loss**: ListNet
- **标签**: 百分位排名
- **特征**: 52 因子（含截面排名 + 行业 + 指数）
- **权重**: Sharpe 加权
- **最佳分数**: 0.0924

## 数据

baostock 免费 A 股数据 → `data/raw/*.parquet` → `data/processed/samples.pkl`。训练前确保数据已下载好（首次约需 20 分钟）。

## 注意

- 时间序列必须按顺序切分，不能随机 shuffle
- seed 对结果影响大（0.33 vs -0.06），必须固定 `--seed 789`
- 多模型集成显著优于单模型
- Docker 镜像需命名为 `bdc2026`，导出为 `中州奶龙.tar`
