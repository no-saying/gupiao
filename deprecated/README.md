# 已淘汰模块 — V9+ 全部废弃 (2026-06-16)

> 核心原则：**纯 LGBM + 旧标签 + 行业中性化 Target**，不做任何花哨后处理。

---

## 一、淘汰清单

| 文件 | 原用途 | 废弃原因 |
|:---|:---|:---|
| `controller_v9.py` | V9.3 控制器 (反中性化+Attack+OOF+Ensemble) | 反中性化有害, Attack 池太小, 固定权重不如简单 LGBM |
| `controller_v6_v8.py` | V6-V8 控制器 (NN+GP+Nash+HRP+HMM) | NN 信噪比太低, 过度复杂 |
| `config_v9.py` | V9.3 完整配置 | NN/V9+ 参数废弃 |
| `config_v6_v8.py` | V6-V8 完整配置 | 同上 |
| `model_nn.py` | NN 架构 (GRU+Transformer+GAT) | NN 不适合此赛题 |
| `train_nn.py` | NN 训练脚本 | 同上 |
| `predict_nn.py` | NN 预测脚本 | 同上 |
| `data_loader.py` | Baostock 数据加载 | 切换纯 Tushare |
| `ensemble.py` | LGBM+CatBoost 集成+HMM+MoE | V9+ 复杂度 |
| `scoring.py` | Nash 均衡+HRP+软聚类 | V9+ 复杂度 |
| `risk.py` | 宏观门控+HMM+尾部检测 | V9+ 复杂度 |
| `utils.py` | 标准化+embedding 缓存 | NN 相关 |
| `validate_v9.py` | V9 消融实验框架 | V9+ 工具 |
| `optuna_tune.py` | Optuna 调参 | RankIC/Top-15 ≠ SCORE |
| `optimize_lgbm.py` | Optuna 单 trial | 同上 |
| `test_window.py` | 滚动窗口 NN 测试 | NN 废弃 |
| `core_init.py` | 旧 core/__init__.py | V9+ 导出 |

## 二、当前有效代码

```
gupiao/
├── controller.py       # 主入口 (~70行)
├── config.py           # 精简配置 (Tushare+LGBM)
├── tushare_loader.py   # Tushare 数据管线
├── score_self.py       # 自评脚本
├── optuna_tune.py      # 调参工具
├── core/
│   ├── data.py         # LGBM 数据构建
│   ├── features.py     # 特征工程
│   ├── alpha158.py     # Alpha158 因子
│   ├── model.py        # LGBM 训练/预测
│   ├── selection.py    # 选股
│   └── validate.py     # Walk-forward 验证
├── data/               # 缓存
├── models/             # 模型文件
└── output/             # 结果
```

## 三、核心教训 (from CLAUDE.md)

| # | 教训 |
|:---|:---|
| 1 | NN 不适合此赛题 (300×999样本×816K参数) |
| 2 | 简单模型+好特征 > 复杂集成 |
| 3 | 行业中性化是核心 (RankIC 0.009→0.146) |
| 4 | RankIC ≠ SCORE |
| 5 | CatBoost 需要 categorical 特征 |
| 6 | Optuna RankIC 调参失败 (RankIC↑33% 但 SCORE 跌负) |
| 7 | Attack 选股依赖池大小 (池太小无效) |
| 8 | Optuna Top-15 调参再次失败 (Top-15 Z↑113% 但 SCORE 跌负) |
