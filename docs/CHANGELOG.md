# 变更记录

## v6.1 (2026-06-14)

**架构重构：**
- controller.py 2061→633行，拆分为 core/ 包（5模块）
- 数据源切换：官方CSV底 → Tushare全量2021-2026
- 事件滤波：硬删除 → 软降权（核心±4天 + 边缘±10天渐变权重）

**优化：**
- Embedding缓存：4处重复加载→1次
- NN特征集：80维→126维（含资金流/融资融券/北向/筹码衍生+Z-Score）
- CatBoost分类特征：1类→10类（SW L1/L2/L3 + 日历）
- Isotonic校准：--calibrate
- 消融实验：--ablation（15机制逐项测试）
- 魔法数字收归 config.py（14组80+参数）

**清理：**
- 删除 data_loader.py（baostock旧管线）
- 删除 g787全部模型、死模型、零权重模型
- 33→22个模型文件

## v6

- Tushare Pro 数据源（11源）+ adj_factor修正
- CatBoost + YetiRank
- Purged K-Fold盲跑（4折，保底250树）
- NN渐进融合（listnet 12模型，背离周降级）
- SW60短窗专家（4模型）
- 52周滚动验证：Sharpe 11.2，WinRate 98%

## v5

- 平滑sigmoid曲面 + HRP + HMM
- Score 0.1273

## v4

- 部分中性化 + 时间衰减权重 + 多日EMA
- Score 0.1474

## v3

- 宏观门控 + 行业自适应 + 动态λ
- Score 0.1267

## v2

- 渐进式风格切换 + Sharpe集成
- Score 0.1256

## v1

- 二元风格切换，纯LGBM
- Score 0.0910
