# 架构文档

## 代码结构

```
controller.py         主入口 — 参数解析 + 流程编排
core/
  ensemble.py         LGBM/CatBoost/NN 训练 + 预测 + 权重融合
  scoring.py          分数校准 + 纳什均衡选股 + 过滤器 + 权重分配
  risk.py             宏观门控 + HMM市场状态 + 极端周检测 + Beta惩罚
  utils.py            _norm + Embedding缓存（4处复用→1次）
  validate.py         滚动窗口验证 + 消融实验
train.py              NN训练入口（10种损失函数）
model.py              NN架构（GRU→Transformer/GAT→评分头）
features.py           特征工程（per-stock + Tushare衍生 + Z-Score + Alpha158）
features_alpha158.py  Alpha158量化因子
tushare_loader.py     Tushare数据管线（14数据源 + hfq复权 + adj_factor）
config.py             超参数（14组，80+参数）
score_self.py         竞赛评分脚本
```

## 数据流

```
Tushare stk_factor_pro(hfq) → OHLCV
  + daily_basic (PE/PB/市值)
  + moneyflow (资金流向)
  + margin_detail (融资融券)
  + hk_hold (北向资金)
  + cyq_perf (筹码分布)
  + stk_holdernumber (股东人数)
  + limit_list_d (涨跌停)
  + index_member_all (SW2021行业)
  + index_daily (6大指数)
       ↓
  build_tushare_only_panel() → 300股×1220天×143列
       ↓
  engineer_features() → 300股×1220天×276列
  add_alpha158_features() → +56因子
       ↓
  make_window_samples() → (N, 300, 60, 126) 时间窗口
       ↓
  ┌─ build_lgbm_data() → LGBM/CatBoost (~380特征)
  └─ NN (~126特征)
```

## 模型矩阵

| 模型 | 特征 | 训练 | 融合权重 |
|------|------|------|---------|
| LGBM | ~380 | Purged K-Fold盲跑 + 时间衰减 | 0.5 |
| CatBoost | ~380+SW分类 | YetiRank | 0.5 |
| NN listnet | ~126 | ListNet + GP优化 | 15-25%渐进 |
| SW60(4) | 57 | Pinball/Focal | 背离加权重 |
| AUX(3) | 57 | Pinball/Top1/Focal | 固定10% |

## 选股管线

```
分数融合 → 极端周检测 → 行业中性化 → Beta惩罚 → 拥挤惩罚
  → 开盘跳空惩罚 → 纳什均衡 → 波动率过滤 → 行业分散
  → 软聚类去重 → HRP权重 → 输出Top5
```
