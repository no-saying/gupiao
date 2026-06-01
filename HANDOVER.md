# 股票组合预测 — 交接文档

## 最终成绩

### 本周 Score (2026-05-25 ~ 2026-05-29)
| 策略 | Score | 胜率 |
|------|:-----:|:----:|
| **LGBM+NN+门控** 🏆 | **0.149** | 5/5 |
| LGBM+NN 纯融合 | 0.144 | 5/5 |
| LightGBM Ranker | 0.130 | 5/5 |
| old fold1 | 0.127 | 3/4 |
| old 3model | 0.107 | — |
| v2 TSCV4 | 0.103 | — |
| old TSCV4 | 0.077 | — |
| NN GP v2 | 0.059 | — |

### 6窗口滚动验证
| 策略 | WinRate | Sharpe | 周均收益 |
|------|:-------:|:------:|:--------:|
| **LightGBM Ranker** 🏆 | **94%** | **1.19** | **+5.77%** |
| GP v2 NN | 61% | 0.31 | +1.79% |
| v2 TSCV4 | 59% | 0.21 | +1.24% |
| old TSCV4 | 53% | 0.18 | +1.09% |
| old fold1 | 47% | 0.06 | +0.30% |

### 数据泄漏检验 ✅
```
    模型     时间段       胜率    Sharpe
W1  Fold1  2024-04~09   38%    -0.20
W2  Fold2  2024-09~25   51%     0.13
W3  Fold2  2025-02~07   56%     0.12
W4  Fold3  2025-08~12   59%     0.23
W5  Fold4  2025-12~26   58%     0.25
──────────────────────────────────────
平均                  52%    0.07
✓ 无泄漏, 信号逐年增强
```

## 文件对应

| 文件 | 策略 | 说明 |
|------|------|------|
| `result_lgbm_nn.csv` | LGBM+NN+门控 | 推荐提交 |
| `result_lgbm.csv` | LightGBM单独 | 长期最稳 |
| `result_nash.csv` | +纳什均衡 | 同+博弈论 |
| `result_gp.csv` | NN GP v2 | NN独立 |
| `result_fold1.csv` | old fold1 | 高风险 |
| `result_3model.csv` | old 3model | 不推荐 |

## 方法

### 分数生成
- **LightGBM Ranker**: lambdarank, 500树, leaves=63, 57特征
- **NN GP**: 12个v2模型(3种子×4折) GP优化权重
- **融合**: 0.7×LGBM + 0.3×NN, 归一化后

### 选股
1. 候选池: 分数 top 40
2. 精度门控: 动量>0, 回撤<8%, 波动率过滤
3. softmax加权: 高分多配

### GP最优权重 (NN ensemble)
```python
g791_f1=0.261, g791_f4=0.261, seed42_f4=0.248
g791_f3=0.128, g787_f4=0.070, seed42_f1=0.032
```

## 运行

```bash
python controller.py --weight softmax              # 默认
python controller.py --ensemble lgbm               # LGBM单独
python controller.py --game 0.3                    # +纳什均衡
python controller.py --ensemble compare            # 对比
```

## 注意事项
- 旧模型(51特征) 与 v2模型(57特征) 不兼容
- controller.py 自动处理特征对齐
- LightGBM 首次运行自动训练并缓存
- 总训练时间 < 2小时, 满足8小时限制
