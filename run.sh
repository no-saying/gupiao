#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  股票组合预测 — 一键运行"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ── 1. 数据 ──
echo ""
echo "[1/4] 数据准备..."
python3 -c "
from tushare_loader import fetch_csi300_stocks, build_tushare_only_panel
from features import engineer_features, make_window_samples
from features_alpha158 import add_alpha158_features
sids = fetch_csi300_stocks()
p = build_tushare_only_panel(sids)
p = engineer_features(p, sids)
p = add_alpha158_features(p)
X, y, m, dates = make_window_samples(p, sids, normalize=False)
print(f'  ✓ {len(X)} 样本, {X.shape[-1]} 特征')
print(f'    训练={int(len(X)*.75)} 验证={int(len(X)*.15)} 测试={int(len(X)*.10)}')
print(f'    日期: {dates[0]} ~ {dates[-1]}')
"

# ── 2. NN训练 ──
echo ""
echo "[2/4] 训练 NN v3..."

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

for seed in 791 42; do
    echo "  --- seed=$seed ---"
    python train.py \
        --loss listnet \
        --epochs 300 \
        --batch-size 32 \
        --augment \
        --seed "$seed" \
        --output "$MODELS_DIR/portfolio_model_${seed}_v3.pt"
done

# ── 3. 预测 ──
echo ""
echo "[3/4] 预测..."
python controller.py --calibrate

# ── 4. 结果 ──
echo ""
echo "[4/4] 结果"
echo "============================================"
cat output/result.csv 2>/dev/null
echo ""
echo "============================================"
echo "  完成 $(date '+%Y-%m-%d %H:%M:%S')"
echo "  结果: output/result.csv"
echo "  日志: output/run_log.csv"
echo "============================================"
