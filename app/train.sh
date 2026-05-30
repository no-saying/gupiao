#!/bin/bash
# =============================================================================
#  train.sh — 模型训练脚本（docker 内运行）
#  从 /app/data/train.csv 读取数据，训练模型，保存到 /app/model/
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  train.sh — 模型训练"
echo "============================================"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"

cd /app/code/src

# ---- train ----
python3 train.py \
    --data /app/data/train.csv \
    --model-dir /app/model \
    --output /app/output/result.csv \
    --seed 789

echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo "  训练完成"
echo "============================================"
