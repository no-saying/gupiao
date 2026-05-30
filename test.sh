#!/bin/bash
# =============================================================================
#  test.sh — 模型预测脚本（docker 内运行）
#  从 /app/data/test.csv 读取测试数据，加载模型，输出到 /app/output/result.csv
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  test.sh — 模型预测"
echo "============================================"
echo "  Start: $(date '+%Y-%m-%d %H:%M:%S')"

cd /app/code/src

# ---- predict (5 min limit) ----
timeout 300 python test.py \
    --data /app/data/test.csv \
    --model-dir /app/model \
    --output /app/output/result.csv

echo "  End: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"
echo "  预测完成"
echo "============================================"
