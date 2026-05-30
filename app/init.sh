#!/bin/bash
# =============================================================================
#  init.sh — 环境初始化脚本（docker 内运行）
#  安装项目依赖，验证数据完整性
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  init.sh — 环境初始化"
echo "============================================"

# ---- install Python deps (offline / from cache) ----
echo "[1/3] 安装 Python 依赖 ..."
pip install --no-cache-dir --quiet \
    numpy pandas scikit-learn tqdm pyarrow openpyxl 2>&1 | tail -1
echo "  依赖安装完成"

# ---- verify data ----
echo "[2/3] 验证数据文件 ..."
for f in /app/data/train.csv /app/data/test.csv; do
    if [ -f "$f" ]; then
        lines=$(wc -l < "$f")
        echo "  $f: ${lines} 行"
    else
        echo "  [WARN] $f 不存在"
    fi
done

# ---- verify model dir ----
echo "[3/3] 验证模型目录 ..."
if [ -d /app/model ]; then
    model_count=$(find /app/model -name "*.pt" 2>/dev/null | wc -l)
    echo "  /app/model: ${model_count} 个模型文件"
else
    echo "  [WARN] /app/model 不存在"
fi

echo "============================================"
echo "  初始化完成"
echo "============================================"
