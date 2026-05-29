#!/bin/bash
# =============================================================================
#  AutoDL Claude Code + DeepSeek V4 一键部署
#  用法: bash a.sh
# =============================================================================
set -euo pipefail

echo "==================== AutoDL Claude Code + DeepSeek V4 ===================="

# ---- 1. 安装 Node.js + npm ----
echo "[1/5] 安装 Node.js 20.x ..."
if command -v node &>/dev/null; then
    echo "  Node.js 已安装: $(node -v)"
elif command -v apt-get &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
    echo "  Node.js 安装完成: $(node -v)"
elif command -v conda &>/dev/null; then
    conda install -y -c conda-forge nodejs=20
    echo "  Node.js (conda) 完成: $(node -v)"
else
    echo "  无法自动安装 Node.js，请手动安装后重试"
    exit 1
fi

# ---- 2. 换 npm 国内源 ----
echo "[2/5] 配置 npm 国内镜像"
npm config set registry https://registry.npmmirror.com

# ---- 3. 安装 Claude Code ----
echo "[3/5] 安装 Claude Code ..."
npm install -g @anthropic-ai/claude-code
claude --version

# ---- 4. 配置 DeepSeek 环境变量 ----
echo "[4/5] 配置 DeepSeek API ..."
DS_KEY="sk-8b78a9e5ac8c4aa3bfd114e25a0cc458"

# 直接写入 ~/.bashrc
if ! grep -q "ANTHROPIC_BASE_URL" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'ENVEOF'

# === DeepSeek API (for Claude Code) ===
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_API_KEY="DSKEY_PLACEHOLDER"
export ANTHROPIC_AUTH_TOKEN="DSKEY_PLACEHOLDER"
export ANTHROPIC_MODEL="deepseek-v4-flash"
ENVEOF
    echo "  已写入 ~/.bashrc"
else
    echo "  配置已存在，跳过写入"
fi

# 替换占位符
sed -i "s/DSKEY_PLACEHOLDER/${DS_KEY}/g" ~/.bashrc

# 当前 shell 直接 export（不用 source ~/.bashrc，避免 PS1 unbound 报错）
export ANTHROPIC_BASE_URL="https://api.deepseek.com/anthropic"
export ANTHROPIC_API_KEY="${DS_KEY}"
export ANTHROPIC_AUTH_TOKEN="${DS_KEY}"
export ANTHROPIC_MODEL="deepseek-v4-flash"
echo "  环境变量已生效:"
echo "    BASE_URL=$ANTHROPIC_BASE_URL"
echo "    MODEL=$ANTHROPIC_MODEL"
echo "    KEY=${DS_KEY:0:12}..."

# ---- 5. 测试连通性 ----
echo "[5/5] 测试 DeepSeek V4 API ..."
curl -s -w "\nHTTP: %{http_code}\n" \
    "$ANTHROPIC_BASE_URL/v1/messages" \
    -H "x-api-key: $ANTHROPIC_AUTH_TOKEN" \
    -H "anthropic-version: 2023-06-01" \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4-flash","max_tokens":16,"messages":[{"role":"user","content":"ping"}]}'

echo ""
echo "==================== 完成！输入 claude 启动 ===================="
