#!/usr/bin/env bash
# =============================================================================
#  AutoDL Claude Code + DeepSeek V4 一键部署
# =============================================================================
#  Usage:
#    bash a.sh                          # 使用内置 API Key
#    bash a.sh --api-key sk-xxx         # 指定 Key
#    DEEPSEEK_KEY=sk-xxx bash a.sh      # 环境变量传入
#    bash a.sh --skip-node              # 跳过 Node.js 安装
#    bash a.sh -h                       # 帮助
# =============================================================================
set -euo pipefail

# ---- color helpers ----
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[FATAL]${NC} $*" >&2; exit 1; }

# ---- defaults ----
API_KEY="${DEEPSEEK_KEY:-sk-8b78a9e5ac8c4aa3bfd114e25a0cc458}"
SKIP_NODE=false
NODE_VERSION=20
CLAUDE_PKG="@anthropic-ai/claude-code"
DEEPSEEK_BASE_URL="https://api.deepseek.com/anthropic"
DEEPSEEK_MODEL="deepseek-v4-flash"

# ---- parse args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)   API_KEY="$2"; shift 2 ;;
        --skip-node) SKIP_NODE=true; shift ;;
        -h|--help)
            sed -n '2,10p' "$0"; exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

# ---- banner ----
cat << 'BANNER'

╔══════════════════════════════════════════════════════╗
║   AutoDL Claude Code + DeepSeek V4 一键部署          ║
╚══════════════════════════════════════════════════════╝
BANNER

# =========================================================================
# Step 1: Install Node.js
# =========================================================================
step1_install_node() {
    info "Step 1/5: Node.js ${NODE_VERSION} ..."

    if command -v node &>/dev/null; then
        info "  已安装: $(node -v)"
        return 0
    fi

    if $SKIP_NODE; then
        warn "  --skip-node 已指定，跳过安装"
        return 0
    fi

    if command -v apt-get &>/dev/null; then
        curl -fsSL "https://deb.nodesource.com/setup_${NODE_VERSION}.x" | bash -
        apt-get install -y nodejs
    elif command -v conda &>/dev/null; then
        conda install -y -c conda-forge "nodejs=${NODE_VERSION}"
    else
        die "无法自动安装 Node.js，请手动安装或用 --skip-node 跳过"
    fi

    # reload PATH (node/npm might be in new location)
    export PATH="/usr/local/bin:$PATH"
    command -v node &>/dev/null || die "Node.js 安装后仍不可用"
    info "  Node.js 安装完成: $(node -v)"
}

step1_install_node

# =========================================================================
# Step 2: npm mirror
# =========================================================================
step2_npm_mirror() {
    info "Step 2/5: npm 国内镜像 ..."
    npm config set registry https://registry.npmmirror.com
    info "  registry -> npmmirror.com"
}
step2_npm_mirror

# =========================================================================
# Step 3: Install Claude Code
# =========================================================================
step3_install_claude() {
    info "Step 3/5: 安装 Claude Code ..."
    npm install -g "$CLAUDE_PKG"
    info "  $(claude --version 2>&1 | head -1)"
}
step3_install_claude

# =========================================================================
# Step 4: Configure DeepSeek
# =========================================================================
step4_config_deepseek() {
    info "Step 4/5: 配置 DeepSeek API ..."

    local rc=~/.bashrc

    if ! grep -q "ANTHROPIC_BASE_URL" "$rc" 2>/dev/null; then
        cat >> "$rc" << 'BASHRC_EOF'

# === DeepSeek API (Claude Code) ===
export ANTHROPIC_BASE_URL="__URL__"
export ANTHROPIC_API_KEY="__KEY__"
export ANTHROPIC_AUTH_TOKEN="__KEY__"
export ANTHROPIC_MODEL="__MODEL__"
BASHRC_EOF
        sed -i "s|__URL__|${DEEPSEEK_BASE_URL}|" "$rc"
        sed -i "s|__KEY__|${API_KEY}|g" "$rc"
        sed -i "s|__MODEL__|${DEEPSEEK_MODEL}|" "$rc"
        info "  已写入 ~/.bashrc"
    else
        warn "  配置已存在，跳过写入"
    fi

    # export for current shell (no source to avoid PS1 unbound error)
    export ANTHROPIC_BASE_URL="$DEEPSEEK_BASE_URL"
    export ANTHROPIC_API_KEY="$API_KEY"
    export ANTHROPIC_AUTH_TOKEN="$API_KEY"
    export ANTHROPIC_MODEL="$DEEPSEEK_MODEL"

    info "  URL:   $ANTHROPIC_BASE_URL"
    info "  Model: $ANTHROPIC_MODEL"
    info "  Key:   ${API_KEY:0:12}***"
}
step4_config_deepseek

# =========================================================================
# Step 5: Test connectivity
# =========================================================================
step5_test_api() {
    info "Step 5/5: 测试 API 连通性 ..."

    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        "${ANTHROPIC_BASE_URL}/v1/messages" \
        -H "x-api-key: ${ANTHROPIC_AUTH_TOKEN}" \
        -H "anthropic-version: 2023-06-01" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"${DEEPSEEK_MODEL}\",\"max_tokens\":16,\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}]}" \
        2>/dev/null || echo "000")

    case "$http_code" in
        200) info "  API 连通正常 (HTTP 200)" ;;
        401) warn "  API Key 无效 (HTTP 401)，请检查 Key" ;;
        *)   warn "  HTTP ${http_code} — 请检查网络和 Key" ;;
    esac
}
step5_test_api

# ---- done ----
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  部署完成！输入 claude 启动                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
