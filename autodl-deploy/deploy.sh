#!/bin/bash
# =============================================================================
#  AutoDL 一键部署脚本：DeepSeek API 驱动的 AI 编程助手
# =============================================================================
#
# 使用方法（在 AutoDL 实例的终端中执行）：
#   bash deploy.sh
#
# 部署内容：
#   1. OpenAI 兼容 API 代理（端口 8000）—— 将 DeepSeek API 包装为标准格式
#   2. Web 编程助手界面（端口 7860）—— 基于 Gradio 的聊天式代码助手
#   3. CLI 命令行工具 —— 终端中直接调用 AI
#   4. systemd 服务 —— 实例重启后自动恢复
#
# 前置条件：
#   - AutoDL GRID 实例（推荐 A100/4090，但纯 API 调用不需要 GPU）
#   - DeepSeek API Key（在 https://platform.deepseek.com 获取）
#
# =============================================================================
set -euo pipefail

# ── 命令行参数解析 ───────────────────────────────────────────────────
usage() {
    echo "Usage: bash deploy.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --api-key KEY    DeepSeek API Key（也可用环境变量 DEEPSEEK_API_KEY）"
    echo "  --port PROXY     代理端口（默认 8000）"
    echo "  --web-port PORT  Web UI 端口（默认 7860）"
    echo "  --install-dir DIR 安装目录（默认 /root/ai-assistant）"
    echo "  -h, --help       显示帮助"
    echo ""
    echo "Examples:"
    echo "  bash deploy.sh --api-key sk-xxx"
    echo "  DEEPSEEK_API_KEY=sk-xxx bash deploy.sh"
    echo "  bash deploy.sh --api-key sk-xxx --port 9000"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --api-key)      DEEPSEEK_API_KEY="$2"; shift 2 ;;
        --port)         PROXY_PORT="$2"; shift 2 ;;
        --web-port)     WEB_PORT="$2"; shift 2 ;;
        --install-dir)  INSTALL_DIR="$2"; shift 2 ;;
        -h|--help)      usage ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

# ── 配色 ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()  { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── 配置 ─────────────────────────────────────────────────────────────
INSTALL_DIR="${INSTALL_DIR:-/root/ai-assistant}"
VENV_DIR="$INSTALL_DIR/venv"
PROXY_PORT="${PROXY_PORT:-8000}"
WEB_PORT="${WEB_PORT:-7860}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-sk-8b78a9e5ac8c4aa3bfd114e25a0cc458}"
LOG_DIR="$INSTALL_DIR/logs"
mkdir -p "$INSTALL_DIR" "$LOG_DIR"

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     AutoDL AI 编程助手 —— DeepSeek API 一键部署             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Step 1: 检查环境 ────────────────────────────────────────────────
log "Step 1/6: 检查系统环境 ..."
OS=$(grep -oP 'PRETTY_NAME="\K[^"]+' /etc/os-release 2>/dev/null || echo "Unknown")
echo "  OS: $OS"
echo "  Python: $(python3 --version 2>/dev/null || echo 'NOT FOUND')"
echo "  pip: $(pip3 --version 2>/dev/null || echo 'NOT FOUND')"
echo "  CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU-only')"

# 确保基础包
pip3 install --quiet --upgrade pip setuptools wheel 2>/dev/null || true

# ── Step 2: 创建虚拟环境 ────────────────────────────────────────────
log "Step 2/6: 创建 Python 虚拟环境 ..."
python3 -m venv "$VENV_DIR" || err "创建虚拟环境失败，请检查 Python 安装"
source "$VENV_DIR/bin/activate"

# ── Step 3: 安装依赖 ────────────────────────────────────────────────
log "Step 3/6: 安装依赖包 ..."
pip install --quiet \
    "fastapi>=0.110.0" \
    "uvicorn[standard]>=0.29.0" \
    "httpx>=0.27.0" \
    "openai>=1.30.0" \
    "gradio>=4.30.0" \
    "python-dotenv>=1.0.0" \
    "pydantic>=2.0.0" \
    "rich>=13.0.0" \
    "aiofiles>=23.0.0" \
    2>&1 | tail -1

echo "  Dependencies installed OK"

# ── Step 4: 生成配置文件 ────────────────────────────────────────────
log "Step 4/6: 配置 DeepSeek API ..."

# 交互式获取 API Key
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo ""
    echo -e "  ${YELLOW}请输入你的 DeepSeek API Key${NC}"
    echo -e "  ${YELLOW}（在 https://platform.deepseek.com/api_keys 获取）${NC}"
    echo -n "  API Key: "
    read -r DEEPSEEK_API_KEY
fi

if [ -z "$DEEPSEEK_API_KEY" ]; then
    err "API Key 不能为空"
fi

# 写入 .env 文件
cat > "$INSTALL_DIR/.env" << EOF
# DeepSeek API Configuration
DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# Service ports
PROXY_PORT=$PROXY_PORT
WEB_PORT=$WEB_PORT
EOF
chmod 600 "$INSTALL_DIR/.env"
log "API Key 已保存到 $INSTALL_DIR/.env"

# ── Step 5: 写入服务代码 ────────────────────────────────────────────
log "Step 5/6: 部署服务代码 ..."

# --- OpenAI 兼容代理服务器 ---
cat > "$INSTALL_DIR/proxy_server.py" << 'PROXY_PY'
#!/usr/bin/env python3
"""
OpenAI-compatible API proxy using DeepSeek backend.

Supports:
  - POST /v1/chat/completions  (OpenAI format -> DeepSeek)
  - POST /v1/completions        (OpenAI format -> DeepSeek)
  - GET  /v1/models             (List available models)
  - GET  /health                (Health check)
"""

import os, sys, json, time, logging
from typing import Optional

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

# ---- Config ----
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "8000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("proxy")

app = FastAPI(title="DeepSeek API Proxy", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ---- Routes ----

@app.get("/health")
async def health():
    return {"status": "ok", "backend": "deepseek", "model": DEEPSEEK_MODEL}

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": DEEPSEEK_MODEL, "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-chat", "object": "model", "owned_by": "deepseek"},
            {"id": "deepseek-reasoner", "object": "model", "owned_by": "deepseek"},
        ],
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Forward chat completion requests to DeepSeek API."""
    body = await request.json()
    api_key = request.headers.get("Authorization", "").replace("Bearer ", "") or DEEPSEEK_API_KEY

    # Force model to DeepSeek if not specified
    if "model" not in body or not body["model"]:
        body["model"] = DEEPSEEK_MODEL
    elif body["model"] not in ("deepseek-chat", "deepseek-reasoner"):
        body["model"] = DEEPSEEK_MODEL  # map any model name to deepseek

    # Remove unsupported params
    for key in ["response_format", "tool_choice"]:
        body.pop(key, None)

    # Remove function calling tools (DeepSeek v3 supports but let's keep simple)
    # Actually, keep them - DeepSeek supports tools now

    stream = body.get("stream", False)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            if stream:
                return await _stream_response(client, body, headers)
            else:
                resp = await client.post(
                    f"{DEEPSEEK_BASE_URL}/chat/completions",
                    json=body, headers=headers,
                )
                return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="DeepSeek API timeout")
    except Exception as e:
        logger.error(f"Proxy error: {e}")
        raise HTTPException(status_code=502, detail=f"DeepSeek API error: {str(e)}")


async def _stream_response(client, body, headers):
    """Stream chunks from DeepSeek back to client."""
    async def generate():
        async with client.stream(
            "POST",
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            json=body, headers=headers,
        ) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk
    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    """Forward text completion requests (maps to chat completions)."""
    body = await request.json()
    prompt = body.pop("prompt", "")
    body["messages"] = [{"role": "user", "content": prompt}]
    body["model"] = DEEPSEEK_MODEL
    return await chat_completions(request)


if __name__ == "__main__":
    logger.info(f"Starting DeepSeek API Proxy on 0.0.0.0:{PROXY_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")
PROXY_PY

# --- Gradio Web UI ---
cat > "$INSTALL_DIR/web_ui.py" << 'WEBUI_PY'
#!/usr/bin/env python3
"""Gradio-based AI coding assistant web UI."""

import os, sys, json, time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import gradio as gr
from openai import OpenAI

# ---- Config ----
API_KEY = os.environ["DEEPSEEK_API_KEY"]
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
WEB_PORT = int(os.environ.get("WEB_PORT", "7860"))

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """你是一个专业的AI编程助手，擅长Python、JavaScript、Go、Rust等语言。
提供代码时请：
1. 给出完整可运行的代码
2. 简短注释关键逻辑
3. 指出潜在的性能问题和安全隐患
4. 代码风格简洁清晰"""


def chat(message: str, history: list, system_prompt: str = SYSTEM_PROMPT):
    """Handle one chat turn."""
    if not message.strip():
        yield history
        return

    messages = [{"role": "system", "content": system_prompt}]
    for h in history or []:
        messages.append({"role": "user", "content": h[0]})
        if h[1]:
            messages.append({"role": "assistant", "content": h[1]})
    messages.append({"role": "user", "content": message})

    try:
        stream = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            stream=True,
            temperature=0.3,
            max_tokens=4096,
        )
        response = ""
        for chunk in stream:
            if chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content
                yield response
    except Exception as e:
        yield f"**错误**: {str(e)}"


def code_review(code: str, language: str = "auto"):
    """AI code review."""
    if not code.strip():
        return "请粘贴需要审查的代码"
    prompt = f"请审查以下{language}代码，找出bug、性能问题和安全隐患，给出改进建议：\n\n```{language}\n{code}\n```"
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是资深代码审查专家，请详细分析代码问题。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"**错误**: {str(e)}"


def explain_code(code: str):
    """Explain what the code does."""
    if not code.strip():
        return "请粘贴需要解释的代码"
    prompt = f"请详细解释以下代码的功能和实现原理：\n\n```\n{code}\n```"
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是耐心的编程导师，擅长用通俗语言解释复杂代码。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"**错误**: {str(e)}"


# ---- UI ----
with gr.Blocks(
    title="AI 编程助手",
    theme=gr.themes.Soft(),
    css="footer {display: none !important}",
) as demo:
    gr.Markdown("""
    # AI 编程助手
    **后端**: DeepSeek API | **模型**: {} | `Ctrl+Enter` 发送
    """.format(MODEL))

    with gr.Tabs():
        # Tab 1: Chat
        with gr.TabItem("对话编程"):
            with gr.Row():
                system_input = gr.Textbox(
                    label="System Prompt", value=SYSTEM_PROMPT, lines=3,
                    info="自定义助手行为",
                )
            chatbot = gr.Chatbot(label="对话", height=500)
            msg = gr.Textbox(
                label="输入你的问题", placeholder="例如：用Python写一个快速排序...",
                lines=3,
            )
            with gr.Row():
                send_btn = gr.Button("发送", variant="primary")
                clear_btn = gr.Button("清空对话")

            send_btn.click(
                chat, [msg, chatbot, system_input], [chatbot],
                queue=True, show_progress="minimal",
            ).then(lambda: "", None, [msg])
            msg.submit(
                chat, [msg, chatbot, system_input], [chatbot],
                queue=True, show_progress="minimal",
            ).then(lambda: "", None, [msg])
            clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

        # Tab 2: Code Review
        with gr.TabItem("代码审查"):
            gr.Markdown("粘贴代码，AI 帮你找 Bug + 安全漏洞 + 性能问题")
            with gr.Row():
                lang = gr.Dropdown(
                    ["auto", "python", "javascript", "go", "rust", "java", "c++", "sql"],
                    value="auto", label="语言",
                )
            review_input = gr.Code(label="待审查代码", language="python", lines=15)
            review_btn = gr.Button("开始审查", variant="primary")
            review_output = gr.Markdown(label="审查结果")
            review_btn.click(code_review, [review_input, lang], [review_output])

        # Tab 3: Explain Code
        with gr.TabItem("代码解释"):
            gr.Markdown("粘贴代码，AI 用通俗语言解释逻辑")
            explain_input = gr.Code(label="待解释代码", language="python", lines=15)
            explain_btn = gr.Button("解释代码", variant="primary")
            explain_output = gr.Markdown(label="解释结果")
            explain_btn.click(explain_code, [explain_input], [explain_output])

    gr.Markdown("---\nPowered by DeepSeek API | AutoDL Deployment")

demo.queue(max_size=32).launch(
    server_name="0.0.0.0",
    server_port=WEB_PORT,
    share=False,  # 用 AutoDL 的自定义服务端口映射
    show_error=True,
)
WEBUI_PY

# --- CLI Tool ---
cat > "$INSTALL_DIR/ai" << 'CLI_PY'
#!/usr/bin/env python3
"""CLI AI coding assistant - direct DeepSeek API calls."""
import os, sys, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
from openai import OpenAI

API_KEY = os.environ["DEEPSEEK_API_KEY"]
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def main():
    if len(sys.argv) < 2:
        print("Usage: ai <question>")
        print("  ai 'write a quicksort in Python'")
        print("  ai --review file.py")
        print("  ai --explain code.py")
        sys.exit(1)

    if sys.argv[1] == "--review" and len(sys.argv) > 2:
        code = Path(sys.argv[2]).read_text()
        prompt = f"Review this code for bugs, performance, and security:\\n\\n```\\n{code}\\n```"
    elif sys.argv[1] == "--explain" and len(sys.argv) > 2:
        code = Path(sys.argv[2]).read_text()
        prompt = f"Explain what this code does:\\n\\n```\\n{code}\\n```"
    else:
        prompt = " ".join(sys.argv[1:])

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful coding assistant. Reply in Chinese if the user asks in Chinese."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    print(resp.choices[0].message.content)

if __name__ == "__main__":
    main()
CLI_PY
chmod +x "$INSTALL_DIR/ai"

# ── Step 6: 配置 systemd 服务 ───────────────────────────────────────
log "Step 6/6: 配置自启动服务 ..."

cat > /etc/systemd/system/ai-proxy.service << EOF
[Unit]
Description=DeepSeek API Proxy Service
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$VENV_DIR/bin/python $INSTALL_DIR/proxy_server.py
Restart=always
RestartSec=5
StandardOutput=append:$LOG_DIR/proxy.log
StandardError=append:$LOG_DIR/proxy.log

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/ai-webui.service << EOF
[Unit]
Description=AI Coding Assistant Web UI
After=network.target ai-proxy.service
Wants=ai-proxy.service

[Service]
Type=simple
User=root
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$VENV_DIR/bin:/usr/local/bin:/usr/bin:/bin
EnvironmentFile=$INSTALL_DIR/.env
ExecStart=$VENV_DIR/bin/python $INSTALL_DIR/web_ui.py
Restart=always
RestartSec=5
StandardOutput=append:$LOG_DIR/webui.log
StandardError=append:$LOG_DIR/webui.log

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ai-proxy ai-webui
systemctl start ai-proxy ai-webui

# 创建 CLI 软链接
ln -sf "$INSTALL_DIR/ai" /usr/local/bin/ai 2>/dev/null || true

# ── 验证部署 ────────────────────────────────────────────────────────
log ""
log "=============================================="
log "  部署完成！"
log "=============================================="
log ""
log "  API Proxy:  http://localhost:${PROXY_PORT}"
log "    测试:     curl http://localhost:${PROXY_PORT}/health"
log ""
log "  Web UI:     http://localhost:${WEB_PORT}"
log "    在 AutoDL 中: 点击「自定义服务」查看外网地址"
log ""
log "  CLI 工具:   ai '你的问题'"
log "    代码审查:  ai --review file.py"
log "    代码解释:  ai --explain code.py"
log ""
log "  管理命令:"
log "    systemctl status ai-proxy   # 查看代理状态"
log "    systemctl status ai-webui   # 查看 Web UI 状态"
log "    systemctl restart ai-proxy  # 重启代理"
log "    journalctl -u ai-proxy -f   # 查看代理日志"
log ""
log "  API Key 配置: $INSTALL_DIR/.env"

# 测试代理是否正常
sleep 2
if curl -s "http://localhost:${PROXY_PORT}/health" 2>/dev/null | grep -q "ok"; then
    echo -e "\n${GREEN}✓ API Proxy 运行正常${NC}"
else
    echo -e "\n${YELLOW}⚠ API Proxy 可能未就绪，请手动检查${NC}"
fi

echo ""
echo -e "${BLUE}部署脚本执行完毕。${NC}"
