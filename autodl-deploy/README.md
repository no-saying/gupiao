# AutoDL + DeepSeek API 一键部署 AI 编程助手

## 快速开始

在 AutoDL 实例的终端中执行：

```bash
# 1. 下载部署脚本
wget https://your-server/deploy.sh -O deploy.sh
# 或者手动上传 autodl-deploy/ 目录到 /root/

# 2. 运行
bash deploy.sh

# 3. 输入 DeepSeek API Key（在 https://platform.deepseek.com/api_keys 获取）
```

## 部署后

| 服务 | 端口 | 用途 |
|------|------|------|
| API Proxy | 8000 | OpenAI 兼容 API 端点 |
| Web UI | 7860 | Gradio 编程助手界面 |
| CLI | - | `ai` 命令直接调用 |

AutoDL 访问 Web UI：实例列表 → 自定义服务 → 查看 7860 端口的外网 URL。

## 使用方式

### Web UI
浏览器打开 AutoDL 提供的外网地址（端口 7860），支持：
- **对话编程**：聊天式代码生成
- **代码审查**：粘贴代码 → AI 找 Bug + 安全漏洞
- **代码解释**：粘贴代码 → AI 解释逻辑

### CLI
```bash
ai '用Python写一个快速排序'
ai --review app.py      # 审查代码
ai --explain utils.py   # 解释代码
```

### API Proxy
兼容 OpenAI SDK：
```python
from openai import OpenAI
client = OpenAI(
    api_key="any-key",  # 代理不做鉴权
    base_url="http://localhost:8000/v1"
)
resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello"}],
)
```

### 配置 Claude Code 使用代理
```bash
export ANTHROPIC_BASE_URL=http://localhost:8000/v1
export ANTHROPIC_API_KEY=any-key
```

## 管理

```bash
systemctl status ai-proxy      # 查看代理状态
systemctl status ai-webui      # 查看 Web UI 状态
systemctl restart ai-proxy     # 重启代理
journalctl -u ai-proxy -f      # 实时日志

# 修改 API Key
vim /root/ai-assistant/.env
systemctl restart ai-proxy ai-webui
```

## 文件结构

```
/root/ai-assistant/
├── .env                # API Key 配置
├── proxy_server.py     # API 代理服务
├── web_ui.py           # Gradio Web UI
├── ai                  # CLI 工具
├── logs/               # 运行日志
└── venv/               # Python 虚拟环境
```
