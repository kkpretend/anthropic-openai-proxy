# Anthropic to OpenAI Proxy

一个轻量协议适配代理：

- 对外暴露 Anthropic Messages API: `POST /v1/messages`
- 对内转发到 OpenAI Chat Completions: `POST {OPENAI_BASE_URL}/chat/completions`
- API key 从入站请求透传：优先读取 `x-api-key`，其次读取 `Authorization: Bearer ...`
- 支持非流式和 `stream: true` 的 SSE 流式响应
- 支持单独把 Anthropic 请求 JSON 转换成 OpenAI 请求 JSON，不请求上游
- 仅依赖 Python 标准库

## 启动

先在 [proxy.py](/home/administrator/full_process/anthropic-openai-proxy/proxy.py) 顶部填写：

```python
OPENAI_BASE_URL = "http://your-openai-compatible-host/v1"
```

如果 `OPENAI_BASE_URL` 留空字符串，才会读取环境变量 `OPENAI_BASE_URL`。

```bash
cd anthropic-openai-proxy
python3 proxy.py --port 8080
```

也可以把代码里的上游完整写到 `/chat/completions`：

```python
OPENAI_BASE_URL = "http://your-openai-compatible-host/v1/chat/completions"
```

临时覆盖可以用命令行参数：

```bash
python3 proxy.py --openai-base-url "http://your-openai-compatible-host/v1"
```

可选环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OPENAI_BASE_URL` | `http://127.0.0.1:8000/v1` | 仅当代码里的 `OPENAI_BASE_URL` 为空时读取 |
| `HOST` | `0.0.0.0` | 监听地址 |
| `PORT` | `8080` | 监听端口 |
| `UPSTREAM_TIMEOUT_SECONDS` | `600` | 上游请求超时 |
| `DEBUG` | `false` | 打印请求日志，不打印 key |

## Anthropic SDK 调用

把 SDK 的 base URL 指向这个代理即可，key 会透传给上游 OpenAI-compatible 服务。

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="sk-your-upstream-key",
    base_url="http://127.0.0.1:8080",
)

message = client.messages.create(
    model="your-openai-model",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "你好，用一句话介绍你自己。"},
    ],
)

print(message.content[0].text)
```

## 只转换不转发

如果只想检查 Anthropic 请求体会被转换成什么 OpenAI 请求体，可以调用：

```bash
curl http://127.0.0.1:8080/convert/anthropic-to-openai \
  -H 'content-type: application/json' \
  -d '{
    "model": "your-openai-model",
    "system": "You are concise.",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]
  }'
```

返回示例：

```json
{
  "model": "your-openai-model",
  "messages": [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "hello"}
  ],
  "stream": false,
  "max_tokens": 1024
}
```

也支持别名：

```text
/v1/convert/anthropic-to-openai
```

CLI 转换：

```bash
python3 proxy.py --convert anthropic-to-openai < anthropic-request.json
```

## curl 验证

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: sk-your-upstream-key' \
  -d '{
    "model": "your-openai-model",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "hello"}
    ]
  }'
```

流式：

```bash
curl -N http://127.0.0.1:8080/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: sk-your-upstream-key' \
  -d '{
    "model": "your-openai-model",
    "max_tokens": 1024,
    "stream": true,
    "messages": [
      {"role": "user", "content": "写三句话"}
    ]
  }'
```

## 字段转换

Anthropic 入参会转换为 OpenAI 入参：

| Anthropic | OpenAI |
| --- | --- |
| `system` | 第一条 `role=system` message |
| `messages[].content` 文本块 | `messages[].content` |
| `max_tokens` | `max_tokens` |
| `stop_sequences` | `stop` |
| `temperature`, `top_p` | 同名字段 |
| `stream` | `stream` |
| `tools[].input_schema` | `tools[].function.parameters` |
| `tool_choice` | OpenAI `tool_choice` |
| assistant `tool_use` | assistant `tool_calls` |
| user `tool_result` | `role=tool` message |

响应会转换回 Anthropic message 格式：

```json
{
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "..."}],
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 0, "output_tokens": 0}
}
```

OpenAI 上游返回 `tool_calls` 时，会转换为 Anthropic `tool_use` content block；流式 `delta.tool_calls` 会转换为 Anthropic SSE 的 `tool_use` / `input_json_delta`。

如果你需要向 OpenAI-compatible 上游传递额外非标准字段，可以放在 `extra_body` 中，它们会被合并进上游请求体：

```json
{
  "model": "your-model",
  "max_tokens": 1024,
  "extra_body": {
    "enable_thinking": false
  },
  "messages": [{"role": "user", "content": "hello"}]
}
```
