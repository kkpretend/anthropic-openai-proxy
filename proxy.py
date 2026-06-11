#!/usr/bin/env python3
"""Anthropic Messages API to OpenAI Chat Completions proxy.

The service exposes an Anthropic-compatible /v1/messages endpoint and forwards
requests to an OpenAI-compatible /chat/completions endpoint. Incoming API keys
are passed through to the upstream service.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib import error, request


# Fill this with your OpenAI-compatible upstream. If left empty, the service
# falls back to the OPENAI_BASE_URL environment variable.
OPENAI_BASE_URL = "https://xhyapi.com"

DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080
DEFAULT_TIMEOUT_SECONDS = 600
ANTHROPIC_MESSAGES_PATH = "/v1/messages"
CONVERT_ANTHROPIC_TO_OPENAI_PATHS = {
    "/convert/anthropic-to-openai",
    "/v1/convert/anthropic-to-openai",
}
CONVERT_OPENAI_TO_ANTHROPIC_PATHS = {
    "/convert/openai-to-anthropic",
    "/v1/convert/openai-to-anthropic",
}
CONVERT_AUTO_PATHS = {
    "/convert/auto",
    "/v1/convert/auto",
}


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def compact_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def normalize_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/responses"):
        return base_url[: -len("/responses")]
    return base_url


def chat_completions_url(base_url: str) -> str:
    return f"{normalize_base_url(base_url)}/responses"


def resolve_openai_base_url(cli_base_url: str | None = None) -> str:
    if cli_base_url:
        return cli_base_url
    if OPENAI_BASE_URL:
        return OPENAI_BASE_URL
    return os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)


def get_incoming_api_key(headers: Any) -> str | None:
    api_key = headers.get("x-api-key")
    if api_key:
        return api_key.strip()

    authorization = headers.get("authorization")
    if not authorization:
        return None

    authorization = authorization.strip()
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return authorization


def anthropic_error(error_type: str, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }


def openai_finish_to_anthropic(finish_reason: str | None) -> str | None:
    if finish_reason is None:
        return None
    return {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "stop_sequence",
        "tool_calls": "tool_use",
        "function_call": "tool_use",
    }.get(finish_reason, finish_reason)


def anthropic_stop_to_openai_stop(payload: dict[str, Any]) -> Any:
    if "stop_sequences" in payload:
        return payload["stop_sequences"]
    if "stop_sequence" in payload:
        return payload["stop_sequence"]
    return None


def content_block_to_openai(block: Any) -> Any:
    if isinstance(block, str):
        return {"type": "text", "text": block}
    if not isinstance(block, dict):
        return {"type": "text", "text": str(block)}

    block_type = block.get("type")
    if block_type == "text":
        return {"type": "text", "text": block.get("text", "")}

    if block_type == "image":
        source = block.get("source") or {}
        if source.get("type") == "base64":
            media_type = source.get("media_type") or "application/octet-stream"
            data = source.get("data") or ""
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            }
        if source.get("type") == "url":
            return {
                "type": "image_url",
                "image_url": {"url": source.get("url") or ""},
            }

    # Keep unknown content blocks readable for text-only upstream models instead
    # of silently dropping user input.
    return {"type": "text", "text": json.dumps(block, ensure_ascii=False)}


def anthropic_content_to_openai(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        converted = [content_block_to_openai(block) for block in content]
        if all(isinstance(item, dict) and item.get("type") == "text" for item in converted):
            return "".join(item.get("text", "") for item in converted)
        return converted
    return str(content)


def tool_result_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(json.dumps(block, ensure_ascii=False))
        return "".join(parts)
    return json.dumps(content, ensure_ascii=False)


def json_dumps_object(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value if value is not None else {}, ensure_ascii=False, separators=(",", ":"))


def parse_tool_arguments(arguments: Any) -> Any:
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    if not isinstance(arguments, str):
        return arguments
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"arguments": arguments}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def anthropic_tool_use_to_openai_tool_call(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": block.get("id") or f"toolu_{uuid.uuid4().hex}",
        "type": "function",
        "function": {
            "name": block.get("name") or "tool",
            "arguments": json_dumps_object(block.get("input")),
        },
    }


def anthropic_message_to_openai_messages(message: dict[str, Any]) -> list[dict[str, Any]]:
    role = message.get("role")
    if role not in {"user", "assistant", "system", "tool"}:
        role = "user"

    content = message.get("content")

    if role == "user" and isinstance(content, list):
        regular_blocks: list[Any] = []
        tool_messages: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id") or block.get("id") or "",
                        "content": tool_result_content_to_text(block.get("content")),
                    }
                )
            else:
                regular_blocks.append(block)

        messages: list[dict[str, Any]] = []
        if regular_blocks:
            messages.append({"role": "user", "content": anthropic_content_to_openai(regular_blocks)})
        messages.extend(tool_messages)
        return messages or [{"role": "user", "content": ""}]

    if role == "assistant" and isinstance(content, list):
        regular_blocks: list[Any] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_calls.append(anthropic_tool_use_to_openai_tool_call(block))
            else:
                regular_blocks.append(block)

        openai_message: dict[str, Any] = {
            "role": "assistant",
            "content": anthropic_content_to_openai(regular_blocks) if regular_blocks else None,
        }
        if tool_calls:
            openai_message["tool_calls"] = tool_calls
        return [openai_message]

    return [{"role": role, "content": anthropic_content_to_openai(content)}]


def system_to_openai_messages(system: Any) -> list[dict[str, Any]]:
    if system is None:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    if isinstance(system, list):
        return [{"role": "system", "content": anthropic_content_to_openai(system)}]
    return [{"role": "system", "content": str(system)}]


def anthropic_messages_to_openai(payload: dict[str, Any]) -> dict[str, Any]:
    messages = system_to_openai_messages(payload.get("system"))

    for message in payload.get("messages", []):
        if not isinstance(message, dict):
            continue
        messages.extend(anthropic_message_to_openai_messages(message))

    openai_payload: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": messages,
        "stream": bool(payload.get("stream", False)),
    }

    field_map = {
        "max_tokens": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "n": "n",
        "response_format": "response_format",
        "seed": "seed",
        "user": "user",
    }
    for anthropic_key, openai_key in field_map.items():
        if anthropic_key in payload and payload[anthropic_key] is not None:
            openai_payload[openai_key] = payload[anthropic_key]

    stop = anthropic_stop_to_openai_stop(payload)
    if stop is not None:
        openai_payload["stop"] = stop

    if "tools" in payload:
        openai_payload["tools"] = anthropic_tools_to_openai(payload["tools"])
    if "tool_choice" in payload:
        openai_payload["tool_choice"] = anthropic_tool_choice_to_openai(payload["tool_choice"])

    extra_body = payload.get("extra_body")
    if isinstance(extra_body, dict):
        openai_payload.update(extra_body)

    return {key: value for key, value in openai_payload.items() if value is not None}


def anthropic_tools_to_openai(tools: Any) -> Any:
    if not isinstance(tools, list):
        return tools

    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            converted.append(tool)
            continue
        name = tool.get("name")
        if not name:
            continue
        function: dict[str, Any] = {
            "name": name,
            "parameters": tool.get("input_schema") or {"type": "object", "properties": {}},
        }
        if tool.get("description"):
            function["description"] = tool["description"]
        converted.append({"type": "function", "function": function})
    return converted


def anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        return tool_choice
    if not isinstance(tool_choice, dict):
        return tool_choice

    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "none":
        return "none"
    if choice_type == "any":
        return "required"
    if choice_type == "tool":
        return {
            "type": "function",
            "function": {"name": tool_choice.get("name") or ""},
        }
    return tool_choice


def openai_finish_to_anthropic_stop(finish_reason: str | None) -> str | None:
    return openai_finish_to_anthropic(finish_reason)


def anthropic_stop_reason_to_openai_finish(stop_reason: str | None) -> str | None:
    if stop_reason is None:
        return None
    return {
        "end_turn": "stop",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "tool_use": "tool_calls",
    }.get(stop_reason, stop_reason)


def openai_stop_to_anthropic_stop_sequences(stop: Any) -> list[str] | None:
    if stop is None:
        return None
    if isinstance(stop, list):
        return [str(item) for item in stop]
    return [str(stop)]


def data_uri_to_anthropic_image_source(url: str) -> dict[str, Any]:
    if url.startswith("data:") and ";base64," in url:
        header, data = url.split(";base64,", 1)
        media_type = header[5:] or "application/octet-stream"
        return {"type": "base64", "media_type": media_type, "data": data}
    return {"type": "url", "url": url}


def openai_content_block_to_anthropic(block: Any) -> dict[str, Any]:
    if isinstance(block, str):
        return {"type": "text", "text": block}
    if not isinstance(block, dict):
        return {"type": "text", "text": str(block)}

    block_type = block.get("type")
    if block_type in {"text", "input_text"}:
        return {"type": "text", "text": block.get("text", "")}

    if block_type in {"image_url", "input_image"}:
        image_url = block.get("image_url") or block.get("image") or {}
        url = image_url.get("url") if isinstance(image_url, dict) else image_url
        return {"type": "image", "source": data_uri_to_anthropic_image_source(str(url or ""))}

    return {"type": "text", "text": json.dumps(block, ensure_ascii=False)}


def openai_content_to_anthropic(content: Any) -> Any:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        converted = [openai_content_block_to_anthropic(block) for block in content]
        if all(block.get("type") == "text" for block in converted):
            return "".join(block.get("text", "") for block in converted)
        return converted
    return str(content)


def content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type in {"text", "input_text"}:
                    parts.append(block.get("text", ""))
                elif block_type == "image_url":
                    parts.append(json.dumps(block, ensure_ascii=False))
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def openai_tool_call_to_anthropic_tool_use(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    return {
        "type": "tool_use",
        "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex}",
        "name": function.get("name") or "tool",
        "input": parse_tool_arguments(function.get("arguments")),
    }


def openai_message_to_anthropic_message(message: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    role = message.get("role")
    content = message.get("content")

    if role in {"system", "developer"}:
        return None, content_to_text(content)

    if role == "tool":
        return (
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.get("tool_call_id") or message.get("id") or "",
                        "content": content_to_text(content),
                    }
                ],
            },
            None,
        )

    if role == "assistant":
        content_blocks: list[dict[str, Any]] = []
        converted_content = openai_content_to_anthropic(content)
        if isinstance(converted_content, str):
            if converted_content:
                content_blocks.append({"type": "text", "text": converted_content})
        elif isinstance(converted_content, list):
            content_blocks.extend(converted_content)

        for tool_call in message.get("tool_calls") or []:
            if isinstance(tool_call, dict):
                content_blocks.append(openai_tool_call_to_anthropic_tool_use(tool_call))

        if content_blocks:
            return {"role": "assistant", "content": content_blocks}, None
        return {"role": "assistant", "content": ""}, None

    anthropic_role = "assistant" if role == "assistant" else "user"
    return {"role": anthropic_role, "content": openai_content_to_anthropic(content)}, None


def openai_tools_to_anthropic(tools: Any) -> Any:
    if not isinstance(tools, list):
        return tools

    converted = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("input_schema"):
            converted.append(tool)
            continue
        function = tool.get("function") or {}
        name = function.get("name") or tool.get("name")
        if not name:
            continue
        anthropic_tool: dict[str, Any] = {
            "name": name,
            "input_schema": function.get("parameters") or {"type": "object", "properties": {}},
        }
        if function.get("description"):
            anthropic_tool["description"] = function["description"]
        converted.append(anthropic_tool)
    return converted


def openai_tool_choice_to_anthropic(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        if tool_choice == "required":
            return {"type": "any"}
        return {"type": tool_choice}
    if not isinstance(tool_choice, dict):
        return tool_choice

    if tool_choice.get("type") == "function":
        function = tool_choice.get("function") or {}
        return {"type": "tool", "name": function.get("name") or ""}
    return tool_choice


def openai_messages_to_anthropic(payload: dict[str, Any]) -> dict[str, Any]:
    anthropic_payload: dict[str, Any] = {
        "model": payload.get("model"),
        "messages": [],
    }

    max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens"))
    if max_tokens is not None:
        anthropic_payload["max_tokens"] = max_tokens

    for key in ("temperature", "top_p", "stream"):
        if key in payload and payload[key] is not None:
            anthropic_payload[key] = payload[key]

    stop_sequences = openai_stop_to_anthropic_stop_sequences(payload.get("stop"))
    if stop_sequences is not None:
        anthropic_payload["stop_sequences"] = stop_sequences

    if payload.get("tools") is not None:
        anthropic_payload["tools"] = openai_tools_to_anthropic(payload["tools"])
    if payload.get("tool_choice") is not None:
        anthropic_payload["tool_choice"] = openai_tool_choice_to_anthropic(payload["tool_choice"])
    if payload.get("user"):
        anthropic_payload["metadata"] = {"user_id": str(payload["user"])}

    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        converted_message, system_text = openai_message_to_anthropic_message(message)
        if system_text:
            system_parts.append(system_text)
        if converted_message is not None:
            messages.append(converted_message)

    if system_parts:
        anthropic_payload["system"] = "\n\n".join(part for part in system_parts if part)
    anthropic_payload["messages"] = messages

    return {key: value for key, value in anthropic_payload.items() if value is not None}


def anthropic_usage_to_openai(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def anthropic_response_to_openai(anthropic_body: dict[str, Any]) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for block in anthropic_body.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "tool_use":
            tool_calls.append(anthropic_tool_use_to_openai_tool_call(block))

    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    if message["content"] is None and not tool_calls:
        message["content"] = ""

    return {
        "id": anthropic_body.get("id") or f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": anthropic_body.get("model"),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": anthropic_stop_reason_to_openai_finish(anthropic_body.get("stop_reason")),
            }
        ],
        "usage": anthropic_usage_to_openai(anthropic_body.get("usage")),
    }


def openai_usage_to_anthropic(usage: dict[str, Any] | None) -> dict[str, int]:
    usage = usage or {}
    return {
        "input_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
    }


def openai_response_to_anthropic(openai_body: dict[str, Any], model: str | None) -> dict[str, Any]:
    choices = openai_body.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    content = message.get("content") or ""

    if isinstance(content, list):
        text = "".join(
            item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
        )
    else:
        text = str(content)

    content_blocks: list[dict[str, Any]] = []
    if text:
        content_blocks.append({"type": "text", "text": text})

    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        content_blocks.append(
            {
                "type": "tool_use",
                "id": tool_call.get("id") or f"toolu_{uuid.uuid4().hex}",
                "name": function.get("name") or "tool",
                "input": parse_tool_arguments(function.get("arguments")),
            }
        )

    if not content_blocks:
        content_blocks.append({"type": "text", "text": ""})

    return {
        "id": openai_body.get("id") or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": openai_body.get("model") or model,
        "content": content_blocks,
        "stop_reason": openai_finish_to_anthropic(choice.get("finish_reason")) or "end_turn",
        "stop_sequence": None,
        "usage": openai_usage_to_anthropic(openai_body.get("usage")),
    }


def content_blocks_contain_type(content: Any, block_types: set[str]) -> bool:
    if not isinstance(content, list):
        return False
    return any(isinstance(block, dict) and block.get("type") in block_types for block in content)


def has_anthropic_tool_schema(payload: dict[str, Any]) -> bool:
    tools = payload.get("tools")
    return isinstance(tools, list) and any(isinstance(tool, dict) and "input_schema" in tool for tool in tools)


def has_openai_tool_schema(payload: dict[str, Any]) -> bool:
    tools = payload.get("tools")
    return isinstance(tools, list) and any(
        isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("function"), dict)
        for tool in tools
    )


def detect_request_format(payload: dict[str, Any]) -> str:
    if payload.get("system") is not None or payload.get("stop_sequences") is not None or payload.get("extra_body"):
        return "anthropic_request"
    if has_anthropic_tool_schema(payload):
        return "anthropic_request"

    tool_choice = payload.get("tool_choice")
    if isinstance(tool_choice, dict) and tool_choice.get("type") in {"any", "tool"}:
        return "anthropic_request"

    if has_openai_tool_schema(payload):
        return "openai_request"
    if any(key in payload for key in ("response_format", "logit_bias", "max_completion_tokens")):
        return "openai_request"
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        return "openai_request"

    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role in {"system", "developer", "tool"}:
            return "openai_request"
        if "tool_calls" in message or "function_call" in message:
            return "openai_request"

        content = message.get("content")
        if content_blocks_contain_type(content, {"tool_use", "tool_result", "image"}):
            return "anthropic_request"
        if content_blocks_contain_type(content, {"image_url", "input_text", "input_image"}):
            return "openai_request"

    # A plain {"model": "...", "messages": [{"role": "user", "content": "..."}]}
    # is valid enough for both APIs. Prefer the proxy's primary use case.
    return "anthropic_request"


def detect_payload_format(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    if payload.get("type") == "message" and isinstance(payload.get("content"), list):
        return "anthropic_response"

    if payload.get("object") in {"chat.completion", "chat.completion.chunk"}:
        return "openai_response"
    if isinstance(payload.get("choices"), list):
        return "openai_response"

    if isinstance(payload.get("messages"), list):
        return detect_request_format(payload)

    raise ValueError("unable to detect Anthropic or OpenAI JSON format")


def convert_anthropic_to_openai_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if detect_payload_format(payload) == "anthropic_response":
        return anthropic_response_to_openai(payload)
    return anthropic_messages_to_openai(payload)


def convert_openai_to_anthropic_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if detect_payload_format(payload) == "openai_response":
        return openai_response_to_anthropic(payload, payload.get("model"))
    return openai_messages_to_anthropic(payload)


def auto_convert_payload(payload: dict[str, Any]) -> dict[str, Any]:
    detected = detect_payload_format(payload)
    if detected in {"anthropic_request", "anthropic_response"}:
        return convert_anthropic_to_openai_payload(payload)
    if detected in {"openai_request", "openai_response"}:
        return convert_openai_to_anthropic_payload(payload)
    raise ValueError(f"unsupported payload format: {detected}")


def build_upstream_headers(api_key: str, incoming_headers: Any, source_format: str = "anthropic") -> dict[str, str]:
    # 转换为 OpenAI 规范时（从 Anthropic → OpenAI）使用 codex
    # 转换为 Anthropic 规范时（从 OpenAI → Anthropic）使用 claude
    user_agent = "codex/1.0" if source_format == "anthropic" else "claude/1.0"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": user_agent,
    }

    organization = incoming_headers.get("openai-organization")
    if organization:
        headers["OpenAI-Organization"] = organization

    project = incoming_headers.get("openai-project")
    if project:
        headers["OpenAI-Project"] = project

    return headers


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = handler.headers.get("Content-Length")
    if not content_length:
        return {}

    raw_body = handler.rfile.read(int(content_length))
    if not raw_body:
        return {}

    body = json.loads(raw_body.decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


class ProxyConfig:
    def __init__(self, openai_base_url: str, timeout_seconds: int, debug: bool = False):
        self.openai_base_url = openai_base_url
        self.timeout_seconds = timeout_seconds
        self.debug = debug


class AnthropicOpenAIProxyHandler(BaseHTTPRequestHandler):
    server_version = "anthropic-openai-proxy/0.1"
    protocol_version = "HTTP/1.1"

    @property
    def config(self) -> ProxyConfig:
        return self.server.config  # type: ignore[attr-defined]

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_json(HTTPStatus.OK, {"ok": True})
            return
        self.send_json(HTTPStatus.NOT_FOUND, anthropic_error("not_found_error", "not found"))

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in CONVERT_ANTHROPIC_TO_OPENAI_PATHS:
            self.convert_payload("anthropic-to-openai")
            return
        if path in CONVERT_OPENAI_TO_ANTHROPIC_PATHS:
            self.convert_payload("openai-to-anthropic")
            return
        if path in CONVERT_AUTO_PATHS:
            self.convert_payload("auto")
            return

        if path != ANTHROPIC_MESSAGES_PATH:
            self.send_json(HTTPStatus.NOT_FOUND, anthropic_error("not_found_error", "not found"))
            return

        try:
            payload = read_json_body(self)
            api_key = get_incoming_api_key(self.headers)
            if not api_key:
                self.send_json(
                    HTTPStatus.UNAUTHORIZED,
                    anthropic_error("authentication_error", "missing x-api-key or Authorization header"),
                )
                return

            openai_payload = anthropic_messages_to_openai(payload)
            if self.config.debug:
                source_format = "anthropic"  # 主要路径是 Anthropic → OpenAI
                ua = "codex/1.0" if source_format == "anthropic" else "claude/1.0"
                print(
                    f"forwarding /v1/messages model={openai_payload.get('model')} "
                    f"stream={openai_payload.get('stream')} upstream={chat_completions_url(self.config.openai_base_url)} "
                    f"User-Agent={ua}",
                    flush=True,
                )

            if openai_payload.get("stream"):
                self.proxy_streaming(openai_payload, api_key)
            else:
                self.proxy_non_streaming(openai_payload, api_key)

        except json.JSONDecodeError:
            self.send_json(HTTPStatus.BAD_REQUEST, anthropic_error("invalid_request_error", "invalid JSON body"))
        except ValueError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, anthropic_error("invalid_request_error", str(exc)))
        except UpstreamHTTPError as exc:
            self.send_json(exc.status, anthropic_error("api_error", exc.body or str(exc)))
        except Exception as exc:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, anthropic_error("api_error", str(exc)))

    def convert_payload(self, mode: str) -> None:
        try:
            payload = read_json_body(self)
            if mode == "anthropic-to-openai":
                converted = convert_anthropic_to_openai_payload(payload)
            elif mode == "openai-to-anthropic":
                converted = convert_openai_to_anthropic_payload(payload)
            elif mode == "auto":
                converted = auto_convert_payload(payload)
            else:
                raise ValueError(f"unknown conversion mode: {mode}")
            self.send_json(HTTPStatus.OK, converted)
        except json.JSONDecodeError:
            self.send_json(HTTPStatus.BAD_REQUEST, anthropic_error("invalid_request_error", "invalid JSON body"))
        except ValueError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, anthropic_error("invalid_request_error", str(exc)))
        except Exception as exc:
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, anthropic_error("api_error", str(exc)))

    def proxy_non_streaming(self, openai_payload: dict[str, Any], api_key: str) -> None:
        upstream_body = self.call_upstream(openai_payload, api_key)
        response = openai_response_to_anthropic(upstream_body, openai_payload.get("model"))
        self.send_json(HTTPStatus.OK, response)

    def proxy_streaming(self, openai_payload: dict[str, Any], api_key: str) -> None:
        req = request.Request(
            chat_completions_url(self.config.openai_base_url),
            data=compact_json(openai_payload),
            headers=build_upstream_headers(api_key, self.headers),
            method="POST",
        )

        try:
            upstream = request.urlopen(req, timeout=self.config.timeout_seconds)
        except error.HTTPError as exc:
            self.send_upstream_http_error(exc)
            return

        message_id = f"msg_{uuid.uuid4().hex}"
        model = openai_payload.get("model")
        output_tokens = 0
        stop_reason = "end_turn"
        next_content_index = 0
        text_block_index: int | None = None
        tool_states: dict[int, dict[str, Any]] = {}

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        try:
            self.write_sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                },
            )

            def start_text_block() -> int:
                nonlocal next_content_index, text_block_index
                if text_block_index is None:
                    text_block_index = next_content_index
                    next_content_index += 1
                    self.write_sse(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": text_block_index,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                return text_block_index

            def stop_text_block() -> None:
                nonlocal text_block_index
                if text_block_index is not None:
                    self.write_sse("content_block_stop", {"type": "content_block_stop", "index": text_block_index})
                    text_block_index = None

            def emit_tool_pending(state: dict[str, Any]) -> None:
                arguments = state.get("arguments") or ""
                emitted_len = state.get("emitted_len") or 0
                pending = arguments[emitted_len:]
                if pending:
                    self.write_sse(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": state["content_index"],
                            "delta": {"type": "input_json_delta", "partial_json": pending},
                        },
                    )
                    state["emitted_len"] = len(arguments)

            def start_tool_block(state: dict[str, Any]) -> None:
                nonlocal next_content_index
                if state.get("content_index") is not None:
                    return
                stop_text_block()
                state["content_index"] = next_content_index
                next_content_index += 1
                self.write_sse(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": state["content_index"],
                        "content_block": {
                            "type": "tool_use",
                            "id": state.get("id") or f"toolu_{uuid.uuid4().hex}",
                            "name": state.get("name") or "tool",
                            "input": {},
                        },
                    },
                )
                emit_tool_pending(state)

            for raw_line in upstream:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if chunk.get("model"):
                    model = chunk["model"]

                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    output_tokens = int(usage["completion_tokens"])

                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    text = delta.get("content")
                    if text:
                        index = start_text_block()
                        self.write_sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": index,
                                "delta": {"type": "text_delta", "text": text},
                            },
                        )

                    for tool_call_delta in delta.get("tool_calls") or []:
                        if not isinstance(tool_call_delta, dict):
                            continue
                        tool_index = int(tool_call_delta.get("index") or 0)
                        state = tool_states.setdefault(
                            tool_index,
                            {
                                "id": None,
                                "name": None,
                                "arguments": "",
                                "emitted_len": 0,
                                "content_index": None,
                            },
                        )
                        if tool_call_delta.get("id"):
                            state["id"] = tool_call_delta["id"]
                        function_delta = tool_call_delta.get("function") or {}
                        if function_delta.get("name"):
                            state["name"] = function_delta["name"]
                        if function_delta.get("arguments"):
                            state["arguments"] += function_delta["arguments"]

                        start_tool_block(state)
                        emit_tool_pending(state)

                    mapped_stop_reason = openai_finish_to_anthropic(choice.get("finish_reason"))
                    if mapped_stop_reason:
                        stop_reason = mapped_stop_reason

            if next_content_index == 0:
                start_text_block()
            stop_text_block()

            for _, state in sorted(tool_states.items(), key=lambda item: item[1].get("content_index") or 0):
                start_tool_block(state)
                emit_tool_pending(state)
                self.write_sse(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": state["content_index"]},
                )

            self.write_sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"output_tokens": output_tokens},
                },
            )
            self.write_sse("message_stop", {"type": "message_stop"})
        finally:
            upstream.close()
            self.close_connection = True

    def call_upstream(self, openai_payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        headers = build_upstream_headers(api_key, self.headers)
        if self.config.debug:
            print(f"\n=== Upstream Request ===", flush=True)
            print(f"URL: {chat_completions_url(self.config.openai_base_url)}", flush=True)
            print(f"Headers: {headers}", flush=True)
            print(f"Body preview: {str(compact_json(openai_payload))[:500]}...", flush=True)
            print(f"=======================\n", flush=True)

        req = request.Request(
            chat_completions_url(self.config.openai_base_url),
            data=compact_json(openai_payload),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.config.timeout_seconds) as response:
                response_body = response.read()
        except error.HTTPError as exc:
            raise UpstreamHTTPError(exc.code, read_http_error_body(exc)) from exc
        except error.URLError as exc:
            raise RuntimeError(f"upstream connection failed: {exc.reason}") from exc

        body = json.loads(response_body.decode("utf-8"))
        if not isinstance(body, dict):
            raise RuntimeError("upstream response must be a JSON object")
        return body

    def send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = compact_json(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_upstream_http_error(self, exc: error.HTTPError) -> None:
        self.send_json(
            exc.code,
            anthropic_error("api_error", read_http_error_body(exc) or f"upstream HTTP {exc.code}"),
        )

    def write_sse(self, event: str, data: dict[str, Any]) -> None:
        payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, separators=(',', ':'))}\n\n"
        self.wfile.write(payload.encode("utf-8"))
        self.wfile.flush()

    def log_message(self, format: str, *args: Any) -> None:
        if self.config.debug:
            super().log_message(format, *args)


class UpstreamHTTPError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(body or f"upstream HTTP {status}")
        self.status = status
        self.body = body


def read_http_error_body(exc: error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def make_server(host: str, port: int, config: ProxyConfig) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), AnthropicOpenAIProxyHandler)
    server.config = config  # type: ignore[attr-defined]
    return server


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Anthropic Messages API to OpenAI Chat Completions proxy")
    parser.add_argument("--host", default=os.getenv("HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", DEFAULT_PORT)))
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("UPSTREAM_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS)),
    )
    parser.add_argument("--debug", action="store_true", default=env_bool("DEBUG", False))
    parser.add_argument(
        "--convert",
        choices=["anthropic-to-openai", "openai-to-anthropic", "auto"],
        help="read a JSON request from stdin, print the converted JSON, and exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.convert:
        payload = json.loads(sys.stdin.read())
        if args.convert == "anthropic-to-openai":
            converted = convert_anthropic_to_openai_payload(payload)
        elif args.convert == "openai-to-anthropic":
            converted = convert_openai_to_anthropic_payload(payload)
        else:
            converted = auto_convert_payload(payload)
        print(json.dumps(converted, ensure_ascii=False, indent=2))
        return

    config = ProxyConfig(
        openai_base_url=normalize_base_url(resolve_openai_base_url(args.openai_base_url)),
        timeout_seconds=args.timeout,
        debug=args.debug,
    )
    server = make_server(args.host, args.port, config)
    print(
        f"anthropic-openai-proxy listening on http://{args.host}:{args.port}; "
        f"upstream={chat_completions_url(config.openai_base_url)}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
