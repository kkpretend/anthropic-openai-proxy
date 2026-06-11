import unittest
import json
import threading
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request

import proxy


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    received_authorization = None
    received_body = None

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length") or 0)
        FakeOpenAIHandler.received_authorization = self.headers.get("Authorization")
        FakeOpenAIHandler.received_body = json.loads(self.rfile.read(content_length).decode("utf-8"))
        body = json.dumps(
            {
                "id": "chatcmpl-test",
                "model": FakeOpenAIHandler.received_body["model"],
                "choices": [{"message": {"content": "hello from upstream"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def start_threaded_server(server):
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return thread


class ProxyConversionTests(unittest.TestCase):
    def test_anthropic_messages_to_openai(self):
        payload = {
            "model": "qwen3-max",
            "system": "You are concise.",
            "max_tokens": 256,
            "temperature": 0,
            "stop_sequences": ["</done>"],
            "extra_body": {"enable_thinking": False},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "text", "text": " world"},
                    ],
                }
            ],
        }

        converted = proxy.anthropic_messages_to_openai(payload)

        self.assertEqual(converted["model"], "qwen3-max")
        self.assertEqual(converted["max_tokens"], 256)
        self.assertEqual(converted["temperature"], 0)
        self.assertEqual(converted["stop"], ["</done>"])
        self.assertEqual(converted["enable_thinking"], False)
        self.assertEqual(
            converted["messages"],
            [
                {"role": "system", "content": "You are concise."},
                {"role": "user", "content": "hello world"},
            ],
        )

    def test_image_content_block_maps_to_openai_image_url(self):
        converted = proxy.anthropic_content_to_openai(
            [
                {"type": "text", "text": "describe"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abc",
                    },
                },
            ]
        )

        self.assertEqual(converted[0], {"type": "text", "text": "describe"})
        self.assertEqual(
            converted[1],
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        )

    def test_openai_response_to_anthropic(self):
        converted = proxy.openai_response_to_anthropic(
            {
                "id": "chatcmpl-1",
                "model": "qwen3-max",
                "choices": [
                    {
                        "message": {"content": "done"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
            "fallback-model",
        )

        self.assertEqual(converted["id"], "chatcmpl-1")
        self.assertEqual(converted["model"], "qwen3-max")
        self.assertEqual(converted["content"], [{"type": "text", "text": "done"}])
        self.assertEqual(converted["stop_reason"], "max_tokens")
        self.assertEqual(converted["usage"], {"input_tokens": 11, "output_tokens": 7})

    def test_tool_schema_and_messages_convert_to_openai(self):
        converted = proxy.anthropic_messages_to_openai(
            {
                "model": "qwen3-max",
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    }
                ],
                "tool_choice": {"type": "tool", "name": "get_weather"},
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "get_weather",
                                "input": {"city": "Beijing"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": "sunny",
                            }
                        ],
                    },
                ],
            }
        )

        self.assertEqual(
            converted["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        )
        self.assertEqual(
            converted["tool_choice"],
            {"type": "function", "function": {"name": "get_weather"}},
        )
        self.assertEqual(
            converted["messages"],
            [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city":"Beijing"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"},
            ],
        )

    def test_openai_tool_calls_convert_to_anthropic(self):
        converted = proxy.openai_response_to_anthropic(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": '{"city":"Beijing"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            "qwen3-max",
        )

        self.assertEqual(converted["stop_reason"], "tool_use")
        self.assertEqual(
            converted["content"],
            [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"city": "Beijing"},
                }
            ],
        )

    def test_api_key_passthrough_headers(self):
        headers = Message()
        headers["x-api-key"] = "sk-test"
        headers["OpenAI-Organization"] = "org_1"

        api_key = proxy.get_incoming_api_key(headers)
        self.assertEqual(api_key, "sk-test")

        upstream_headers = proxy.build_upstream_headers(api_key, headers)
        self.assertEqual(upstream_headers["Authorization"], "Bearer sk-test")
        self.assertEqual(upstream_headers["OpenAI-Organization"], "org_1")

    def test_normalize_base_url_accepts_full_chat_url(self):
        self.assertEqual(
            proxy.chat_completions_url("http://host/v1/chat/completions"),
            "http://host/v1/chat/completions",
        )


class ProxyHTTPTests(unittest.TestCase):
    def test_messages_endpoint_forwards_to_openai_with_passthrough_key(self):
        FakeOpenAIHandler.received_authorization = None
        FakeOpenAIHandler.received_body = None

        upstream = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
        upstream_port = upstream.server_address[1]
        proxy_server = proxy.make_server(
            "127.0.0.1",
            0,
            proxy.ProxyConfig(f"http://127.0.0.1:{upstream_port}/v1", 10),
        )
        proxy_port = proxy_server.server_address[1]

        start_threaded_server(upstream)
        start_threaded_server(proxy_server)
        try:
            body = json.dumps(
                {
                    "model": "qwen3-max",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "hello"}],
                }
            ).encode("utf-8")
            req = request.Request(
                f"http://127.0.0.1:{proxy_port}/v1/messages",
                data=body,
                headers={"Content-Type": "application/json", "x-api-key": "sk-test"},
                method="POST",
            )
            with request.urlopen(req, timeout=10) as response:
                converted = json.loads(response.read().decode("utf-8"))

            self.assertEqual(FakeOpenAIHandler.received_authorization, "Bearer sk-test")
            self.assertEqual(FakeOpenAIHandler.received_body["model"], "qwen3-max")
            self.assertEqual(
                FakeOpenAIHandler.received_body["messages"],
                [{"role": "user", "content": "hello"}],
            )
            self.assertEqual(converted["content"], [{"type": "text", "text": "hello from upstream"}])
            self.assertEqual(converted["usage"], {"input_tokens": 3, "output_tokens": 2})
        finally:
            proxy_server.shutdown()
            upstream.shutdown()
            proxy_server.server_close()
            upstream.server_close()

    def test_convert_endpoint_returns_openai_payload_without_api_key(self):
        proxy_server = proxy.make_server(
            "127.0.0.1",
            0,
            proxy.ProxyConfig("http://127.0.0.1:1/v1", 10),
        )
        proxy_port = proxy_server.server_address[1]

        start_threaded_server(proxy_server)
        try:
            body = json.dumps(
                {
                    "model": "qwen3-max",
                    "system": "Be brief.",
                    "max_tokens": 32,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "hello"}],
                        }
                    ],
                }
            ).encode("utf-8")
            req = request.Request(
                f"http://127.0.0.1:{proxy_port}/convert/anthropic-to-openai",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(req, timeout=10) as response:
                converted = json.loads(response.read().decode("utf-8"))

            self.assertEqual(
                converted,
                {
                    "model": "qwen3-max",
                    "messages": [
                        {"role": "system", "content": "Be brief."},
                        {"role": "user", "content": "hello"},
                    ],
                    "stream": False,
                    "max_tokens": 32,
                },
            )
        finally:
            proxy_server.shutdown()
            proxy_server.server_close()


if __name__ == "__main__":
    unittest.main()
