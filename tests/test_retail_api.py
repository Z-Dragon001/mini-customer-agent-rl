import json
import os
import unittest
from unittest.mock import patch

from openai import OpenAI
from openai.types.chat import ChatCompletionChunk
import httpx

from mini_customer_agent.retail.api import complete_chat
from mini_customer_agent.retail.config import endpoint_kwargs, generation_kwargs, simulator_kwargs


class FakeStream:
    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    def __iter__(self):
        return iter(self.chunks)

    def close(self):
        self.closed = True


def chunk(delta=None, finish=None, usage=None):
    return ChatCompletionChunk(id="fake", created=0, model="qwen3.7-flash", object="chat.completion.chunk",
                               choices=[] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}],
                               usage=usage)


class RetailAPITests(unittest.TestCase):
    def setUp(self):
        self.spec = {"model": "qwen3.7-flash", "api_key_env": "ROLE_KEY", "api_key_env_fallback": "SHARED_KEY",
                     "base_url_env": "ROLE_URL", "base_url_env_fallback": "SHARED_URL", "base_url": "https://default.test/v1",
                     "temperature": .2, "max_tokens": 2048, "extra_body": {"enable_thinking": False}}

    def test_shared_credentials_and_role_override(self):
        with patch.dict(os.environ, {"SHARED_KEY": "shared", "SHARED_URL": "https://shared.test/v1"}, clear=True):
            self.assertEqual(endpoint_kwargs(self.spec)["api_key"], "shared")
            self.assertEqual(endpoint_kwargs(self.spec)["base_url"], "https://shared.test/v1")
            with patch.dict(os.environ, {"ROLE_KEY": "role", "ROLE_URL": "https://role.test/v1"}):
                self.assertEqual(endpoint_kwargs(self.spec)["api_key"], "role")
                self.assertEqual(endpoint_kwargs(self.spec)["base_url"], "https://role.test/v1")

    def test_stream_accumulates_fragmented_tools_reasoning_and_usage(self):
        stream = FakeStream([
            chunk({"role": "assistant", "reasoning_content": "Test reasoning."}),
            chunk({"tool_calls": [{"index": 0, "id": "call0", "type": "function", "function": {"name": "get_", "arguments": '{"order_'}},
                                   {"index": 1, "id": "call1", "type": "function", "function": {"name": "calculate", "arguments": '{"expression":'}}]}),
            chunk({"tool_calls": [{"index": 0, "function": {"name": "order_details", "arguments": 'id":"A"}'}},
                                   {"index": 1, "function": {"arguments": '"1+1"}'}}]}),
            chunk({}, "tool_calls"),
            chunk(usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}),
        ])
        result = complete_chat(lambda **kw: stream, stream=True)
        self.assertTrue(stream.closed)
        self.assertEqual(result.usage.total_tokens, 30)
        self.assertEqual(result.choices[0].message.reasoning_content, "Test reasoning.")
        calls = result.choices[0].message.tool_calls
        self.assertEqual(calls[0].function.name, "get_order_details")
        self.assertEqual(json.loads(calls[0].function.arguments), {"order_id": "A"})
        self.assertEqual(calls[1].function.arguments, '{"expression":"1+1"}')

    def test_incomplete_stream_is_not_a_successful_answer(self):
        stream = FakeStream([chunk({"content": "unfinished"})])
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            complete_chat(lambda **kw: stream, stream=True)
        self.assertTrue(stream.closed)

    def test_nonstream_art_response_keeps_original_choice(self):
        value = object()
        self.assertIs(complete_chat(lambda **kw: value, stream=False), value)

    def test_real_sdk_request_contains_provider_parameters(self):
        bodies = []
        def handle(request):
            bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0,
                "model": "qwen3.7-flash", "choices": [{"index": 0, "finish_reason": "stop",
                                                       "message": {"role": "assistant", "content": "ok"}}]})
        with OpenAI(api_key="test", base_url="https://api.test/v1", http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            result = complete_chat(client.chat.completions.create, model=self.spec["model"],
                                   messages=[{"role": "user", "content": "test"}], **generation_kwargs(self.spec))
        self.assertEqual(result.choices[0].message.content, "ok")
        self.assertIs(bodies[0]["enable_thinking"], False)
        self.assertNotIn("extra_body", bodies[0])

    def test_simulator_and_judge_receive_thinking_options(self):
        with patch.dict(os.environ, {"SHARED_KEY": "test"}, clear=True):
            args = simulator_kwargs(self.spec)
        self.assertIs(args["extra_body"]["enable_thinking"], False)
        self.assertNotIn("stream", args)
        with self.assertRaisesRegex(ValueError, "stream=false"):
            simulator_kwargs({**self.spec, "stream": True, "api_key_env": "PATH"})

    def test_real_sdk_stream_keeps_text_separate_from_reasoning(self):
        chunks = [chunk({"reasoning_content": "Synthetic reasoning."}), chunk({"content": "Hello"}),
                  chunk({}, "stop"), chunk(usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3})]
        sse = "".join("data: " + c.model_dump_json() + "\n\n" for c in chunks) + "data: [DONE]\n\n"
        def handle(request):
            body = json.loads(request.content)
            self.assertTrue(body["stream"])
            self.assertEqual(body["stream_options"], {"include_usage": True})
            return httpx.Response(200, content=sse, headers={"content-type": "text/event-stream"})
        displayed = []
        with OpenAI(api_key="test", base_url="https://api.test/v1", http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
            result = complete_chat(client.chat.completions.create, model=self.spec["model"],
                                   messages=[{"role": "user", "content": "hi"}],
                                   on_delta=lambda field, value: displayed.append((field, value)),
                                   **generation_kwargs({**self.spec, "stream": True}))
        self.assertEqual(result.choices[0].message.content, "Hello")
        self.assertEqual(result.choices[0].message.reasoning_content, "Synthetic reasoning.")
        self.assertEqual(result.usage.total_tokens, 3)
        self.assertEqual(displayed[0][0], "reasoning_content")

    def test_provider_options_cannot_replace_episode_messages(self):
        with self.assertRaises(ValueError):
            generation_kwargs({**self.spec, "extra_body": {"messages": []}})
        with self.assertRaises(ValueError):
            generation_kwargs({**self.spec, "extra_body": {"enable_thinking": "false"}})


if __name__ == "__main__":
    unittest.main()
