import os
import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("DOCLING_ADDRESS", "localhost:5001")

import main


class TestStreamChunkResponse(unittest.TestCase):
    def test_request_size_includes_images_tools_and_reasoning(self):
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "\u0442\u0435\u043a\u0441\u0442"},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + "A" * 10000}},
            ]},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ]
        for choice in ("none", {"type": "function", "function": {"name": "websearch"}}):
            client = MagicMock()
            client.chat.completions.create.return_value.__iter__.return_value = iter([])
            with patch.object(main, "OpenAI", return_value=client), patch.object(
                main.logger, "info"
            ) as info:
                list(main.proc_llm_stream_responses(
                    {}, "size-test", messages, choice,
                    "https://example.test/v1", "secret-key", None,
                    "model", 0, 0, "none",
                ))
            payload = client.chat.completions.create.call_args.kwargs
            expected = len(json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"))
            size_log = next(c.args for c in info.call_args_list if "estimated_json_bytes" in c.args[0])
            self.assertEqual(size_log[1:], ("size-test", "model", 11, 6, 1, expected))
            self.assertNotIn("secret-key", str(size_log))
            self.assertNotIn("base64", str(size_log))

    def test_reasoning_effort_is_sent_without_tools(self):
        client = MagicMock()
        stream = MagicMock()
        stream.__iter__.return_value = iter([])
        client.chat.completions.create.return_value = stream

        with patch.object(main, "OpenAI", return_value=client):
            list(main.proc_llm_stream_responses(
                {"input": 0, "output": 0},
                "test",
                [{"role": "user", "content": "hello"}],
                "none",
                "https://example.test/v1",
                "key",
                None,
                "gpt-5.6-sol",
                5,
                0,
                "none",
            ))

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertNotIn("tools", kwargs)

    def test_reasoning_effort_is_sent_with_tools(self):
        client = MagicMock()
        stream = MagicMock()
        stream.__iter__.return_value = iter([])
        client.chat.completions.create.return_value = stream
        tool_choice = {
            "type": "function",
            "function": {"name": "websearch"},
        }

        with patch.object(main, "OpenAI", return_value=client):
            list(main.proc_llm_stream_responses(
                {"input": 0, "output": 0},
                "test",
                [{"role": "user", "content": "hello"}],
                tool_choice,
                "https://example.test/v1",
                "key",
                None,
                "gpt-5.6-sol",
                5,
                0,
                "none",
            ))

        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["reasoning_effort"], "none")
        self.assertEqual(kwargs["tool_choice"], tool_choice)
        self.assertEqual(kwargs["tools"], main.TOOLS)

    def test_tiered_cost_uses_standard_price_at_threshold(self):
        price = {
            "input": 0.000004,
            "cached_input": 0.0000004,
            "output": 0.000020,
            "long_context_threshold": 272000,
            "long_context_input": 0.000008,
            "long_context_cached_input": 0.0000008,
            "long_context_output": 0.000030,
        }

        cost = main.calculate_token_cost(price, 272000, 1000, 2000)

        expected = 270000 * 0.000004 + 2000 * 0.0000004 + 1000 * 0.000020
        self.assertAlmostEqual(cost, expected)

    def test_tiered_cost_uses_long_context_price_above_threshold(self):
        price = {
            "input": 0.000004,
            "cached_input": 0.0000004,
            "output": 0.000020,
            "long_context_threshold": 272000,
            "long_context_input": 0.000008,
            "long_context_cached_input": 0.0000008,
            "long_context_output": 0.000030,
        }

        cost = main.calculate_token_cost(price, 272001, 1000, 2000)

        expected = 270001 * 0.000008 + 2000 * 0.0000008 + 1000 * 0.000030
        self.assertAlmostEqual(cost, expected)

    def test_flat_and_split_price_formats_remain_supported(self):
        self.assertEqual(main.calculate_token_cost(0.1, 2, 3), 0.5)
        self.assertEqual(
            main.calculate_token_cost(
                {"input": 0.01, "output": 0.02}, 2, 3
            ),
            0.08,
        )

    def test_usage_cached_tokens_are_priced_separately(self):
        chunk = SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=10,
                total_tokens=110,
                prompt_tokens_details=SimpleNamespace(cached_tokens=40),
            ),
        )
        price = {
            "input": 0.01,
            "cached_input": 0.001,
            "output": 0.02,
        }

        response, _ = main.responses_from_llm_chunk(
            price, "test", chunk, 0, 0
        )

        self.assertTrue(response.HasField("complete"))
        self.assertAlmostEqual(
            response.complete.expected_cost_usd,
            60 * 0.01 + 40 * 0.001 + 10 * 0.02,
        )

    def test_content_with_usage_is_returned_as_generate(self):
        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content="Раз два три",
                        reasoning_content=None,
                    ),
                    finish_reason=None,
                )
            ],
            usage=SimpleNamespace(
                completion_tokens=3,
                prompt_tokens=14,
                total_tokens=557,
            ),
        )

        response, delta_content = main.responses_from_llm_chunk(
            0.0, "test", chunk, 0, 0
        )

        self.assertTrue(response.HasField("generate"))
        self.assertEqual(response.generate.content, "Раз два три")
        self.assertEqual(delta_content, "Раз два три")

    def test_usage_without_content_is_returned_as_complete(self):
        chunk = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                completion_tokens=3,
                prompt_tokens=14,
                total_tokens=557,
            ),
        )

        response, delta_content = main.responses_from_llm_chunk(
            0.000001, "test", chunk, 0, 0
        )

        self.assertTrue(response.HasField("complete"))
        self.assertEqual(response.complete.completion_tokens, 3)
        self.assertEqual(response.complete.prompt_tokens, 14)
        self.assertEqual(response.complete.total_tokens, 557)
        self.assertIsNone(delta_content)

    def test_google_one_chunk_tool_call_completes_on_stop(self):
        first_chunk = SimpleNamespace(
            object="chat.completion.chunk",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="websearch",
                                    arguments='{"query":"погода в Алматы"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
        final_chunk = SimpleNamespace(
            object="chat.completion.chunk",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )

        response, item, call_id, name, args = (
            main.function_call_responses_from_llm_chunk(
                "test", first_chunk, "", "", ""
            )
        )
        self.assertTrue(response.HasField("function_call_added"))
        self.assertIsNone(item)
        self.assertEqual(call_id, "call_1")
        self.assertEqual(name, "websearch")
        self.assertEqual(args, '{"query":"погода в Алматы"}')

        response, item, call_id, name, args = (
            main.function_call_responses_from_llm_chunk(
                "test", final_chunk, call_id, name, args
            )
        )
        self.assertTrue(response.HasField("function_call_complete"))
        self.assertEqual(item["tool_calls"][0]["function"]["name"], "websearch")
        self.assertEqual(
            item["tool_calls"][0]["function"]["arguments"],
            '{"query":"погода в Алматы"}',
        )
        self.assertIsNone(call_id)
        self.assertIsNone(name)
        self.assertIsNone(args)

    def test_openai_split_tool_call_arguments_wait_for_finish(self):
        first_chunk = SimpleNamespace(
            object="chat.completion.chunk",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="websearch",
                                    arguments="",
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
        args_chunk = SimpleNamespace(
            object="chat.completion.chunk",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                id=None,
                                function=SimpleNamespace(
                                    name=None,
                                    arguments='{"query":"weather',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
        final_chunk = SimpleNamespace(
            object="chat.completion.chunk",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(tool_calls=None),
                    finish_reason="tool_calls",
                )
            ],
        )

        response, item, call_id, name, args = (
            main.function_call_responses_from_llm_chunk(
                "test", first_chunk, "", "", ""
            )
        )
        self.assertTrue(response.HasField("function_call_added"))
        self.assertIsNone(item)

        response, item, call_id, name, args = (
            main.function_call_responses_from_llm_chunk(
                "test", args_chunk, call_id, name, args
            )
        )
        self.assertTrue(response.HasField("function_call_delta"))
        self.assertIsNone(item)
        self.assertEqual(args, '{"query":"weather')

        response, item, call_id, name, args = (
            main.function_call_responses_from_llm_chunk(
                "test", final_chunk, call_id, name, args
            )
        )
        self.assertTrue(response.HasField("function_call_complete"))
        self.assertEqual(
            item["tool_calls"][0]["function"]["arguments"],
            '{"query":"weather',
        )
        self.assertIsNone(call_id)
        self.assertIsNone(name)
        self.assertIsNone(args)

    def test_websearch_tool_payload_is_compact(self):
        payload = main.websearch_tool_payload(
            "погода",
            [
                {
                    "title": "Weather",
                    "url": "https://example.test",
                    "content": "Sunny, +31 C",
                    "raw_content": "large noisy page text",
                    "score": 0.9,
                }
            ],
        )

        self.assertEqual(
            payload,
            {
                "query": "погода",
                "results": [
                    {
                        "title": "Weather",
                        "url": "https://example.test",
                        "content": "Sunny, +31 C",
                    }
                ],
            },
        )

    def test_call_function_websearch_returns_google_friendly_content(self):
        with patch.object(main, "websearch") as websearch:
            websearch.return_value = [
                {
                    "title": "Weather",
                    "url": "https://example.test",
                    "content": "Sunny, +31 C",
                    "raw_content": "large noisy page text",
                }
            ]

            result, meta = main.call_function(
                "test", "websearch", {"query": "погода"}
            )

        payload = json.loads(result)
        self.assertEqual(payload["query"], "погода")
        self.assertEqual(payload["results"][0]["content"], "Sunny, +31 C")
        self.assertNotIn("raw_content", payload["results"][0])
        self.assertEqual(len(meta.websearch.item), 1)

    def test_empty_websearch_is_not_returned_as_bare_list(self):
        payload = main.websearch_tool_payload("weather", [])

        self.assertEqual(payload["query"], "weather")
        self.assertEqual(payload["results"], [])
        self.assertIn("No websearch results", payload["message"])

    def test_google_tool_call_extra_content_is_preserved(self):
        chunk = SimpleNamespace(
            object="chat.completion.chunk",
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="websearch",
                                    arguments='{"query":"погода в Алматы"}',
                                ),
                                extra_content={
                                    "google": {
                                        "thought_signature": "signature"
                                    }
                                },
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )

        response, item, call_id, name, args = (
            main.function_call_responses_from_llm_chunk(
                "test", chunk, "", "", ""
            )
        )

        self.assertTrue(response.HasField("function_call_complete"))
        self.assertEqual(
            item["tool_calls"][0]["extra_content"]["google"]["thought_signature"],
            "signature",
        )
        self.assertIsNone(call_id)
        self.assertIsNone(name)
        self.assertIsNone(args)


if __name__ == "__main__":
    unittest.main()
