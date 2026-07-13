import os
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DOCLING_ADDRESS", "localhost:5001")

import main


class TestStreamChunkResponse(unittest.TestCase):
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
