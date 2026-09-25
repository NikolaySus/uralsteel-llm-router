import json
import os
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch


os.environ.update({
    "INFERENCE_API_OPENAIVLM_MODEL": "gpt-5.6-sol",
    "INFERENCE_API_OPENAIVLM_REASONING_EFFORT": "none",
    "INFERENCE_API_OPENAIVLM_BASE_URL": "https://example.test/v1",
    "INFERENCE_API_OPENAIVLM_KEY": "test",
    "INFERENCE_API_OPENAIMINI_MODEL": "test-mini",
    "INFERENCE_API_OPENAIMINI_BASE_URL": "https://example.test/v1",
    "INFERENCE_API_OPENAIMINI_KEY": "test",
    "INFERENCE_API_OPENAIMINI_VLM": "true",
    "INFERENCE_API_DEEPSEEK_MODEL": "test-deepseek",
    "INFERENCE_API_DEEPSEEK_BASE_URL": "https://example.test/v1",
    "INFERENCE_API_DEEPSEEK_KEY": "test",
    "INFERENCE_API_OPENROUTERGEMINI_MODEL": "gemini-3.1-pro-preview",
    "INFERENCE_API_OPENROUTERGEMINI_BASE_URL": "https://openrouter.ai/api/v1",
    "INFERENCE_API_OPENROUTERGEMINI_KEY": "test",
    "INFERENCE_API_OPENROUTERGEMINI_VLM": "true",
    "INFERENCE_API_OPENROUTERCLAUDE_MODEL": "claude-opus-5",
    "INFERENCE_API_OPENROUTERCLAUDE_BASE_URL": "https://openrouter.ai/api/v1",
    "INFERENCE_API_OPENROUTERCLAUDE_KEY": "test",
    "INFERENCE_API_OPENROUTERCLAUDE_VLM": "true",
})

import main


for env_name, env_value in os.environ.items():
    if not env_name.startswith(main.CONST):
        continue
    api_and_case = env_name[main.CONST_LEN:].lower()
    delim = api_and_case.find("_")
    api = api_and_case[:delim]
    case = api_and_case[delim + 1:]
    main.ALL_API_VARS.setdefault(api, {})[case] = env_value
    if case in {"model", "rename"}:
        main.MODEL_TO_API[env_value] = api


class TestModelImageSupport(unittest.TestCase):
    def test_new_documents_are_compacted_before_chunks_and_storage(self):
        markdown = "|  A  |  B  |\n|----------|----------|\n|  C  |  D  |\n"
        compact = "| A | B |\n| --- | --- |\n| C | D |\n"
        for from_url in (False, True):
            with self.subTest(markdown_url=from_url):
                request = SimpleNamespace(
                    msg="User    text", history=[], text2text_model="gpt-5.6-sol",
                    function="", images_urls=[],
                    documents_urls=[] if from_url else ["https://files.test/a.docx"],
                    markdown_urls=[SimpleNamespace(
                        original_name="a.docx", url="https://files.test/a.md"
                    )] if from_url else [],
                )
                storage = Mock()
                http = Mock()
                http.get.return_value.text = markdown
                with patch.object(main, "generate_chat_name", return_value=None), patch.object(
                    main, "convert_to_md", return_value=("a.docx", markdown)
                ), patch.object(main, "Minio", return_value=storage), patch.object(
                    main.httpx, "Client"
                ) as http_class:
                    http_class.return_value.__enter__.return_value = http
                    stream = main.LlmServicer().NewMessage(request, Mock())
                    responses = []
                    for response in stream:
                        responses.append(response)
                        if response.user_message_uid:
                            break
                    stream.close()
                payload = json.loads(storage.put_object.call_args.args[2].getvalue())
                text = "".join(b.get("text", "") for b in payload["content"])
                self.assertIn(compact, text)
                self.assertEqual(payload["content"][-1]["text"], "User    text")
                chunks = "".join(r.markdown_chunk.markdown_chunk for r in responses)
                self.assertEqual(chunks, "" if from_url else compact)

    def test_history_is_compacted_without_writing_objects(self):
        original = {"role": "user", "content": [{"type": "text", "text": (
            '# FILE "a.docx" BEGIN\n|  A  | B |\n|------|------|\n'
            '# FILE "a.docx" END'
        )}]}
        storage = Mock()
        storage.get_object.return_value = BytesIO(json.dumps(original).encode())
        current = {"role": "user", "content": "next"}
        with patch.object(main, "Minio", return_value=storage):
            messages, _ = main.build_messages_from_history(["stored"], current, "model")
        self.assertIn("| --- | --- |", messages[1]["content"][0]["text"])
        self.assertEqual(messages[-1], current)
        storage.put_object.assert_not_called()

    def test_env_flag_marks_openaimini_as_image_capable(self):
        self.assertTrue(main.model_supports_images("test-mini"))

    def test_model_without_env_flag_is_not_image_capable(self):
        self.assertFalse(main.model_supports_images("test-deepseek"))

    def test_openaivlm_is_image_capable_by_default(self):
        main.ALL_API_VARS["openaivlm"].pop("vlm", None)
        self.assertTrue(main.model_supports_images("gpt-5.6-sol"))

    def test_openrouter_models_are_added_and_image_capable(self):
        models = main.add_configured_text2text_models([])

        self.assertIn("gpt-5.6-sol", models)
        self.assertIn("gemini-3.1-pro-preview", models)
        self.assertIn("claude-opus-5", models)
        self.assertNotIn("gpt-5.5", models)
        self.assertNotIn("anthropic/claude-opus-4.7", models)
        self.assertNotIn("google/gemini-3.1-pro-preview", models)
        self.assertEqual(main.MODEL_TO_API["gpt-5.6-sol"], "openaivlm")
        self.assertEqual(
            main.MODEL_TO_API["gemini-3.1-pro-preview"],
            "openroutergemini",
        )
        self.assertEqual(
            main.MODEL_TO_API["claude-opus-5"],
            "openrouterclaude",
        )
        self.assertTrue(
            main.model_supports_images("gemini-3.1-pro-preview")
        )
        self.assertTrue(main.model_supports_images("claude-opus-5"))

    def test_text2text_models_fall_back_to_configured_models(self):
        context = Mock()
        with patch.object(
            main,
            "available_models",
            side_effect=RuntimeError("Yandex unavailable"),
        ):
            response = main.LlmServicer().AvailableModelsText2Text(
                None,
                context,
            )

        self.assertIn("gpt-5.6-sol", response.strings)
        self.assertIn("gemini-3.1-pro-preview", response.strings)
        self.assertIn("claude-opus-5", response.strings)
        context.set_code.assert_not_called()

    def test_text2text_models_include_and_route_yandex_models(self):
        with patch.object(
            main,
            "available_models",
            return_value=["yandex-test-model"],
        ), patch.dict(main.MODEL_TO_API, {}, clear=True):
            models = main.get_text2text_models()

            self.assertIn("yandex-test-model", models)
            self.assertEqual(
                main.MODEL_TO_API["yandex-test-model"],
                "yandexai",
            )
            self.assertIn("gpt-5.6-sol", models)

    def test_openrouter_models_have_price_coefficients(self):
        with open("config.json", encoding="utf-8") as config_file:
            prices = json.load(config_file)["prices_coefs"]

        self.assertEqual(
            prices["openroutergemini"],
            {
                "input": 0.000002,
                "cached_input": 0.0000002,
                "output": 0.000012,
                "long_context_threshold": 200000,
                "long_context_input": 0.000004,
                "long_context_cached_input": 0.0000004,
                "long_context_output": 0.000018,
            },
        )
        self.assertEqual(
            prices["openrouterclaude"],
            {"input": 0.000005, "output": 0.000025},
        )

    def test_tool_call_done_with_none_arguments_uses_accumulated_args(self):
        added_chunk = SimpleNamespace(
            object="chat.completion.chunk",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="websearch",
                                    arguments=None,
                                ),
                            )
                        ]
                    ),
                )
            ],
        )
        delta_chunk = SimpleNamespace(
            object="chat.completion.chunk",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                id=None,
                                function=SimpleNamespace(
                                    name=None,
                                    arguments='{"query":"Moscow news"}',
                                ),
                            )
                        ]
                    ),
                )
            ],
        )
        unrelated_chunk = SimpleNamespace(
            object="chat.completion.chunk",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(tool_calls=None),
                )
            ],
        )
        done_chunk = SimpleNamespace(
            id="chunk-1",
            object="chat.completion.chunk",
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    delta=SimpleNamespace(
                        tool_calls=[
                            SimpleNamespace(
                                id="call-1",
                                function=SimpleNamespace(
                                    name="websearch",
                                    arguments=None,
                                ),
                            )
                        ]
                    ),
                )
            ],
        )

        _, _, call_id, name, args = main.function_call_responses_from_llm_chunk(
            "test", added_chunk
        )
        _, _, call_id, name, args = main.function_call_responses_from_llm_chunk(
            "test", delta_chunk, call_id, name, args
        )
        _, _, call_id, name, args = main.function_call_responses_from_llm_chunk(
            "test", unrelated_chunk, call_id, name, args
        )
        response, item, _, _, _ = main.function_call_responses_from_llm_chunk(
            "test", done_chunk, call_id, name, args
        )

        self.assertTrue(response.HasField("function_call_complete"))
        self.assertEqual(item["content"], "")
        self.assertEqual(item["tool_calls"][0]["id"], "call-1")
        function_args = item["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(function_args, '{"query":"Moscow news"}')
        self.assertEqual(json.loads(function_args), {"query": "Moscow news"})

    def test_parse_tool_arguments_uses_user_text_fallback(self):
        tool_call = {
            "type": "function",
            "function": {
                "name": "websearch",
                "arguments": None,
            },
        }

        args = main.parse_tool_arguments(
            "test",
            tool_call,
            "Какая сейчас погода в Алма Ате?",
        )

        self.assertEqual(args, {"query": "Какая сейчас погода в Алма Ате?"})

    def test_fallback_tool_arguments_can_be_written_back_as_json(self):
        tool_call = {
            "id": "call-1",
            "type": "function",
            "function": {
                "name": "websearch",
                "arguments": None,
            },
        }
        args = main.parse_tool_arguments("test", tool_call, "погода")
        tool_call["function"]["arguments"] = json.dumps(
            args,
            ensure_ascii=False,
        )

        self.assertEqual(tool_call["function"]["arguments"], '{"query": "погода"}')

    def test_message_content_length_accepts_tool_call_content_none(self):
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [],
        }

        self.assertEqual(main.message_content_length(message), 0)

    def test_normalize_tool_call_item_fills_provider_required_fields(self):
        item = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": None,
                    "type": "function",
                    "function": {
                        "name": None,
                        "arguments": None,
                    },
                }
            ],
        }
        function_tool = {
            "type": "function",
            "function": {"name": "websearch"},
        }

        tool_call = main.normalize_tool_call_item(
            item,
            function_tool,
            {"query": "погода"},
        )

        self.assertEqual(item["content"], "")
        self.assertTrue(tool_call["id"].startswith("call_"))
        self.assertEqual(tool_call["function"]["name"], "websearch")
        self.assertEqual(tool_call["function"]["arguments"], '{"query": "погода"}')

    def test_selected_tool_name_reads_forced_tool_choice(self):
        function_tool = {
            "type": "function",
            "function": {"name": "websearch"},
        }

        self.assertEqual(main.selected_tool_name(function_tool), "websearch")

    def test_unknown_tool_returns_result_and_no_metadata(self):
        result, meta = main.call_function("test", None, {})

        self.assertIsNone(meta)
        self.assertEqual(json.loads(result), {"error": "Unknown tool None"})

    def test_is_large_context_error_detects_context_length_error(self):
        error = (
            "This endpoint's maximum context length is 1000000 tokens. "
            "However, you requested about 5452453 tokens."
        )

        self.assertTrue(main.is_large_context_error(error))

    def test_is_large_context_error_detects_string_length_error(self):
        error = (
            "Invalid 'messages[1].content[1].text': string too long. "
            "Expected a string with maximum length 10485760, "
            "but got a string with length 16841667 instead."
        )

        self.assertTrue(main.is_large_context_error(error))

    def test_is_large_context_error_ignores_other_provider_errors(self):
        self.assertFalse(main.is_large_context_error("Error code: 402"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
