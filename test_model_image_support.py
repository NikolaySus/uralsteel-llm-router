import json
import os
import unittest
from types import SimpleNamespace


os.environ.update({
    "INFERENCE_API_OPENAIVLM_MODEL": "test-vlm",
    "INFERENCE_API_OPENAIVLM_BASE_URL": "https://example.test/v1",
    "INFERENCE_API_OPENAIVLM_KEY": "test",
    "INFERENCE_API_OPENAIMINI_MODEL": "test-mini",
    "INFERENCE_API_OPENAIMINI_BASE_URL": "https://example.test/v1",
    "INFERENCE_API_OPENAIMINI_KEY": "test",
    "INFERENCE_API_OPENAIMINI_VLM": "true",
    "INFERENCE_API_DEEPSEEK_MODEL": "test-deepseek",
    "INFERENCE_API_DEEPSEEK_BASE_URL": "https://example.test/v1",
    "INFERENCE_API_DEEPSEEK_KEY": "test",
    "INFERENCE_API_OPENROUTERGEMINI_MODEL": "google/gemini-3.1-pro-preview",
    "INFERENCE_API_OPENROUTERGEMINI_BASE_URL": "https://openrouter.ai/api/v1",
    "INFERENCE_API_OPENROUTERGEMINI_KEY": "test",
    "INFERENCE_API_OPENROUTERGEMINI_VLM": "true",
    "INFERENCE_API_OPENROUTERCLAUDE_MODEL": "anthropic/claude-opus-4.7",
    "INFERENCE_API_OPENROUTERCLAUDE_BASE_URL": "https://openrouter.ai/api/v1",
    "INFERENCE_API_OPENROUTERCLAUDE_KEY": "test",
    "INFERENCE_API_OPENROUTERCLAUDE_VLM": "true",
})

import main


class TestModelImageSupport(unittest.TestCase):
    def test_env_flag_marks_openaimini_as_image_capable(self):
        self.assertTrue(main.model_supports_images("test-mini"))

    def test_model_without_env_flag_is_not_image_capable(self):
        self.assertFalse(main.model_supports_images("test-deepseek"))

    def test_openaivlm_is_image_capable_by_default(self):
        main.ALL_API_VARS["openaivlm"].pop("vlm", None)
        self.assertTrue(main.model_supports_images("test-vlm"))

    def test_openrouter_models_are_added_and_image_capable(self):
        models = main.add_configured_text2text_models([])

        self.assertIn("google/gemini-3.1-pro-preview", models)
        self.assertIn("anthropic/claude-opus-4.7", models)
        self.assertEqual(
            main.MODEL_TO_API["google/gemini-3.1-pro-preview"],
            "openroutergemini",
        )
        self.assertEqual(
            main.MODEL_TO_API["anthropic/claude-opus-4.7"],
            "openrouterclaude",
        )
        self.assertTrue(
            main.model_supports_images("google/gemini-3.1-pro-preview")
        )
        self.assertTrue(main.model_supports_images("anthropic/claude-opus-4.7"))

    def test_openrouter_models_have_price_coefficients(self):
        with open("config.json", encoding="utf-8") as config_file:
            prices = json.load(config_file)["prices_coefs"]

        self.assertEqual(
            prices["openroutergemini"],
            {"input": 0.000002, "output": 0.000012},
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
        response, item, _, _, _ = main.function_call_responses_from_llm_chunk(
            "test", done_chunk, call_id, name, args
        )

        self.assertTrue(response.HasField("function_call_complete"))
        function_args = item["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(function_args, '{"query":"Moscow news"}')
        self.assertEqual(json.loads(function_args), {"query": "Moscow news"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
