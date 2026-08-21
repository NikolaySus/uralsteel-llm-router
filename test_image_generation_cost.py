import unittest
from types import SimpleNamespace

import main


class TestImageGenerationCost(unittest.TestCase):
    def setUp(self):
        self.original = dict(main.ALL_API_VARS["openaiimgen"])
        main.ALL_API_VARS["openaiimgen"]["image_price_config"] = {
            "text_input": 0.000005,
            "cached_text_input": 0.00000125,
            "image_input": 0.000008,
            "cached_image_input": 0.000002,
            "image_output": 0.00003,
            "fallback": 0.125,
        }

    def tearDown(self):
        main.ALL_API_VARS["openaiimgen"].clear()
        main.ALL_API_VARS["openaiimgen"].update(self.original)

    def test_calculates_cost_from_image_usage(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens_details=SimpleNamespace(
                    text_tokens=100,
                    image_tokens=20,
                    cached_text_tokens=10,
                    cached_image_tokens=5,
                ),
                output_tokens_details=SimpleNamespace(image_tokens=4000),
                output_tokens=4000,
            )
        )

        cost = main.image_generation_expected_cost(response)

        expected = (
            100 * 0.000005 +
            10 * 0.00000125 +
            20 * 0.000008 +
            5 * 0.000002 +
            4000 * 0.00003
        )
        self.assertAlmostEqual(cost, expected)

    def test_falls_back_when_usage_is_missing(self):
        response = SimpleNamespace(usage=None)

        self.assertEqual(main.image_generation_expected_cost(response), 0.125)

    def test_uses_total_input_tokens_when_details_are_missing(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=200, output_tokens=1000)
        )

        cost = main.image_generation_expected_cost(response)

        self.assertAlmostEqual(cost, 200 * 0.000005 + 1000 * 0.00003)

    def test_falls_back_to_legacy_scalar_price(self):
        main.ALL_API_VARS["openaiimgen"].pop("image_price_config", None)
        main.ALL_API_VARS["openaiimgen"]["price_coef"] = 0.042

        self.assertEqual(main.image_generation_expected_cost(object()), 0.042)


if __name__ == "__main__":
    unittest.main()
