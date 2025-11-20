import unittest
import json
import os

from main import build_user_message


TEST_IMAGE_URL = os.environ.get('TEST_IMAGE_URL', '')
TEST_DOCX_URL = os.environ.get('TEST_DOCX_URL', '')


class TestLlmService(unittest.TestCase):
    """Юнит тесты."""

    def test_01_build_user_message(self):
        user_message = "Найди 10 отличий:"
        md_name = "ДЗ Горкунов Н.М. ИУ5-83Б.docx.md.tmp"
        content = None
        with open(md_name, 'r', encoding='utf-8') as file:
            content = file.read()
        md_docs = {
            md_name: content,
            md_name + "_v2": content
        }
        images_urls = [TEST_IMAGE_URL, TEST_IMAGE_URL]
        res, must_true = build_user_message(user_message, md_docs, images_urls)
        assert must_true, "There are images for sure!"
        with open('output.json.tmp', 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
