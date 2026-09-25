import copy
import unittest

from document_markdown import (
    normalize_document_markdown, normalize_history_documents,
)
from util import build_user_message


class TestDocumentMarkdown(unittest.TestCase):
    def test_huge_padding_and_images(self):
        image = "data:image/png;base64," + "A" * 100000
        padding = " " * 100000
        markdown = (
            f"| ![Image]({image}) | Title{padding} |\n"
            f"|{'-' * 100000}|:{'-' * 100000}:|\n"
            f"| Value{padding}|12 mm{padding}|\n"
        )
        result = normalize_document_markdown(markdown)
        self.assertEqual(result, (
            f"| ![Image]({image}) | Title |\n"
            "| --- | :---: |\n| Value | 12 mm |\n"
        ))
        self.assertEqual(normalize_document_markdown(result), result)

    def test_cells_escaping_code_unicode_and_empty_values(self):
        markdown = (
            "|  Name  | Value | Empty |\r\n"
            "|:--------|------:|:------:|\r\n"
            "| a\\|b | `x|y` | |\r\n"
            "| ``a`|b`` | \u0442\u0435\u043a\u0441\u0442  12 | |\r\n"
        )
        expected = (
            "| Name | Value | Empty |\r\n"
            "| :--- | ---: | :---: |\r\n"
            "| a\\|b | `x|y` |  |\r\n"
            "| ``a`|b`` | \u0442\u0435\u043a\u0441\u0442  12 |  |\r\n"
        )
        self.assertEqual(normalize_document_markdown(markdown), expected)

    def test_code_and_non_table_text_are_unchanged(self):
        table = "|  A  | B |\n|------|---|\n|  C  | D |\n"
        for fence in ("```", "~~~~"):
            markdown = fence + "text\n" + table + fence + "\n"
            self.assertEqual(normalize_document_markdown(markdown), markdown)
        indented = "".join("    " + line for line in table.splitlines(True))
        text = "ordinary    spaces\n|  not a table |\n\n" + indented
        self.assertEqual(normalize_document_markdown(text), text)

    def test_no_outer_pipes_and_no_final_newline(self):
        self.assertEqual(
            normalize_document_markdown(" A | B \n ---|--- \n C | D "),
            "| A | B |\n| --- | --- |\n| C | D |",
        )

    def test_history_preserves_images_user_text_and_original(self):
        image = "data:image/png;base64,AAAA"
        markdown = (
            f"| ![Image]({image}) |  Heading  |\n"
            "|----------------|-----------------|\n"
            "|  Cell  |  Value  |\n"
        )
        message, _ = build_user_message(
            "User    text", {"test.docx": markdown, "second.docx": markdown},
            None, {},
        )
        original = copy.deepcopy(message)
        result = normalize_history_documents(message)
        self.assertEqual(message, original)
        self.assertEqual(result["content"][-1], message["content"][-1])
        images = lambda m: [b for b in m["content"] if b["type"] == "image_url"]
        self.assertEqual(images(result), images(message))
        text = "".join(b.get("text", "") for b in result["content"])
        self.assertIn("| --- | --- |", text)
        self.assertNotIn("----------------", text)
        self.assertEqual(normalize_history_documents(result), result)
        assistant = {**message, "role": "assistant"}
        self.assertIs(normalize_history_documents(assistant), assistant)

    def test_ambiguous_history_is_preserved(self):
        for text in (
            '# FILE "a" BEGIN\n| A | B |\n|------|------|',
            '# FILE "a" BEGIN\n| A | B |\n|------|------|\n# FILE "b" END',
            'prefix # FILE "a" BEGIN\n| A | B |\n|------|------|\n# FILE "a" END',
            '# FILE "a" BEGIN\n# FILE "b" BEGIN\n| A | B |\n|------|------|\n# FILE "a" END',
        ):
            message = {"role": "user", "content": [{"type": "text", "text": text}]}
            self.assertEqual(normalize_history_documents(message), message)


if __name__ == "__main__":
    unittest.main()
