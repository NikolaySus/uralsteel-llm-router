import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

import util


def make_response(
    status, *, content=b"", json_data=None, url="https://files.test/a"
):
    request = httpx.Request("GET", url)
    if json_data is not None:
        return httpx.Response(status, json=json_data, request=request)
    return httpx.Response(status, content=content, request=request)


class AsyncClientMock:
    def __init__(self, get_response, post_response):
        self.get = AsyncMock(return_value=get_response)
        self.post = AsyncMock(return_value=post_response)
        self.aclose = AsyncMock()


class TestDoclingConversion(unittest.IsolatedAsyncioTestCase):
    async def test_uploads_downloaded_bytes_to_file_endpoint(self):
        file_data = b"PK\x03\x04word/" + b"content"
        client = AsyncClientMock(
            make_response(200, content=file_data),
            make_response(
                200,
                json_data={
                    "status": "success",
                    "errors": [],
                    "document": {
                        "filename": "report.docx",
                        "md_content": "converted",
                    },
                },
                url="http://docling:5001/v1/convert/file",
            ),
        )

        with patch.object(util.httpx, "AsyncClient", return_value=client):
            result = await util.convert_to_md_async(
                "https://files.test/report.docx?signature=secret",
                "docling:5001",
            )

        self.assertEqual(result, ("report.docx", "converted"))
        client.get.assert_awaited_once()
        self.assertEqual(
            client.post.await_args.args[0],
            "http://docling:5001/v1/convert/file",
        )
        request_data = client.post.await_args.kwargs
        self.assertEqual(request_data["data"]["from_formats"], "docx")
        self.assertEqual(request_data["data"]["to_formats"], "md")
        self.assertEqual(
            request_data["files"]["files"][:2], ("report.docx", file_data)
        )
        client.aclose.assert_awaited_once()

    async def test_detected_format_gets_synthetic_filename(self):
        file_data = b"PK\x03\x04word/" + b"content"
        client = AsyncClientMock(
            make_response(200, content=file_data),
            make_response(
                200,
                json_data={
                    "status": "success",
                    "errors": [],
                    "document": {"md_content": "converted"},
                },
                url="http://docling:5001/v1/convert/file",
            ),
        )

        with patch.object(util.httpx, "AsyncClient", return_value=client):
            result = await util.convert_to_md_async(
                "https://files.test/download", "docling:5001"
            )

        self.assertEqual(result, ("document.docx", "converted"))
        upload = client.post.await_args.kwargs["files"]["files"]
        self.assertEqual(upload[0], "document.docx")

    async def test_download_http_error_does_not_call_docling(self):
        client = AsyncClientMock(
            make_response(403, content=b"denied"), MagicMock()
        )

        with patch.object(util.httpx, "AsyncClient", return_value=client):
            result = await util.convert_to_md_async(
                "https://files.test/report.md?signature=secret",
                "docling:5001",
            )

        self.assertEqual(result, (None, None))
        client.post.assert_not_awaited()
        client.aclose.assert_awaited_once()

    async def test_docling_http_error_uses_docx_fallback(self):
        file_data = b"PK\x03\x04word/" + b"content"
        client = AsyncClientMock(
            make_response(200, content=file_data),
            make_response(
                504,
                content=b"gateway timeout",
                url="http://docling:5001/v1/convert/file",
            ),
        )

        with (
            patch.object(util.httpx, "AsyncClient", return_value=client),
            patch.object(
                util,
                "docx_to_markdown_via_markitdown",
                return_value="fallback",
            ) as fallback,
        ):
            result = await util.convert_to_md_async(
                "https://files.test/report.docx", "docling:5001"
            )

        self.assertEqual(result, ("report.docx", "fallback"))
        fallback.assert_called_once_with(file_data, "docx")

    async def test_missing_markdown_uses_docx_fallback(self):
        file_data = b"PK\x03\x04word/" + b"content"
        client = AsyncClientMock(
            make_response(200, content=file_data),
            make_response(
                200,
                json_data={
                    "status": "success",
                    "errors": [],
                    "document": {"filename": "report.docx"},
                },
                url="http://docling:5001/v1/convert/file",
            ),
        )

        with (
            patch.object(util.httpx, "AsyncClient", return_value=client),
            patch.object(
                util,
                "docx_to_markdown_via_markitdown",
                return_value="fallback",
            ) as fallback,
        ):
            result = await util.convert_to_md_async(
                "https://files.test/report.docx", "docling:5001"
            )

        self.assertEqual(result, ("report.docx", "fallback"))
        fallback.assert_called_once_with(file_data, "docx")

    async def test_pdf_size_filter_uses_downloaded_bytes_for_fallback(self):
        file_data = b"%PDF" + (b"x" * 100)
        client = AsyncClientMock(
            make_response(200, content=file_data),
            make_response(
                200,
                json_data={
                    "status": "success",
                    "errors": [],
                    "document": {
                        "filename": "a.pdf",
                        "md_content": "short",
                    },
                },
                url="http://docling:5001/v1/convert/file",
            ),
        )

        with (
            patch.object(util.httpx, "AsyncClient", return_value=client),
            patch.object(
                util,
                "pdf_bytes_to_b64_images",
                return_value=["data:image/png;base64,page"],
            ) as fallback,
        ):
            filename, markdown = await util.convert_to_md_async(
                "https://files.test/a.pdf", "docling:5001"
            )

        self.assertEqual(filename, "a.pdf")
        self.assertIn("data:image/png;base64,page", markdown)
        fallback.assert_called_once_with(file_data)

    def test_safe_http_text_redacts_query_and_truncates(self):
        value = "failed https://files.test/a?signature=secret " + ("x" * 2000)
        safe = util._safe_http_text(value)

        self.assertNotIn("secret", safe)
        self.assertIn("?<redacted>", safe)
        self.assertLessEqual(len(safe), 1027)


if __name__ == "__main__":
    unittest.main()
