import unittest
import os

os.environ['DOCLING_ADDRESS'] = 'localhost:5001'

import main

TEST_PDF_URL = 'http://localhost:9001/api/v1/download-shared-object/aHR0cDovLzEyNy4wLjAuMTo5MDAwL3RtcC90ZXN0LnBkZj9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPVlMSThRTVY4UFUxQVMwR0wxODRXJTJGMjAyNTEyMjQlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjUxMjI0VDAxNDM1MlomWC1BbXotRXhwaXJlcz00MzIwMCZYLUFtei1TZWN1cml0eS1Ub2tlbj1leUpoYkdjaU9pSklVelV4TWlJc0luUjVjQ0k2SWtwWFZDSjkuZXlKaFkyTmxjM05MWlhraU9pSlpURWs0VVUxV09GQlZNVUZUTUVkTU1UZzBWeUlzSW1WNGNDSTZNVGMyTmpVNE1ESTJPQ3dpY0dGeVpXNTBJam9pYldsdWFXOWhaRzFwYmlKOS5rb2hsVGtzVHZQQmdhOTZFVlVZNW1Zc0FyelpHaFVMdUVrX1Z0VDJoY3JFYkliZDlOVmhmNFBnOV9oaUxSR3RENWVMU3NuRVhJQWw1WUNNWnJoNGtvUSZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QmdmVyc2lvbklkPW51bGwmWC1BbXotU2lnbmF0dXJlPTdmOWQ2OTFlMThkNDkyMzdhMzRjMDhmMTU2YmUyMTdmOGY1MGU4YmI5ZDM0YmQwZDk4MTUxOGJkMjZmNDI1MTQ'


class TestLlmService(unittest.TestCase):
    """Юнит тесты."""

    def test_01_pdf_to_md(self):
        """Test PDF to markdown conversion."""
        # Call main.convert_to_md with test PDF
        filename, md_content = main.convert_to_md(TEST_PDF_URL)
        
        # Check that both values are not None
        self.assertIsNotNone(filename, "Filename should not be None")
        self.assertIsNotNone(md_content, "Markdown content should not be None")

        # Write md_content to file for inspection
        output_filename = f"{filename}_test.md"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"Markdown content written to: {output_filename}")

        # Additional assertions about the content
        self.assertGreater(len(md_content), 0, "Markdown content should not be empty")
        self.assertIsInstance(filename, str, "Filename should be a string")
        self.assertIsInstance(md_content, str, "Markdown content should be a string")


if __name__ == "__main__":
    unittest.main(verbosity=2)
