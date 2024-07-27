import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest

from utils import KeywordExtractor, KeywordExtractorError


class TestKeywordExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = KeywordExtractor()

    def test_extract_keywords_earth_based(self):
        text = "On September 15, 2023, Sarah Johnson gave a presentation about climate change at the United Nations headquarters in New York City."
        result = self.extractor.extract_keywords(text, num_keywords=5)
        self.assertIn("keywords", result)
        self.assertIn("when", result)
        self.assertIn("who", result)
        self.assertIn("where", result)

        # Check for correct identification of entities
        self.assertIn("Sarah Johnson", result["who"])
        self.assertIn("New York City", result["where"])
        self.assertIn("September 15, 2023", result["when"])

        # Check for relevant keywords
        self.assertTrue(
            set(["presentation"]).issubset(set(result["keywords"]))
        )

    def test_extract_keywords_empty_input(self):
        with self.assertRaises(KeywordExtractorError):
            self.extractor.extract_keywords("")

    def test_extract_keywords_invalid_num_keywords(self):
        with self.assertRaises(KeywordExtractorError):
            self.extractor.extract_keywords("Test text", num_keywords=0)

    def test_extract_keywords_no_entities(self):
        text = "This is a simple test text without any named entities."
        result = self.extractor.extract_keywords(text, num_keywords=2)
        self.assertEqual(len(result["keywords"]), 2)
        self.assertEqual(result["who"], [])
        self.assertEqual(result["where"], [])
        self.assertEqual(result["when"], [])


if __name__ == "__main__":
    unittest.main()
