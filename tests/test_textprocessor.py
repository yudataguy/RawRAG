import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
from typing import List

from utils import TextProcessor, TextProcessorError


class TestTextProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = TextProcessor()

    def test_text_splitter_normal(self):
        text = "This is a test sentence. It should be split into two parts."
        result = self.processor.text_splitter(text, char_limit=35)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "This is a test sentence.")
        self.assertEqual(result[1], "It should be split into two parts.")

    def test_text_splitter_with_overlap(self):
        text = "First sentence. Second sentence. Third sentence."
        result = self.processor.text_splitter(text, char_limit=20, overlap=5)
        self.assertEqual(len(result), 3)
        self.assertTrue(result[1].startswith("ence. Second"))
        self.assertTrue(result[2].startswith("ence. Third"))

    def test_text_splitter_empty_input(self):
        with self.assertRaises(TextProcessorError):
            self.processor.text_splitter("")

    def test_text_splitter_invalid_char_limit(self):
        with self.assertRaises(TextProcessorError):
            self.processor.text_splitter("Test", char_limit=0)

    def test_tokenize_normal(self):
        text = "Hello, world! This is a test."
        result = self.processor.tokenize(text)
        print(result)
        self.assertIsInstance(result, List)
        self.assertEqual(len(result), 6)
        self.assertEqual(result, ["Hello", "world", "This", "is", "a", "test"])

    def test_tokenize_empty_input(self):
        with self.assertRaises(TextProcessorError):
            self.processor.tokenize("")

    def test_tokenize_non_string_input(self):
        with self.assertRaises(TextProcessorError):
            self.processor.tokenize(123)


if __name__ == "__main__":
    unittest.main()
    unittest.main()
