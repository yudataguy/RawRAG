import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest

from utils import SearchEngine, SearchEngineError


class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.search_engine = SearchEngine()
        self.documents = [
            "The quick brown fox jumps over the lazy dog",
            "A quick brown dog outpaces a fast fox",
            "The lazy dog sleeps all day",
        ]

    def test_calculate_idf(self):
        tokenized_docs = [doc.split() for doc in self.documents]
        idf = self.search_engine.calculate_idf(tokenized_docs)
        self.assertIsInstance(idf, dict)
        self.assertGreater(
            idf["sleeps"], idf["quick"]
        )  # 'sleeps' appears in fewer documents

    def test_calculate_idf_empty_corpus(self):
        with self.assertRaises(SearchEngineError):
            self.search_engine.calculate_idf([])

    def test_bm25_search_normal(self):
        query = "quick brown fox"
        results = self.search_engine.bm25_search(query, self.documents, k=2)
        print(results)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], 1)  # First document should be the most relevant

    def test_bm25_search_empty_query(self):
        with self.assertRaises(SearchEngineError):
            self.search_engine.bm25_search("", self.documents)

    def test_bm25_search_empty_documents(self):
        with self.assertRaises(SearchEngineError):
            self.search_engine.bm25_search("test", [])

    def test_bm25_search_invalid_k(self):
        with self.assertRaises(SearchEngineError):
            self.search_engine.bm25_search("test", self.documents, k=0)


if __name__ == "__main__":
    unittest.main()
    unittest.main()
