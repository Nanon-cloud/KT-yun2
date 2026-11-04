import unittest

from research_assistant import chunk_page_text, retrieve_relevant_chunks


class ResearchAssistantTests(unittest.TestCase):
    def test_chunk_page_text_returns_chunks(self):
        text = ("Quantitative finance uses momentum, volatility, and Sharpe ratio " * 80).strip()
        chunks = chunk_page_text(text, chunk_chars=300, overlap=50)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))

    def test_retrieval_prefers_relevant_passage(self):
        chunks = [
            {"text": "Backtest overfitting occurs when a strategy is tuned to historical noise and fails out of sample.",
             "source": "Backtest paper.pdf", "page": 4},
            {"text": "Portfolio diversification spreads capital across assets and sectors.",
             "source": "Portfolio paper.pdf", "page": 2},
        ]
        result = retrieve_relevant_chunks("What is backtest overfitting?", chunks=chunks, top_k=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source"], "Backtest paper.pdf")
        self.assertEqual(result[0]["page"], 4)
        self.assertGreater(result[0]["similarity"], 0)

    def test_empty_query_returns_no_chunks(self):
        self.assertEqual(retrieve_relevant_chunks("", chunks=[{"text": "finance"}]), [])


if __name__ == "__main__":
    unittest.main()
