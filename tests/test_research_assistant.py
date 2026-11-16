import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from research_assistant import (
    build_prompt,
    chunk_page_text,
    clean_model_output,
    generate_local_answer,
    is_local_url,
    list_local_models,
    normalise_local_base_url,
    retrieve_relevant_chunks,
)


class _FakeLocalLLM(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible server (the API Ollama / LM Studio expose)."""
    last_request: dict = {}
    reply = "Momentum is a 6-month return [paper.pdf, p. 3]."

    def log_message(self, *args):  # silence test output
        pass

    def _send(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"data": [{"id": "llama3.2:latest"}, {"id": "qwen2.5:3b"}]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).last_request = body
        if body.get("model") == "missing-model":
            self._send(404, {"error": "model not found"})
        elif self.path == "/v1/chat/completions":
            self._send(200, {"choices": [{"message": {"role": "assistant", "content": type(self).reply}}]})
        else:
            self._send(404, {"error": "not found"})


class LocalLLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _FakeLocalLLM)
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        _FakeLocalLLM.reply = "Momentum is a 6-month return [paper.pdf, p. 3]."

    def test_generates_answer_and_sends_evidence(self):
        chunks = [{"text": "Momentum ranks stocks by past return.", "source": "paper.pdf", "page": 3, "similarity": 0.5}]
        answer = generate_local_answer(self.url, "llama3.2", "What is momentum?", "AAA score 80", chunks)
        self.assertIn("[paper.pdf, p. 3]", answer)
        sent = _FakeLocalLLM.last_request
        self.assertEqual(sent["model"], "llama3.2")
        self.assertEqual(sent["messages"][0]["role"], "system")
        user_prompt = sent["messages"][1]["content"]
        self.assertIn("AAA score 80", user_prompt)
        self.assertIn("paper.pdf", user_prompt)
        self.assertIn("What is momentum?", user_prompt)

    def test_strips_hidden_reasoning(self):
        _FakeLocalLLM.reply = "<think>secret chain of thought</think>Final answer."
        self.assertEqual(generate_local_answer(self.url, "m", "q", "ctx", []), "Final answer.")
        self.assertEqual(clean_model_output("<think>x</think> ok"), "ok")

    def test_lists_models(self):
        self.assertEqual(list_local_models(self.url), ["llama3.2:latest", "qwen2.5:3b"])

    def test_unknown_model_gives_pull_hint(self):
        with self.assertRaises(RuntimeError) as ctx:
            generate_local_answer(self.url, "missing-model", "q", "ctx", [])
        self.assertIn("ollama pull missing-model", str(ctx.exception))

    def test_server_not_running_gives_start_hint(self):
        with self.assertRaises(RuntimeError) as ctx:
            generate_local_answer("http://127.0.0.1:9", "m", "q", "ctx", [], timeout=2)
        self.assertIn("Cannot reach a local AI server", str(ctx.exception))

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            generate_local_answer(self.url, "m", "  ", "ctx", [])

    def test_local_prompt_is_compact(self):
        long_chunk = [{"text": "x" * 5000, "source": "a.pdf", "page": 1, "similarity": 0.1}]
        compact = build_prompt("q", "ctx", long_chunk, max_chunk_chars=1100)
        full = build_prompt("q", "ctx", long_chunk)
        self.assertLess(len(compact), len(full))
        self.assertLess(len(compact), 2500)

    def test_url_helpers(self):
        self.assertEqual(normalise_local_base_url("localhost:11434"), "http://localhost:11434/v1")
        self.assertEqual(normalise_local_base_url("http://localhost:1234/v1/"), "http://localhost:1234/v1")
        self.assertTrue(is_local_url("http://localhost:11434/v1"))
        self.assertFalse(is_local_url("https://api.example.com/v1"))


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
