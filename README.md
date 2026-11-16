# Quantitative Stock Decision Assistant — LLM + Research Papers

A Streamlit-based quantitative-finance decision-support prototype. Python calculates the stock metrics and rankings; a **local AI model running on your own computer** (via Ollama or LM Studio) explains the calculated results and can use passages retrieved from the group's uploaded research papers. Gemini is kept only as an optional cloud alternative.

> **Educational research tool only.** It is not personalised financial advice, a guaranteed forecast, or an automatic buy/sell system. A higher relative score is not proof that a stock is undervalued or will outperform.

## What changed in this version

### Deterministic quantitative engine

The existing `quant_engine.py` remains responsible for price data, returns, six-month momentum, Sharpe ratio, moving-average trend, annualised volatility, maximum drawdown, peer-relative factor scores, and the lagged monthly momentum backtest. The LLM does not alter or calculate these values.

### Local AI connection (default)

The AI Assistant tab talks to a **local language model** through the OpenAI-compatible HTTP API that Ollama, LM Studio and llama.cpp's server all provide. No cloud account, API key, or internet connection is needed for the explanation step. The request includes:

1. The user's question.
2. A compact, current summary of the metrics produced by the Python quantitative engine.
3. Up to six recent conversation messages (shortened for small models).
4. Up to five relevant paper passages retrieved locally from the Research Library (shortened to fit small context windows).

The local model explains and contextualises results. It does not fetch market prices, execute trades, or change stock rankings. The default is `llama3.2` on Ollama (`http://localhost:11434/v1`); change `LOCAL_LLM_MODEL` / `LOCAL_LLM_BASE_URL` in `.env` or in the AI settings panel. The panel has a **Test connection** button that lists the installed models and tells you what to run if the server or model is missing. If the address you enter is not on your own computer, the app warns you that your data would leave it.

### Optional: Gemini (cloud)

Choose *Gemini (optional cloud API)* in the AI provider selector (or set `LLM_PROVIDER=gemini`). This needs `python -m pip install -r requirements-gemini.txt`, a `GEMINI_API_KEY`, and sends the question, metric summary and selected passages to Google. The default model name is `gemini-3.8-flash`; change `GEMINI_MODEL` if needed.

### Local PDF knowledge base (RAG)

- Upload PDF papers through the Research Library tab.
- The app extracts selectable text using `pypdf` and preserves page numbers.
- Text is divided into overlapping passages and saved as JSON metadata/index files under `data/index/`.
- Original PDFs are stored under `data/papers/`.
- At query time, `scikit-learn` TF-IDF and cosine similarity retrieve relevant passages locally.
- The app passes only the selected passages to the language model and displays the retrieved source filenames and pages.

The first version does not include OCR, so scanned/image-only PDFs without selectable text cannot be indexed. It also does not use vector embeddings; TF-IDF is a lightweight lexical retrieval baseline suitable for a small class project.

## Setup in VS Code (macOS)

1. Extract the ZIP and open the `quantitative_stock_decision_assistant_llm` folder in VS Code.
2. Open **Terminal → New Terminal**. Run `pwd` and `ls`; you should see `app.py`, `quant_engine.py`, and `requirements.txt`.
3. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

5. Install a local AI runtime. The easiest is [Ollama](https://ollama.com): install it, then in a second terminal run:

```bash
ollama pull llama3.2
ollama serve        # skip if the Ollama app is already running
```

   Any other OpenAI-compatible local server also works (for example LM Studio at `http://localhost:1234/v1`). Larger models answer better; `llama3.2` (3B) is a small default that fits on a laptop.
6. Copy `.env.example` to `.env` (the defaults already point at Ollama):

```bash
cp .env.example .env
```

7. Start the app:

```bash
python -m streamlit run app.py
```

Streamlit normally opens `http://localhost:8501`.

You can also change the local server address and model name in the AI Assistant tab. (If you prefer Gemini, enter an API key there for the current session only; it is never displayed in chat messages.)

## Suggested first use

1. Start with **Demo dataset** to explore the dashboard. The prices are synthetic and must not be treated as live equities data.
2. Open **Research paper library** and upload the group's PDFs, for example:
   - `Deep Learning in the Stock Market: A Systematic Survey of Practice, Backtesting, and Applications`.
   - `Empirical Asset Pricing via Machine Learning: Evidence from the European Stock Market`.
   - `The Probability of Backtest Overfitting`.
3. Confirm that each PDF shows an indexed status, page count, and passage count.
4. Make sure Ollama is running with a model pulled, then use **Test connection** in the AI Assistant settings.
5. Open **AI assistant** and ask questions such as:
   - "Explain the difference between momentum and Sharpe ratio in this ranking."
   - "What are the limitations of backtesting a momentum strategy? Use the research papers."
   - "Compare the leading stocks in the current table, using their quantitative metrics."
6. For real-stock research, select **Yahoo Finance**, load a ticker universe, and check the displayed data period and source before asking the AI to interpret it.

## How the communication with the LLM works

```text
User question
      |
      +------------------------------+
      |                              |
      v                              v
Python quantitative engine     Local PDF library
(stock metrics and scores)     (PDF -> page chunks -> TF-IDF retrieval)
      |                              |
      +---------------+--------------+
                      |
                      v
        Context sent to the LOCAL model
     (question + metrics + retrieved passages)
                      |
                      v
        Local LLM (Ollama / LM Studio) explanation
                      |
                      v
             Answer shown in Streamlit
             with retrieved PDF/page sources
```

## Privacy and GitHub

- Original PDFs and local indexes remain inside `data/` on the computer running the app.
- With the default local provider, the question, metric summary and selected PDF passages stay on your computer. Only fetching Yahoo Finance prices uses the internet, and the demo and CSV data sources work fully offline.
- If you opt into Gemini, the question, current metric summary, and selected relevant PDF passages are sent to Google's Gemini API (never the entire PDF library).
- `.env`, API keys, original PDFs, and local indexes are excluded by `.gitignore`. Do not commit secrets or research PDFs to a public repository.
- If you use the optional Gemini provider, review Google's current Gemini API terms and your institution's data rules before uploading confidential material.

## Tests

Run the tests (no API key, internet connection or local model required):

```bash
python -m unittest discover -s tests -v
```

The tests cover text chunking, local retrieval, and the local-AI client (answer parsing, hidden-reasoning removal, model listing, and helpful errors when the server or model is missing). The local-AI client is tested against a small mock of the OpenAI-compatible API; a real model is only used when you run the app. The Gemini request is not called during tests.

## Known limitations

- Answer quality depends on the local model. Small models (3B parameters) can miss details or ignore the citation format; use a larger model if your computer allows it. Local models also have small context windows, which is why the app shortens passages for them.

- TF-IDF retrieval is lexical, not semantic embedding retrieval; it may miss relevant passages expressed with different vocabulary.
- Scanned PDFs require OCR, which is not included.
- Yahoo Finance data may be unavailable, delayed, or subject to provider terms.
- The score weights are initial educational assumptions. They have not been proven optimal through out-of-sample research.
- The backtest is simplified and does not fully model taxes, market impact, liquidity, delisted securities, or survivorship bias.
- The system is a research aid, not a stock-picking oracle. Check the data, assumptions, and evidence yourself.

## Project structure

```text
quantitative_stock_decision_assistant_llm/
├── app.py
├── quant_engine.py
├── research_assistant.py
├── requirements.txt
├── requirements-gemini.txt   (optional cloud provider)
├── .env.example
├── .gitignore
├── README.md
└── tests/
    └── test_research_assistant.py
```
