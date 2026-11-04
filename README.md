# Quantitative Stock Decision Assistant — LLM + Research Papers

A Streamlit-based quantitative-finance decision-support prototype. Python calculates the stock metrics and rankings; a Gemini LLM explains the calculated results and can use passages retrieved from the group's uploaded research papers.

> **Educational research tool only.** It is not personalised financial advice, a guaranteed forecast, or an automatic buy/sell system. A higher relative score is not proof that a stock is undervalued or will outperform.

## What changed in this version

### Deterministic quantitative engine

The existing `quant_engine.py` remains responsible for price data, returns, six-month momentum, Sharpe ratio, moving-average trend, annualised volatility, maximum drawdown, peer-relative factor scores, and the lagged monthly momentum backtest. The LLM does not alter or calculate these values.

### Gemini LLM connection

The AI Assistant tab calls Google's Gemini API through the official `google-genai` Python SDK. The request includes:

1. The user's question.
2. A compact, current summary of the metrics produced by the Python quantitative engine.
3. Up to six recent conversation messages.
4. Up to five relevant paper passages retrieved locally from the Research Library.

Gemini is used to explain and contextualise results. It does not fetch market prices, execute trades, or change stock rankings. The default model is `gemini-3.8-flash`; change `GEMINI_MODEL` if a different model is enabled for your API key.

### Local PDF knowledge base (RAG)

- Upload PDF papers through the Research Library tab.
- The app extracts selectable text using `pypdf` and preserves page numbers.
- Text is divided into overlapping passages and saved as JSON metadata/index files under `data/index/`.
- Original PDFs are stored under `data/papers/`.
- At query time, `scikit-learn` TF-IDF and cosine similarity retrieve relevant passages locally.
- The app sends only the selected passages to Gemini for answer generation and displays the retrieved source filenames and pages.

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

5. Create a Gemini API key in [Google AI Studio](https://aistudio.google.com/app/apikey).
6. Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
```

Then open `.env` in VS Code and replace the placeholder:

```text
GEMINI_API_KEY=your_real_key_here
GEMINI_MODEL=gemini-3.8-flash
```

7. Start the app:

```bash
python -m streamlit run app.py
```

Streamlit normally opens `http://localhost:8501`.

Alternatively, you can enter the Gemini API key in the AI Assistant tab for the current app session only. The key is never displayed in chat messages.

## Suggested first use

1. Start with **Demo dataset** to explore the dashboard. The prices are synthetic and must not be treated as live equities data.
2. Open **Research paper library** and upload the group's PDFs, for example:
   - `Deep Learning in the Stock Market: A Systematic Survey of Practice, Backtesting, and Applications`.
   - `Empirical Asset Pricing via Machine Learning: Evidence from the European Stock Market`.
   - `The Probability of Backtest Overfitting`.
3. Confirm that each PDF shows an indexed status, page count, and passage count.
4. Configure the Gemini API key in `.env` or the AI Assistant settings.
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
             Context sent to Gemini API
             (question + metrics + retrieved passages)
                      |
                      v
              Gemini generated explanation
                      |
                      v
             Answer shown in Streamlit
             with retrieved PDF/page sources
```

## Privacy and GitHub

- Original PDFs and local indexes remain inside `data/` on the computer running the app.
- When using Gemini, the user question, current metric summary, and selected relevant PDF passages are sent to Google's Gemini API. The entire PDF library is not sent as a single upload by this app.
- Internet access and a valid Gemini API key are required for LLM responses. Screener calculations, PDF storage, and TF-IDF retrieval work locally.
- `.env`, API keys, original PDFs, and local indexes are excluded by `.gitignore`. Do not commit secrets or research PDFs to a public repository.
- Review Google's current Gemini API terms and your institution's data rules before uploading confidential material.

## Tests

Run lightweight local tests (no Gemini API key required):

```bash
python -m unittest discover -s tests -v
```

The tests cover the text chunking and local retrieval logic. The Gemini request itself requires a valid API key and an internet connection, so it is not called during tests.

## Known limitations

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
├── .env.example
├── .gitignore
├── README.md
└── tests/
    └── test_research_assistant.py
```
