
# Quantitative Stock Decision Assistant — Local AI + Research Papers

A Streamlit-based quantitative-finance decision-support platform. Python calculates stock metrics and rankings, while a locally running Qwen3 model explains the results and answers questions using relevant passages from uploaded academic research papers.

> **Educational research tool only.** This application is not personalized financial advice, a guaranteed forecast, or an automatic buy/sell system.

## 1. Project Features

### Quantitative Finance Engine

The `quant_engine.py` module performs the numerical calculations independently of the LLM, including:

- Stock price retrieval
- Returns and six-month momentum
- Sharpe ratio
- Moving-average price trend
- Annualized volatility
- Maximum drawdown
- Peer-relative factor scores and composite ranking
- Historical strategy backtesting

The LLM does not calculate, modify, or determine the stock rankings.

### Local AI with Ollama and Qwen3

The application uses Ollama to communicate with a locally running Qwen3 8B language model.

- Model: `qwen3:8b`
- Runtime: Ollama
- API endpoint: `http://127.0.0.1:11434/api/chat`
- Communication: Python sends HTTP requests to the local Ollama service.
- No Gemini API key or cloud LLM API is required.

The LLM interprets the quantitative results and generates natural-language explanations. It does not independently fetch stock prices or alter the numerical calculations.

### Research Papers and RAG

The application includes a local research-paper library.

1. Users upload PDF research papers through the Research Library interface.
2. `pypdf` extracts text while preserving page numbers.
3. Text is divided into overlapping passages.
4. Scikit-learn TF-IDF and cosine similarity retrieve relevant passages.
5. Python sends the selected passages, the user's question, and quantitative results to the local Ollama model.
6. Qwen3 generates an answer based on the supplied evidence.

The local research library includes papers such as:

- Deep Learning in the Stock Market: A Systematic Survey of Practice, Backtesting, and Applications.
- Empirical Asset Pricing via Machine Learning: Evidence from the European Stock Market.
- The Probability of Backtest Overfitting.

The application displays retrieved document names and page references to support its answers.

## 2. System Architecture

```text
                 USER
                   |
                   v
           Streamlit Interface
                   |
          +--------+--------+
          |                 |
          v                 v
  Quantitative Engine    PDF Library
     (Python)             (Local)
          |                 |
          |            PDF Extraction
          |                 |
          |            TF-IDF Retrieval
          |                 |
          +--------+--------+
                   |
                   v
          Python AI Assistant
                   |
             HTTP Request
                   |
                   v
          Local Ollama API
          localhost:11434
                   |
                   v
              Qwen3 8B
                   |
          Natural-language
             explanation
                   |
                   v
          Streamlit Interface
```

## 3. Installation Requirements

- macOS or another supported operating system
- Python
- Ollama
- Qwen3 8B model
- Project dependencies listed in `requirements.txt`

## 4. Setup Instructions (macOS)

### Step 1: Install Ollama

Download Ollama:

https://ollama.com/download

Install and open the application.

### Step 2: Download the local AI model

Open Terminal and run:

```bash
ollama pull qwen3:8b
```

Verify the model:

```bash
ollama list
```

### Step 3: Open the project in VS Code

Open the `KT-yun2` project folder.

Open Terminal inside VS Code.

### Step 4: Create and activate a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 5: Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 6: Start the application

Make sure the Ollama application is running, then execute:

```bash
python -m streamlit run app.py
```

The application normally opens at:

http://localhost:8501

No Gemini API key or `.env` configuration is required.

## 5. Suggested First Use

1. Start with the demo dataset to explore the dashboard. Demo prices are synthetic and must not be treated as real stock prices.
2. Open the Research Library and confirm that the uploaded research papers have been indexed.
3. Open the AI Assistant.
4. Ask questions such as:
   - Explain the difference between momentum and Sharpe ratio in this ranking.
   - What are the limitations of backtesting a momentum strategy? Use the research papers.
   - Compare the leading stocks using their quantitative metrics.
5. For real-stock research, select Yahoo Finance and check the displayed data period and source before interpreting the results.

## 6. How Python Communicates with the Local LLM

The application uses the Ollama HTTP API.

In `research_assistant.py`, Python constructs a request containing:

- The system instructions
- The user's question
- Current quantitative results
- Relevant retrieved research passages
- Recent conversation history

Python sends the request to:

```text
http://127.0.0.1:11434/api/chat
```

Ollama processes the request using the locally installed Qwen3 model and returns a generated response.

Python then returns the response to Streamlit for display.

The quantitative engine remains independent of this process.

## 7. Privacy and Data Handling

- The LLM runs locally through Ollama.
- User questions, quantitative context, and retrieved research passages are processed by the local model through the local API.
- PDF extraction and TF-IDF retrieval are performed locally.
- The application does not require a cloud LLM API key.
- Yahoo Finance market-data retrieval requires an internet connection and sends data requests to the external data provider.
- Uploaded documents and generated indexes are stored on the computer running the application.

## 8. Testing

Run the existing unit tests:

```bash
python -m unittest discover -s tests -v
```

Check Python syntax:

```bash
python -m py_compile app.py research_assistant.py quant_engine.py
```

Check that Ollama is running:

```bash
ollama list
```

## 9. Known Limitations

- TF-IDF retrieval is lexical rather than semantic embedding retrieval.
- Scanned or image-only PDFs are not supported by the current text-extraction process.
- Qwen3's explanations may contain errors and should be checked against the underlying data and cited research.
- Yahoo Finance data may be delayed, incomplete, or unavailable.
- Stock scoring weights are initial educational assumptions and are not proven optimal.
- Backtesting does not fully model taxes, market impact, liquidity, delisted securities, or survivorship bias.
- Historical performance does not guarantee future results.

## 10. Project Structure

```text
KT-yun2/
├── app.py
├── quant_engine.py
├── research_assistant.py
├── requirements.txt
├── README.md
├── .env.example
├── .gitignore
├── tests/
└── research paper PDFs
```
