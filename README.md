# Quantitative Stock Decision Assistant

A standalone Streamlit application for comparing stocks using transparent, historical quantitative-finance indicators. It includes a peer-relative score, price charts, CSV import, and a lagged monthly momentum backtest.

> **Educational use only.** The platform is a research and comparison tool, not personalised financial advice, a guaranteed forecast, or an automatic buy/sell system. A higher score is relative to the selected peer group and historical period.

## Features

- **Stock screener:** compare a list of ticker symbols using six-month momentum, Sharpe ratio, price trend, annualised volatility, and maximum drawdown.
- **Transparent score:** the initial score weights are Momentum 30%, Sharpe 25%, Price trend 20%, Low volatility 15%, and Drawdown control 10%. Factors are converted into peer-relative percentiles before weighting.
- **Individual analysis:** view historical adjusted prices with 50- and 200-day moving averages.
- **Backtesting:** evaluate a monthly, cross-sectional six-month momentum strategy against an equal-weight benchmark. The signal is lagged and estimated transaction costs are deducted.
- **CSV import:** use your own historical price data if live data is unavailable.
- **Offline demo:** explore the interface with synthetic data. Demo prices are clearly marked and must never be interpreted as real market data.
- **Exports:** download rankings, price history, and backtest results as CSV.

## Requirements

- macOS, Windows, or Linux
- Python 3.10 or newer (Python 3.11 or 3.12 recommended)
- Internet connection for Yahoo Finance data; not required for demo data or CSV imports

## Setup in VS Code

1. Download and unzip this project.
2. In VS Code, choose **File → Open Folder** and open `quantitative_stock_decision_assistant`.
3. Open **Terminal → New Terminal**. Confirm the terminal is in this project directory by running `pwd` and `ls`. You should see `app.py`, `quant_engine.py`, and `requirements.txt`.
4. Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with:

```powershell
.venv\Scripts\Activate.ps1
```

5. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

6. Start the app:

```bash
python -m streamlit run app.py
```

7. Streamlit will print a local URL (normally `http://localhost:8501`) and may open it in your browser.

To stop the app, click the terminal running Streamlit and press `Ctrl+C`.

## How to use

### Option A — Demo data

Select **Demo dataset** in the left sidebar. The app generates synthetic, reproducible prices so the dashboard can be explored without internet access. The app marks this data as synthetic.

### Option B — Yahoo Finance

1. Select **Yahoo Finance** in the sidebar.
2. Enter ticker symbols separated by commas, for example `AAPL, MSFT, NVDA, AMZN, GOOGL, META`.
3. Select a historical period and annual risk-free-rate assumption.
4. Click **Load / refresh market data**.

The app retrieves auto-adjusted daily closing prices through `yfinance`. yfinance is an open-source interface to Yahoo Finance's publicly available APIs and is intended for research/educational and personal use. It is not affiliated with or endorsed by Yahoo. Review the data provider's current terms and limitations before using data beyond personal research. Data may be incomplete, delayed, or unavailable.

### Option C — Upload CSV files

Upload one or more CSV files. Supported formats:

**One file per ticker** (ticker is taken from the filename):

```csv
Date,Close
2024-01-02,100.25
2024-01-03,101.10
```

`Adj Close` is also accepted and takes precedence over `Close` if both exist.

**One combined wide file:**

```csv
Date,AAPL,MSFT,NVDA
2024-01-02,185.64,370.87,48.17
2024-01-03,184.25,373.25,49.48
```

Use adjusted closing prices if your data provider supplies them. Prices should be daily, positive, and consistently denominated. The app does not independently adjust uploaded data for dividends or splits.

## Score methodology

For each stock, the application calculates:

- **Period return:** last price divided by first price in the selected data, minus one.
- **6-month momentum:** latest adjusted price divided by the price 126 trading observations earlier, minus one.
- **Sharpe ratio:** (mean daily return × 252 − annual risk-free rate) / (standard deviation of daily returns × √252).
- **Price trend:** 50 points for price above the 50-day average and 50 points for price above the 200-day average, using whichever averages are available.
- **Annualised volatility:** daily return standard deviation × √252.
- **Maximum drawdown:** minimum of price divided by its running maximum, minus one.

Each factor is converted to a percentile within the selected stock universe. The weighted score is thus **relative**, not an absolute probability of future gains. If a factor is missing for a stock, available weights are re-scaled for that stock. Scores and ranks can change when the selected peer group changes.

The weights are initial project assumptions, not empirically proven optimal values. Test alternative weights and periods as part of the research project.

## Backtest methodology and limitations

The backtest resamples prices to month-end observations. At each month, it uses the six-month momentum signal from the prior month-end to select the top N stocks for the following month. The selected stocks are equally weighted. A simple transaction-cost estimate is deducted using one-way portfolio turnover and the entered cost in basis points. The benchmark is an equal-weight portfolio of the same selected stock universe.

This is a simplified educational backtest. It does not currently model taxes, market impact, detailed slippage, delisted stocks, survivorship bias in the user's chosen universe, liquidity constraints, or corporate-action issues in arbitrary CSV uploads. It does not guarantee future performance. Avoid tuning a strategy repeatedly on the same test period; use a separate out-of-sample period.

## Project structure

```text
quantitative_stock_decision_assistant/
├── app.py             # Streamlit interface
├── quant_engine.py    # Data loading, indicators, scoring and backtesting
├── requirements.txt   # Python dependencies
├── README.md          # Setup and methodology
└── tests/
    └── test_quant_engine.py
```

## GitHub safety

Do not commit API keys, passwords, `.env` files, private datasets, or personal account information. This version does not require an API key. Check data provider terms before redistributing market data.
