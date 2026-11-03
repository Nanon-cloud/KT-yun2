# Quantitative Stock Decision Assistant

## 1. Project overview

The **Quantitative Stock Decision Assistant** is an educational, data-driven platform designed to help users compare publicly traded stocks and identify companies that may be worth further investment research.

The platform applies quantitative finance techniques to historical market data. It calculates measurable indicators, compares stocks using a transparent scoring framework, evaluates simple investment strategies through backtesting, and explains the results in plain language.

The platform is intended to support investment research and decision-making. It does **not** guarantee returns, predict prices with certainty, or provide personalised financial advice. A high score is a screening result—not an instruction to buy.

## 2. Problem and motivation

Investors often face a large amount of financial information when comparing stocks. Looking at price movements or individual indicators in isolation can make it difficult to compare opportunities consistently. It can also encourage decisions based on recent performance without properly considering risk.

Quantitative finance provides a structured way to analyse market data using statistics, mathematical models, and systematic rules. This project aims to make several of those methods accessible through one simple interface.

## 3. Research question

**How can quantitative indicators be combined to compare stocks and identify potentially attractive investment candidates while accounting for risk?**

A related research question is:

**Does a systematic, momentum-based stock-selection strategy perform better than a simple benchmark after accounting for risk and transaction costs?**

These questions can be refined depending on the stocks, market, and time period selected by the group.

## 4. Target users

- Students learning quantitative finance and investment analysis.
- Beginner investors who want a more systematic way to compare stocks.
- Researchers who want to test simple investment hypotheses using historical data.

## 5. Main platform features

### 5.1 Stock screener and comparison

Users select a group of stocks and a historical analysis period. The platform retrieves or imports historical price data, calculates quantitative indicators, and presents the results in a comparison table.

Possible features include:

- Search for stocks by ticker symbol.
- Compare several stocks over the same period.
- View historical price and return charts.
- Sort stocks by individual indicators or an overall quantitative score.
- Review the data period and latest available observation date.

### 5.2 Quantitative indicators

The first version can calculate the following indicators:

| Indicator | What it measures | How it may be used |
|---|---|---|
| Historical return | Change in price over a selected period | Measures past performance; does not guarantee future returns. |
| Momentum | Strength of recent performance over a defined lookback period | Identifies stocks with relatively strong or weak recent trends. |
| Moving averages | Average price over selected windows, such as 50 and 200 trading days | Helps assess the direction of a price trend. |
| Annualised volatility | Historical variability of returns | Provides a measure of how much returns have fluctuated. |
| Sharpe ratio | Excess return relative to return volatility | Helps compare historical risk-adjusted performance. |
| Maximum drawdown | Largest peak-to-trough decline over the period | Shows the severity of the worst historical decline. |

The platform should state the formula, lookback window, risk-free-rate assumption, and data frequency used for every metric. For example, an annualised Sharpe ratio calculated from daily returns depends on the annualisation convention and the chosen risk-free rate.

### 5.3 Transparent quantitative score

To create an easy-to-read comparison, the first version can use a weighted score based on five factors:

| Factor | Proposed weight | Interpretation |
|---|---:|---|
| Momentum | 30% | Rewards stronger recent performance relative to the comparison group. |
| Sharpe ratio | 25% | Rewards stronger historical return per unit of volatility. |
| Price trend | 20% | Considers whether price is above selected moving averages. |
| Low volatility | 15% | Rewards lower historical return variability. |
| Maximum drawdown | 10% | Rewards smaller historical peak-to-trough losses. |
| **Total** | **100%** | |

These weights are **initial design choices for an educational prototype**, not empirically proven optimal weights. Before combining the factors, the platform should put them on comparable scales, explain the normalisation method, and ensure that higher scores consistently represent the preferred direction. The group should test alternative weights and examine whether the rankings are sensitive to those choices.

The resulting score is relative to the selected stocks and time period. It should not be interpreted as an objective probability that a stock will rise.

### 5.4 Strategy backtesting

Users can test a simple rule-based strategy using historical data. An initial example is a momentum strategy that selects the stocks with the strongest trailing returns and rebalances at a defined interval.

The backtest can report:

- Cumulative strategy return.
- Annualised return.
- Annualised volatility.
- Sharpe ratio.
- Maximum drawdown.
- Number of trades and assumed transaction costs.
- Performance against a stated benchmark.

The backtest must avoid look-ahead bias: a strategy may only use information that would have been available at the time of each decision. It should also account for transaction costs and clearly describe assumptions about rebalancing, missing data, dividends, and survivorship bias where relevant. Historical performance is not a guarantee of future results.

### 5.5 Research-paper library and AI explanations (optional extension)

The platform may include a library where users upload academic papers and course materials related to quantitative finance. The papers can provide background for explanations of concepts such as momentum, asset pricing, machine learning, and backtest overfitting.

An optional AI assistant can answer questions using retrieved passages from the uploaded documents and show the source document and page number. AI explanations should distinguish between information found in the research papers and calculations produced by the platform. The assistant should not invent citations or treat an academic paper as proof that a particular stock is suitable to buy.

## 6. Example user workflow

1. The user opens the platform and chooses a market or enters stock ticker symbols.
2. The user selects a historical date range and comparison settings.
3. The platform obtains historical price data or accepts a user-provided CSV file.
4. The platform calculates returns, momentum, moving averages, volatility, Sharpe ratio, and maximum drawdown.
5. The platform normalises the selected factors and calculates a transparent comparison score.
6. The user reviews the ranking, charts, individual indicators, and risk measures.
7. The user optionally runs a backtest to evaluate a strategy under stated assumptions.
8. The user can consult the research-paper library or AI explanation feature to understand the quantitative results.
9. The platform presents limitations and encourages further research before any investment decision.

## 7. Data and technical approach

### Data inputs

The platform needs historical stock prices, preferably adjusted prices when the data source provides them, so that returns can account for corporate actions such as dividends and stock splits. It should record the data provider, ticker, currency, frequency, date range, and retrieval date.

For the prototype, the group can use a market-data library or a CSV upload. Data-provider coverage, licensing, delays, and availability vary, so the chosen source should be checked and documented. Missing or inconsistent observations should be handled explicitly rather than silently filled.

### Suggested technology stack

| Component | Suggested technology | Purpose |
|---|---|---|
| User interface | Streamlit | Build an interactive dashboard in Python. |
| Data processing | pandas and NumPy | Clean price data and calculate returns and indicators. |
| Quantitative analysis | scikit-learn or custom Python functions | Normalise factors and support optional models. |
| Charts | Plotly | Display price, performance, and risk charts. |
| Market data | A selected data API/library or CSV upload | Supply historical prices. |
| Optional AI explanations | A language-model API | Explain calculations and summarise uploaded research with source references. |
| Local storage | CSV, SQLite, or local files | Save settings, uploaded documents, and analysis results. |

The core scoring and backtesting calculations should be deterministic Python functions. They should not depend on an AI language model. This makes results easier to reproduce, test, and audit.

## 8. Evaluation plan

The group can evaluate the project in several ways:

1. **Calculation checks:** Verify indicator calculations against manually calculated examples or a trusted finance reference.
2. **Ranking transparency:** Confirm that users can inspect each factor and understand how the overall score was produced.
3. **Backtest integrity:** Check for look-ahead bias, unrealistic execution assumptions, and omitted transaction costs.
4. **Out-of-sample testing:** Evaluate a strategy on a period that was not used to design or tune it.
5. **Benchmark comparison:** Compare risk and return with an appropriate benchmark over the same dates.
6. **Sensitivity analysis:** Change factor weights and lookback periods to see whether the results change substantially.
7. **Usability testing:** Ask users whether they can understand the rankings, indicators, assumptions, and limitations.

A successful project is not one that always identifies a winning stock. It is one that produces clear, reproducible comparisons and communicates uncertainty and risk honestly.

## 9. Scope of the first prototype

To keep the project achievable, the minimum viable version should include:

- A simple Streamlit dashboard.
- A way to enter or select a small set of stock tickers.
- Historical price data from one documented source or CSV upload.
- Calculations for returns, momentum, moving averages, volatility, Sharpe ratio, and maximum drawdown.
- A transparent weighted scoring table.
- At least one price/performance chart.
- Clear data dates, assumptions, and risk disclaimers.

Backtesting, AI-generated explanations, and a research-paper library can be added after the core calculations work correctly.

## 10. Expected deliverables

1. A functional quantitative stock comparison platform.
2. A documented methodology explaining the selected indicators, formulas, weights, and assumptions.
3. Stock comparison outputs and visualisations.
4. A backtesting demonstration, if included in the final scope.
5. A short evaluation discussing performance, limitations, and possible improvements.
6. A public GitHub repository containing functional source code and setup instructions, but excluding API keys and private credentials.

## 11. Important limitations and responsible use

- A quantitative score is a screening tool, not a buy/sell instruction.
- Historical returns and statistical relationships may not continue in the future.
- Results depend on data quality, date range, stock universe, formulas, and model assumptions.
- Stocks with missing data or very short histories may not be comparable.
- A backtest can look successful because of overfitting, look-ahead bias, survivorship bias, or unrealistic trading assumptions.
- AI-generated explanations can be incorrect. Users should be able to inspect the underlying numbers and source documents.
- The platform is an educational research tool and does not provide personalised financial advice.

## 12. Short project description for the assignment

**Our project is a Quantitative Stock Decision Assistant that helps users compare stocks and identify potentially attractive investment candidates using quantitative finance. The platform analyses historical stock data through indicators such as momentum, Sharpe ratio, volatility, moving averages, and maximum drawdown. It combines these factors into a transparent scoring system, allowing users to rank and compare stocks while considering risk. The platform may also include strategy backtesting and AI-assisted explanations supported by academic research papers. Its purpose is to support systematic investment research rather than guarantee returns or make decisions on behalf of users.**
