from __future__ import annotations

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from research_assistant import (
    delete_library_document,
    generate_gemini_answer,
    index_pdf_bytes,
    list_library_documents,
    make_quant_context,
    retrieve_relevant_chunks,
)
from quant_engine import (
    DEFAULT_WEIGHTS,
    add_composite_score,
    calculate_backtest,
    calculate_metrics,
    fetch_yahoo_prices,
    load_uploaded_csvs,
    make_demo_prices,
)

load_dotenv(override=False)

st.set_page_config(
    page_title="Quantitative Stock Decision Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background: #f7f9fc; }
    [data-testid="stSidebar"] { background: #101b2d; }
    [data-testid="stSidebar"] * { color: #edf3fa !important; }
    .hero { padding: 1.25rem 1.5rem; border-radius: 18px; background: linear-gradient(120deg,#10223c,#174d62); color: white; margin-bottom: 1rem; }
    .hero h1 { color: white; font-size: 2rem; margin: 0; }
    .hero p { color: #d6e7f2; margin: .45rem 0 0 0; }
    .small-note { color: #526174; font-size: .88rem; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e4eaf1; padding: 12px 14px; border-radius: 12px; }
    div[data-testid="stDataFrame"] { border: 1px solid #e4eaf1; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


def fmt_pct(value):
    return "—" if pd.isna(value) else f"{value:.1%}"


def fmt_num(value, digits=2):
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def render_research_library() -> None:
    """Upload, index, list, and delete local quantitative-finance PDFs."""
    st.subheader("Research paper library")
    st.write(
        "Upload academic papers and reports. The app extracts selectable PDF text, "
        "splits it into page-labelled passages, and builds a local TF-IDF search index. "
        "Files and the index stay in this project's data/ folder."
    )
    uploaded_pdfs = st.file_uploader(
        "Upload research papers (PDF)",
        type=["pdf"],
        accept_multiple_files=True,
        key="research_pdf_uploader",
        help="Maximum 25 MB per PDF. Scanned/image-only PDFs are not OCR-processed in this version.",
    )
    if st.button("Upload and index PDFs", type="primary", disabled=not uploaded_pdfs,
                 key="index_research_pdfs"):
        results = []
        progress = st.progress(0)
        for number, uploaded in enumerate(uploaded_pdfs, start=1):
            try:
                result = index_pdf_bytes(uploaded.name, uploaded.getvalue())
                results.append(result)
            except Exception as exc:
                st.error(f"{uploaded.name}: {exc}")
            progress.progress(number / len(uploaded_pdfs))
        for result in results:
            if result["status"] == "indexed":
                st.success(
                    f"Indexed {result['filename']} — {result['pages']} pages, "
                    f"{result['chunks']} searchable passages."
                )
            else:
                st.info(f"Already in library: {result['filename']}")

    documents = list_library_documents()
    st.markdown("### Documents currently indexed")
    if not documents:
        st.info("No research papers uploaded yet. Add your group's quantitative-finance PDFs above.")
    else:
        summary = pd.DataFrame([
            {
                "Document": d["filename"],
                "File size": _format_bytes(d["size_bytes"]),
                "PDF pages": d["pages"],
                "Pages with text": d["pages_with_text"],
                "Searchable passages": d["chunks"],
                "Status": "Indexed",
            }
            for d in documents
        ])
        st.dataframe(summary, use_container_width=True, hide_index=True)
        for document in documents:
            left, right = st.columns([5, 1])
            left.caption(f"{document['filename']} · SHA-256 {document['sha256'][:12]}…")
            if right.button("Delete", key=f"delete_paper_{document['sha256']}"):
                if delete_library_document(document["sha256"]):
                    st.success(f"Deleted {document['filename']} from the local library.")
                    st.rerun()
                else:
                    st.error(f"Could not delete {document['filename']}.")
    st.caption(
        "Privacy note: the original PDFs and local search index are stored on this computer. "
        "When you ask Gemini a question, only the relevant retrieved passages are sent to the Gemini API."
    )


def render_ai_assistant(quant_context: str, is_demo: bool) -> None:
    """Render LLM chat grounded in quantitative metrics and local paper excerpts."""
    st.subheader("AI quantitative-finance research assistant")
    st.write(
        "Gemini explains the app's calculated metrics and can consult relevant passages "
        "from your uploaded papers. Python remains responsible for the numerical scores; "
        "the LLM does not calculate or alter the ranking."
    )
    if is_demo:
        st.warning("The current stock universe uses synthetic demo prices. Do not ask the AI to assess these as real securities.")

    env_key = os.getenv("GEMINI_API_KEY", "").strip()
    default_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    with st.expander("Gemini connection settings", expanded=not bool(env_key)):
        if env_key:
            st.success("GEMINI_API_KEY was found in your .env / environment.")
        else:
            st.info("Add GEMINI_API_KEY to a local .env file, or enter it below for this session only.")
        typed_key = st.text_input(
            "Gemini API key", value="", type="password", key="gemini_api_key_session",
            placeholder="Paste your API key (kept only in this Streamlit session)",
        )
        model_name = st.text_input("Gemini model", value=default_model, key="gemini_model_name")
        api_key = typed_key.strip() or env_key
        st.caption("Get an API key from Google AI Studio. Never commit your API key or .env file to GitHub.")

    if "ai_research_messages" not in st.session_state:
        st.session_state["ai_research_messages"] = []

    for message in st.session_state["ai_research_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Research passages retrieved"):
                    for source in message["sources"]:
                        st.markdown(
                            f"- **{source['source']}**, page {source['page']} "
                            f"(retrieval similarity: {source['similarity']:.2f})"
                        )

    question = st.chat_input("Ask about the stock ranking, risk metrics, strategy, or uploaded papers…", key="quant_ai_chat_input")
    if question:
        st.session_state["ai_research_messages"].append({"role": "user", "content": question})
        if not api_key:
            answer = "Please configure a Gemini API key in the settings above. You can create one in Google AI Studio."
            st.session_state["ai_research_messages"].append({"role": "assistant", "content": answer})
            st.rerun()
        with st.spinner("Retrieving research passages and asking Gemini to analyse the evidence…"):
            try:
                retrieved = retrieve_relevant_chunks(question, top_k=5)
                previous = st.session_state["ai_research_messages"][:-1]
                answer = generate_gemini_answer(
                    api_key=api_key,
                    model=model_name,
                    question=question,
                    quant_context=quant_context,
                    retrieved_chunks=retrieved,
                    conversation=previous,
                )
                sources = [
                    {"source": item.get("source", "Unknown"),
                     "page": item.get("page", "?"),
                     "similarity": item.get("similarity", 0.0)}
                    for item in retrieved
                ]
                st.session_state["ai_research_messages"].append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                st.session_state["ai_research_messages"].append(
                    {"role": "assistant", "content": f"I couldn't complete the Gemini request. {exc}"}
                )
        st.rerun()

    col1, col2 = st.columns([1, 5])
    if col1.button("Clear chat", key="clear_ai_research_chat"):
        st.session_state["ai_research_messages"] = []
        st.rerun()
    col2.caption("Suggested question: Compare the two highest-ranked stocks and explain the risk/return trade-offs using the quantitative results and research papers.")


st.markdown("""
<div class="hero">
  <h1>📈 Quantitative Stock Decision Assistant</h1>
  <p>Compare stocks systematically using momentum, risk-adjusted performance, price trends and historical drawdowns.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Research workspace")
st.sidebar.caption("Choose a data source and define the stock universe.")
data_mode = st.sidebar.radio(
    "Price data source",
    ["Demo dataset", "Yahoo Finance", "Upload CSV"],
    help="Demo data is synthetic. Yahoo Finance data is retrieved through yfinance. CSV upload lets you analyse your own data.",
)

prices = None
source_note = ""
fetch_errors: list[str] = []

if data_mode == "Demo dataset":
    prices = make_demo_prices()
    source_note = prices.attrs.get("source", "Synthetic demo data")
    st.sidebar.info("Demo prices are synthetic. They are only for testing the interface and are not real stocks.")
elif data_mode == "Yahoo Finance":
    default_tickers = "AAPL, MSFT, NVDA, AMZN, GOOGL, META"
    ticker_text = st.sidebar.text_area("Ticker symbols", value=default_tickers, height=90,
                                       help="Separate symbols with commas. Example: AAPL, MSFT, NVDA")
    period = st.sidebar.selectbox("Historical period", ["1y", "2y", "5y", "10y", "max"], index=2)
    risk_free_rate = st.sidebar.number_input("Annual risk-free rate (%)", min_value=0.0, max_value=20.0,
                                              value=4.0, step=0.25) / 100
    if st.sidebar.button("Load / refresh market data", type="primary", use_container_width=True):
        tickers = [t.strip().upper() for t in ticker_text.replace("\n", ",").split(",") if t.strip()]
        with st.spinner("Retrieving historical prices..."):
            try:
                data = fetch_yahoo_prices(tickers, period)
                st.session_state["loaded_prices"] = data
                st.session_state["loaded_source"] = data.attrs.get("source", "Yahoo Finance via yfinance")
                st.session_state["loaded_errors"] = data.attrs.get("fetch_errors", [])
                st.session_state["loaded_rf"] = risk_free_rate
            except Exception as exc:
                st.error(str(exc))
    prices = st.session_state.get("loaded_prices")
    source_note = st.session_state.get("loaded_source", "Load market data using the sidebar button.")
    fetch_errors = st.session_state.get("loaded_errors", [])
    risk_free_rate = st.session_state.get("loaded_rf", 0.04)
else:
    csv_files = st.sidebar.file_uploader("Upload price CSV files", type=["csv"], accept_multiple_files=True,
                                         help="One CSV per ticker with Date and Close/Adj Close columns, or one wide CSV with Date and ticker columns.")
    risk_free_rate = st.sidebar.number_input("Annual risk-free rate (%)", min_value=0.0, max_value=20.0,
                                              value=4.0, step=0.25) / 100
    if csv_files:
        try:
            prices = load_uploaded_csvs(csv_files)
            source_note = prices.attrs.get("source", "User-uploaded CSV files")
        except Exception as exc:
            st.sidebar.error(str(exc))

if prices is None:
    st.info("Choose **Demo dataset** to explore the dashboard immediately, or choose **Yahoo Finance** / **Upload CSV** to analyse historical prices.")
    st.markdown("### How to start")
    st.markdown("1. Select **Demo dataset** to explore the app.\n2. Switch to **Yahoo Finance** and load ticker symbols, or upload your own CSV files.\n3. Review the peer-relative score and inspect the individual risk and return factors.\n4. Use Backtesting to test a simple momentum strategy against an equal-weight benchmark.")
    ai_tab, library_tab = st.tabs(["🤖 AI research assistant", "📚 Research paper library"])
    with ai_tab:
        render_ai_assistant("No stock-price data is currently loaded. Analyse the user's question using uploaded research excerpts only, and clearly state that no live quantitative metrics are available.", False)
    with library_tab:
        render_research_library()
    st.stop()

if data_mode != "Yahoo Finance":
    risk_free_rate = locals().get("risk_free_rate", 0.04)

try:
    metrics = calculate_metrics(prices, risk_free_rate=risk_free_rate)
    scores = add_composite_score(metrics)
except Exception as exc:
    st.error(f"Unable to analyse this price data: {exc}")
    st.stop()

start_date = prices.dropna(how="all").index.min().date()
end_date = prices.dropna(how="all").index.max().date()
valid_tickers = list(prices.columns)

st.caption(f"**Data source:** {source_note}  ·  **Period:** {start_date} to {end_date}  ·  **Stocks analysed:** {len(valid_tickers)}")
if data_mode == "Demo dataset":
    st.warning("DEMO MODE — all prices are synthetic. Do not interpret the ranking as a real investment signal.")
if fetch_errors:
    with st.expander("Some tickers could not be loaded"):
        for err in fetch_errors:
            st.write(f"- {err}")

screen_tab, detail_tab, backtest_tab, ai_tab, library_tab, method_tab = st.tabs(
    ["🏆 Stock screener", "📊 Stock analysis", "🧪 Backtesting", "🤖 AI assistant", "📚 Research library", "📘 Methodology"]
)

with screen_tab:
    st.subheader("Stock ranking")
    st.write("The score compares the selected stocks with one another. A high score means stronger relative factor readings in this selected universe and period; it does not mean a stock is objectively cheap or certain to rise.")

    c1, c2, c3, c4 = st.columns(4)
    best_ticker = scores.index[0]
    c1.metric("Highest relative score", best_ticker)
    c2.metric("Quant score", f"{scores.iloc[0]['Quant score']:.1f}/100" if pd.notna(scores.iloc[0]["Quant score"]) else "—")
    c3.metric("Stocks compared", len(scores))
    c4.metric("Last price date", str(end_date))

    table_cols = [
        "Quant score", "Relative assessment", "Period return", "6M momentum",
        "Sharpe ratio", "Annualised volatility", "Maximum drawdown", "Price trend",
    ]
    display = scores[table_cols].copy()
    st.dataframe(
        display,
        use_container_width=True,
        column_config={
            "Quant score": st.column_config.ProgressColumn("Quant score", min_value=0, max_value=100, format="%.1f"),
            "Period return": st.column_config.NumberColumn(format="%.1%"),
            "6M momentum": st.column_config.NumberColumn(format="%.1%"),
            "Sharpe ratio": st.column_config.NumberColumn(format="%.2f"),
            "Annualised volatility": st.column_config.NumberColumn(format="%.1%"),
            "Maximum drawdown": st.column_config.NumberColumn(format="%.1%"),
            "Price trend": st.column_config.NumberColumn(format="%.0f/100"),
        },
    )

    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Relative factor scores")
        factor_cols = ["Momentum score", "Sharpe score", "Trend score", "Low-volatility score", "Drawdown score"]
        factor_labels = ["Momentum", "Sharpe", "Trend", "Low volatility", "Drawdown control"]
        factor_long = scores[factor_cols].rename(columns=dict(zip(factor_cols, factor_labels))).reset_index().melt(
            id_vars="Ticker", var_name="Factor", value_name="Percentile score"
        )
        chart = px.bar(factor_long, x="Ticker", y="Percentile score", color="Factor", barmode="group",
                       range_y=[0, 105], title="Peer-relative factor percentiles")
        chart.update_layout(height=390, legend_title_text="Factor", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(chart, use_container_width=True)
    with right:
        st.markdown("#### Composite score")
        bar = px.bar(scores.reset_index(), x="Ticker", y="Quant score", color="Quant score",
                     color_continuous_scale="Tealgrn", range_y=[0, 100])
        bar.update_layout(height=390, coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(bar, use_container_width=True)

    st.download_button("Download ranking as CSV", scores.to_csv().encode("utf-8"),
                       file_name="quant_stock_ranking.csv", mime="text/csv")
    st.download_button("Download prices as CSV", prices.to_csv(index_label="Date").encode("utf-8"),
                       file_name="historical_prices.csv", mime="text/csv")

with detail_tab:
    st.subheader("Individual stock analysis")
    selected = st.selectbox("Choose a stock", valid_tickers)
    series = prices[selected].dropna()
    row = scores.loc[selected]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest price", f"{series.iloc[-1]:,.2f}")
    k2.metric("Period return", fmt_pct(row["Period return"]))
    k3.metric("Annualised volatility", fmt_pct(row["Annualised volatility"]))
    k4.metric("Quant score", f"{row['Quant score']:.1f}/100" if pd.notna(row["Quant score"]) else "—")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name=f"{selected} adjusted close"))
    if series.notna().sum() >= 50:
        fig.add_trace(go.Scatter(x=series.index, y=series.rolling(50).mean(), mode="lines", name="50-day moving average"))
    if series.notna().sum() >= 200:
        fig.add_trace(go.Scatter(x=series.index, y=series.rolling(200).mean(), mode="lines", name="200-day moving average"))
    fig.update_layout(title=f"{selected} — historical price and trend", height=430, yaxis_title="Adjusted price", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Quantitative readings")
    detail = pd.DataFrame({
        "Metric": ["6-month momentum", "Sharpe ratio", "Price trend score", "Maximum drawdown", "50-day moving average", "200-day moving average", "Observations"],
        "Value": [fmt_pct(row["6M momentum"]), fmt_num(row["Sharpe ratio"]), f"{row['Price trend']:.0f}/100" if pd.notna(row["Price trend"]) else "—",
                  fmt_pct(row["Maximum drawdown"]), fmt_num(row["SMA 50"]), fmt_num(row["SMA 200"]), str(int(row["Observations"]))],
    })
    st.table(detail)
    assessment = row["Relative assessment"] if pd.notna(row["Relative assessment"]) else "Not enough data"
    st.info(f"**Relative assessment: {assessment}.** This is a comparison against the currently selected peer group, not a buy recommendation. Review the underlying metrics and assumptions before drawing conclusions.")

with backtest_tab:
    st.subheader("6-month momentum strategy backtest")
    st.write("At each month-end, the strategy ranks stocks using information available through the prior month-end, then equally weights the top-ranked stocks for the next month. Estimated transaction costs are deducted using portfolio turnover.")
    b1, b2 = st.columns(2)
    top_n = b1.slider("Number of stocks to hold", 1, max(1, min(10, len(valid_tickers))), min(3, len(valid_tickers)))
    cost_bps = b2.number_input("Transaction cost per traded notional (basis points)", min_value=0.0, max_value=200.0, value=10.0, step=1.0)
    try:
        bt, summary = calculate_backtest(prices, top_n=top_n, transaction_cost_bps=cost_bps, risk_free_rate=risk_free_rate)
        st.line_chart(bt[["Strategy wealth", "Benchmark wealth"]], height=380)
        st.caption("Growth of 1 unit of starting capital. The benchmark is an equal-weight portfolio of the same uploaded/selected stock universe.")
        summary_df = pd.DataFrame(summary).T
        summary_formats = {c: "{:.2%}" for c in ["Cumulative return", "Annualised return", "Annualised volatility", "Maximum drawdown"]}
        summary_formats["Sharpe ratio"] = "{:.2f}"
        st.dataframe(summary_df.style.format(summary_formats), use_container_width=True)
        st.download_button("Download backtest results", bt.to_csv().encode("utf-8"),
                           file_name="momentum_backtest.csv", mime="text/csv")
    except Exception as exc:
        st.warning(f"Backtest unavailable: {exc}")
        st.caption("Try a longer price-history period or upload more historical observations. At least eight month-end observations are required for the default six-month lookback.")

with ai_tab:
    quant_context = make_quant_context(
        scores=scores,
        source_note=source_note,
        start_date=start_date,
        end_date=end_date,
        risk_free_rate=risk_free_rate,
        is_demo=(data_mode == "Demo dataset"),
    )
    render_ai_assistant(quant_context, is_demo=(data_mode == "Demo dataset"))

with library_tab:
    render_research_library()

with method_tab:
    st.subheader("How the score is calculated")
    st.write("The first prototype uses a transparent, rule-based, cross-sectional score. Each factor is converted to a percentile within the stocks selected by the user. The weighted average is therefore relative to the selected comparison group and time period.")
    weights_df = pd.DataFrame({"Factor": list(DEFAULT_WEIGHTS.keys()), "Weight": list(DEFAULT_WEIGHTS.values())})
    st.dataframe(weights_df, hide_index=True, use_container_width=True,
                 column_config={"Weight": st.column_config.NumberColumn(format="%.0%")})
    st.markdown("""
**Factor definitions**

- **Momentum (30%)** — six-month price return, calculated as the latest available adjusted close divided by the adjusted close 126 trading observations earlier, minus one.
- **Sharpe ratio (25%)** — annualised average daily return minus the user-selected annual risk-free rate, divided by annualised daily volatility.
- **Price trend (20%)** — 50 points when the latest price is above its 50-day moving average and 50 points when it is above its 200-day moving average. If a moving average is unavailable, only available tests are used.
- **Low volatility (15%)** — lower annualised standard deviation of daily returns receives a higher peer percentile.
- **Drawdown control (10%)** — a smaller (less negative) maximum peak-to-trough drawdown receives a higher peer percentile.

**Important interpretation**

- Percentile scores depend on the selected peer group. Adding or removing tickers can change the score without changing a stock's own price history.
- Equal values receive a middle score. Missing factor values are excluded and the remaining available weights are re-scaled for that stock.
- The scoring weights are initial educational design choices, not proven optimal parameters.
- This first version focuses on price-based factors; it does not yet include valuation, financial statements, earnings quality, or company fundamentals.
- Yahoo Finance data accessed via yfinance may be delayed, incomplete, or unavailable. Check data dates and source terms before relying on it.

**How the LLM integration works**

- The quantitative engine calculates all numerical indicators, factor percentiles, composite scores, and backtest results in Python.
- PDF papers are stored locally, parsed into page-labelled text chunks, and indexed with TF-IDF. Retrieval is local and does not require an embedding API.
- When a user asks a question, the app retrieves up to five relevant paper passages and sends those passages, the user's question, recent chat context, and a compact summary of the currently displayed stock metrics to the Gemini API using the official Google Gen AI Python SDK.
- Gemini explains the supplied figures and research excerpts. It does not fetch prices, recalculate the score, or change the ranking.
- The API key is read from a local `.env` file or entered for the current session. Do not commit API keys or `.env` to GitHub.

**Backtest assumptions**

- Monthly rebalancing, six-month cross-sectional momentum signal, equal weights among selected stocks.
- Signals are lagged one month so the strategy does not select holdings using the return it is about to earn.
- Transaction costs are estimated using one-way portfolio turnover and the user-entered basis-point assumption.
- The benchmark is equal-weighted across the selected universe; it is not necessarily a market index.
- The backtest does not currently model taxes, slippage beyond the input cost, delistings, survivorship bias in the chosen universe, short selling, or liquidity constraints.
""")
    st.download_button("Download methodology and current score table", scores.to_csv().encode("utf-8"),
                       file_name="quant_methodology_stock_metrics.csv", mime="text/csv")

st.divider()
st.caption("Educational research tool only — not personalised financial advice or a guaranteed buy/sell signal. Historical performance does not guarantee future results.")
