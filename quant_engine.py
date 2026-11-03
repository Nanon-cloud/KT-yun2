"""Core quantitative-finance calculations for the Stock Decision Assistant.

This module intentionally contains deterministic calculations only. It does not
make buy/sell decisions and does not use a language model.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS = 252
DEFAULT_WEIGHTS = {
    "Momentum": 0.30,
    "Sharpe ratio": 0.25,
    "Price trend": 0.20,
    "Low volatility": 0.15,
    "Drawdown control": 0.10,
}


def _standardize_price_frame(prices: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, numeric, positive-price data with a DatetimeIndex."""
    if prices is None or prices.empty:
        raise ValueError("No price data was supplied.")
    frame = prices.copy()
    frame.index = pd.to_datetime(frame.index, errors="coerce")
    frame = frame.loc[~frame.index.isna()]
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame = frame.where(frame > 0)
    frame = frame.dropna(axis=1, how="all")
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("No usable positive closing prices were found.")
    frame.columns = [str(col).upper().strip() for col in frame.columns]
    return frame


def fetch_yahoo_prices(tickers: Iterable[str], period: str = "5y") -> pd.DataFrame:
    """Fetch adjusted daily close prices using yfinance.

    yfinance is intended for research/educational and personal use. Availability
    and terms are controlled by the data provider and can change.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - surfaced in UI
        raise RuntimeError("yfinance is not installed. Run: pip install -r requirements.txt") from exc

    cleaned = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
    if not cleaned:
        raise ValueError("Enter at least one ticker symbol.")

    series_list: dict[str, pd.Series] = {}
    errors: list[str] = []
    for ticker in cleaned:
        try:
            hist = yf.Ticker(ticker).history(period=period, auto_adjust=True, actions=False)
            if hist is None or hist.empty or "Close" not in hist.columns:
                errors.append(f"{ticker}: no price history returned")
                continue
            close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
            close.name = ticker
            if len(close) < 30:
                errors.append(f"{ticker}: fewer than 30 observations")
                continue
            series_list[ticker] = close
        except Exception as exc:  # provider failures vary
            errors.append(f"{ticker}: {str(exc)[:140]}")

    if not series_list:
        details = "; ".join(errors) if errors else "No data returned."
        raise RuntimeError(
            "Could not retrieve prices. Check the ticker symbols and internet connection. "
            "You can also use CSV upload. Details: " + details
        )

    frame = pd.concat(series_list.values(), axis=1).sort_index()
    frame = frame.dropna(how="all")
    frame.attrs["source"] = "Yahoo Finance via yfinance (auto-adjusted daily close)"
    frame.attrs["fetch_errors"] = errors
    return _standardize_price_frame(frame)


def load_uploaded_csvs(files: list) -> pd.DataFrame:
    """Load one CSV per ticker or a combined wide CSV.

    Supported individual CSV format: Date, Close (or Adj Close).
    Supported combined format: Date, AAPL, MSFT, NVDA, ...
    """
    if not files:
        raise ValueError("Choose one or more CSV files first.")

    frames: list[pd.DataFrame] = []
    for uploaded in files:
        raw = pd.read_csv(BytesIO(uploaded.getvalue()))
        if raw.empty:
            continue
        raw.columns = [str(c).strip() for c in raw.columns]
        lower_to_original = {str(c).strip().lower(): c for c in raw.columns}
        date_col = next(
            (lower_to_original[k] for k in ("date", "datetime", "timestamp", "time") if k in lower_to_original),
            None,
        )
        if date_col is None:
            raise ValueError(f"{uploaded.name}: include a Date column.")
        dates = pd.to_datetime(raw[date_col], errors="coerce")
        if dates.notna().sum() < 2:
            raise ValueError(f"{uploaded.name}: the Date column could not be parsed.")

        close_col = next(
            (lower_to_original[k] for k in ("adj close", "adj_close", "adjusted close", "close", "price") if k in lower_to_original),
            None,
        )
        if close_col is not None:
            ticker = Path(uploaded.name).stem.upper().replace(" ", "_")
            piece = pd.DataFrame({ticker: pd.to_numeric(raw[close_col], errors="coerce").values}, index=dates)
            frames.append(piece)
        else:
            value_cols = [c for c in raw.columns if c != date_col]
            if not value_cols:
                raise ValueError(f"{uploaded.name}: no price columns found.")
            piece = raw[value_cols].apply(pd.to_numeric, errors="coerce")
            piece.index = dates
            piece.columns = [str(c).upper().strip() for c in piece.columns]
            frames.append(piece)

    if not frames:
        raise ValueError("No valid rows were found in the uploaded CSV files.")
    combined = pd.concat(frames, axis=1).sort_index()
    combined = combined.groupby(level=0).last()
    combined.attrs["source"] = "User-uploaded CSV files"
    return _standardize_price_frame(combined)


def make_demo_prices() -> pd.DataFrame:
    """Create clearly synthetic sample data so the dashboard works offline."""
    rng = np.random.default_rng(19)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=756)
    specs = {
        "DEMO_A": (0.00055, 0.018),
        "DEMO_B": (0.00035, 0.012),
        "DEMO_C": (0.00070, 0.025),
        "DEMO_D": (0.00020, 0.009),
        "DEMO_E": (0.00045, 0.020),
        "DEMO_F": (0.00010, 0.015),
    }
    data: dict[str, np.ndarray] = {}
    common = rng.normal(0.00025, 0.006, len(dates))
    for ticker, (mu, sigma) in specs.items():
        idio = rng.normal(0, sigma, len(dates))
        returns = mu + 0.25 * common + idio
        returns = np.clip(returns, -0.20, 0.20)
        data[ticker] = 100 * np.cumprod(1 + returns)
    frame = pd.DataFrame(data, index=dates)
    frame.attrs["source"] = "Synthetic demonstration data — not real securities or market prices"
    return frame


def calculate_metrics(prices: pd.DataFrame, risk_free_rate: float = 0.04) -> pd.DataFrame:
    """Calculate historical metrics for each ticker in a price matrix."""
    frame = _standardize_price_frame(prices)
    rows: list[dict] = []
    for ticker in frame.columns:
        p = frame[ticker].dropna()
        r = p.pct_change().dropna()
        if len(p) < 2 or r.empty:
            continue

        current = float(p.iloc[-1])
        total_return = current / float(p.iloc[0]) - 1
        momentum_6m = current / float(p.iloc[-127]) - 1 if len(p) >= 127 else np.nan
        sma50 = float(p.tail(50).mean()) if len(p) >= 50 else np.nan
        sma200 = float(p.tail(200).mean()) if len(p) >= 200 else np.nan
        trend_parts = []
        if np.isfinite(sma50):
            trend_parts.append(float(current > sma50))
        if np.isfinite(sma200):
            trend_parts.append(float(current > sma200))
        trend_score = float(np.mean(trend_parts) * 100) if trend_parts else np.nan

        daily_vol = float(r.std(ddof=1)) if len(r) > 1 else np.nan
        annual_vol = daily_vol * np.sqrt(TRADING_DAYS) if np.isfinite(daily_vol) else np.nan
        annual_return = float(r.mean() * TRADING_DAYS)
        sharpe = (
            (annual_return - float(risk_free_rate)) / annual_vol
            if np.isfinite(annual_vol) and annual_vol > 0
            else np.nan
        )
        drawdown = p / p.cummax() - 1
        max_drawdown = float(drawdown.min())

        rows.append({
            "Ticker": ticker,
            "Latest price": current,
            "Period return": total_return,
            "6M momentum": momentum_6m,
            "SMA 50": sma50,
            "SMA 200": sma200,
            "Price trend": trend_score,
            "Annualised volatility": annual_vol,
            "Sharpe ratio": sharpe,
            "Maximum drawdown": max_drawdown,
            "Observations": int(len(p)),
            "Last date": p.index[-1].date().isoformat(),
        })

    if not rows:
        raise ValueError("Not enough usable price data to calculate metrics.")
    return pd.DataFrame(rows).set_index("Ticker")


def _percentile_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    valid = values.dropna()
    result = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.empty:
        return result
    if valid.nunique() == 1:
        result.loc[valid.index] = 50.0
    else:
        ranks = valid.rank(pct=True, method="average", ascending=not higher_is_better)
        result.loc[valid.index] = ranks * 100
    return result


def add_composite_score(metrics: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Add relative percentile factor scores and a weighted overall score."""
    weights = weights or DEFAULT_WEIGHTS
    result = metrics.copy()
    factor_sources = {
        "Momentum score": ("6M momentum", True),
        "Sharpe score": ("Sharpe ratio", True),
        "Trend score": ("Price trend", True),
        "Low-volatility score": ("Annualised volatility", False),
        "Drawdown score": ("Maximum drawdown", True),
    }
    score_cols = []
    for score_name, (source_name, higher_is_better) in factor_sources.items():
        if source_name in result.columns:
            result[score_name] = _percentile_score(result[source_name], higher_is_better)
            score_cols.append(score_name)

    mapping = {
        "Momentum score": weights.get("Momentum", 0.30),
        "Sharpe score": weights.get("Sharpe ratio", 0.25),
        "Trend score": weights.get("Price trend", 0.20),
        "Low-volatility score": weights.get("Low volatility", 0.15),
        "Drawdown score": weights.get("Drawdown control", 0.10),
    }
    numerator = pd.Series(0.0, index=result.index)
    denominator = pd.Series(0.0, index=result.index)
    for col in score_cols:
        weight = float(mapping.get(col, 0))
        available = result[col].notna()
        numerator.loc[available] += result.loc[available, col] * weight
        denominator.loc[available] += weight
    result["Quant score"] = (numerator / denominator.replace(0, np.nan)).round(1)
    result["Relative assessment"] = pd.cut(
        result["Quant score"],
        bins=[-np.inf, 40, 60, 80, np.inf],
        labels=["Lower relative score", "Middle of peer group", "Above peer group", "Highest relative range"],
    ).astype("string")
    result = result.sort_values("Quant score", ascending=False, na_position="last")
    return result


def calculate_backtest(
    prices: pd.DataFrame,
    top_n: int = 3,
    transaction_cost_bps: float = 10.0,
    risk_free_rate: float = 0.04,
    lookback_months: int = 6,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Monthly cross-sectional momentum backtest with lagged signals.

    Signals available at the previous month-end select equal-weight holdings for
    the following month. A simple one-way turnover transaction-cost estimate is
    deducted. This remains a simplified educational backtest.
    """
    frame = _standardize_price_frame(prices)
    monthly = frame.resample("ME").last()
    monthly_returns = monthly.pct_change(fill_method=None)
    signals = monthly.pct_change(periods=lookback_months, fill_method=None).shift(1)
    if len(monthly) <= lookback_months + 1:
        raise ValueError(
            f"Backtesting needs more history. Use at least {lookback_months + 2} month-end observations."
        )

    ranks = signals.rank(axis=1, ascending=False, method="first")
    eligible = ranks.le(max(1, int(top_n))) & signals.notna()
    weights = eligible.astype(float).div(eligible.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    turnover = weights.diff().abs().sum(axis=1) / 2.0
    if len(turnover):
        turnover.iloc[0] = weights.iloc[0].abs().sum()  # initial portfolio establishment
    costs = turnover * (float(transaction_cost_bps) / 10000.0)

    held_returns = (weights * monthly_returns.fillna(0.0)).sum(axis=1)
    strategy_returns = held_returns - costs
    benchmark_returns = monthly_returns.mean(axis=1, skipna=True).fillna(0.0)

    results = pd.DataFrame({
        "Strategy return": strategy_returns,
        "Equal-weight benchmark": benchmark_returns,
        "Turnover": turnover,
        "Estimated costs": costs,
    })
    active_rows = weights.sum(axis=1) > 0
    results = results.loc[active_rows].copy()
    if results.empty:
        raise ValueError("No valid momentum signals were available for this period.")

    def summarize(series: pd.Series) -> dict[str, float]:
        s = series.dropna()
        if s.empty:
            return {"Cumulative return": np.nan, "Annualised return": np.nan,
                    "Annualised volatility": np.nan, "Sharpe ratio": np.nan,
                    "Maximum drawdown": np.nan}
        wealth = (1 + s).cumprod()
        years = len(s) / 12.0
        annual_return = float(wealth.iloc[-1] ** (1 / years) - 1) if years > 0 and wealth.iloc[-1] > 0 else np.nan
        annual_vol = float(s.std(ddof=1) * np.sqrt(12)) if len(s) > 1 else np.nan
        sharpe = (float(s.mean() * 12) - risk_free_rate) / annual_vol if np.isfinite(annual_vol) and annual_vol > 0 else np.nan
        mdd = float((wealth / wealth.cummax() - 1).min())
        return {
            "Cumulative return": float(wealth.iloc[-1] - 1),
            "Annualised return": annual_return,
            "Annualised volatility": annual_vol,
            "Sharpe ratio": sharpe,
            "Maximum drawdown": mdd,
        }

    summary = {
        "Momentum strategy": summarize(results["Strategy return"]),
        "Equal-weight benchmark": summarize(results["Equal-weight benchmark"]),
    }
    results["Strategy wealth"] = (1 + results["Strategy return"]).cumprod()
    results["Benchmark wealth"] = (1 + results["Equal-weight benchmark"]).cumprod()
    return results, summary
