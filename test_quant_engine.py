import numpy as np
import pandas as pd

from quant_engine import calculate_metrics, add_composite_score, calculate_backtest, make_demo_prices


def test_metrics_and_scores():
    prices = make_demo_prices()
    metrics = calculate_metrics(prices, risk_free_rate=0.04)
    scored = add_composite_score(metrics)
    assert len(scored) == prices.shape[1]
    assert scored["Quant score"].between(0, 100).all()
    assert "Sharpe ratio" in scored.columns


def test_backtest_returns_results():
    prices = make_demo_prices()
    results, summary = calculate_backtest(prices, top_n=2, transaction_cost_bps=10)
    assert not results.empty
    assert "Momentum strategy" in summary
    assert np.isfinite(summary["Momentum strategy"]["Cumulative return"])
