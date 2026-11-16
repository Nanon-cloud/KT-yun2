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


def test_percentile_score_direction():
    from quant_engine import _percentile_score

    values = pd.Series({"low": 1.0, "mid": 2.0, "high": 3.0})
    better_high = _percentile_score(values, higher_is_better=True)
    assert better_high["high"] > better_high["mid"] > better_high["low"]
    better_low = _percentile_score(values, higher_is_better=False)
    assert better_low["low"] > better_low["mid"] > better_low["high"]


def test_best_metrics_get_best_factor_scores():
    metrics = pd.DataFrame(
        {
            "6M momentum": [0.30, 0.10, -0.10],
            "Sharpe ratio": [1.2, 0.5, -0.2],
            "Price trend": [100.0, 50.0, 0.0],
            "Annualised volatility": [0.15, 0.25, 0.40],
            "Maximum drawdown": [-0.10, -0.25, -0.50],
        },
        index=["good", "middle", "bad"],
    )
    scored = add_composite_score(metrics)
    assert list(scored.index) == ["good", "middle", "bad"]
    for col in ["Momentum score", "Sharpe score", "Trend score", "Low-volatility score", "Drawdown score"]:
        assert scored.loc["good", col] > scored.loc["bad", col], col
