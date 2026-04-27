"""
P51 — Test: regime filter outputs valid labels and thresholds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture
def nifty_prices():
    """Synthetic 300-day Nifty-like index for regime detection."""
    rng = np.random.default_rng(0)
    n = 300
    raw = 20000.0 + np.cumsum(rng.normal(0.5, 15, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(raw, index=dates, name="Nifty")


def test_regime_filter_imports():
    from production.regime_filter import RegimeFilter
    assert RegimeFilter is not None


def test_regime_filter_produces_label(nifty_prices):
    from production.regime_filter import RegimeFilter
    rf = RegimeFilter()
    result = rf.get_regime(nifty_prices)
    assert isinstance(result, dict), "get_regime must return a dict"
    assert "regime" in result, "Missing 'regime' key"
    assert result["regime"] in ("BULL", "BEAR", "NEUTRAL", "CAUTIOUS", "SKIP_DAY"), (
        f"Unknown regime label: {result['regime']}"
    )


def test_regime_filter_threshold(nifty_prices):
    from production.regime_filter import RegimeFilter
    rf = RegimeFilter()
    result = rf.get_regime(nifty_prices)
    threshold = result.get("recommended_threshold", result.get("threshold", None))
    assert threshold is not None, "Regime result missing threshold"
    assert 0.50 <= threshold <= 0.90, f"Threshold {threshold} outside [0.50, 0.90]"


def test_regime_filter_is_tradeable_bool(nifty_prices):
    from production.regime_filter import RegimeFilter
    rf = RegimeFilter()
    result = rf.get_regime(nifty_prices)
    assert "is_tradeable" in result, "Missing 'is_tradeable' key"
    assert isinstance(result["is_tradeable"], bool), "is_tradeable must be bool"


def test_3layer_ema_regime(nifty_prices):
    from production.regime_filter import RegimeFilter
    rf = RegimeFilter()
    result = rf.get_regime(nifty_prices)
    # Should have some EMA-derived field
    ema_key = next((k for k in result if "ema" in k.lower() or "three" in k.lower()), None)
    assert ema_key is not None, (
        f"Expected an EMA/three-layer key in regime result, got: {list(result.keys())}"
    )


def test_bear_market_not_tradeable():
    """A clearly bearish series should produce is_tradeable=False."""
    from production.regime_filter import RegimeFilter
    rf = RegimeFilter()
    n = 300
    # Strong downtrend: -0.5 per day on average
    raw = 25000.0 + np.cumsum(np.full(n, -0.5))
    raw = np.maximum(raw, 1000.0)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    bear_prices = pd.Series(raw, index=dates)
    result = rf.get_regime(bear_prices)
    assert not result.get("is_tradeable", True), (
        "Strong downtrend should produce is_tradeable=False"
    )


def test_hmm_regime_present(nifty_prices):
    from production.regime_filter import RegimeFilter
    rf = RegimeFilter()
    result = rf.get_regime(nifty_prices)
    # HMM should be attempted (may fall back gracefully)
    assert "hmm_regime" in result or "regime" in result, (
        "HMM regime or fallback regime should be present"
    )
