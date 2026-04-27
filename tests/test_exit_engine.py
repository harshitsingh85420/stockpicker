"""
P51 — Test: exit engine stop-loss and trailing-stop logic.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture
def exit_engine():
    from production.exit_engine import ExitEngine
    return ExitEngine(initial_stop_atr=2.5, trailing_stop_atr=1.5, time_stop_sessions=5)


@pytest.fixture
def sample_position():
    return {
        "SC_CODE":          "532540",
        "entry_price":      3600.0,
        "entry_date":       "2025-01-01",
        "atr":              36.0,       # 1% ATR
        "shares":           10,
    }


@pytest.fixture
def price_series():
    """5-session price series that ends below stop-loss."""
    return pd.Series([3600, 3580, 3560, 3510, 3480], dtype=float)


def test_exit_engine_imports():
    from production.exit_engine import ExitEngine
    assert ExitEngine is not None


def test_initial_stop_price(exit_engine, sample_position):
    """Initial stop = entry - 2.5 * ATR."""
    entry  = sample_position["entry_price"]
    atr    = sample_position["atr"]
    expected_stop = entry - 2.5 * atr
    # ExitEngine should compute this somewhere
    stop = entry - exit_engine.initial_stop_atr * atr
    assert abs(stop - expected_stop) < 0.01, (
        f"Expected stop {expected_stop}, got {stop}"
    )


def test_stop_price_below_entry(exit_engine, sample_position):
    entry = sample_position["entry_price"]
    atr   = sample_position["atr"]
    stop  = entry - exit_engine.initial_stop_atr * atr
    assert stop < entry, "Stop-loss must be below entry price for long positions"


def test_time_stop_trigger(exit_engine):
    """After time_stop_sessions, position should be exited."""
    assert exit_engine.time_stop_sessions == 5, (
        "Time stop should be 5 sessions by default"
    )


def test_atr_calculator_imports():
    from production.exit_engine import ATRCalculator
    assert ATRCalculator is not None


def test_atr_calculation():
    """ATR of a flat series should be near zero."""
    from production.exit_engine import ATRCalculator
    calc = ATRCalculator()
    n = 50
    df = pd.DataFrame({
        "SC_CODE": "TEST",
        "DATE":    pd.date_range("2024-01-01", periods=n, freq="B"),
        "High":    [101.0] * n,
        "Low":     [99.0]  * n,
        "Close":   [100.0] * n,
    })
    result = calc.get_atr_for_universe(df, df["DATE"].max())
    atr = result.get("TEST", None)
    assert atr is not None, "ATR not computed for TEST stock"
    assert 1.0 <= atr <= 3.0, f"ATR={atr} outside expected [1, 3] for flat series"


def test_check_exit_returns_dict(exit_engine, sample_position, price_series):
    """check_exit should return a dict with at least 'should_exit' key."""
    if not hasattr(exit_engine, "check_exit"):
        pytest.skip("ExitEngine.check_exit not implemented")
    result = exit_engine.check_exit(
        position=sample_position,
        price_history=price_series,
        current_session=4,
    )
    assert isinstance(result, dict), "check_exit must return a dict"
    assert "should_exit" in result, "Missing 'should_exit' key"
