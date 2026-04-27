"""
P51 — Test: friction model round-trip cost calculations.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture
def fm():
    from production.friction_model import FrictionModel
    return FrictionModel(instrument_type="equity_delivery")


def test_friction_model_imports():
    from production.friction_model import FrictionModel
    assert FrictionModel is not None


def test_round_trip_cost_structure(fm):
    result = fm.calculate_round_trip_cost(100_000.0)
    assert isinstance(result, dict), "round_trip_cost must return a dict"
    assert "total_pct" in result, "Missing 'total_pct' key"
    assert "total_inr" in result, "Missing 'total_inr' key"


def test_round_trip_cost_positive(fm):
    result = fm.calculate_round_trip_cost(100_000.0)
    assert result["total_pct"] > 0, "Friction should be positive"
    assert result["total_inr"] > 0, "Friction INR should be positive"


def test_round_trip_cost_reasonable(fm):
    """BSE equity delivery costs should be between 0.1% and 2% per round trip."""
    result = fm.calculate_round_trip_cost(100_000.0)
    assert 0.001 <= result["total_pct"] <= 0.02, (
        f"Friction {result['total_pct']:.4%} outside expected 0.1%–2% range"
    )


def test_friction_scales_with_value(fm):
    small = fm.calculate_round_trip_cost(10_000.0)
    large = fm.calculate_round_trip_cost(1_000_000.0)
    # Larger trades should have smaller % friction (impact spreads)
    # or at least not more than small trades
    assert large["total_pct"] <= small["total_pct"] * 2, (
        "Friction for large trade unexpectedly much higher than small trade"
    )


def test_friction_zero_value(fm):
    """Should not crash on zero position value."""
    result = fm.calculate_round_trip_cost(0.0)
    assert isinstance(result, dict)


def test_net_expected_return_calculation(fm):
    """Net return = prob * avg_gain - friction."""
    prob = 0.65
    result = fm.calculate_round_trip_cost(100_000.0)
    net = prob * 0.05 - result["total_pct"]
    assert net > -0.05, "Net expected return wildly negative — check friction model"
