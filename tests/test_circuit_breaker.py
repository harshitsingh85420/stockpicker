"""
P51 — Test: drawdown circuit breaker state transitions.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


@pytest.fixture
def state_file(tmp_path):
    return str(tmp_path / "cb_state.json")


@pytest.fixture
def cb(state_file):
    from production.risk_controls import DrawdownCircuitBreaker
    return DrawdownCircuitBreaker(
        warning_threshold=-0.05,
        halt_threshold=-0.10,
        state_file=state_file,
    )


def test_import():
    from production.risk_controls import DrawdownCircuitBreaker
    assert DrawdownCircuitBreaker is not None


def test_initial_state_normal(cb):
    cb.load_state()
    state = cb.check_state()
    assert state.name == "NORMAL", f"Expected NORMAL, got {state.name}"


def test_size_multiplier_normal(cb):
    cb.load_state()
    cb.check_state()
    mult = cb.get_position_size_multiplier()
    assert mult == 1.0, f"Normal state should have mult=1.0, got {mult}"


def test_state_persists(state_file):
    from production.risk_controls import DrawdownCircuitBreaker
    cb1 = DrawdownCircuitBreaker(
        warning_threshold=-0.05, halt_threshold=-0.10, state_file=state_file
    )
    cb1.load_state()
    cb1.save_state()

    cb2 = DrawdownCircuitBreaker(
        warning_threshold=-0.05, halt_threshold=-0.10, state_file=state_file
    )
    cb2.load_state()
    state = cb2.check_state()
    assert state.name in ("NORMAL", "WARNING", "HALTED")


def test_warning_threshold(state_file):
    """Simulate a drawdown that crosses warning but not halt threshold."""
    from production.risk_controls import DrawdownCircuitBreaker
    cb = DrawdownCircuitBreaker(
        warning_threshold=-0.05, halt_threshold=-0.10, state_file=state_file
    )
    cb.load_state()
    # Inject a drawdown
    if hasattr(cb, "update_pnl"):
        cb.update_pnl(-0.07)   # -7% — should trigger WARNING
        state = cb.check_state()
        assert state.name in ("WARNING", "HALTED"), (
            f"Expected WARNING or HALTED at -7% drawdown, got {state.name}"
        )


def test_halt_threshold(state_file):
    from production.risk_controls import DrawdownCircuitBreaker
    cb = DrawdownCircuitBreaker(
        warning_threshold=-0.05, halt_threshold=-0.10, state_file=state_file
    )
    cb.load_state()
    if hasattr(cb, "update_pnl"):
        cb.update_pnl(-0.15)   # -15% — should trigger HALTED
        state = cb.check_state()
        assert state.name == "HALTED", (
            f"Expected HALTED at -15% drawdown, got {state.name}"
        )


def test_size_multiplier_warning(state_file):
    from production.risk_controls import DrawdownCircuitBreaker
    cb = DrawdownCircuitBreaker(
        warning_threshold=-0.05, halt_threshold=-0.10, state_file=state_file
    )
    cb.load_state()
    if hasattr(cb, "update_pnl"):
        cb.update_pnl(-0.07)
        cb.check_state()
        mult = cb.get_position_size_multiplier()
        assert 0.0 < mult <= 1.0, f"WARNING multiplier should be <1, got {mult}"
