"""
production/circuit_breaker.py -- P22: Portfolio-level drawdown circuit breaker.

Monitors the portfolio equity curve and halts new position-taking when the
drawdown from the rolling peak exceeds configurable thresholds.

Three states:
  NORMAL  -- drawdown within tolerance; full trading allowed.
  WARNING -- drawdown >= warning_threshold; position-size reduced by 50%.
  HALTED  -- drawdown >= halt_threshold; NO new positions allowed.

Recovery: once in HALTED or WARNING, the circuit breaker returns to NORMAL
only when the equity recovers above recovery_threshold from the trough.

State is persisted to disk between runs so the circuit breaker survives
process restarts (critical for a production daily scheduler).

Usage
-----
from production.circuit_breaker import DrawdownCircuitBreaker

cb = DrawdownCircuitBreaker()
cb.update(current_portfolio_value=980_000)
if cb.is_halted():
    logger.critical("Circuit breaker HALTED -- no new trades today.")
elif cb.in_warning():
    size_mult = 0.5
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_ROOT      = Path(__file__).parent.parent
STATE_PATH = _ROOT / "stock_picker_data" / "circuit_breaker_state.json"


class DrawdownCircuitBreaker:
    """
    Portfolio equity drawdown circuit breaker.

    Parameters
    ----------
    warning_threshold  : Drawdown fraction that triggers WARNING (default -5%).
    halt_threshold     : Drawdown fraction that triggers HALT (default -10%).
    recovery_threshold : Drawdown below which recovery restores NORMAL (default -5%).
    state_path         : JSON file for persistent state (default stock_picker_data/...).
    history_tail_len   : How many equity values to keep in the state history.
    """

    STATES = ("NORMAL", "WARNING", "HALTED")

    def __init__(
        self,
        warning_threshold:  float = -0.05,
        halt_threshold:     float = -0.10,
        recovery_threshold: float = -0.05,
        state_path: Optional[Path] = None,
        history_tail_len: int = 20,
    ) -> None:
        self.warning_threshold  = warning_threshold
        self.halt_threshold     = halt_threshold
        self.recovery_threshold = recovery_threshold
        self.state_path         = state_path or STATE_PATH
        self.history_tail_len   = history_tail_len

        self._state        = "NORMAL"
        self._peak_value:  Optional[float] = None
        self._trough_value: Optional[float] = None
        self._history:     List[float] = []

        self._load_state()

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, current_portfolio_value: float) -> Dict[str, Any]:
        """
        Feed the latest portfolio value and update state.

        Parameters
        ----------
        current_portfolio_value : Total mark-to-market value of the portfolio.

        Returns
        -------
        dict with keys: state, drawdown_from_peak, action_required.
        """
        val = float(current_portfolio_value)
        self._history.append(val)
        if len(self._history) > self.history_tail_len:
            self._history = self._history[-self.history_tail_len:]

        # Update rolling peak
        if self._peak_value is None or val > self._peak_value:
            self._peak_value = val

        drawdown = (val - self._peak_value) / self._peak_value if self._peak_value else 0.0

        # --- State machine ---
        if self._state == "HALTED":
            # Recovery: only exit HALTED when equity recovers past recovery_threshold
            if self._trough_value and val >= self._trough_value * (1 - self.recovery_threshold):
                self._state = "NORMAL"
                logger.info("P22: Circuit breaker recovered to NORMAL (val=%.0f, trough=%.0f)", val, self._trough_value)
            # else remain HALTED

        elif self._state == "WARNING":
            if drawdown <= self.halt_threshold:
                self._state = "HALTED"
                self._trough_value = val
                logger.critical("P22: Circuit breaker HALTED -- drawdown=%.2f%%", drawdown * 100)
            elif drawdown > self.warning_threshold:
                self._state = "NORMAL"
                logger.info("P22: Circuit breaker recovered WARNING -> NORMAL")

        else:  # NORMAL
            if drawdown <= self.halt_threshold:
                self._state = "HALTED"
                self._trough_value = val
                logger.critical("P22: Circuit breaker HALTED -- drawdown=%.2f%%", drawdown * 100)
            elif drawdown <= self.warning_threshold:
                self._state = "WARNING"
                logger.warning("P22: Circuit breaker WARNING -- drawdown=%.2f%%", drawdown * 100)

        self._save_state(val)

        action = {
            "NORMAL":  "TRADE_NORMAL",
            "WARNING": "REDUCE_SIZE",
            "HALTED":  "NO_NEW_TRADES",
        }[self._state]

        result = {
            "state":              self._state,
            "drawdown_from_peak": round(drawdown, 6),
            "peak_value":         self._peak_value,
            "current_value":      val,
            "action_required":    action,
        }
        logger.info(
            "P22 CircuitBreaker: state=%s  drawdown=%.2f%%  peak=%.0f  val=%.0f",
            self._state, drawdown * 100, self._peak_value or 0, val,
        )
        return result

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_halted(self) -> bool:
        return self._state == "HALTED"

    def in_warning(self) -> bool:
        return self._state == "WARNING"

    @property
    def state(self) -> str:
        return self._state

    def size_multiplier(self) -> float:
        """Return position-size multiplier based on current state."""
        return {"NORMAL": 1.0, "WARNING": 0.5, "HALTED": 0.0}[self._state]

    def reset(self, new_peak: Optional[float] = None) -> None:
        """
        Manually reset circuit breaker (e.g. after capital injection or new fiscal year).
        """
        self._state        = "NORMAL"
        self._peak_value   = new_peak
        self._trough_value = None
        self._history      = []
        self._save_state(new_peak or 0.0)
        logger.info("P22: Circuit breaker manually reset. new_peak=%s", new_peak)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_state(self, current_value: float) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state":              self._state,
            "peak_value":         self._peak_value,
            "trough_value":       self._trough_value,
            "current_value":      current_value,
            "warning_threshold":  self.warning_threshold,
            "halt_threshold":     self.halt_threshold,
            "recovery_threshold": self.recovery_threshold,
            "saved_at":           datetime.now().isoformat(),
            "history_tail":       self._history[-5:],
        }
        try:
            self.state_path.write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            logger.warning("P22: State save failed: %s", exc)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text())
            # Only restore if thresholds match (guard against stale state with different params)
            if (
                abs(payload.get("warning_threshold", 0) - self.warning_threshold) < 1e-9
                and abs(payload.get("halt_threshold", 0) - self.halt_threshold) < 1e-9
            ):
                self._state        = payload.get("state", "NORMAL")
                self._peak_value   = payload.get("peak_value")
                self._trough_value = payload.get("trough_value")
                self._history      = payload.get("history_tail", [])
                logger.info(
                    "P22: Circuit breaker state restored: %s (peak=%.0f)",
                    self._state, self._peak_value or 0,
                )
        except Exception as exc:
            logger.warning("P22: State load failed (%s) -- starting fresh.", exc)
