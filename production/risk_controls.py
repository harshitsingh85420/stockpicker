"""
risk_controls.py
================
Production risk controls for the Indian market stock picker.

Handles:
  - Gap 11: Drawdown Circuit Breaker
  - Gap 18: Human Override Kill Switch

Usage:
    from production.risk_controls import RiskManager, is_safe_to_trade

    rm = RiskManager()
    can_trade, reason = rm.pre_trade_check(trade_value=50000, portfolio_value=500000)
"""

import os
import json
import logging
import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CircuitBreakerState(Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    HALTED = "HALTED"


# ---------------------------------------------------------------------------
# DrawdownCircuitBreaker
# ---------------------------------------------------------------------------

class DrawdownCircuitBreaker:
    """
    Monitors portfolio drawdown and trips a circuit breaker when thresholds
    are breached.

    States
    ------
    NORMAL  : drawdown > warning_threshold               (full position sizing)
    WARNING : warning_threshold >= drawdown > halt_threshold  (50% sizing)
    HALTED  : drawdown <= halt_threshold                 (no new trades)

    Parameters
    ----------
    warning_threshold : float
        Drawdown level (negative) that triggers WARNING state.  Default -5%.
    halt_threshold : float
        Drawdown level (negative) that triggers HALTED state.  Default -10%.
    recovery_threshold : float
        HALTED -> WARNING recovery requires drawdown to improve above this.
    state_file : str
        JSON file used to persist state across process restarts.
    """

    def __init__(
        self,
        warning_threshold: float = -0.05,
        halt_threshold: float = -0.10,
        recovery_threshold: float = -0.05,
        state_file: str = "stock_picker_data/circuit_breaker_state.json",
    ) -> None:
        if warning_threshold >= 0 or halt_threshold >= 0:
            raise ValueError("Thresholds must be negative fractions, e.g. -0.05")
        if halt_threshold >= warning_threshold:
            raise ValueError("halt_threshold must be more negative than warning_threshold")

        self.warning_threshold = warning_threshold
        self.halt_threshold = halt_threshold
        self.recovery_threshold = recovery_threshold
        self.state_file = Path(state_file)

        # Runtime history: list of (timestamp_iso, value) tuples
        self._history: List[Tuple[str, float]] = []
        self._peak_value: Optional[float] = None
        self._current_value: Optional[float] = None
        self._state: CircuitBreakerState = CircuitBreakerState.NORMAL

        # Attempt to restore persisted state
        self.load_state()

    # ------------------------------------------------------------------
    # Core value tracking
    # ------------------------------------------------------------------

    def update_portfolio_value(
        self,
        current_value: float,
        timestamp: Optional[datetime.datetime] = None,
    ) -> None:
        """Record a new portfolio value snapshot and update the peak."""
        if current_value <= 0:
            raise ValueError(f"Portfolio value must be positive, got {current_value}")

        ts = (timestamp or datetime.datetime.now()).isoformat()
        self._history.append((ts, current_value))
        self._current_value = current_value

        # Keep only last 500 snapshots to bound memory usage
        if len(self._history) > 500:
            self._history = self._history[-500:]

        if self._peak_value is None or current_value > self._peak_value:
            self._peak_value = current_value
            logger.debug("New portfolio peak: %.2f", self._peak_value)

        # Refresh state
        new_state = self.check_state()
        if new_state != self._state:
            logger.warning(
                "Circuit breaker state change: %s -> %s  (drawdown=%.2f%%)",
                self._state.value,
                new_state.value,
                self.compute_current_drawdown() * 100,
            )
            self._state = new_state

        self.save_state()

    def compute_current_drawdown(self) -> float:
        """
        Return current drawdown from peak as a negative fraction.
        Returns 0.0 if no history is available.
        """
        if self._peak_value is None or self._current_value is None:
            return 0.0
        if self._peak_value == 0:
            return 0.0
        return (self._current_value - self._peak_value) / self._peak_value

    # ------------------------------------------------------------------
    # State logic
    # ------------------------------------------------------------------

    def check_state(self) -> CircuitBreakerState:
        """Determine state purely from current drawdown vs thresholds."""
        drawdown = self.compute_current_drawdown()
        if drawdown <= self.halt_threshold:
            return CircuitBreakerState.HALTED
        if drawdown <= self.warning_threshold:
            return CircuitBreakerState.WARNING
        return CircuitBreakerState.NORMAL

    def get_position_size_multiplier(self) -> float:
        """
        Return a multiplier [0, 1] that should be applied to intended
        position sizes based on the current circuit breaker state.
        """
        multipliers = {
            CircuitBreakerState.NORMAL: 1.0,
            CircuitBreakerState.WARNING: 0.5,
            CircuitBreakerState.HALTED: 0.0,
        }
        return multipliers[self._state]

    def can_open_new_trades(self) -> bool:
        """Return True if the system is allowed to open new positions."""
        return self._state in (CircuitBreakerState.NORMAL, CircuitBreakerState.WARNING)

    def reset_if_recovered(self) -> bool:
        """
        If currently HALTED and drawdown has improved above
        ``recovery_threshold``, transition back to WARNING.

        Returns True if a recovery transition was made.
        """
        if self._state != CircuitBreakerState.HALTED:
            return False

        drawdown = self.compute_current_drawdown()
        if drawdown > self.recovery_threshold:
            logger.info(
                "Circuit breaker recovering: HALTED -> WARNING  (drawdown=%.2f%%)",
                drawdown * 100,
            )
            self._state = CircuitBreakerState.WARNING
            self.save_state()
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self, path: Optional[str] = None) -> None:
        """Persist circuit breaker state to a JSON file."""
        target = Path(path) if path else self.state_file
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state": self._state.value,
            "peak_value": self._peak_value,
            "current_value": self._current_value,
            "warning_threshold": self.warning_threshold,
            "halt_threshold": self.halt_threshold,
            "recovery_threshold": self.recovery_threshold,
            "saved_at": datetime.datetime.now().isoformat(),
            "history_tail": self._history[-20:],  # last 20 snapshots
        }
        try:
            with open(target, "w") as fh:
                json.dump(payload, fh, indent=2)
        except OSError as exc:
            logger.error("Failed to save circuit breaker state: %s", exc)

    def load_state(self, path: Optional[str] = None) -> None:
        """Restore circuit breaker state from a JSON file if it exists."""
        target = Path(path) if path else self.state_file
        if not target.exists():
            return
        try:
            with open(target) as fh:
                data = json.load(fh)
            self._state = CircuitBreakerState(data.get("state", "NORMAL"))
            self._peak_value = data.get("peak_value")
            self._current_value = data.get("current_value")
            # Restore saved thresholds only if they match constructor values
            logger.info(
                "Loaded circuit breaker state from %s: %s", target, self._state.value
            )
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Could not load circuit breaker state (%s), starting fresh.", exc)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_status_report(self) -> Dict[str, Any]:
        """Return a human-readable status dictionary."""
        return {
            "state": self._state.value,
            "current_drawdown": self.compute_current_drawdown(),
            "current_drawdown_pct": f"{self.compute_current_drawdown() * 100:.2f}%",
            "peak_value": self._peak_value,
            "current_value": self._current_value,
            "warning_threshold": self.warning_threshold,
            "halt_threshold": self.halt_threshold,
            "position_size_multiplier": self.get_position_size_multiplier(),
            "can_open_new_trades": self.can_open_new_trades(),
        }


# ---------------------------------------------------------------------------
# KillSwitch
# ---------------------------------------------------------------------------

class KillSwitch:
    """
    Human override kill switch.

    Activated by either:
      1. Setting an environment variable to '1', 'true', or 'yes'.
      2. Creating a flag file on disk.

    Both mechanisms are checked on every ``is_active()`` call, making the
    kill switch reliable even across subprocess boundaries.

    Parameters
    ----------
    env_var : str
        Name of the environment variable to check.
    flag_file : str
        Path to the sentinel file that, if present, halts all trading.
    """

    def __init__(
        self,
        env_var: str = "STOCKPICKER_KILL",
        flag_file: str = "stock_picker_data/NO_TRADE_TODAY",
    ) -> None:
        self.env_var = env_var
        self.flag_file = Path(flag_file)

    # ------------------------------------------------------------------
    # Core checks
    # ------------------------------------------------------------------

    def is_active(self) -> bool:
        """Return True if the kill switch is currently active."""
        return self._env_is_set() or self._flag_file_exists()

    def _env_is_set(self) -> bool:
        val = os.environ.get(self.env_var, "").strip().lower()
        return val in ("1", "true", "yes")

    def _flag_file_exists(self) -> bool:
        return self.flag_file.exists()

    # ------------------------------------------------------------------
    # Activation / deactivation
    # ------------------------------------------------------------------

    def activate(self, reason: str = "Manual override") -> None:
        """
        Create the flag file, which immediately halts all trading.
        The file stores a timestamp and reason for audit purposes.
        """
        self.flag_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "activated_at": datetime.datetime.now().isoformat(),
            "reason": reason,
            "activated_by": "KillSwitch.activate()",
        }
        try:
            with open(self.flag_file, "w") as fh:
                json.dump(payload, fh, indent=2)
            logger.critical("KILL SWITCH ACTIVATED: %s  [file=%s]", reason, self.flag_file)
        except OSError as exc:
            logger.error("Failed to create kill switch flag file: %s", exc)
            raise

    def deactivate(self) -> None:
        """Remove the flag file to re-enable trading."""
        if self.flag_file.exists():
            try:
                self.flag_file.unlink()
                logger.info("Kill switch deactivated, flag file removed: %s", self.flag_file)
            except OSError as exc:
                logger.error("Failed to remove kill switch flag file: %s", exc)
                raise
        else:
            logger.info("Kill switch already inactive (flag file not present).")

    # ------------------------------------------------------------------
    # Reason / status
    # ------------------------------------------------------------------

    def get_reason(self) -> str:
        """Return a human-readable reason why the kill switch is active."""
        if self._env_is_set():
            return f"Environment variable {self.env_var} is set to a truthy value."
        if self._flag_file_exists():
            try:
                with open(self.flag_file) as fh:
                    data = json.load(fh)
                return (
                    f"Flag file present. Activated at {data.get('activated_at', 'unknown')}. "
                    f"Reason: {data.get('reason', 'unknown')}"
                )
            except (OSError, json.JSONDecodeError):
                return f"Flag file present at {self.flag_file} (could not parse reason)."
        return "Kill switch is NOT active."

    def check_and_log(self) -> bool:
        """
        Check kill switch status, log the result, and return the active state.

        Returns
        -------
        bool
            True if the kill switch is active (trading should be halted).
        """
        active = self.is_active()
        if active:
            logger.critical("KILL SWITCH IS ACTIVE: %s", self.get_reason())
        else:
            logger.debug("Kill switch is inactive — trading allowed.")
        return active


# ---------------------------------------------------------------------------
# RiskManager (combines circuit breaker + kill switch)
# ---------------------------------------------------------------------------

class RiskManager:
    """
    High-level risk manager that combines the DrawdownCircuitBreaker and
    KillSwitch into a single pre-trade / post-trade interface.

    Parameters
    ----------
    circuit_breaker : DrawdownCircuitBreaker, optional
        Provide a pre-configured instance or let RiskManager create a default.
    kill_switch : KillSwitch, optional
        Provide a pre-configured instance or let RiskManager create a default.
    max_single_loss_pct : float
        Maximum allowed loss on a single trade as a fraction of portfolio.
        Default 5%.
    """

    def __init__(
        self,
        circuit_breaker: Optional[DrawdownCircuitBreaker] = None,
        kill_switch: Optional[KillSwitch] = None,
        max_single_loss_pct: float = 0.05,
    ) -> None:
        self.circuit_breaker = circuit_breaker or DrawdownCircuitBreaker()
        self.kill_switch = kill_switch or KillSwitch()
        self.max_single_loss_pct = max_single_loss_pct

    # ------------------------------------------------------------------
    # Pre-trade gate
    # ------------------------------------------------------------------

    def pre_trade_check(
        self, trade_value: float, portfolio_value: float
    ) -> Tuple[bool, str]:
        """
        Run all pre-trade risk checks.

        Parameters
        ----------
        trade_value : float
            Notional value of the proposed trade.
        portfolio_value : float
            Current total portfolio value.

        Returns
        -------
        (can_trade, reason) : (bool, str)
        """
        # 1. Kill switch
        if self.kill_switch.is_active():
            reason = f"Kill switch active: {self.kill_switch.get_reason()}"
            logger.warning("PRE-TRADE BLOCKED — %s", reason)
            return False, reason

        # 2. Circuit breaker
        if not self.circuit_breaker.can_open_new_trades():
            reason = (
                f"Circuit breaker HALTED — drawdown is "
                f"{self.circuit_breaker.compute_current_drawdown() * 100:.2f}%"
            )
            logger.warning("PRE-TRADE BLOCKED — %s", reason)
            return False, reason

        # 3. Single-trade size limit
        if portfolio_value > 0:
            trade_pct = trade_value / portfolio_value
            if trade_pct > self.max_single_loss_pct:
                reason = (
                    f"Trade value {trade_value:,.0f} is "
                    f"{trade_pct * 100:.1f}% of portfolio — exceeds "
                    f"max_single_loss_pct {self.max_single_loss_pct * 100:.1f}%"
                )
                logger.warning("PRE-TRADE BLOCKED — %s", reason)
                return False, reason

        # 4. Apply position size multiplier (just log; caller sizes the position)
        multiplier = self.circuit_breaker.get_position_size_multiplier()
        if multiplier < 1.0:
            logger.info(
                "Circuit breaker WARNING: position size multiplier = %.1f", multiplier
            )

        return True, "All risk checks passed."

    # ------------------------------------------------------------------
    # Post-trade update
    # ------------------------------------------------------------------

    def post_trade_update(
        self, realized_pnl: float, portfolio_value: float
    ) -> None:
        """
        Update the circuit breaker with the latest portfolio value after
        a trade is executed.

        Parameters
        ----------
        realized_pnl : float
            P&L of the completed trade (positive = profit, negative = loss).
        portfolio_value : float
            New total portfolio value after the trade.
        """
        self.circuit_breaker.update_portfolio_value(portfolio_value)
        if realized_pnl < 0:
            logger.info(
                "Post-trade loss recorded: %.2f  |  Portfolio: %.2f",
                realized_pnl,
                portfolio_value,
            )
        # Attempt recovery if we were halted
        self.circuit_breaker.reset_if_recovered()

    # ------------------------------------------------------------------
    # Emergency close
    # ------------------------------------------------------------------

    def emergency_close_all(
        self, positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Mark all open positions for immediate closure.

        Parameters
        ----------
        positions : list of dict
            Each dict should have at minimum 'sc_code' and 'quantity'.

        Returns
        -------
        list of dict
            Same positions with 'action' set to 'EMERGENCY_CLOSE' and a
            timestamp added.
        """
        ts = datetime.datetime.now().isoformat()
        marked = []
        for pos in positions:
            updated = dict(pos)
            updated["action"] = "EMERGENCY_CLOSE"
            updated["marked_at"] = ts
            updated["reason"] = "RiskManager.emergency_close_all() invoked"
            marked.append(updated)
            logger.critical(
                "EMERGENCY CLOSE: sc_code=%s qty=%s",
                pos.get("sc_code", "?"),
                pos.get("quantity", "?"),
            )
        if not self.kill_switch.is_active():
            self.kill_switch.activate("Emergency close triggered by RiskManager")
        return marked

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Return a combined status report from all risk subsystems."""
        cb_report = self.circuit_breaker.get_status_report()
        return {
            "kill_switch_active": self.kill_switch.is_active(),
            "kill_switch_reason": self.kill_switch.get_reason(),
            "circuit_breaker": cb_report,
            "max_single_loss_pct": self.max_single_loss_pct,
            "overall_safe_to_trade": (
                not self.kill_switch.is_active()
                and self.circuit_breaker.can_open_new_trades()
            ),
            "checked_at": datetime.datetime.now().isoformat(),
        }


# ---------------------------------------------------------------------------
# Module-level convenience helper
# ---------------------------------------------------------------------------

def is_safe_to_trade() -> bool:
    """
    One-liner safety check.  Creates default RiskManager instances and
    verifies that neither the kill switch nor the circuit breaker is
    blocking trading.

    Returns True only if it is safe to open new trades.
    """
    rm = RiskManager()
    status = rm.get_comprehensive_status()
    safe = bool(status["overall_safe_to_trade"])
    if not safe:
        logger.warning("is_safe_to_trade() -> False: %s", status)
    return safe


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )

    print("=" * 60)
    print("DrawdownCircuitBreaker demo")
    print("=" * 60)

    cb = DrawdownCircuitBreaker(
        warning_threshold=-0.05,
        halt_threshold=-0.10,
        state_file="stock_picker_data/demo_cb_state.json",
    )

    # Simulate portfolio values
    values = [100_000, 98_000, 96_000, 94_000, 91_000, 89_000, 87_000]
    for v in values:
        cb.update_portfolio_value(v)
        report = cb.get_status_report()
        print(
            f"  Value={v:>8,}  Drawdown={report['current_drawdown_pct']:>8}  "
            f"State={report['state']:>8}  Multiplier={report['position_size_multiplier']}"
        )

    print()
    print("=" * 60)
    print("KillSwitch demo")
    print("=" * 60)

    ks = KillSwitch(flag_file="stock_picker_data/demo_NO_TRADE")
    print(f"  Initially active: {ks.is_active()}")
    ks.activate(reason="Demo activation")
    print(f"  After activate:   {ks.is_active()}")
    print(f"  Reason:           {ks.get_reason()}")
    ks.deactivate()
    print(f"  After deactivate: {ks.is_active()}")

    print()
    print("=" * 60)
    print("RiskManager.pre_trade_check demo")
    print("=" * 60)

    rm = RiskManager(circuit_breaker=cb, kill_switch=ks, max_single_loss_pct=0.05)
    can, reason = rm.pre_trade_check(trade_value=50_000, portfolio_value=500_000)
    print(f"  can_trade={can}  reason={reason}")

    print()
    print("is_safe_to_trade():", is_safe_to_trade())
