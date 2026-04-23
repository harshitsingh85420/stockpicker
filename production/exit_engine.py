"""
exit_engine.py - Gap 7: Dynamic Exit Engine
============================================
Production module for managing trade exits in Indian equity markets.

Implements a multi-layered exit strategy combining:
  - Initial ATR-based hard stop loss
  - Trailing stop loss activated on a configurable gain threshold
  - Time-based stop for stagnant positions
  - Manual exit support

Designed to work with BSE BhavCopy data (sc_code, sc_name, OPEN, HIGH, LOW, CLOSE).

Author: StockPicker System
Date: 2026-04-23
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLIPPAGE_PCT = 0.001          # 0.1% slippage on exits (market impact, BSE)
MIN_ATR_VALUE = 0.01          # Guard against zero-ATR stocks


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """
    Represents a single open or closed trade position.

    Attributes
    ----------
    sc_code : str
        BSE scrip code (e.g. '500325').
    sc_name : str
        Human-readable scrip name (e.g. 'RELIANCE').
    entry_price : float
        Price at which the position was entered.
    entry_date : date
        Calendar date of entry.
    atr_at_entry : float
        ATR value recorded at the time of entry (used for initial stop).
    shares : int
        Number of shares held.
    stop_loss : float
        Current hard stop-loss price level. Updated as trailing stop moves up.
    target_price : float
        Initial price target (e.g. entry + 3x ATR). Informational.
    trailing_stop_activated : bool
        True once the position has crossed the trail_activation_pct gain threshold.
    trailing_high : float
        Highest price observed since entry (used to calculate trailing stop).
    sessions_held : int
        Number of update_position() calls completed (proxy for trading sessions).
    status : str
        One of: 'OPEN', 'STOPPED_OUT', 'TARGET_HIT', 'TIME_STOP', 'MANUAL'.
    """

    sc_code: str
    sc_name: str
    entry_price: float
    entry_date: date
    atr_at_entry: float
    shares: int
    stop_loss: float = 0.0
    target_price: float = 0.0
    trailing_stop_activated: bool = False
    trailing_high: float = 0.0
    sessions_held: int = 0
    status: str = "OPEN"

    def __post_init__(self):
        if self.trailing_high == 0.0:
            self.trailing_high = self.entry_price
        if self.stop_loss == 0.0:
            # Will be set properly by DynamicExitEngine.set_stop_loss()
            self.stop_loss = self.entry_price * 0.95

    @property
    def is_open(self) -> bool:
        """Return True if the position has not yet been closed."""
        return self.status == "OPEN"

    @property
    def position_value(self) -> float:
        """Current position value at entry price (for reference)."""
        return self.entry_price * self.shares


# ---------------------------------------------------------------------------
# ATR Calculator
# ---------------------------------------------------------------------------

class ATRCalculator:
    """
    Average True Range (ATR) calculator for Indian equity data.

    ATR is computed using Wilder's smoothing (EWM with alpha = 1/period),
    which matches most charting platforms and is robust to gap-up/gap-down opens
    common in Indian markets after corporate actions.
    """

    @staticmethod
    def compute(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Compute ATR for a stock given OHLC data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: HIGH, LOW, CLOSE (case-insensitive).
            Rows should be sorted ascending by date.
        period : int
            Look-back window for Wilder's ATR. Default 14 sessions.

        Returns
        -------
        pd.Series
            ATR values aligned with df's index. First (period-1) values will be NaN.

        Raises
        ------
        KeyError
            If required OHLC columns are missing.
        ValueError
            If df has fewer rows than `period`.
        """
        df = df.copy()
        # Normalise column names
        df.columns = [c.upper() for c in df.columns]

        required = {"HIGH", "LOW", "CLOSE"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"ATRCalculator.compute: missing columns {missing}")

        if len(df) < period:
            raise ValueError(
                f"ATRCalculator.compute: need at least {period} rows, got {len(df)}"
            )

        high = df["HIGH"]
        low = df["LOW"]
        close_prev = df["CLOSE"].shift(1)

        true_range = pd.concat(
            [
                high - low,
                (high - close_prev).abs(),
                (low - close_prev).abs(),
            ],
            axis=1,
        ).max(axis=1)

        # Wilder's smoothing: equivalent to EWM with alpha = 1/period, adjust=False
        atr = true_range.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
        return atr

    @classmethod
    def get_latest_atr(
        cls,
        bhav_df: pd.DataFrame,
        sc_code: str,
        period: int = 14,
    ) -> float:
        """
        Extract the most recent ATR for a single scrip from a BhavCopy DataFrame.

        Parameters
        ----------
        bhav_df : pd.DataFrame
            BhavCopy data with columns SC_CODE, HIGH, LOW, CLOSE and a date column
            (ZDATE, DATE, or the DataFrame index).
        sc_code : str
            BSE scrip code to filter on.
        period : int
            ATR period.

        Returns
        -------
        float
            Latest ATR value. Returns MIN_ATR_VALUE if insufficient data.
        """
        stock_df = bhav_df[bhav_df["SC_CODE"].astype(str) == str(sc_code)].copy()

        # Sort by date
        date_col = _detect_date_column(stock_df)
        if date_col:
            stock_df = stock_df.sort_values(date_col)

        if len(stock_df) < period:
            logger.warning(
                "sc_code %s: only %d rows available (need %d). Returning MIN_ATR.",
                sc_code, len(stock_df), period,
            )
            return MIN_ATR_VALUE

        try:
            atr_series = cls.compute(stock_df, period=period)
            latest = atr_series.dropna().iloc[-1]
            return max(float(latest), MIN_ATR_VALUE)
        except Exception as exc:
            logger.error("get_latest_atr failed for %s: %s", sc_code, exc)
            return MIN_ATR_VALUE

    @classmethod
    def get_atr_for_universe(
        cls,
        bhav_df: pd.DataFrame,
        reference_date,
        period: int = 14,
    ) -> Dict[str, float]:
        """
        Compute the latest ATR for every scrip in the BhavCopy universe.

        Parameters
        ----------
        bhav_df : pd.DataFrame
            Full BhavCopy data, potentially multi-date.
        reference_date : str | date | datetime
            Include only rows on or before this date.
        period : int
            ATR period.

        Returns
        -------
        dict
            {sc_code: atr_value} for all scrips with sufficient history.
        """
        df = bhav_df.copy()
        date_col = _detect_date_column(df)

        if date_col:
            ref = pd.to_datetime(reference_date)
            df[date_col] = pd.to_datetime(df[date_col])
            df = df[df[date_col] <= ref]

        atr_map: Dict[str, float] = {}
        for sc_code, group in df.groupby("SC_CODE"):
            group = group.sort_values(date_col) if date_col else group
            try:
                atr_series = cls.compute(group, period=period)
                val = atr_series.dropna().iloc[-1]
                atr_map[str(sc_code)] = max(float(val), MIN_ATR_VALUE)
            except Exception as exc:
                logger.debug("Skipping %s in get_atr_for_universe: %s", sc_code, exc)

        logger.info("ATR computed for %d scrips up to %s.", len(atr_map), reference_date)
        return atr_map


# ---------------------------------------------------------------------------
# Dynamic Exit Engine
# ---------------------------------------------------------------------------

class DynamicExitEngine:
    """
    Multi-layered dynamic exit engine for Indian equity positions.

    Exit hierarchy (evaluated in order each session):
    1. Time stop  - close stagnant positions after `time_stop_sessions` with < 1% gain.
    2. Hard stop  - close if price falls below the current stop_loss level.
    3. Trailing stop - once activated, ratchet up stop_loss; close on violation.

    Parameters
    ----------
    initial_stop_atr : float
        Multiplier applied to ATR at entry to set the initial hard stop below entry.
        Default 2.5x ATR.
    trailing_stop_atr : float
        Multiplier applied to current ATR to set trailing stop below trailing high.
        Default 1.5x ATR.
    time_stop_sessions : int
        Number of completed sessions before a time-based exit is considered.
        Default 5 sessions.
    trail_activation_pct : float
        Minimum unrealised gain (as a fraction of entry price) required before the
        trailing stop activates. Default 0.02 (2%).
    """

    def __init__(
        self,
        initial_stop_atr: float = 2.5,
        trailing_stop_atr: float = 1.5,
        time_stop_sessions: int = 5,
        trail_activation_pct: float = 0.02,
    ):
        if initial_stop_atr <= 0:
            raise ValueError("initial_stop_atr must be positive.")
        if trailing_stop_atr <= 0:
            raise ValueError("trailing_stop_atr must be positive.")
        if time_stop_sessions < 1:
            raise ValueError("time_stop_sessions must be >= 1.")
        if not (0 < trail_activation_pct < 1):
            raise ValueError("trail_activation_pct must be between 0 and 1 exclusive.")

        self.initial_stop_atr = initial_stop_atr
        self.trailing_stop_atr = trailing_stop_atr
        self.time_stop_sessions = time_stop_sessions
        self.trail_activation_pct = trail_activation_pct

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def set_stop_loss(self, position: Position, atr: float) -> float:
        """
        Set the initial hard stop loss for a position.

        stop_loss = entry_price - (initial_stop_atr * atr)

        Parameters
        ----------
        position : Position
            The position to update in-place.
        atr : float
            Current ATR value for the scrip.

        Returns
        -------
        float
            The computed stop-loss price.
        """
        atr = max(atr, MIN_ATR_VALUE)
        stop = position.entry_price - (self.initial_stop_atr * atr)
        stop = max(stop, 0.01)          # Price cannot be negative
        position.stop_loss = stop
        position.atr_at_entry = atr
        logger.debug(
            "%s: initial stop set at %.4f (entry=%.4f, atr=%.4f, mult=%.1f)",
            position.sc_code, stop, position.entry_price, atr, self.initial_stop_atr,
        )
        return stop

    def open_position(
        self,
        sc_code: str,
        sc_name: str,
        entry_price: float,
        entry_date: date,
        atr: float,
        shares: int,
    ) -> Position:
        """
        Create and initialise a new Position with stop loss and target set.

        Target is calculated as entry + 3 * ATR (a 1:3 risk/reward target).

        Parameters
        ----------
        sc_code : str
            BSE scrip code.
        sc_name : str
            Scrip display name.
        entry_price : float
            Execution price of the trade.
        entry_date : date
            Date the position was opened.
        atr : float
            ATR at entry (used for stop and target calculation).
        shares : int
            Number of shares purchased.

        Returns
        -------
        Position
            Fully initialised open position.
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if shares < 1:
            raise ValueError(f"shares must be >= 1, got {shares}")

        atr = max(atr, MIN_ATR_VALUE)
        pos = Position(
            sc_code=str(sc_code),
            sc_name=str(sc_name),
            entry_price=float(entry_price),
            entry_date=entry_date,
            atr_at_entry=atr,
            shares=int(shares),
            trailing_high=float(entry_price),
        )
        self.set_stop_loss(pos, atr)
        pos.target_price = entry_price + (3.0 * self.initial_stop_atr * atr)

        logger.info(
            "Opened position: %s %s | entry=%.2f | stop=%.2f | target=%.2f | shares=%d",
            sc_code, sc_name, entry_price, pos.stop_loss, pos.target_price, shares,
        )
        return pos

    def update_position(
        self,
        position: Position,
        current_price: float,
        current_date: date,
        current_atr: float,
    ) -> Position:
        """
        Evaluate exit conditions and update position state for one trading session.

        Exit logic (evaluated sequentially, first match wins):
        1. Time stop  : sessions_held >= time_stop_sessions AND gain < 1%
        2. Hard stop  : current_price <= stop_loss
        3. Trailing   : gain >= trail_activation_pct -> activate/update trailing stop
                        if current_price <= trail_stop -> STOPPED_OUT

        sessions_held is incremented at the end regardless of exit.

        Parameters
        ----------
        position : Position
            The position to update (modified in-place).
        current_price : float
            Latest traded / closing price for the scrip.
        current_date : date
            Today's date (for logging/audit).
        current_atr : float
            Current session's ATR for dynamic trail computation.

        Returns
        -------
        Position
            The updated position (same object, mutated in-place).
        """
        if not position.is_open:
            return position   # Already exited; nothing to do

        current_atr = max(current_atr, MIN_ATR_VALUE)
        gain_pct = (current_price - position.entry_price) / position.entry_price

        # ---- 1. Time stop ------------------------------------------------
        if (
            position.sessions_held >= self.time_stop_sessions
            and gain_pct < 0.01
        ):
            position.status = "TIME_STOP"
            logger.info(
                "%s TIME_STOP on %s | sessions=%d | gain=%.2f%%",
                position.sc_code, current_date,
                position.sessions_held, gain_pct * 100,
            )
            position.sessions_held += 1
            return position

        # ---- 2. Hard stop ------------------------------------------------
        if current_price <= position.stop_loss:
            position.status = "STOPPED_OUT"
            logger.info(
                "%s STOPPED_OUT on %s | price=%.4f <= stop=%.4f",
                position.sc_code, current_date, current_price, position.stop_loss,
            )
            position.sessions_held += 1
            return position

        # ---- 3. Trailing stop --------------------------------------------
        if gain_pct >= self.trail_activation_pct:
            position.trailing_stop_activated = True
            position.trailing_high = max(position.trailing_high, current_price)

            trail_stop = position.trailing_high - (self.trailing_stop_atr * current_atr)
            trail_stop = max(trail_stop, position.stop_loss)  # never move stop down

            if current_price <= trail_stop:
                position.status = "STOPPED_OUT"
                position.stop_loss = trail_stop
                logger.info(
                    "%s TRAILING STOP on %s | price=%.4f <= trail_stop=%.4f "
                    "(trail_high=%.4f, atr=%.4f)",
                    position.sc_code, current_date, current_price, trail_stop,
                    position.trailing_high, current_atr,
                )
            else:
                # Ratchet stop up
                position.stop_loss = trail_stop
                logger.debug(
                    "%s trailing stop ratcheted to %.4f (trail_high=%.4f)",
                    position.sc_code, trail_stop, position.trailing_high,
                )

        position.sessions_held += 1
        return position

    # ------------------------------------------------------------------
    # Portfolio-level update
    # ------------------------------------------------------------------

    def update_portfolio(
        self,
        positions_dict: Dict[str, Position],
        latest_prices_df: pd.DataFrame,
        current_date: date,
        atr_df: Dict[str, float],
    ) -> Dict[str, Position]:
        """
        Update all open positions in the portfolio for a single trading session.

        Parameters
        ----------
        positions_dict : dict
            {sc_code: Position} mapping.
        latest_prices_df : pd.DataFrame
            Must contain columns SC_CODE and CLOSE (latest closing prices).
        current_date : date
            Today's session date.
        atr_df : dict
            {sc_code: atr_value} pre-computed ATR for the session.

        Returns
        -------
        dict
            Updated {sc_code: Position} mapping.
        """
        price_map: Dict[str, float] = {}
        if not latest_prices_df.empty:
            for _, row in latest_prices_df.iterrows():
                code = str(row.get("SC_CODE", ""))
                price_map[code] = float(row.get("CLOSE", 0.0))

        closed_count = 0
        for sc_code, position in positions_dict.items():
            if not position.is_open:
                continue

            price = price_map.get(str(sc_code))
            if price is None or price <= 0:
                logger.warning(
                    "No price found for %s on %s; skipping update.", sc_code, current_date
                )
                continue

            atr_val = atr_df.get(str(sc_code), position.atr_at_entry)
            self.update_position(position, price, current_date, atr_val)

            if not position.is_open:
                closed_count += 1

        open_count = sum(1 for p in positions_dict.values() if p.is_open)
        logger.info(
            "Portfolio update %s: %d open, %d closed this session.",
            current_date, open_count, closed_count,
        )
        return positions_dict

    # ------------------------------------------------------------------
    # Exit utilities
    # ------------------------------------------------------------------

    def calculate_exit_price(self, position: Position, reason: str) -> float:
        """
        Estimate the actual exit price after slippage.

        For STOPPED_OUT/TIME_STOP: sell at stop_loss with adverse slippage.
        For TARGET_HIT: sell at target_price with adverse slippage.
        For MANUAL: sell at trailing_high (last known high) with slippage.

        Parameters
        ----------
        position : Position
            The position being exited.
        reason : str
            Exit reason string (matches position.status).

        Returns
        -------
        float
            Estimated fill price after slippage.
        """
        reason = reason.upper()

        if reason in ("STOPPED_OUT", "TIME_STOP"):
            raw = position.stop_loss
            exit_price = raw * (1 - SLIPPAGE_PCT)
        elif reason == "TARGET_HIT":
            raw = position.target_price
            exit_price = raw * (1 - SLIPPAGE_PCT)
        elif reason == "MANUAL":
            raw = position.trailing_high
            exit_price = raw * (1 - SLIPPAGE_PCT)
        else:
            logger.warning("Unknown exit reason '%s'; using trailing_high.", reason)
            exit_price = position.trailing_high * (1 - SLIPPAGE_PCT)

        return max(exit_price, 0.01)

    def get_pnl(self, position: Position, exit_price: float) -> dict:
        """
        Calculate P&L metrics for a closed (or hypothetically closed) position.

        Parameters
        ----------
        position : Position
            The position (open or closed).
        exit_price : float
            Actual or estimated exit price per share.

        Returns
        -------
        dict
            Keys:
            - gross_pnl     : float, total rupee P&L (exit - entry) * shares
            - pnl_pct       : float, percentage gain/loss from entry
            - sessions_held : int, number of sessions the position was held
            - exit_reason   : str, position.status at time of calculation
            - entry_price   : float
            - exit_price    : float
            - shares        : int
            - max_adverse   : float, max possible loss if stop had been hit from entry
        """
        gross_pnl = (exit_price - position.entry_price) * position.shares
        pnl_pct = (exit_price - position.entry_price) / position.entry_price * 100
        max_adverse = (position.stop_loss - position.entry_price) * position.shares

        return {
            "gross_pnl": round(gross_pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "sessions_held": position.sessions_held,
            "exit_reason": position.status,
            "entry_price": position.entry_price,
            "exit_price": round(exit_price, 4),
            "shares": position.shares,
            "max_adverse_excursion": round(max_adverse, 2),
        }


# ---------------------------------------------------------------------------
# Module-level utility
# ---------------------------------------------------------------------------

def calculate_risk_reward(
    entry: float,
    stop_loss: float,
    target: float,
) -> dict:
    """
    Compute risk, reward, and risk/reward ratio for a trade setup.

    Parameters
    ----------
    entry : float
        Entry price of the trade.
    stop_loss : float
        Hard stop-loss price level.
    target : float
        Target exit price.

    Returns
    -------
    dict
        Keys:
        - risk        : float, rupee risk per share (entry - stop_loss)
        - reward      : float, rupee reward per share (target - entry)
        - ratio       : float, reward / risk (higher is better; aim for >= 2.0)
        - risk_pct    : float, % risk from entry to stop
        - reward_pct  : float, % reward from entry to target

    Raises
    ------
    ValueError
        If entry <= stop_loss, or target <= entry (degenerate setup).
    """
    if entry <= stop_loss:
        raise ValueError(
            f"entry ({entry}) must be greater than stop_loss ({stop_loss})"
        )
    if target <= entry:
        raise ValueError(
            f"target ({target}) must be greater than entry ({entry})"
        )

    risk = entry - stop_loss
    reward = target - entry

    return {
        "risk": round(risk, 4),
        "reward": round(reward, 4),
        "ratio": round(reward / risk, 4),
        "risk_pct": round(risk / entry * 100, 4),
        "reward_pct": round(reward / entry * 100, 4),
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _detect_date_column(df: pd.DataFrame) -> Optional[str]:
    """Return the name of the date column in the DataFrame, or None."""
    candidates = ["ZDATE", "DATE", "TIMESTAMP", "DT", "DATE1"]
    cols_upper = {c.upper(): c for c in df.columns}
    for c in candidates:
        if c in cols_upper:
            return cols_upper[c]
    return None


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    engine = DynamicExitEngine(
        initial_stop_atr=2.5,
        trailing_stop_atr=1.5,
        time_stop_sessions=5,
        trail_activation_pct=0.02,
    )

    # --- Open a position ---
    entry_date = date(2026, 4, 1)
    pos = engine.open_position(
        sc_code="500325",
        sc_name="RELIANCE",
        entry_price=2800.0,
        entry_date=entry_date,
        atr=45.0,
        shares=10,
    )
    print(f"\nOpened: {pos.sc_name} | stop={pos.stop_loss:.2f} | target={pos.target_price:.2f}")

    rr = calculate_risk_reward(pos.entry_price, pos.stop_loss, pos.target_price)
    print(f"Risk/Reward: {rr}")

    # --- Simulate 7 sessions ---
    prices = [2810, 2830, 2870, 2910, 2880, 2950, 2920]
    atrs = [44, 43, 46, 48, 50, 47, 49]
    for i, (price, atr) in enumerate(zip(prices, atrs)):
        session_date = date(2026, 4, i + 2)
        engine.update_position(pos, float(price), session_date, float(atr))
        print(
            f"  Session {i+1}: price={price} | stop={pos.stop_loss:.2f} "
            f"| trail_high={pos.trailing_high:.2f} | status={pos.status}"
        )
        if not pos.is_open:
            break

    # If still OPEN after all sessions, treat as MANUAL exit at current price
    exit_reason = pos.status if not pos.is_open else "MANUAL"
    exit_price = engine.calculate_exit_price(pos, exit_reason)
    pnl = engine.get_pnl(pos, exit_price)
    print(f"\nFinal status: {pos.status} | exit_reason used: {exit_reason}")
    print(f"P&L Report: {pnl}")
