"""
production/paper_trader.py -- P24: Paper trading pipeline + monitoring dashboard.

Simulates live trade execution end-to-end without real money:

  1. On each trading day, loads bhav data and runs the full signal pipeline.
  2. Opens simulated positions with realistic fills (close price + slippage).
  3. Updates existing positions against current prices (P&L mark-to-market).
  4. Applies exit engine logic (stop, trail, time-stop).
  5. Logs all activity to a JSON trade log.
  6. Prints a terminal dashboard with open positions, daily P&L, and stats.

Usage
-----
from production.paper_trader import PaperTrader

pt = PaperTrader(capital=500_000)
pt.run_day(bhav_df, signal_date='2026-04-25')
pt.print_dashboard()
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT     = Path(__file__).parent.parent
LOG_DIR   = _ROOT / "stock_picker_data" / "paper_trades"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SLIPPAGE  = 0.001   # 0.1% fill slippage (buy high / sell low)
STT_PCT   = 0.001   # Securities Transaction Tax (equity delivery)
BROKERAGE = 20.0    # flat Rs 20 per order (Zerodha-style)


class PaperTrader:
    """
    P24 -- Simulated live trading runner.

    Parameters
    ----------
    capital          : Starting paper capital in INR.
    max_positions    : Maximum concurrent open positions.
    prob_threshold   : Minimum signal probability to enter.
    atr_stop_mult    : ATR multiplier for initial stop (default 1.5).
    trail_activation : Gain fraction to activate trailing stop (default 0.03).
    trail_atr_mult   : ATR multiplier for trailing stop distance (default 1.0).
    time_stop_days   : Sessions before time-stop fires (default 5).
    log_prefix       : Prefix for the JSON trade log filename.
    """

    def __init__(
        self,
        capital:          float = 500_000,
        max_positions:    int   = 10,
        prob_threshold:   float = 0.62,
        atr_stop_mult:    float = 1.5,
        trail_activation: float = 0.03,
        trail_atr_mult:   float = 1.0,
        time_stop_days:   int   = 5,
        log_prefix:       str   = "paper_trades",
    ) -> None:
        self.capital          = float(capital)
        self.initial_capital  = float(capital)
        self.max_positions    = max_positions
        self.prob_threshold   = prob_threshold
        self.atr_stop_mult    = atr_stop_mult
        self.trail_activation = trail_activation
        self.trail_atr_mult   = trail_atr_mult
        self.time_stop_days   = time_stop_days
        self.log_path         = LOG_DIR / f"{log_prefix}.json"

        self.open_positions: List[Dict[str, Any]] = []
        self.closed_trades:  List[Dict[str, Any]] = []
        self._daily_log:     List[Dict[str, Any]] = []

        self._load_state()

    # ------------------------------------------------------------------
    # Daily run
    # ------------------------------------------------------------------

    def run_day(
        self,
        bhav_df:     pd.DataFrame,
        signal_date: Any,
        picks_df:    Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Execute one paper trading day.

        Parameters
        ----------
        bhav_df     : BhavCopy for this date (needs SC_CODE, Close, ATR14 or High/Low).
        signal_date : The trading date being processed.
        picks_df    : Pre-computed picks (SC_CODE, Probability, Close).
                      If None, exits are still processed but no new entries.

        Returns
        -------
        dict with keys: date, n_entries, n_exits, daily_pnl, portfolio_value.
        """
        sig_date = pd.Timestamp(signal_date)
        price_map = self._build_price_map(bhav_df, sig_date)
        atr_map   = self._build_atr_map(bhav_df, sig_date)

        daily_pnl = 0.0
        n_exits   = 0
        n_entries = 0

        # --- 1. Update and exit existing positions ---
        still_open = []
        for pos in self.open_positions:
            sc = pos["sc_code"]
            price = price_map.get(sc)
            if price is None:
                still_open.append(pos)
                continue

            atr = atr_map.get(sc, pos.get("atr_at_entry", 5.0))
            pos["sessions_held"] += 1
            gain_pct = (price - pos["entry_price"]) / pos["entry_price"]

            exit_reason = None

            # Time stop
            if pos["sessions_held"] >= self.time_stop_days and gain_pct < 0.01:
                exit_reason = "TIME_STOP"

            # Hard stop
            elif price <= pos["stop_loss"]:
                exit_reason = "HARD_STOP"

            # Trailing stop
            elif gain_pct >= self.trail_activation:
                pos["trailing_high"] = max(pos["trailing_high"], price)
                trail_stop = pos["trailing_high"] - self.trail_atr_mult * atr
                trail_stop = max(trail_stop, pos["stop_loss"])
                pos["stop_loss"] = trail_stop
                if price <= trail_stop:
                    exit_reason = "TRAIL_STOP"

            if exit_reason:
                pnl = self._close_position(pos, price, exit_reason, sig_date)
                daily_pnl += pnl
                n_exits += 1
            else:
                pos["current_price"] = price
                pos["unrealised_pnl"] = (price - pos["entry_price"]) * pos["shares"]
                still_open.append(pos)

        self.open_positions = still_open

        # --- 2. Enter new positions from picks ---
        if picks_df is not None and not picks_df.empty:
            open_codes = {p["sc_code"] for p in self.open_positions}
            candidates = picks_df[
                (picks_df.get("Probability", picks_df.get("probability", pd.Series(dtype=float))) >= self.prob_threshold)
            ].copy() if ("Probability" in picks_df.columns or "probability" in picks_df.columns) else picks_df.copy()

            prob_col = "Probability" if "Probability" in candidates.columns else "probability"
            code_col = "SC_CODE" if "SC_CODE" in candidates.columns else "sc_code"

            for _, row in candidates.iterrows():
                if len(self.open_positions) >= self.max_positions:
                    break
                sc = str(row[code_col])
                if sc in open_codes:
                    continue
                price = price_map.get(sc, row.get("Close", row.get("close", None)))
                if price is None or price <= 0:
                    continue
                atr = atr_map.get(sc, price * 0.02)
                pos = self._open_position(sc, row.get("SC_NAME", sc), price, atr, sig_date)
                if pos:
                    self.open_positions.append(pos)
                    open_codes.add(sc)
                    n_entries += 1

        # --- 3. Mark-to-market unrealised ---
        portfolio_value = self.capital + sum(
            p.get("unrealised_pnl", 0) for p in self.open_positions
        ) + sum(p["shares"] * p.get("current_price", p["entry_price"]) for p in self.open_positions)

        day_result = {
            "date":            str(sig_date.date()),
            "n_entries":       n_entries,
            "n_exits":         n_exits,
            "daily_realised":  round(daily_pnl, 2),
            "open_positions":  len(self.open_positions),
            "portfolio_value": round(portfolio_value, 2),
        }
        self._daily_log.append(day_result)
        self._save_state()
        logger.info(
            "P24 PaperTrader [%s]: entries=%d exits=%d pnl=%.0f portfolio=%.0f",
            sig_date.date(), n_entries, n_exits, daily_pnl, portfolio_value,
        )
        return day_result

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def print_dashboard(self) -> None:
        total_unrealised = sum(
            (p.get("current_price", p["entry_price"]) - p["entry_price"]) * p["shares"]
            for p in self.open_positions
        )
        total_realised = sum(t["pnl"] for t in self.closed_trades)
        n_wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        win_rate = n_wins / len(self.closed_trades) if self.closed_trades else None
        portfolio_val = self.capital + total_unrealised + sum(
            p["shares"] * p.get("current_price", p["entry_price"]) for p in self.open_positions
        )

        print("=" * 70)
        print("  P24 PAPER TRADING DASHBOARD")
        print("=" * 70)
        print(f"  Capital (free):   Rs {self.capital:>12,.0f}")
        print(f"  Portfolio value:  Rs {portfolio_val:>12,.0f}  ({(portfolio_val/self.initial_capital-1)*100:+.2f}%)")
        print(f"  Realised P&L:     Rs {total_realised:>+12,.0f}")
        print(f"  Unrealised P&L:   Rs {total_unrealised:>+12,.0f}")
        print(f"  Closed trades:    {len(self.closed_trades)}  Win rate: {f'{win_rate:.1%}' if win_rate is not None else 'N/A'}")
        print(f"  Open positions:   {len(self.open_positions)}")
        if self.open_positions:
            print()
            print(f"  {'SC_CODE':<12} {'Entry':>8} {'Current':>8} {'Stop':>8} {'Sess':>5} {'Unreal P&L':>12}")
            print("  " + "-" * 58)
            for p in self.open_positions:
                curr  = p.get("current_price", p["entry_price"])
                unr   = (curr - p["entry_price"]) * p["shares"]
                print(
                    f"  {p['sc_code']:<12} {p['entry_price']:>8.2f} {curr:>8.2f} "
                    f"{p['stop_loss']:>8.2f} {p['sessions_held']:>5}  {unr:>+11,.0f}"
                )
        print("=" * 70)

    def get_summary(self) -> Dict[str, Any]:
        total_realised = sum(t["pnl"] for t in self.closed_trades)
        n_wins = sum(1 for t in self.closed_trades if t["pnl"] > 0)
        return {
            "initial_capital":  self.initial_capital,
            "free_capital":     round(self.capital, 2),
            "n_open":           len(self.open_positions),
            "n_closed":         len(self.closed_trades),
            "total_realised":   round(total_realised, 2),
            "win_rate":         round(n_wins / len(self.closed_trades), 4) if self.closed_trades else None,
            "n_trading_days":   len(self._daily_log),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_position(self, sc_code, sc_name, price, atr, entry_date) -> Optional[Dict]:
        fill = price * (1 + SLIPPAGE)
        stop = max(0.01, fill - self.atr_stop_mult * atr)
        per_share_alloc = self.capital / (self.max_positions - len(self.open_positions))
        shares = max(1, int(per_share_alloc / fill))
        cost = shares * fill + BROKERAGE + shares * fill * STT_PCT

        if cost > self.capital:
            return None

        self.capital -= cost
        return {
            "sc_code":        sc_code,
            "sc_name":        sc_name,
            "entry_price":    round(fill, 2),
            "entry_date":     str(entry_date.date()),
            "stop_loss":      round(stop, 2),
            "trailing_high":  fill,
            "atr_at_entry":   round(atr, 2),
            "shares":         shares,
            "cost":           round(cost, 2),
            "sessions_held":  0,
            "current_price":  fill,
            "unrealised_pnl": 0.0,
        }

    def _close_position(self, pos, price, reason, close_date) -> float:
        fill = price * (1 - SLIPPAGE)
        proceeds = pos["shares"] * fill - BROKERAGE - pos["shares"] * fill * STT_PCT
        pnl = proceeds - pos["cost"]
        self.capital += proceeds
        trade = {
            "sc_code":     pos["sc_code"],
            "entry_price": pos["entry_price"],
            "exit_price":  round(fill, 2),
            "shares":      pos["shares"],
            "pnl":         round(pnl, 2),
            "reason":      reason,
            "sessions":    pos["sessions_held"],
            "entry_date":  pos["entry_date"],
            "exit_date":   str(close_date.date()),
        }
        self.closed_trades.append(trade)
        logger.info(
            "P24 CLOSE %s: %s @ %.2f  pnl=%.0f  reason=%s",
            pos["sc_code"], close_date.date(), fill, pnl, reason,
        )
        return pnl

    def _build_price_map(self, bhav_df, sig_date) -> Dict[str, float]:
        df = bhav_df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        day = df[df["DATE"] == sig_date]
        if day.empty:
            day = df[df["DATE"] == df["DATE"].max()]
        return dict(zip(day["SC_CODE"].astype(str), day["Close"].astype(float)))

    def _build_atr_map(self, bhav_df, sig_date) -> Dict[str, float]:
        df = bhav_df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        day = df[df["DATE"] == sig_date]
        if day.empty:
            day = df[df["DATE"] == df["DATE"].max()]
        if "ATR14" in day.columns:
            return dict(zip(day["SC_CODE"].astype(str), day["ATR14"].fillna(0).astype(float)))
        return {}

    def _save_state(self) -> None:
        try:
            state = {
                "capital":         self.capital,
                "initial_capital": self.initial_capital,
                "open_positions":  self.open_positions,
                "closed_trades":   self.closed_trades[-200:],
                "daily_log":       self._daily_log[-60:],
                "saved_at":        datetime.now().isoformat(),
            }
            self.log_path.write_text(json.dumps(state, indent=2, default=str))
        except Exception as exc:
            logger.warning("P24: State save failed: %s", exc)

    def _load_state(self) -> None:
        if not self.log_path.exists():
            return
        try:
            state = json.loads(self.log_path.read_text())
            self.capital         = state.get("capital", self.capital)
            self.initial_capital = state.get("initial_capital", self.initial_capital)
            self.open_positions  = state.get("open_positions", [])
            self.closed_trades   = state.get("closed_trades", [])
            self._daily_log      = state.get("daily_log", [])
            logger.info(
                "P24: Paper trader state loaded (%d open, %d closed trades)",
                len(self.open_positions), len(self.closed_trades),
            )
        except Exception as exc:
            logger.warning("P24: State load failed (%s) -- starting fresh.", exc)
