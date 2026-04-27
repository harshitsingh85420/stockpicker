"""
position_sizer.py - Gap 8: Risk-Based Position Sizing
======================================================
Production module for calculating trade sizes in Indian equity markets.

All sizing logic is rooted in the classic 1% risk-per-trade rule:
  shares = floor((capital * risk_pct) / (entry - stop_loss))

Additional guardrails:
  - Max position concentration (max_position_pct of total capital)
  - Portfolio-level capital saturation check (max_capital_deployed_pct)
  - ATR-based dynamic stop calculation for sizing when no explicit stop is provided

Designed to integrate directly with universe_filter.py output and
exit_engine.py stop-loss levels.

Author: StockPicker System
Date: 2026-04-23
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level simple helpers
# ---------------------------------------------------------------------------

def risk_based_shares(
    capital: float,
    risk_pct: float,
    entry: float,
    stop: float,
) -> int:
    """
    Compute share count using the 1%-risk rule.

    shares = floor((capital * risk_pct) / (entry - stop))

    Parameters
    ----------
    capital : float
        Available capital in INR.
    risk_pct : float
        Fraction of capital to risk (e.g. 0.01 for 1%).
    entry : float
        Entry price per share.
    stop : float
        Hard stop-loss price per share.

    Returns
    -------
    int
        Number of shares to buy (>= 0). Returns 0 if stop >= entry.
    """
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    return max(0, math.floor((capital * risk_pct) / risk_per_share))


def equal_weight_allocation(
    capital: float,
    n_positions: int,
    price: float,
) -> int:
    """
    Compute shares for an equal-weight (naive) portfolio allocation.

    Parameters
    ----------
    capital : float
        Total capital to distribute equally across positions.
    n_positions : int
        Number of positions to fund.
    price : float
        Current market price per share.

    Returns
    -------
    int
        Shares per position (floor). Returns 0 if price or n_positions is zero.
    """
    if n_positions <= 0 or price <= 0:
        return 0
    allocation_per_position = capital / n_positions
    return max(0, math.floor(allocation_per_position / price))


# ---------------------------------------------------------------------------
# PositionSizer
# ---------------------------------------------------------------------------

class PositionSizer:
    """
    Risk-aware position sizer for Indian equity portfolios.

    Each trade is sized so the maximum loss (if stop is hit) equals
    risk_pct_per_trade * total_capital, subject to:
      - Position value <= max_position_pct * available_capital
      - Total open positions <= max_positions
      - At least min_shares purchased

    Parameters
    ----------
    total_capital : float
        Total portfolio capital in INR. Used as the denominator for risk %.
        Default: 10,00,000 (10 lakh).
    risk_pct_per_trade : float
        Fraction of total_capital to risk on one trade. Default 0.01 (1%).
    max_position_pct : float
        Maximum fraction of available_capital in a single position. Default 0.10 (10%).
    max_positions : int
        Maximum number of concurrent open positions. Default 15.
    min_shares : int
        Minimum number of shares per position. Default 1.
    """

    def __init__(
        self,
        total_capital: float = 1_000_000.0,
        risk_pct_per_trade: float = 0.01,
        max_position_pct: float = 0.10,
        max_positions: int = 15,
        min_shares: int = 1,
    ):
        if total_capital <= 0:
            raise ValueError("total_capital must be positive.")
        if not (0 < risk_pct_per_trade < 1):
            raise ValueError("risk_pct_per_trade must be between 0 and 1 exclusive.")
        if not (0 < max_position_pct <= 1):
            raise ValueError("max_position_pct must be between 0 and 1.")
        if max_positions < 1:
            raise ValueError("max_positions must be >= 1.")
        if min_shares < 1:
            raise ValueError("min_shares must be >= 1.")

        self.total_capital = float(total_capital)
        self.risk_pct_per_trade = float(risk_pct_per_trade)
        self.max_position_pct = float(max_position_pct)
        self.max_positions = int(max_positions)
        self.min_shares = int(min_shares)

    # ------------------------------------------------------------------
    # Core sizing methods
    # ------------------------------------------------------------------

    def calculate_shares(
        self,
        entry_price: float,
        stop_loss: float,
        available_capital: float,
    ) -> int:
        """
        Calculate the optimal number of shares using the fixed-risk rule.

        Formula:
          raw_shares = floor((total_capital * risk_pct) / (entry - stop))
          max_shares  = floor((available_capital * max_position_pct) / entry)
          shares      = min(raw_shares, max_shares)

        Parameters
        ----------
        entry_price : float
            Intended entry price per share.
        stop_loss : float
            Hard stop-loss price per share.
        available_capital : float
            Cash currently available to deploy (not already invested).

        Returns
        -------
        int
            Number of shares to buy. Returns 0 if the trade is unfeasible
            (stop >= entry, or resulting position value > available capital).

        Notes
        -----
        Uses self.total_capital for risk sizing so a 1% loss never changes
        even as available_capital shrinks mid-portfolio build.
        """
        risk_per_share = entry_price - stop_loss
        if risk_per_share <= 0:
            logger.warning(
                "calculate_shares: stop_loss (%.4f) >= entry_price (%.4f). "
                "Returning 0 shares.",
                stop_loss, entry_price,
            )
            return 0

        # Risk-rule shares
        raw_shares = math.floor(
            (self.total_capital * self.risk_pct_per_trade) / risk_per_share
        )

        # Position-concentration cap
        max_position_value = available_capital * self.max_position_pct
        max_shares_by_value = math.floor(max_position_value / entry_price)

        shares = min(raw_shares, max_shares_by_value)

        # Cannot afford even one share
        if entry_price > available_capital:
            logger.warning(
                "entry_price (%.2f) > available_capital (%.2f). Returning 0 shares.",
                entry_price, available_capital,
            )
            return 0

        shares = max(shares, 0)

        # Enforce minimum only if we can afford it
        if 0 < shares < self.min_shares:
            if self.min_shares * entry_price <= max_position_value:
                shares = self.min_shares
            else:
                shares = 0

        logger.debug(
            "Shares: entry=%.2f | stop=%.2f | risk/share=%.4f | raw=%d | capped=%d",
            entry_price, stop_loss, risk_per_share, raw_shares, shares,
        )
        return shares

    def calculate_atr_position_size(
        self,
        entry_price: float,
        atr: float,
        available_capital: float,
        atr_stop_multiplier: float = 2.5,
    ) -> int:
        """
        Calculate position size using an ATR-derived stop loss.

        Derives stop_loss = entry - (atr_stop_multiplier * atr), then calls
        calculate_shares().

        Parameters
        ----------
        entry_price : float
            Intended entry price per share.
        atr : float
            Average True Range for the stock (same price units as entry_price).
        available_capital : float
            Cash currently available for deployment.
        atr_stop_multiplier : float
            Multiplier applied to ATR to set the stop distance. Default 2.5.

        Returns
        -------
        int
            Number of shares to buy.
        """
        if atr <= 0:
            logger.warning(
                "ATR is zero or negative (%.4f) for entry %.2f. Returning 0 shares.",
                atr, entry_price,
            )
            return 0

        stop_loss = entry_price - (atr_stop_multiplier * atr)
        stop_loss = max(stop_loss, 0.01)

        return self.calculate_shares(entry_price, stop_loss, available_capital)

    # ------------------------------------------------------------------
    # Portfolio-level allocation
    # ------------------------------------------------------------------

    def calculate_portfolio_allocation(
        self,
        picks_df: pd.DataFrame,
        available_capital: float,
        atr_dict: Dict[str, float],
        atr_stop_multiplier: float = 2.5,
    ) -> pd.DataFrame:
        """
        Compute position sizing for a full set of stock picks.

        Adds the following columns to picks_df (returns a copy):
          - Stop_Loss       : ATR-derived hard stop price
          - Shares          : Recommended share count
          - Position_Value  : Shares * Entry price (INR)
          - Risk_Amount     : (Entry - Stop) * Shares (INR at risk per position)
          - Risk_Pct        : Risk_Amount / total_capital * 100 (%)
          - Allocation_Pct  : Position_Value / available_capital * 100 (%)

        Parameters
        ----------
        picks_df : pd.DataFrame
            Must contain columns: SC_CODE, Entry_Price (or CLOSE as fallback).
            ATR is sourced from atr_dict; missing entries default to 2% of price.
        available_capital : float
            Cash available for new positions.
        atr_dict : dict
            {sc_code: atr_value} from ATRCalculator.get_atr_for_universe().
        atr_stop_multiplier : float
            ATR multiplier to use for stop derivation.

        Returns
        -------
        pd.DataFrame
            picks_df with sizing columns appended.
        """
        df = picks_df.copy()

        # Resolve entry price column
        price_col = "Entry_Price" if "Entry_Price" in df.columns else "CLOSE"
        if price_col not in df.columns:
            raise KeyError(
                f"picks_df must contain 'Entry_Price' or 'CLOSE'; found {list(df.columns)}"
            )

        sc_col = "SC_CODE" if "SC_CODE" in df.columns else df.columns[0]

        stop_list, shares_list, pos_val_list, risk_amt_list, risk_pct_list, alloc_pct_list = (
            [], [], [], [], [], []
        )

        for _, row in df.iterrows():
            sc_code = str(row[sc_col])
            entry = float(row[price_col])

            atr = atr_dict.get(sc_code, entry * 0.02)   # fallback: 2% of price
            stop = max(entry - atr_stop_multiplier * atr, 0.01)

            shares = self.calculate_shares(entry, stop, available_capital)

            pos_val = shares * entry
            risk_amt = (entry - stop) * shares
            risk_pct = (risk_amt / self.total_capital) * 100 if self.total_capital > 0 else 0
            alloc_pct = (pos_val / available_capital) * 100 if available_capital > 0 else 0

            stop_list.append(round(stop, 4))
            shares_list.append(shares)
            pos_val_list.append(round(pos_val, 2))
            risk_amt_list.append(round(risk_amt, 2))
            risk_pct_list.append(round(risk_pct, 4))
            alloc_pct_list.append(round(alloc_pct, 4))

        df["Stop_Loss"] = stop_list
        df["Shares"] = shares_list
        df["Position_Value"] = pos_val_list
        df["Risk_Amount"] = risk_amt_list
        df["Risk_Pct"] = risk_pct_list
        df["Allocation_Pct"] = alloc_pct_list

        logger.info(
            "Portfolio allocation: %d picks sized | total deployed = %.2f",
            len(df), df["Position_Value"].sum(),
        )
        return df

    def check_capital_saturation(
        self,
        current_positions: List[Dict],
        picks_df: pd.DataFrame,
        max_capital_deployed_pct: float = 0.70,
    ) -> pd.DataFrame:
        """
        Filter new picks that would breach the maximum capital deployment limit.

        The function greedily adds picks (in their existing order) until either
        the capital threshold or max_positions is reached.

        Parameters
        ----------
        current_positions : list of dict
            Existing open positions, each with keys 'Position_Value' and
            optionally 'SC_CODE'. Represents already-deployed capital.
        picks_df : pd.DataFrame
            New candidate picks with at minimum a 'Position_Value' column
            (added by calculate_portfolio_allocation).
        max_capital_deployed_pct : float
            Maximum fraction of total_capital that may be deployed at once.
            Default 0.70 (70%).

        Returns
        -------
        pd.DataFrame
            Subset of picks_df that can be added without exceeding limits.
        """
        if "Position_Value" not in picks_df.columns:
            raise KeyError(
                "picks_df must have 'Position_Value' column. "
                "Run calculate_portfolio_allocation first."
            )

        max_deployable = self.total_capital * max_capital_deployed_pct
        already_deployed = sum(
            float(p.get("Position_Value", 0)) for p in current_positions
        )
        current_open = len(current_positions)

        approved_indices = []
        running_deployed = already_deployed

        for idx, row in picks_df.iterrows():
            if current_open + len(approved_indices) >= self.max_positions:
                logger.info(
                    "Capital saturation: max_positions=%d reached. Stopping.",
                    self.max_positions,
                )
                break

            candidate_value = float(row["Position_Value"])
            if running_deployed + candidate_value > max_deployable:
                logger.debug(
                    "Skipping %s: would deploy %.0f > limit %.0f",
                    row.get("SC_CODE", idx), running_deployed + candidate_value, max_deployable,
                )
                continue

            approved_indices.append(idx)
            running_deployed += candidate_value

        filtered = picks_df.loc[approved_indices].copy()
        logger.info(
            "Capital saturation check: %d/%d picks approved | deployed=%.0f/%.0f (%.1f%%)",
            len(filtered), len(picks_df),
            running_deployed, max_deployable,
            running_deployed / self.total_capital * 100,
        )
        return filtered

    def get_sizing_report(self, picks_df: pd.DataFrame) -> dict:
        """
        Generate a summary report for a sized set of picks.

        Parameters
        ----------
        picks_df : pd.DataFrame
            Output of calculate_portfolio_allocation() — must contain
            Position_Value, Risk_Amount, Risk_Pct, Shares columns.

        Returns
        -------
        dict
            Keys:
            - total_capital_deployed : float, sum of Position_Value
            - pct_of_capital_deployed : float, as % of self.total_capital
            - num_positions           : int, number of picks with Shares > 0
            - avg_risk_per_trade      : float, mean Risk_Amount in INR
            - total_risk_amount       : float, sum of Risk_Amount (worst-case if all stops hit)
            - max_drawdown_estimate   : float, total_risk / total_capital * 100 (%)
            - avg_allocation_pct      : float, mean position size as % of capital
            - largest_position_value  : float, biggest single position in INR
        """
        required = {"Position_Value", "Risk_Amount", "Shares"}
        missing = required - set(picks_df.columns)
        if missing:
            raise KeyError(
                f"picks_df is missing columns: {missing}. "
                "Run calculate_portfolio_allocation() first."
            )

        active = picks_df[picks_df["Shares"] > 0]

        total_deployed = active["Position_Value"].sum()
        total_risk = active["Risk_Amount"].sum()
        num_pos = len(active)

        return {
            "total_capital_deployed": round(total_deployed, 2),
            "pct_of_capital_deployed": round(total_deployed / self.total_capital * 100, 4),
            "num_positions": num_pos,
            "avg_risk_per_trade": round(active["Risk_Amount"].mean(), 2) if num_pos else 0.0,
            "total_risk_amount": round(total_risk, 2),
            "max_drawdown_estimate": round(total_risk / self.total_capital * 100, 4),
            "avg_allocation_pct": round(
                active["Allocation_Pct"].mean(), 4
            ) if "Allocation_Pct" in active.columns and num_pos else 0.0,
            "largest_position_value": round(active["Position_Value"].max(), 2) if num_pos else 0.0,
        }


# ---------------------------------------------------------------------------
# P19 -- ATR-based position sizing with half-Kelly criterion
# ---------------------------------------------------------------------------

def kelly_fraction(
    win_prob: float,
    avg_win:  float,
    avg_loss: float,
    half_kelly: bool = True,
) -> float:
    """
    Compute (optionally halved) Kelly criterion fraction.

    Kelly formula:  f* = (b*p - q) / b
      b = avg_win / avg_loss  (win/loss ratio)
      p = win probability
      q = 1 - p

    Parameters
    ----------
    win_prob   : Estimated probability of a winning trade (e.g. from model).
    avg_win    : Average win size as a fraction (e.g. 0.05 = 5%).
    avg_loss   : Average loss size as a fraction (e.g. 0.02 = 2%).
    half_kelly : Use half-Kelly to reduce volatility (default True).

    Returns
    -------
    float in [0, 1] -- fraction of capital to allocate.
    """
    if avg_loss <= 0 or avg_win <= 0 or not (0 < win_prob < 1):
        return 0.0
    b = avg_win / avg_loss
    q = 1.0 - win_prob
    f = (b * win_prob - q) / b
    f = max(0.0, f)   # Kelly can be negative -- floor at 0 (no trade)
    if half_kelly:
        f *= 0.5
    return round(min(f, 1.0), 6)


def atr_kelly_shares(
    capital:      float,
    entry:        float,
    atr:          float,
    win_prob:     float,
    avg_win_atr:  float = 2.0,
    avg_loss_atr: float = 1.0,
    atr_stop_mult: float = 1.0,
    max_position_pct: float = 0.10,
    half_kelly:   bool = True,
) -> Dict:
    """
    P19 -- Combine ATR stop-sizing with half-Kelly position sizing.

    Step 1: Set stop-loss at entry - atr_stop_mult * ATR.
    Step 2: Compute Kelly fraction from win_prob and ATR-derived R-multiple.
    Step 3: Translate Kelly capital fraction into share count.
    Step 4: Apply max_position_pct cap.

    Parameters
    ----------
    capital        : Available capital in INR.
    entry          : Entry price per share.
    atr            : 14-day ATR of the stock.
    win_prob       : Model-predicted win probability for this trade.
    avg_win_atr    : Typical profit as ATR multiples (default 2.0).
    avg_loss_atr   : Typical loss as ATR multiples (default 1.0).
    atr_stop_mult  : ATR multiple for stop (default 1.0 ATR below entry).
    max_position_pct: Cap on fraction of capital per position (default 10%).
    half_kelly     : Use half-Kelly (default True -- industry standard for live trading).

    Returns
    -------
    dict with keys:
        shares, stop_price, kelly_fraction, position_value,
        position_pct, risk_per_share, total_risk
    """
    if entry <= 0 or atr <= 0 or capital <= 0:
        return {"shares": 0, "stop_price": 0.0, "kelly_fraction": 0.0,
                "position_value": 0.0, "position_pct": 0.0,
                "risk_per_share": 0.0, "total_risk": 0.0}

    stop_price = max(0.01, entry - atr_stop_mult * atr)
    risk_per_share = entry - stop_price

    # Express win/loss as fraction of entry price
    avg_win_pct  = avg_win_atr  * atr / entry
    avg_loss_pct = avg_loss_atr * atr / entry

    kf = kelly_fraction(win_prob, avg_win_pct, avg_loss_pct, half_kelly=half_kelly)

    # Capital fraction * total capital = position value
    max_val   = capital * max_position_pct
    kelly_val = capital * kf
    pos_val   = min(kelly_val, max_val)
    shares    = max(0, math.floor(pos_val / entry))
    actual_val = shares * entry

    return {
        "shares":          shares,
        "stop_price":      round(stop_price, 2),
        "kelly_fraction":  kf,
        "position_value":  round(actual_val, 2),
        "position_pct":    round(actual_val / capital, 6) if capital > 0 else 0.0,
        "risk_per_share":  round(risk_per_share, 2),
        "total_risk":      round(shares * risk_per_share, 2),
    }


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    print("=" * 65)
    print("  Position Sizer - Demo / Self-Test")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Standalone helper functions
    # ------------------------------------------------------------------
    print("\n[1] Module-level helpers")
    shares = risk_based_shares(
        capital=1_000_000,
        risk_pct=0.01,
        entry=2800.0,
        stop=2687.5,    # 2.5 * 45 ATR below entry
    )
    print(f"  risk_based_shares(1M, 1%, 2800, 2687.5) = {shares} shares")

    eq_shares = equal_weight_allocation(
        capital=1_000_000,
        n_positions=10,
        price=2800.0,
    )
    print(f"  equal_weight_allocation(1M, 10 pos, 2800) = {eq_shares} shares/position")

    # ------------------------------------------------------------------
    # 2. PositionSizer instance
    # ------------------------------------------------------------------
    print("\n[2] PositionSizer initialised")
    sizer = PositionSizer(
        total_capital=1_000_000,
        risk_pct_per_trade=0.01,
        max_position_pct=0.10,
        max_positions=15,
        min_shares=1,
    )
    print(f"  total_capital = {sizer.total_capital:,.0f}")
    print(f"  risk_pct      = {sizer.risk_pct_per_trade:.1%}")
    print(f"  max_position  = {sizer.max_position_pct:.1%}")

    # ------------------------------------------------------------------
    # 3. Single stock sizing
    # ------------------------------------------------------------------
    print("\n[3] Single stock sizing (ATR-based)")
    entry, atr = 1540.0, 28.0
    n = sizer.calculate_atr_position_size(entry, atr, available_capital=900_000)
    stop = entry - 2.5 * atr
    print(f"  Entry={entry}, ATR={atr}, Stop={stop:.2f}")
    print(f"  Shares={n} | Position Value={n * entry:,.2f} | Risk={n * (entry - stop):,.2f}")

    # ------------------------------------------------------------------
    # 4. Portfolio allocation across multiple picks
    # ------------------------------------------------------------------
    print("\n[4] Portfolio allocation for sample picks")

    picks_data = {
        "SC_CODE":    ["500325", "532540", "500696", "540777", "500010",
                       "532978", "500112", "500570", "500180", "500900"],
        "SC_NAME":    ["RELIANCE", "TCS", "HINDUNILVR", "HDFC BANK", "HDFC",
                       "BAJAJ FIN", "SBI", "TITAN", "ICICI BANK", "WIPRO"],
        "Entry_Price":[2800.0, 3500.0, 2650.0, 1620.0, 2940.0,
                       6800.0,  610.0, 3250.0, 1050.0,  490.0],
    }

    atr_dict = {
        "500325": 45.0, "532540": 58.0, "500696": 42.0,
        "540777": 26.0, "500010": 47.0, "532978": 110.0,
        "500112": 12.0, "500570": 55.0, "500180":  18.0,
        "500900":  9.0,
    }

    picks_df = pd.DataFrame(picks_data)
    sized_df = sizer.calculate_portfolio_allocation(
        picks_df,
        available_capital=900_000,
        atr_dict=atr_dict,
    )

    display_cols = [
        "SC_NAME", "Entry_Price", "Stop_Loss", "Shares",
        "Position_Value", "Risk_Amount", "Risk_Pct", "Allocation_Pct",
    ]
    print(sized_df[display_cols].to_string(index=False))

    # ------------------------------------------------------------------
    # 5. Capital saturation check
    # ------------------------------------------------------------------
    print("\n[5] Capital saturation check (max 70% deployed)")

    current_open = [
        {"SC_CODE": "999001", "Position_Value": 150_000},
        {"SC_CODE": "999002", "Position_Value": 120_000},
    ]
    approved = sizer.check_capital_saturation(
        current_positions=current_open,
        picks_df=sized_df,
        max_capital_deployed_pct=0.70,
    )
    print(f"  Already deployed : {sum(p['Position_Value'] for p in current_open):,.0f}")
    print(f"  Picks approved   : {len(approved)} / {len(sized_df)}")
    if len(approved):
        print(f"  New capital used : {approved['Position_Value'].sum():,.2f}")

    # ------------------------------------------------------------------
    # 6. Sizing report
    # ------------------------------------------------------------------
    print("\n[6] Sizing report")
    report = sizer.get_sizing_report(sized_df)
    for k, v in report.items():
        print(f"  {k:<30} : {v}")

    print("\nDone.")
