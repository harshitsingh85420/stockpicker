"""
friction_model.py
Gap 10: Full Friction Layer
Gap 20: Dynamic Cost Manager

Production module for Indian market transaction cost modeling.
Implements all SEBI/NSE/BSE/Finance Act charges as of April 2026.

Key rate changes effective April 1, 2026 (Finance Act 2025):
  - Futures STT: 0.0005 (0.05%), up from 0.02%
  - Options STT on premium: 0.0015 (0.15%), up from 0.1%
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 1. TransactionCostConfig
# ---------------------------------------------------------------------------

@dataclass
class TransactionCostConfig:
    """
    Complete Indian transaction cost configuration.
    All rates are expressed as decimals (fractions of trade value).

    Last updated: 2026-04-01 to reflect Finance Act 2025 changes.
    """

    # --- STT (Securities Transaction Tax) ---
    # Charged on the sell side for most instrument types.
    # Delivery buy: exempt from STT since 2004.
    stt_delivery_buy: float = 0.0          # No STT on buy leg for delivery
    stt_delivery_sell: float = 0.001       # 0.1% on sell turnover
    stt_intraday_sell: float = 0.00025     # 0.025% on sell turnover
    stt_futures_sell: float = 0.0005       # 0.05% (UP from 0.02%, effective Apr 1 2026)
    stt_options_sell_premium: float = 0.0015  # 0.15% on premium (UP from 0.1%, effective Apr 1 2026)

    # --- Exchange Transaction Charges (per rupee of turnover) ---
    exchange_nse_equity: float = 0.0000297    # NSE equity segment
    exchange_bse_equity: float = 0.0000375    # BSE equity segment
    exchange_nse_futures: float = 0.000019    # NSE F&O futures
    exchange_nse_options: float = 0.0000495   # NSE F&O options (on premium)

    # --- SEBI Turnover Fees ---
    # ₹10 per crore = 0.000001 per rupee of turnover
    sebi_charges: float = 0.000001

    # --- Stamp Duty (collected by state govts via exchanges, buy side only) ---
    stamp_duty_buy_equity: float = 0.00015   # 0.015% on buy value
    stamp_duty_buy_futures: float = 0.00002  # 0.002% on buy value
    stamp_duty_buy_options: float = 0.00003  # 0.003% on buy value (on premium)

    # --- GST (on brokerage + exchange charges + SEBI charges) ---
    gst_rate: float = 0.18  # 18%

    # --- Brokerage (lower of flat or percentage) ---
    brokerage_flat_per_trade: float = 20.0   # ₹20 per executed order
    brokerage_percentage: float = 0.0003     # 0.03% of trade value

    # --- IPFT (Investor Protection Fund Trust) ---
    ipft_nse: float = 0.000001   # ₹10 per crore on NSE
    ipft_bse: float = 0.000001   # ₹10 per crore on BSE

    # --- Metadata ---
    version: str = "2026-04-01"  # Date of last rate update

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TransactionCostConfig":
        # Accept only known fields to stay forward-compatible
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# 2. FrictionModel  (Gap 10)
# ---------------------------------------------------------------------------

class FrictionModel:
    """
    Full friction layer for Indian equity and derivatives trading.

    Supported instrument types:
      'equity_delivery'  - Cash delivery (T+1/T+2)
      'equity_intraday'  - MIS/BTST intraday trades
      'futures'          - Equity/Index futures
      'options'          - Equity/Index options (value = premium turnover)
    """

    VALID_INSTRUMENT_TYPES = (
        "equity_delivery",
        "equity_intraday",
        "futures",
        "options",
    )

    def __init__(
        self,
        config: Optional[TransactionCostConfig] = None,
        instrument_type: str = "equity_delivery",
    ) -> None:
        if instrument_type not in self.VALID_INSTRUMENT_TYPES:
            raise ValueError(
                f"instrument_type must be one of {self.VALID_INSTRUMENT_TYPES}"
            )
        self.config = config or TransactionCostConfig()
        self.instrument_type = instrument_type

    # ------------------------------------------------------------------
    # Component calculations
    # ------------------------------------------------------------------

    def calculate_brokerage(self, trade_value: float) -> float:
        """
        Return brokerage cost.
        Uses the lower of flat ₹20 per order or 0.03% of trade value.
        """
        flat = self.config.brokerage_flat_per_trade
        pct = trade_value * self.config.brokerage_percentage
        return min(flat, pct)

    def calculate_stt(
        self,
        trade_value: float,
        is_buy: bool,
        instrument_type: Optional[str] = None,
    ) -> float:
        """
        Return STT (Securities Transaction Tax).

        For delivery buy, STT is zero.
        For options, trade_value should be the premium turnover.
        """
        itype = instrument_type or self.instrument_type
        cfg = self.config

        if itype == "equity_delivery":
            return 0.0 if is_buy else trade_value * cfg.stt_delivery_sell

        if itype == "equity_intraday":
            # STT charged on sell side only for intraday
            return 0.0 if is_buy else trade_value * cfg.stt_intraday_sell

        if itype == "futures":
            # STT on sell side only
            return 0.0 if is_buy else trade_value * cfg.stt_futures_sell

        if itype == "options":
            # STT on sell side premium; options exercise STT handled separately
            return 0.0 if is_buy else trade_value * cfg.stt_options_sell_premium

        return 0.0

    def calculate_exchange_charges(
        self,
        trade_value: float,
        exchange: str = "NSE",
        instrument_type: Optional[str] = None,
    ) -> float:
        """
        Return exchange transaction charges.
        """
        itype = instrument_type or self.instrument_type
        cfg = self.config
        exch = exchange.upper()

        if itype in ("equity_delivery", "equity_intraday"):
            rate = cfg.exchange_nse_equity if exch == "NSE" else cfg.exchange_bse_equity
        elif itype == "futures":
            rate = cfg.exchange_nse_futures
        elif itype == "options":
            rate = cfg.exchange_nse_options
        else:
            rate = cfg.exchange_nse_equity

        return trade_value * rate

    def calculate_stamp_duty(
        self,
        trade_value: float,
        is_buy: bool,
        instrument_type: Optional[str] = None,
    ) -> float:
        """
        Return stamp duty (charged on buy side only).
        """
        if not is_buy:
            return 0.0

        itype = instrument_type or self.instrument_type
        cfg = self.config

        if itype in ("equity_delivery", "equity_intraday"):
            return trade_value * cfg.stamp_duty_buy_equity
        if itype == "futures":
            return trade_value * cfg.stamp_duty_buy_futures
        if itype == "options":
            return trade_value * cfg.stamp_duty_buy_options

        return trade_value * cfg.stamp_duty_buy_equity

    def calculate_sebi_charges(self, trade_value: float) -> float:
        """Return SEBI turnover fee (₹10 per crore = 0.000001)."""
        return trade_value * self.config.sebi_charges

    def calculate_gst(
        self,
        brokerage: float,
        exchange_charges: float,
        sebi_charges: float,
    ) -> float:
        """Return GST (18%) on brokerage + exchange charges + SEBI charges."""
        taxable_base = brokerage + exchange_charges + sebi_charges
        return taxable_base * self.config.gst_rate

    def _calculate_ipft(self, trade_value: float, exchange: str = "NSE") -> float:
        """Return IPFT charge."""
        exch = exchange.upper()
        rate = self.config.ipft_nse if exch == "NSE" else self.config.ipft_bse
        return trade_value * rate

    # ------------------------------------------------------------------
    # Composite calculations
    # ------------------------------------------------------------------

    def calculate_total_cost(
        self,
        trade_value: float,
        is_buy: bool,
        exchange: str = "NSE",
    ) -> dict:
        """
        Calculate all-in cost for a single leg (buy or sell).

        Returns
        -------
        dict with keys:
            brokerage, stt, exchange_charges, stamp_duty,
            sebi_charges, gst, ipft, total, total_pct
        """
        brokerage = self.calculate_brokerage(trade_value)
        stt = self.calculate_stt(trade_value, is_buy)
        exchange_charges = self.calculate_exchange_charges(trade_value, exchange)
        stamp_duty = self.calculate_stamp_duty(trade_value, is_buy)
        sebi_charges = self.calculate_sebi_charges(trade_value)
        gst = self.calculate_gst(brokerage, exchange_charges, sebi_charges)
        ipft = self._calculate_ipft(trade_value, exchange)

        total = brokerage + stt + exchange_charges + stamp_duty + sebi_charges + gst + ipft
        total_pct = total / trade_value if trade_value > 0 else 0.0

        return {
            "brokerage": round(brokerage, 4),
            "stt": round(stt, 4),
            "exchange_charges": round(exchange_charges, 4),
            "stamp_duty": round(stamp_duty, 4),
            "sebi_charges": round(sebi_charges, 4),
            "gst": round(gst, 4),
            "ipft": round(ipft, 4),
            "total": round(total, 4),
            "total_pct": round(total_pct, 6),  # raw fraction (0.002 = 0.2%)
        }

    def calculate_round_trip_cost(
        self,
        entry_value: float,
        exit_value: Optional[float] = None,
        exchange: str = "NSE",
    ) -> dict:
        """
        Calculate round-trip (entry + exit) transaction costs.

        Parameters
        ----------
        entry_value : float
            Value of the buy/entry leg in ₹.
        exit_value : float, optional
            Value of the sell/exit leg in ₹. Defaults to entry_value.
        exchange : str
            'NSE' or 'BSE'.

        Returns
        -------
        dict with entry_costs, exit_costs, total_costs, total_pct,
              breakeven_return_pct
        """
        if exit_value is None:
            exit_value = entry_value

        entry_costs = self.calculate_total_cost(entry_value, is_buy=True, exchange=exchange)
        exit_costs = self.calculate_total_cost(exit_value, is_buy=False, exchange=exchange)

        total_cost = entry_costs["total"] + exit_costs["total"]
        avg_value = (entry_value + exit_value) / 2
        total_pct = total_cost / avg_value if avg_value > 0 else 0.0

        # Breakeven return: the gross return needed to cover all friction costs
        breakeven_return_pct = (total_cost / entry_value) * 100 if entry_value > 0 else 0.0

        total_costs_breakdown = {
            k: round(entry_costs[k] + exit_costs[k], 4)
            for k in ("brokerage", "stt", "exchange_charges", "stamp_duty",
                      "sebi_charges", "gst", "ipft")
        }
        total_costs_breakdown["total"] = round(total_cost, 4)
        total_costs_breakdown["total_pct"] = round(total_pct, 6)

        return {
            "entry_costs": entry_costs,
            "exit_costs": exit_costs,
            "total_costs": total_costs_breakdown,
            "total_pct": round(total_pct, 6),           # raw fraction
            "breakeven_return_pct": round(breakeven_return_pct / 100, 6),  # raw fraction
        }

    def adjust_returns_for_friction(
        self,
        returns_series: pd.Series,
        avg_position_value: float,
        holding_period_days: int = 5,
    ) -> pd.Series:
        """
        Adjust a returns series for transaction costs and slippage.

        Assumption: one round trip per holding period of `holding_period_days`.
        Annual friction cost is scaled proportionally.

        Parameters
        ----------
        returns_series : pd.Series
            Daily or periodic gross returns (as decimals, e.g., 0.01 = 1%).
        avg_position_value : float
            Average position size in ₹ used to compute brokerage impact.
        holding_period_days : int
            Average number of trading days a position is held.

        Returns
        -------
        pd.Series of net returns after friction costs.
        """
        rt = self.calculate_round_trip_cost(avg_position_value)
        # Cost per period as fraction of position value
        round_trip_cost_pct = rt["breakeven_return_pct"] / 100.0

        # Distribute the round-trip cost evenly across holding days
        daily_friction = round_trip_cost_pct / max(holding_period_days, 1)

        adjusted = returns_series - daily_friction
        return adjusted


# ---------------------------------------------------------------------------
# 3. SlippageModel
# ---------------------------------------------------------------------------

class SlippageModel:
    """
    Market impact and slippage model for Indian equity markets.

    Uses a square-root market impact model (Almgren-Chriss style)
    suitable for liquid NSE/BSE mid- and large-cap stocks.
    """

    def __init__(
        self,
        base_slippage_pct: float = 0.002,
        adv_impact_coeff: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        base_slippage_pct : float
            Base slippage (bid-ask + market impact floor), default 0.2%.
        adv_impact_coeff : float
            Coefficient for the square-root ADV impact term.
        """
        self.base_slippage_pct = base_slippage_pct
        self.adv_impact_coeff = adv_impact_coeff

    def estimate_slippage(
        self,
        order_value: float,
        adv_value: float,
        is_buy: bool = True,
    ) -> float:
        """
        Estimate total slippage as a fraction of order value.

        Formula
        -------
        slippage = base_slippage_pct + adv_impact_coeff * sqrt(order_value / adv_value)

        A buy order is assumed to move the price up (positive slippage cost).
        A sell order is assumed to move the price down (also a cost).

        Parameters
        ----------
        order_value : float
            Value of the order in ₹.
        adv_value : float
            Average daily value traded (ADV) in ₹.
        is_buy : bool
            Direction of order (both directions incur a cost).

        Returns
        -------
        float : slippage as a positive fraction (e.g., 0.003 = 0.3%)
        """
        if adv_value <= 0:
            return self.base_slippage_pct

        participation_ratio = order_value / adv_value
        impact = self.adv_impact_coeff * math.sqrt(participation_ratio)
        return self.base_slippage_pct + impact

    def calculate_impact_cost(
        self,
        shares: float,
        avg_daily_volume: float,
        price: float,
        is_buy: bool = True,
    ) -> float:
        """
        Calculate total impact cost in ₹.

        Parameters
        ----------
        shares : float
            Number of shares to trade.
        avg_daily_volume : float
            Average daily traded volume in shares.
        price : float
            Current market price per share in ₹.
        is_buy : bool
            Direction of order.

        Returns
        -------
        float : impact cost in ₹
        """
        order_value = shares * price
        adv_value = avg_daily_volume * price
        slippage_fraction = self.estimate_slippage(order_value, adv_value, is_buy)
        return order_value * slippage_fraction

    def estimate_bid_ask_spread(
        self,
        high: float,
        low: float,
        close: float,
    ) -> float:
        """
        Estimate bid-ask spread as a percentage of closing price.

        Uses the Corwin-Schultz (2012) high-low spread estimator simplified:
            spread_pct ≈ (high - low) / close * 0.5

        This is a rough proxy; actual bid-ask spreads require order book data.

        Parameters
        ----------
        high, low, close : float
            OHLC data for the session.

        Returns
        -------
        float : estimated spread as fraction of close price (e.g., 0.002 = 0.2%)
        """
        if close <= 0:
            return 0.0
        return ((high - low) / close) * 0.5


# ---------------------------------------------------------------------------
# 4. DynamicCostManager  (Gap 20)
# ---------------------------------------------------------------------------

class DynamicCostManager:
    """
    Dynamic cost manager that persists transaction cost configuration
    to a JSON file and tracks rate change history.

    Designed to be updated after Finance Act announcements (typically
    Union Budget in February, effective April 1).
    """

    def __init__(self, config_file: str = "production/cost_config.json") -> None:
        self.config_file = config_file
        self._history: List[dict] = []
        self.config = TransactionCostConfig()
        self.load_config()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_config(self) -> None:
        """Persist current config and rate-change history to JSON."""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        payload = {
            "config": self.config.to_dict(),
            "rate_history": self._history,
        }
        with open(self.config_file, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

    def load_config(self) -> None:
        """Load config from JSON file if it exists; otherwise use defaults."""
        if not os.path.exists(self.config_file):
            return
        try:
            with open(self.config_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self.config = TransactionCostConfig.from_dict(payload.get("config", {}))
            self._history = payload.get("rate_history", [])
        except (json.JSONDecodeError, KeyError) as exc:
            # Log warning; fall back to defaults
            print(f"[DynamicCostManager] Warning: could not load config ({exc}). Using defaults.")

    # ------------------------------------------------------------------
    # Rate management
    # ------------------------------------------------------------------

    def update_stt_rate(
        self,
        instrument_type: str,
        rate: float,
        effective_date: str,
        source: str = "Manual",
    ) -> None:
        """
        Update STT for a specific instrument type.

        Parameters
        ----------
        instrument_type : str
            One of: 'delivery_buy', 'delivery_sell', 'intraday_sell',
                    'futures_sell', 'options_sell_premium'
        rate : float
            New rate as decimal (e.g., 0.001 for 0.1%).
        effective_date : str
            ISO date string (e.g., '2026-04-01').
        source : str
            Source reference (e.g., 'Finance Act 2025', 'SEBI Circular').
        """
        field_map = {
            "delivery_buy": "stt_delivery_buy",
            "delivery_sell": "stt_delivery_sell",
            "intraday_sell": "stt_intraday_sell",
            "futures_sell": "stt_futures_sell",
            "options_sell_premium": "stt_options_sell_premium",
        }
        if instrument_type not in field_map:
            raise ValueError(
                f"instrument_type must be one of: {list(field_map.keys())}"
            )

        attr = field_map[instrument_type]
        old_rate = getattr(self.config, attr)
        setattr(self.config, attr, rate)

        self._history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "effective_date": effective_date,
            "field": attr,
            "instrument_type": instrument_type,
            "old_rate": old_rate,
            "new_rate": rate,
            "source": source,
        })

    def get_current_rates(self) -> dict:
        """Return all current cost rates as a flat dictionary."""
        return self.config.to_dict()

    def apply_finance_act_update(
        self,
        rates_dict: dict,
        effective_date: str,
    ) -> None:
        """
        Bulk update rates following a Finance Act/Budget announcement.

        Parameters
        ----------
        rates_dict : dict
            Dict of {field_name: new_rate} matching TransactionCostConfig fields.
        effective_date : str
            ISO date string of when rates take effect.
        """
        for field_name, new_rate in rates_dict.items():
            if not hasattr(self.config, field_name):
                print(f"[DynamicCostManager] Unknown field '{field_name}', skipping.")
                continue
            old_rate = getattr(self.config, field_name)
            setattr(self.config, field_name, new_rate)
            self._history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "effective_date": effective_date,
                "field": field_name,
                "old_rate": old_rate,
                "new_rate": new_rate,
                "source": "Finance Act bulk update",
            })
        # Update version to reflect the effective date
        self.config.version = effective_date
        self.save_config()

    def get_rate_history(self) -> List[dict]:
        """Return the full list of rate changes with dates and sources."""
        return list(self._history)


# ---------------------------------------------------------------------------
# 5. Module-level convenience function
# ---------------------------------------------------------------------------

def estimate_net_return(
    gross_return: float,
    position_value: float,
    instrument_type: str = "equity_delivery",
    exchange: str = "NSE",
) -> float:
    """
    Estimate net return after all transaction costs for a single round trip.

    Parameters
    ----------
    gross_return : float
        Gross return as a decimal (e.g., 0.05 = 5%).
    position_value : float
        Position size in ₹.
    instrument_type : str
        One of the supported instrument types.
    exchange : str
        'NSE' or 'BSE'.

    Returns
    -------
    float : Net return as decimal. Can be negative if costs exceed gross return.
    """
    model = FrictionModel(instrument_type=instrument_type)
    entry_value = position_value
    exit_value = position_value * (1 + gross_return)

    rt = model.calculate_round_trip_cost(entry_value, exit_value, exchange)
    total_cost_fraction = rt["total_costs"]["total"] / position_value

    return gross_return - total_cost_fraction


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io

    # Use UTF-8 on the output stream so the rupee symbol prints correctly
    # on terminals that support it; fall back gracefully on cp1252 Windows consoles.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    INR = "Rs."  # safe ASCII fallback used in print statements below

    print("=" * 65)
    print(f"FRICTION MODEL DEMO -- Round-trip costs for {INR}1,00,000 trade")
    print(f"Config version: {TransactionCostConfig().version}")
    print("=" * 65)

    TRADE_VALUE = 100_000  # Rs. 1 lakh

    for itype in FrictionModel.VALID_INSTRUMENT_TYPES:
        model = FrictionModel(instrument_type=itype)
        rt = model.calculate_round_trip_cost(TRADE_VALUE, exchange="NSE")

        print(f"\nInstrument: {itype.upper()}")
        print("  Entry (BUY) costs:")
        for k, v in rt["entry_costs"].items():
            if k == "total_pct":
                print(f"    {k:22s}: {v:.4f}%")
            else:
                print(f"    {k:22s}: {INR}{v:,.4f}")
        print("  Exit (SELL) costs:")
        for k, v in rt["exit_costs"].items():
            if k == "total_pct":
                print(f"    {k:22s}: {v:.4f}%")
            else:
                print(f"    {k:22s}: {INR}{v:,.4f}")
        print("  --- Round-trip totals ---")
        tc = rt["total_costs"]
        for k, v in tc.items():
            if k == "total_pct":
                print(f"    {k:22s}: {v:.4f}%")
            else:
                print(f"    {k:22s}: {INR}{v:,.4f}")
        print(f"  Breakeven gross return needed: {rt['breakeven_return_pct']:.4f}%")

    print("\n" + "=" * 65)
    print("SLIPPAGE MODEL DEMO")
    print("=" * 65)
    slip = SlippageModel()
    order_val = 500_000   # Rs. 5 lakh order
    adv_val = 50_000_000  # Rs. 5 crore ADV
    slippage_pct = slip.estimate_slippage(order_val, adv_val)
    print(f"Order {INR}{order_val:,.0f} | ADV {INR}{adv_val:,.0f}")
    print(f"Estimated slippage: {slippage_pct*100:.4f}%")
    spread = slip.estimate_bid_ask_spread(high=1050, low=1030, close=1045)
    print(f"Estimated bid-ask spread (H=1050, L=1030, C=1045): {spread*100:.4f}%")

    print("\n" + "=" * 65)
    print("NET RETURN ESTIMATION")
    print("=" * 65)
    gross = 0.03   # 3% gross return
    net = estimate_net_return(gross, TRADE_VALUE, "equity_delivery", "NSE")
    print(f"Gross return:  {gross*100:.2f}%")
    print(f"Net return:    {net*100:.4f}%")
    print(f"Friction drag: {(gross - net)*100:.4f}%")
