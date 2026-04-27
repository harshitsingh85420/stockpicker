"""
production/capital_buckets.py -- P23: Capital bucket simulation.

Splits total trading capital across three risk buckets and tracks their
equity curves independently.  Each bucket has its own allocation fraction,
max-drawdown limit, and rebalancing rule.

Buckets
-------
  AGGRESSIVE : High-confidence picks (prob >= 0.75), larger position sizes.
               Target: 40% of capital.  Max drawdown: 15%.
  MODERATE   : Standard picks (prob in [0.62, 0.75)), normal sizing.
               Target: 40% of capital.  Max drawdown: 10%.
  DEFENSIVE  : Cash / liquid ETF / very-high-confidence only.
               Target: 20% of capital.  Max drawdown: 5%.

Rebalancing: if any bucket drifts > drift_threshold from its target weight,
a rebalancing event is logged and the next allocation adjusts capital flows.

Usage
-----
from production.capital_buckets import CapitalBucketSimulator

sim = CapitalBucketSimulator(total_capital=1_000_000)
sim.allocate_pick(sc_code='500325', probability=0.78, position_value=50_000)
sim.record_return(sc_code='500325', gross_return=0.04)
report = sim.get_report()
sim.print_report(report)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bucket definitions
# ---------------------------------------------------------------------------

BUCKET_DEFS: Dict[str, Dict[str, Any]] = {
    "AGGRESSIVE": {
        "target_weight":   0.40,
        "prob_min":        0.75,
        "prob_max":        1.00,
        "max_drawdown":   -0.15,
        "size_multiplier": 1.20,
    },
    "MODERATE": {
        "target_weight":   0.40,
        "prob_min":        0.62,
        "prob_max":        0.75,
        "max_drawdown":   -0.10,
        "size_multiplier": 1.00,
    },
    "DEFENSIVE": {
        "target_weight":   0.20,
        "prob_min":        0.00,
        "prob_max":        0.62,
        "max_drawdown":   -0.05,
        "size_multiplier": 0.60,
    },
}


@dataclass
class Bucket:
    name:           str
    target_weight:  float
    max_drawdown:   float
    size_multiplier: float
    capital:        float = 0.0          # current capital in bucket
    peak_capital:   float = 0.0          # all-time high for this bucket
    realized_pnl:   float = 0.0          # cumulative closed P&L
    n_trades:       int = 0
    n_wins:         int = 0
    open_positions: Dict[str, float] = field(default_factory=dict)  # sc_code -> cost

    @property
    def drawdown(self) -> float:
        if self.peak_capital <= 0:
            return 0.0
        return (self.capital - self.peak_capital) / self.peak_capital

    @property
    def is_breached(self) -> bool:
        return self.drawdown <= self.max_drawdown

    @property
    def deployed(self) -> float:
        return sum(self.open_positions.values())

    @property
    def available(self) -> float:
        return max(0.0, self.capital - self.deployed)


class CapitalBucketSimulator:
    """
    P23 -- Simulate capital allocation across AGGRESSIVE / MODERATE / DEFENSIVE buckets.

    Parameters
    ----------
    total_capital     : Total initial trading capital (INR).
    drift_threshold   : Rebalance if bucket weight drifts by more than this from target.
    """

    def __init__(
        self,
        total_capital:   float = 1_000_000,
        drift_threshold: float = 0.10,
    ) -> None:
        self.total_capital   = float(total_capital)
        self.drift_threshold = drift_threshold
        self._rebalance_log: List[Dict[str, Any]] = []

        # Initialise buckets
        self.buckets: Dict[str, Bucket] = {}
        for name, cfg in BUCKET_DEFS.items():
            alloc = self.total_capital * cfg["target_weight"]
            b = Bucket(
                name            = name,
                target_weight   = cfg["target_weight"],
                max_drawdown    = cfg["max_drawdown"],
                size_multiplier = cfg["size_multiplier"],
                capital         = alloc,
                peak_capital    = alloc,
            )
            self.buckets[name] = b

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def classify_pick(self, probability: float) -> str:
        """Return the bucket name for a given model probability."""
        for name, cfg in BUCKET_DEFS.items():
            if cfg["prob_min"] <= probability < cfg["prob_max"]:
                return name
        return "DEFENSIVE"   # fallback

    # ------------------------------------------------------------------
    # Capital flows
    # ------------------------------------------------------------------

    def allocate_pick(
        self,
        sc_code:        str,
        probability:    float,
        position_value: float,
    ) -> Dict[str, Any]:
        """
        Allocate capital for a new pick to the appropriate bucket.

        Parameters
        ----------
        sc_code         : BSE scrip code.
        probability     : Model probability for this pick.
        position_value  : Requested position size in INR.

        Returns
        -------
        dict with keys: bucket, allocated_value, size_multiplier, status.
        """
        bucket_name = self.classify_pick(probability)
        bucket = self.buckets[bucket_name]

        if bucket.is_breached:
            logger.warning(
                "P23: Bucket %s breached (drawdown=%.2f%%) -- rejecting %s",
                bucket_name, bucket.drawdown * 100, sc_code,
            )
            return {
                "bucket": bucket_name, "allocated_value": 0.0,
                "size_multiplier": 0.0, "status": "REJECTED_DRAWDOWN",
            }

        adjusted = position_value * bucket.size_multiplier
        allocated = min(adjusted, bucket.available)

        if allocated <= 0:
            return {
                "bucket": bucket_name, "allocated_value": 0.0,
                "size_multiplier": bucket.size_multiplier,
                "status": "REJECTED_INSUFFICIENT_CAPITAL",
            }

        bucket.open_positions[sc_code] = allocated
        bucket.n_trades += 1
        logger.info(
            "P23: %s -> bucket=%s  allocated=%.0f  avail=%.0f",
            sc_code, bucket_name, allocated, bucket.available,
        )
        return {
            "bucket":          bucket_name,
            "allocated_value": round(allocated, 2),
            "size_multiplier": bucket.size_multiplier,
            "status":          "ALLOCATED",
        }

    def record_return(
        self,
        sc_code:      str,
        gross_return: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Close a position and record its P&L in the owning bucket.

        Parameters
        ----------
        sc_code      : BSE scrip code of the closing trade.
        gross_return : Gross fractional return (e.g. 0.04 = +4%).

        Returns
        -------
        dict with keys: bucket, pnl, new_capital, drawdown.  None if not found.
        """
        for name, bucket in self.buckets.items():
            if sc_code in bucket.open_positions:
                cost = bucket.open_positions.pop(sc_code)
                pnl  = cost * gross_return
                bucket.capital     += pnl
                bucket.realized_pnl += pnl
                bucket.peak_capital = max(bucket.peak_capital, bucket.capital)
                if pnl > 0:
                    bucket.n_wins += 1

                # Check rebalance need
                self._check_rebalance()

                logger.info(
                    "P23: %s closed in %s | pnl=%.0f (%.2f%%) | bucket_cap=%.0f",
                    sc_code, name, pnl, gross_return * 100, bucket.capital,
                )
                return {
                    "bucket":      name,
                    "pnl":         round(pnl, 2),
                    "new_capital": round(bucket.capital, 2),
                    "drawdown":    round(bucket.drawdown, 6),
                }
        logger.warning("P23: sc_code %s not found in any bucket.", sc_code)
        return None

    # ------------------------------------------------------------------
    # Rebalancing
    # ------------------------------------------------------------------

    def _check_rebalance(self) -> None:
        """Log a rebalance event if any bucket drifts beyond threshold."""
        total = sum(b.capital for b in self.buckets.values())
        if total <= 0:
            return
        for name, bucket in self.buckets.items():
            actual_weight = bucket.capital / total
            drift = actual_weight - bucket.target_weight
            if abs(drift) > self.drift_threshold:
                event = {
                    "bucket":         name,
                    "actual_weight":  round(actual_weight, 4),
                    "target_weight":  bucket.target_weight,
                    "drift":          round(drift, 4),
                }
                self._rebalance_log.append(event)
                logger.warning(
                    "P23 REBALANCE: bucket=%s drift=%.1f%% (actual=%.1f%% vs target=%.1f%%)",
                    name, drift * 100, actual_weight * 100, bucket.target_weight * 100,
                )

    def rebalance_now(self) -> Dict[str, Any]:
        """
        Force an immediate rebalance: move capital between buckets to restore
        target weights.  Only affects free (undeployed) capital.

        Returns a dict describing flows.
        """
        total = sum(b.capital for b in self.buckets.values())
        if total <= 0:
            return {"flows": [], "total": 0.0}

        flows = []
        for name, bucket in self.buckets.items():
            target_cap  = total * bucket.target_weight
            delta       = target_cap - bucket.capital
            bucket.capital += delta
            bucket.peak_capital = max(bucket.peak_capital, bucket.capital)
            flows.append({"bucket": name, "delta": round(delta, 2), "new_capital": round(bucket.capital, 2)})
            logger.info("P23 Rebalance: %s delta=%.0f -> %.0f", name, delta, bucket.capital)

        return {"flows": flows, "total": round(total, 2)}

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> Dict[str, Any]:
        total = sum(b.capital for b in self.buckets.values())
        bucket_reports = {}
        for name, b in self.buckets.items():
            w = b.capital / total if total > 0 else 0.0
            bucket_reports[name] = {
                "capital":        round(b.capital, 2),
                "weight":         round(w, 4),
                "target_weight":  b.target_weight,
                "drawdown":       round(b.drawdown, 4),
                "is_breached":    b.is_breached,
                "realized_pnl":   round(b.realized_pnl, 2),
                "n_trades":       b.n_trades,
                "win_rate":       round(b.n_wins / b.n_trades, 4) if b.n_trades else None,
                "deployed":       round(b.deployed, 2),
                "available":      round(b.available, 2),
            }
        return {
            "total_capital":    round(total, 2),
            "initial_capital":  self.total_capital,
            "total_pnl":        round(total - self.total_capital, 2),
            "total_return_pct": round((total / self.total_capital - 1) * 100, 4) if self.total_capital else 0.0,
            "buckets":          bucket_reports,
            "n_rebalance_events": len(self._rebalance_log),
        }

    def print_report(self, report: Optional[Dict] = None) -> None:
        if report is None:
            report = self.get_report()
        print("=" * 65)
        print(f"  P23 CAPITAL BUCKET SIMULATION")
        print(f"  Total: {report['total_capital']:,.0f}  PnL: {report['total_pnl']:+,.0f}  "
              f"({report['total_return_pct']:+.2f}%)")
        print("=" * 65)
        print(f"  {'Bucket':<12} {'Capital':>10} {'Weight':>7} {'Target':>7} "
              f"{'DD':>7} {'PnL':>10} {'WR':>6}")
        print("-" * 65)
        for name, b in report["buckets"].items():
            wr = f"{b['win_rate']:.1%}" if b['win_rate'] is not None else "  N/A"
            breached = " !!" if b['is_breached'] else ""
            print(
                f"  {name:<12} {b['capital']:>10,.0f} {b['weight']:>6.1%} "
                f"{b['target_weight']:>6.1%} {b['drawdown']:>6.1%} "
                f"{b['realized_pnl']:>+10,.0f} {wr:>6}{breached}"
            )
        print("=" * 65)
        print(f"  Rebalance events: {report['n_rebalance_events']}")
