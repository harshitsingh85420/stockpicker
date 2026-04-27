"""
risk_analytics.py
=================
P49 — Portfolio-level risk monitoring: VaR, correlation gate, stress tests.

Usage:
    from production.risk_analytics import PortfolioRiskMonitor
    risk = PortfolioRiskMonitor()
    var_report   = risk.compute_var(picks_df)
    ok           = risk.check_correlation_gate("532540", picks_df)
    stress       = risk.run_stress_test(picks_df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent
_CACHE_DIR = _ROOT / "stock_picker_data" / "cache" / "bse"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_return_history(sc_codes: list, lookback: int = 252) -> pd.DataFrame:
    """
    Load cached daily close prices for given symbols and return a wide
    DataFrame of daily returns (rows=dates, cols=SC_CODE).
    Returns empty DataFrame if cache unavailable.
    """
    frames = []
    for pkl in sorted(_CACHE_DIR.glob("bhav_bse_*.pkl"))[-lookback:]:
        try:
            df = pd.read_pickle(pkl)
            if "SC_CODE" not in df.columns or "Close" not in df.columns:
                continue
            sub = df[df["SC_CODE"].isin(sc_codes)][["SC_CODE", "DATE", "Close"]]
            frames.append(sub)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return pd.DataFrame()

    pivot = (combined
             .pivot_table(index="DATE", columns="SC_CODE", values="Close", aggfunc="last")
             .sort_index())
    returns = pivot.pct_change().dropna(how="all")
    return returns


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class PortfolioRiskMonitor:
    """
    P49: Three risk methods that gate or size portfolio picks.

    Parameters
    ----------
    cache_dir : optional override for BSE cache directory
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR

    # ------------------------------------------------------------------
    def compute_var(
        self,
        positions_df: pd.DataFrame,
        confidence: float = 0.95,
        lookback: int = 252,
    ) -> Dict:
        """
        Historical Value-at-Risk for the current position set.

        Parameters
        ----------
        positions_df : DataFrame with columns SC_CODE, Position_Value
        confidence   : VaR confidence level (default 0.95)
        lookback     : trading days of return history to use

        Returns dict with:
            var_pct     — portfolio VaR as fraction (e.g. 0.032 = 3.2%)
            var_inr     — absolute INR loss at VaR threshold
            cvar_pct    — Conditional VaR (expected shortfall)
            cvar_inr    — absolute CVaR
            n_stocks    — number of stocks in portfolio
            method      — 'historical' | 'proxy'
        """
        if positions_df is None or positions_df.empty:
            return {"var_pct": 0.0, "var_inr": 0.0, "cvar_pct": 0.0,
                    "cvar_inr": 0.0, "n_stocks": 0, "method": "none"}

        codes = positions_df["SC_CODE"].tolist()
        total_value = positions_df["Position_Value"].sum() if "Position_Value" in positions_df.columns else 1.0

        returns = _load_return_history(codes, lookback)

        if returns.empty or len(returns) < 20:
            # Proxy: assume 3% daily VaR if no history
            logger.warning("P49: Insufficient return history — using proxy VaR.")
            var_pct = 0.03
            cvar_pct = 0.045
            return {
                "var_pct":  var_pct,
                "var_inr":  var_pct * total_value,
                "cvar_pct": cvar_pct,
                "cvar_inr": cvar_pct * total_value,
                "n_stocks": len(codes),
                "method":   "proxy",
            }

        # Weights proportional to position values
        wt_series = positions_df.set_index("SC_CODE")["Position_Value"] if "Position_Value" in positions_df.columns else None
        common_codes = [c for c in codes if c in returns.columns]
        ret_sub = returns[common_codes].fillna(0)

        if wt_series is not None:
            wts = wt_series.reindex(common_codes).fillna(0)
            wts = wts / wts.sum() if wts.sum() > 0 else pd.Series(1.0 / len(common_codes), index=common_codes)
        else:
            wts = pd.Series(1.0 / len(common_codes), index=common_codes)

        port_returns = ret_sub.dot(wts)
        alpha = 1.0 - confidence
        var_pct = float(-np.percentile(port_returns, alpha * 100))
        cvar_pct = float(-port_returns[port_returns <= -var_pct].mean()) if len(port_returns[port_returns <= -var_pct]) > 0 else var_pct * 1.5

        result = {
            "var_pct":  round(var_pct, 5),
            "var_inr":  round(var_pct * total_value, 2),
            "cvar_pct": round(cvar_pct, 5),
            "cvar_inr": round(cvar_pct * total_value, 2),
            "n_stocks": len(common_codes),
            "method":   "historical",
        }
        logger.info(
            "P49 VaR(%.0f%%): %.2f%%  CVaR: %.2f%%  on Rs.%.0f portfolio",
            confidence * 100, var_pct * 100, cvar_pct * 100, total_value,
        )
        return result

    # ------------------------------------------------------------------
    def check_correlation_gate(
        self,
        new_pick: str,
        existing_positions: pd.DataFrame,
        max_correlation: float = 0.70,
        lookback: int = 21,
    ) -> bool:
        """
        Return True if new_pick's 21-day return correlation with every
        existing position is below max_correlation (safe to add).
        Return False if it is too correlated with any existing holding.

        Parameters
        ----------
        new_pick          : SC_CODE of the candidate stock
        existing_positions: DataFrame with SC_CODE column
        max_correlation   : reject if corr > this threshold (default 0.70)
        lookback          : rolling window in trading days
        """
        if existing_positions is None or existing_positions.empty:
            return True

        existing_codes = existing_positions["SC_CODE"].tolist()
        all_codes = [new_pick] + existing_codes
        returns = _load_return_history(all_codes, lookback + 5)

        if returns.empty or new_pick not in returns.columns:
            logger.debug("P49: No return data for %s — gate passes by default.", new_pick)
            return True

        ret_new = returns[new_pick].dropna()
        for code in existing_codes:
            if code not in returns.columns:
                continue
            ret_existing = returns[code].dropna()
            common_idx = ret_new.index.intersection(ret_existing.index)
            if len(common_idx) < 5:
                continue
            corr = float(ret_new.loc[common_idx].corr(ret_existing.loc[common_idx]))
            if corr > max_correlation:
                logger.info(
                    "P49 Correlation gate BLOCK: %s vs %s corr=%.2f > %.2f",
                    new_pick, code, corr, max_correlation,
                )
                return False

        return True

    # ------------------------------------------------------------------
    def run_stress_test(self, positions_df: pd.DataFrame) -> Dict:
        """
        Apply three pre-defined market shock scenarios to the current portfolio.

        Scenarios:
            1. Nifty -10%  : broad market correction (beta-driven loss estimate)
            2. March 2020  : -35% crash (historical worst-case for Indian market)
            3. Sector shock: -20% to the largest sector in the portfolio

        Returns dict with scenario names as keys and estimated portfolio loss (INR).
        """
        if positions_df is None or positions_df.empty:
            return {}

        total_value = (positions_df["Position_Value"].sum()
                       if "Position_Value" in positions_df.columns else 0.0)

        scenarios = {
            "nifty_minus_10pct": {
                "description": "Broad market -10% (beta=1 assumption)",
                "shock_pct":   -0.10,
            },
            "march_2020_crash": {
                "description": "India March-2020 style crash -35%",
                "shock_pct":   -0.35,
            },
            "sector_shock_minus_20pct": {
                "description": "Largest-sector -20%, others -5%",
                "shock_pct":   None,   # handled separately below
            },
        }

        results: Dict = {}

        # --- Scenario 1 & 2: flat shock to whole portfolio ---
        for key, info in scenarios.items():
            if info["shock_pct"] is not None:
                loss = total_value * abs(info["shock_pct"])
                results[key] = {
                    "description":   info["description"],
                    "portfolio_loss_inr":  round(loss, 2),
                    "portfolio_loss_pct":  abs(info["shock_pct"]),
                    "residual_value_inr":  round(total_value - loss, 2),
                }

        # --- Scenario 3: sector shock ---
        if "Sector" in positions_df.columns and "Position_Value" in positions_df.columns:
            sector_totals = positions_df.groupby("Sector")["Position_Value"].sum()
            if not sector_totals.empty:
                top_sector = sector_totals.idxmax()
                top_val    = sector_totals.max()
                other_val  = total_value - top_val
                loss = top_val * 0.20 + other_val * 0.05
                results["sector_shock_minus_20pct"] = {
                    "description":         f"Sector {top_sector} -20%, others -5%",
                    "portfolio_loss_inr":  round(loss, 2),
                    "portfolio_loss_pct":  round(loss / total_value, 4) if total_value > 0 else 0,
                    "residual_value_inr":  round(total_value - loss, 2),
                    "top_sector":          top_sector,
                    "top_sector_value":    round(top_val, 2),
                }
        else:
            # No sector info — apply flat -20% to whole portfolio
            loss = total_value * 0.20
            results["sector_shock_minus_20pct"] = {
                "description":         "Sector shock -20% (no sector data, flat applied)",
                "portfolio_loss_inr":  round(loss, 2),
                "portfolio_loss_pct":  0.20,
                "residual_value_inr":  round(total_value - loss, 2),
            }

        logger.info(
            "P49 Stress test | "
            "nifty-10%%=Rs.%.0f | "
            "mar20-35%%=Rs.%.0f | "
            "sector-20%%=Rs.%.0f",
            results.get("nifty_minus_10pct", {}).get("portfolio_loss_inr", 0),
            results.get("march_2020_crash",  {}).get("portfolio_loss_inr", 0),
            results.get("sector_shock_minus_20pct", {}).get("portfolio_loss_inr", 0),
        )
        return results
