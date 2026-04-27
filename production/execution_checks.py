"""
production/execution_checks.py  --  P09: Execution reality checks.

Four pre-trade filters that remove signals that look profitable in a
close-to-close backtest but are likely unexecutable in practice:

  1. Volume constraint  -- Skip stocks where today's volume is < 2x the
                           estimated order size (i.e. order > 50% of ADV).
                           Prevents filling illiquid orders at all.

  2. Circuit filter     -- Skip stocks that hit the BSE upper or lower
                           circuit limit today (close return >= +19.5% or
                           <= -19.5%).  These cannot be traded at market.

  3. Gap avoidance      -- Skip stocks with an overnight gap > 4% (open vs
                           previous close).  Large gaps indicate event risk
                           or thin liquidity at open.

  4. Liquidity gate     -- Require that the position value (shares * price)
                           is <= 1% of the 20-day average daily value traded.
                           Prevents impactful trades in thin markets.

All four checks are consolidated in the single function:
    apply_execution_checks(picks_df, bhav_df, reference_date, ...)

which returns a filtered picks DataFrame and a rejection report dict.

The module is designed to be called in:
  - walk_forward_backtest._predict_block()  (backtest loop)
  - trade_orchestrator._l7_execution()      (live pipeline)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CRORE = 10_000_000


# ---------------------------------------------------------------------------
# Individual checks (each returns a set of SC_CODEs that FAIL the check)
# ---------------------------------------------------------------------------

def _failing_volume_constraint(
    picks_df: pd.DataFrame,
    bhav_df: pd.DataFrame,
    reference_date: Any,
    shares_col: str = "Shares",
    assumed_shares: int = 500,
    adv_lookback: int = 20,
    max_adv_fraction: float = 0.50,
) -> set:
    """
    Return SC_CODEs where the estimated order size exceeds max_adv_fraction
    of the 20-day average daily volume.

    Parameters
    ----------
    picks_df        : Picks DataFrame with SC_CODE and optionally Shares.
    bhav_df         : Full BhavCopy with SC_CODE, DATE, Volume.
    reference_date  : Snapshot date for computing ADV.
    shares_col      : Column in picks_df for order size (default 'Shares').
    assumed_shares  : Fallback order size if Shares column absent (default 500).
    adv_lookback    : Trading days for ADV computation (default 20).
    max_adv_fraction: Maximum order / ADV fraction allowed (default 0.50).
    """
    bhav_c = bhav_df.copy()
    bhav_c["DATE"] = pd.to_datetime(bhav_c["DATE"])
    ref = pd.Timestamp(reference_date)
    cutoff = ref - pd.offsets.BDay(adv_lookback)

    window = bhav_c[(bhav_c["DATE"] >= cutoff) & (bhav_c["DATE"] <= ref)]
    adv = window.groupby("SC_CODE")["Volume"].mean().fillna(0)

    failing = set()
    for _, row in picks_df.iterrows():
        sc = str(row["SC_CODE"])
        order_sz = (
            int(row[shares_col])
            if shares_col in picks_df.columns and pd.notna(row.get(shares_col))
            else assumed_shares
        )
        stock_adv = adv.get(sc, 0)
        if stock_adv <= 0 or order_sz / stock_adv > max_adv_fraction:
            failing.add(sc)
    return failing


def _failing_circuit_filter(
    picks_df: pd.DataFrame,
    bhav_df: pd.DataFrame,
    reference_date: Any,
    circuit_pct: float = 0.195,
) -> set:
    """
    Return SC_CODEs that hit BSE upper or lower circuit on reference_date.

    Circuit detection: |close_return| >= circuit_pct (default 19.5%).
    """
    bhav_c = bhav_df.copy()
    bhav_c["DATE"] = pd.to_datetime(bhav_c["DATE"])
    ref = pd.Timestamp(reference_date)

    # Need today and yesterday
    today_df = bhav_c[bhav_c["DATE"] == ref][["SC_CODE", "Close"]].copy()
    prev_df  = bhav_c[bhav_c["DATE"] < ref].sort_values("DATE")
    prev_close = (
        prev_df.groupby("SC_CODE")["Close"].last().rename("prev_close")
    )

    merged = today_df.merge(prev_close, on="SC_CODE", how="left")
    merged["daily_ret"] = (merged["Close"] - merged["prev_close"]) / (
        merged["prev_close"].replace(0, np.nan)
    )

    circuit_mask = merged["daily_ret"].abs() >= circuit_pct
    codes_in_picks = set(picks_df["SC_CODE"].astype(str))
    failing = set(
        merged[circuit_mask]["SC_CODE"].astype(str).tolist()
    ) & codes_in_picks
    return failing


def _failing_gap_avoidance(
    picks_df: pd.DataFrame,
    bhav_df: pd.DataFrame,
    reference_date: Any,
    max_gap_pct: float = 0.04,
) -> set:
    """
    Return SC_CODEs with an overnight gap > max_gap_pct on reference_date.

    Gap = |Open_today / Close_yesterday - 1|.
    """
    bhav_c = bhav_df.copy()
    bhav_c["DATE"] = pd.to_datetime(bhav_c["DATE"])
    ref = pd.Timestamp(reference_date)

    today_df = bhav_c[bhav_c["DATE"] == ref][["SC_CODE", "Open"]].copy()
    prev_df  = bhav_c[bhav_c["DATE"] < ref].sort_values("DATE")
    prev_close = (
        prev_df.groupby("SC_CODE")["Close"].last().rename("prev_close")
    )

    merged = today_df.merge(prev_close, on="SC_CODE", how="left")
    merged["gap_pct"] = ((merged["Open"] / merged["prev_close"]) - 1).abs()

    codes_in_picks = set(picks_df["SC_CODE"].astype(str))
    failing = set(
        merged[merged["gap_pct"] > max_gap_pct]["SC_CODE"].astype(str).tolist()
    ) & codes_in_picks
    return failing


def _failing_liquidity_gate(
    picks_df: pd.DataFrame,
    bhav_df: pd.DataFrame,
    reference_date: Any,
    shares_col: str = "Shares",
    assumed_shares: int = 500,
    adv_lookback: int = 20,
    max_impact_fraction: float = 0.01,
) -> set:
    """
    Return SC_CODEs where position_value > max_impact_fraction of the
    20-day average daily value traded.

    This is a tighter ADV check in value (Rs) terms rather than share volume.
    """
    bhav_c = bhav_df.copy()
    bhav_c["DATE"] = pd.to_datetime(bhav_c["DATE"])
    ref = pd.Timestamp(reference_date)
    cutoff = ref - pd.offsets.BDay(adv_lookback)

    window = bhav_c[(bhav_c["DATE"] >= cutoff) & (bhav_c["DATE"] <= ref)].copy()
    if "Value" not in window.columns:
        if "Volume" in window.columns:
            window["Value"] = window["Close"] * window["Volume"]
        else:
            return set()   # can't compute; skip check

    adv_value = window.groupby("SC_CODE")["Value"].mean().fillna(0)

    # Get latest close
    latest_close = (
        bhav_c[bhav_c["DATE"] <= ref]
        .sort_values("DATE")
        .groupby("SC_CODE")["Close"]
        .last()
    )

    failing = set()
    for _, row in picks_df.iterrows():
        sc = str(row["SC_CODE"])
        order_sz = (
            int(row[shares_col])
            if shares_col in picks_df.columns and pd.notna(row.get(shares_col))
            else assumed_shares
        )
        price = latest_close.get(sc, row.get("Close", 100))
        pos_val = order_sz * float(price)
        adv_val = adv_value.get(sc, 0)
        if adv_val <= 0 or pos_val / adv_val > max_impact_fraction:
            failing.add(sc)
    return failing


# ---------------------------------------------------------------------------
# Combined gate
# ---------------------------------------------------------------------------

def apply_execution_checks(
    picks_df: pd.DataFrame,
    bhav_df: pd.DataFrame,
    reference_date: Any,
    # Volume constraint
    max_adv_fraction: float = 0.50,
    adv_lookback: int = 20,
    assumed_shares: int = 500,
    # Circuit filter
    circuit_pct: float = 0.195,
    # Gap avoidance
    max_gap_pct: float = 0.04,
    # Liquidity gate
    max_impact_fraction: float = 0.01,
    # Which checks to run
    check_volume: bool = True,
    check_circuit: bool = True,
    check_gap: bool = True,
    check_liquidity: bool = True,
) -> tuple:
    """
    Apply all four execution reality checks to picks_df.

    Parameters
    ----------
    picks_df        : DataFrame with SC_CODE (and optionally Shares, Close).
    bhav_df         : Full BhavCopy with SC_CODE, DATE, Open, High, Low,
                      Close, Volume (and optionally Value).
    reference_date  : The signal generation date.
    max_adv_fraction: Max order / ADV allowed (default 0.50 = 50%).
    adv_lookback    : Lookback sessions for ADV (default 20).
    assumed_shares  : Default shares if Shares column absent (default 500).
    circuit_pct     : Circuit limit threshold (default 0.195 = 19.5%).
    max_gap_pct     : Max overnight gap allowed (default 0.04 = 4%).
    max_impact_fraction: Max position_value / ADV_value (default 0.01 = 1%).
    check_volume    : Run volume constraint check (default True).
    check_circuit   : Run circuit filter (default True).
    check_gap       : Run gap avoidance (default True).
    check_liquidity : Run liquidity gate (default True).

    Returns
    -------
    (filtered_df, report)
        filtered_df  -- picks that passed all checks
        report       -- dict with per-check rejection counts and codes
    """
    if picks_df.empty:
        return picks_df, {
            "n_before": 0, "n_after": 0,
            "volume_rejected": 0, "circuit_rejected": 0,
            "gap_rejected": 0, "liquidity_rejected": 0,
        }

    n_before = len(picks_df)
    rejected_union: set = set()
    report = {"n_before": n_before}

    # 1. Volume constraint
    if check_volume:
        try:
            rej = _failing_volume_constraint(
                picks_df, bhav_df, reference_date,
                assumed_shares=assumed_shares,
                adv_lookback=adv_lookback,
                max_adv_fraction=max_adv_fraction,
            )
            report["volume_rejected"] = len(rej)
            rejected_union |= rej
            if rej:
                logger.debug(
                    "P09 [%s] volume: %d picks rejected (order > %.0f%% ADV): %s",
                    reference_date, len(rej), max_adv_fraction * 100, sorted(rej)[:5],
                )
        except Exception as e:
            logger.debug("P09 volume check skipped: %s", e)
            report["volume_rejected"] = 0
    else:
        report["volume_rejected"] = 0

    # 2. Circuit filter
    if check_circuit:
        try:
            rej = _failing_circuit_filter(
                picks_df, bhav_df, reference_date, circuit_pct=circuit_pct,
            )
            report["circuit_rejected"] = len(rej)
            rejected_union |= rej
            if rej:
                logger.debug(
                    "P09 [%s] circuit: %d picks on circuit limit: %s",
                    reference_date, len(rej), sorted(rej)[:5],
                )
        except Exception as e:
            logger.debug("P09 circuit check skipped: %s", e)
            report["circuit_rejected"] = 0
    else:
        report["circuit_rejected"] = 0

    # 3. Gap avoidance
    if check_gap:
        try:
            rej = _failing_gap_avoidance(
                picks_df, bhav_df, reference_date, max_gap_pct=max_gap_pct,
            )
            report["gap_rejected"] = len(rej)
            rejected_union |= rej
            if rej:
                logger.debug(
                    "P09 [%s] gap: %d picks with gap > %.1f%%: %s",
                    reference_date, len(rej), max_gap_pct * 100, sorted(rej)[:5],
                )
        except Exception as e:
            logger.debug("P09 gap check skipped: %s", e)
            report["gap_rejected"] = 0
    else:
        report["gap_rejected"] = 0

    # 4. Liquidity gate
    if check_liquidity:
        try:
            rej = _failing_liquidity_gate(
                picks_df, bhav_df, reference_date,
                assumed_shares=assumed_shares,
                adv_lookback=adv_lookback,
                max_impact_fraction=max_impact_fraction,
            )
            report["liquidity_rejected"] = len(rej)
            rejected_union |= rej
            if rej:
                logger.debug(
                    "P09 [%s] liquidity: %d picks exceed market impact limit: %s",
                    reference_date, len(rej), sorted(rej)[:5],
                )
        except Exception as e:
            logger.debug("P09 liquidity check skipped: %s", e)
            report["liquidity_rejected"] = 0
    else:
        report["liquidity_rejected"] = 0

    # Apply rejections
    filtered = picks_df[~picks_df["SC_CODE"].astype(str).isin(rejected_union)].copy()
    report["n_after"]   = len(filtered)
    report["n_rejected"] = n_before - len(filtered)

    if report["n_rejected"] > 0:
        logger.debug(
            "P09 [%s]: %d -> %d picks after execution checks "
            "(vol=%d, circuit=%d, gap=%d, liq=%d)",
            reference_date,
            n_before, len(filtered),
            report["volume_rejected"],
            report["circuit_rejected"],
            report["gap_rejected"],
            report["liquidity_rejected"],
        )

    return filtered, report
