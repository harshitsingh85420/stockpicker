"""
universe_filter.py
==================
Gap 2 — Universe & Tradability Gate for BSE/NSE production trading systems.

Screens a BhavCopy DataFrame down to a universe of stocks that are actually
tradable given minimum liquidity, price, and value-traded thresholds.

Also computes per-stock Average Daily Volume (ADV) and the maximum number of
shares that can be traded without meaningful market impact (ADV cap).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Conversion constant
CRORE = 10_000_000  # 1 crore = 10 million rupees


# ===========================================================================
# Helper utilities
# ===========================================================================
def _to_datetime(date: Any) -> datetime:
    """Coerce *date* (str, datetime, Timestamp) to a Python datetime."""
    if isinstance(date, datetime):
        return date
    return pd.Timestamp(date).to_pydatetime()


def _ensure_value_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a ``Value`` column exists (Close * Volume) if absent.

    BSE BhavCopy sometimes uses ``TDQ_QTY`` instead of ``Volume`` and may
    or may not have a pre-computed ``Value`` column.
    """
    out = df.copy()
    if "Value" not in out.columns:
        vol_col = "Volume" if "Volume" in out.columns else "TDQ_QTY"
        if vol_col in out.columns:
            out["Value"] = out["Close"] * out[vol_col]
            logger.debug(
                "Derived Value column from Close * %s", vol_col
            )
        else:
            logger.warning(
                "Neither Volume nor TDQ_QTY found; Value will be NaN."
            )
            out["Value"] = np.nan
    if "Volume" not in out.columns and "TDQ_QTY" in out.columns:
        out["Volume"] = out["TDQ_QTY"]
    return out


def _lookback_window(
    df: pd.DataFrame,
    reference_date: Any,
    lookback: int,
) -> pd.DataFrame:
    """
    Return rows in *df* within the *lookback* business days ending on
    (and including) *reference_date*.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``DATE`` column (datetime-compatible).
    reference_date : date-like
    lookback : int
        Number of trading days to look back.

    Returns
    -------
    pd.DataFrame
        Subset of *df*.
    """
    ref = pd.Timestamp(reference_date)
    past = ref - pd.offsets.BDay(lookback)
    mask = (df["DATE"] >= past) & (df["DATE"] <= ref)
    return df[mask]


# ===========================================================================
# TradabilityGate
# ===========================================================================
class TradabilityGate:
    """
    Screen stocks from a BhavCopy DataFrame for minimum tradability.

    Parameters
    ----------
    min_value_crore : float
        Minimum 20-day average daily value traded in crores (default 2.0).
    min_price : float
        Minimum latest closing price in INR (default 20.0).
    min_avg_volume : int
        Minimum 20-day average daily volume in shares (default 10 000).
    lookback_days : int
        Number of trading days to use for rolling averages (default 20).
    max_adv_participation : float
        Maximum fraction of ADV that can be traded in one order (default 0.01,
        i.e. 1 % of ADV).
    """

    def __init__(
        self,
        min_value_crore: float = 2.0,
        min_price: float = 20.0,
        min_avg_volume: int = 10_000,
        lookback_days: int = 20,
        max_adv_participation: float = 0.01,
    ) -> None:
        self.min_value_crore = min_value_crore
        self.min_price = min_price
        self.min_avg_volume = min_avg_volume
        self.lookback_days = lookback_days
        self.max_adv_participation = max_adv_participation

        logger.info(
            "TradabilityGate initialised: min_price=%.2f, "
            "min_avg_vol=%d, min_value_cr=%.2f, lookback=%d, "
            "adv_participation=%.4f",
            min_price, min_avg_volume, min_value_crore,
            lookback_days, max_adv_participation,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def apply(
        self,
        bhav_df: pd.DataFrame,
        reference_date: Any,
    ) -> pd.DataFrame:
        """
        Apply all tradability filters and return only tradable stocks.

        Filters applied sequentially:
        1. Latest close >= ``min_price``
        2. 20-day avg volume >= ``min_avg_volume``
        3. 20-day avg value traded >= ``min_value_crore`` crores

        Parameters
        ----------
        bhav_df : pd.DataFrame
            Multi-stock BhavCopy with columns:
            SC_CODE, DATE, Open, High, Low, Close, Volume (or TDQ_QTY),
            and optionally Value.
        reference_date : date-like
            Snapshot date for computing latest price and rolling averages.

        Returns
        -------
        pd.DataFrame
            Subset of *bhav_df* (all dates) for stocks that pass all filters.
        """
        df = _ensure_value_col(bhav_df.copy())
        df["DATE"] = pd.to_datetime(df["DATE"])
        ref = pd.Timestamp(reference_date)

        # Step 1 – price
        price_pass = self.filter_by_price(df, ref)
        price_codes = set(price_pass["SC_CODE"].unique())

        # Step 2 – volume
        vol_pass = self.filter_by_volume(df, ref)
        vol_codes = set(vol_pass["SC_CODE"].unique())

        # Step 3 – value
        val_pass = self.filter_by_value_traded(df, ref)
        val_codes = set(val_pass["SC_CODE"].unique())

        tradable_codes = price_codes & vol_codes & val_codes

        result = df[df["SC_CODE"].isin(tradable_codes)].copy()
        logger.info(
            "apply(%s): %d -> %d tradable stocks (price=%d, vol=%d, value=%d)",
            reference_date,
            df["SC_CODE"].nunique(),
            len(tradable_codes),
            len(price_codes),
            len(vol_codes),
            len(val_codes),
        )
        return result

    # ------------------------------------------------------------------
    def filter_by_price(
        self,
        bhav_df: pd.DataFrame,
        reference_date: Any,
    ) -> pd.DataFrame:
        """
        Filter stocks where the most recent close on or before *reference_date*
        is >= ``min_price``.

        Parameters
        ----------
        bhav_df : pd.DataFrame
            Must have SC_CODE, DATE, Close.
        reference_date : date-like

        Returns
        -------
        pd.DataFrame
            Rows for qualifying stocks (all dates retained).
        """
        df = bhav_df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        ref = pd.Timestamp(reference_date)

        snap = df[df["DATE"] <= ref].copy()
        if snap.empty:
            logger.warning(
                "filter_by_price: no data on or before %s", reference_date
            )
            return df.iloc[0:0]

        latest = (
            snap.sort_values("DATE")
            .groupby("SC_CODE", sort=False)["Close"]
            .last()
        )
        qualifying = latest[latest >= self.min_price].index.tolist()

        before = df["SC_CODE"].nunique()
        result = df[df["SC_CODE"].isin(qualifying)]
        logger.debug(
            "filter_by_price(>= %.2f): %d -> %d stocks",
            self.min_price, before, len(qualifying),
        )
        return result

    # ------------------------------------------------------------------
    def filter_by_volume(
        self,
        bhav_df: pd.DataFrame,
        reference_date: Any,
    ) -> pd.DataFrame:
        """
        Filter stocks where the ``lookback_days``-day average daily volume is
        >= ``min_avg_volume`` shares.

        Parameters
        ----------
        bhav_df : pd.DataFrame
            Must have SC_CODE, DATE, Volume.
        reference_date : date-like

        Returns
        -------
        pd.DataFrame
            Rows for qualifying stocks (all dates retained).
        """
        df = bhav_df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"])

        window = _lookback_window(df, reference_date, self.lookback_days)
        if window.empty:
            logger.warning(
                "filter_by_volume: no data in lookback window for %s",
                reference_date,
            )
            return df.iloc[0:0]

        avg_vol = (
            window.groupby("SC_CODE")["Volume"]
            .mean()
            .fillna(0)
        )
        qualifying = avg_vol[avg_vol >= self.min_avg_volume].index.tolist()

        before = df["SC_CODE"].nunique()
        result = df[df["SC_CODE"].isin(qualifying)]
        logger.debug(
            "filter_by_volume(>= %d): %d -> %d stocks",
            self.min_avg_volume, before, len(qualifying),
        )
        return result

    # ------------------------------------------------------------------
    def filter_by_value_traded(
        self,
        bhav_df: pd.DataFrame,
        reference_date: Any,
    ) -> pd.DataFrame:
        """
        Filter stocks where the ``lookback_days``-day average daily value
        traded (in crores) is >= ``min_value_crore``.

        Value = Close * Volume when a pre-computed Value column is absent.

        Parameters
        ----------
        bhav_df : pd.DataFrame
            Must have SC_CODE, DATE, and Value (or Close + Volume).
        reference_date : date-like

        Returns
        -------
        pd.DataFrame
            Rows for qualifying stocks (all dates retained).
        """
        df = _ensure_value_col(bhav_df.copy())
        df["DATE"] = pd.to_datetime(df["DATE"])

        window = _lookback_window(df, reference_date, self.lookback_days)
        if window.empty:
            logger.warning(
                "filter_by_value_traded: no data in lookback window for %s",
                reference_date,
            )
            return df.iloc[0:0]

        avg_value = (
            window.groupby("SC_CODE")["Value"]
            .mean()
            .fillna(0)
        )
        # Convert rupees -> crores
        avg_value_crore = avg_value / CRORE
        qualifying = avg_value_crore[
            avg_value_crore >= self.min_value_crore
        ].index.tolist()

        before = df["SC_CODE"].nunique()
        result = df[df["SC_CODE"].isin(qualifying)]
        logger.debug(
            "filter_by_value_traded(>= %.2f Cr): %d -> %d stocks",
            self.min_value_crore, before, len(qualifying),
        )
        return result

    # ------------------------------------------------------------------
    def compute_adv(
        self,
        bhav_df: pd.DataFrame,
        sc_code: str,
        lookback: Optional[int] = None,
    ) -> float:
        """
        Compute the Average Daily Volume for a single stock.

        Parameters
        ----------
        bhav_df : pd.DataFrame
            Full multi-stock BhavCopy.
        sc_code : str
            Stock to compute ADV for.
        lookback : int, optional
            Overrides ``self.lookback_days`` if supplied.

        Returns
        -------
        float
            Average daily volume in shares.  Returns 0.0 if no data found.
        """
        lb = lookback if lookback is not None else self.lookback_days
        df = bhav_df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE")

        stock_df = df[df["SC_CODE"].astype(str) == str(sc_code)]
        if stock_df.empty:
            logger.warning("compute_adv: SC_CODE=%s not found", sc_code)
            return 0.0

        recent = stock_df.tail(lb)
        if "Volume" not in recent.columns:
            logger.warning(
                "compute_adv: SC_CODE=%s has no Volume column", sc_code
            )
            return 0.0

        adv = float(recent["Volume"].mean())
        logger.debug("compute_adv(SC_CODE=%s, lb=%d): %.0f", sc_code, lb, adv)
        return adv

    # ------------------------------------------------------------------
    def get_max_tradeable_shares(
        self,
        bhav_df: pd.DataFrame,
        sc_code: str,
        lookback: Optional[int] = None,
    ) -> float:
        """
        Return the maximum number of shares that can be traded in a single
        order without exceeding the ADV participation cap.

        max_shares = ADV * max_adv_participation

        Parameters
        ----------
        bhav_df : pd.DataFrame
        sc_code : str
        lookback : int, optional

        Returns
        -------
        float
            Maximum tradeable shares (floored to nearest whole share).
        """
        adv = self.compute_adv(bhav_df, sc_code, lookback)
        max_shares = adv * self.max_adv_participation
        logger.debug(
            "get_max_tradeable_shares(SC_CODE=%s): ADV=%.0f, "
            "participation=%.4f, max_shares=%.0f",
            sc_code, adv, self.max_adv_participation, max_shares,
        )
        return float(int(max_shares))

    # ------------------------------------------------------------------
    def get_filter_report(
        self,
        bhav_df: pd.DataFrame,
        reference_date: Any,
    ) -> Dict[str, int]:
        """
        Return a summary dict with stock counts at each filter stage.

        Parameters
        ----------
        bhav_df : pd.DataFrame
        reference_date : date-like

        Returns
        -------
        dict with keys:
            total_stocks, after_price_filter, after_volume_filter,
            after_value_filter, final_count
        """
        df = _ensure_value_col(bhav_df.copy())
        df["DATE"] = pd.to_datetime(df["DATE"])
        ref = pd.Timestamp(reference_date)

        total = df["SC_CODE"].nunique()

        price_pass = self.filter_by_price(df, ref)
        n_price = price_pass["SC_CODE"].nunique()

        vol_pass = self.filter_by_volume(price_pass, ref)
        n_vol = vol_pass["SC_CODE"].nunique()

        val_pass = self.filter_by_value_traded(vol_pass, ref)
        n_val = val_pass["SC_CODE"].nunique()

        report = {
            "total_stocks": total,
            "after_price_filter": n_price,
            "after_volume_filter": n_vol,
            "after_value_filter": n_val,
            "final_count": n_val,
        }
        logger.info("get_filter_report(%s): %s", reference_date, report)
        return report


# ===========================================================================
# Module-level convenience function
# ===========================================================================
def apply_tradability_gate(
    bhav_df: pd.DataFrame,
    reference_date: Any,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Convenience wrapper: instantiate ``TradabilityGate`` and apply it.

    Parameters
    ----------
    bhav_df : pd.DataFrame
        Multi-stock BhavCopy.
    reference_date : date-like
        Snapshot date for filters.
    **kwargs
        Passed directly to ``TradabilityGate.__init__``.
        Valid keys: min_value_crore, min_price, min_avg_volume,
        lookback_days, max_adv_participation.

    Returns
    -------
    pd.DataFrame
        Filtered BhavCopy with only tradable stocks.

    Examples
    --------
    >>> tradable = apply_tradability_gate(bhav, "2024-03-15",
    ...                                  min_price=50, min_value_crore=5)
    """
    gate = TradabilityGate(**kwargs)
    return gate.apply(bhav_df, reference_date)


# ===========================================================================
# Smoke-test / demo
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # -----------------------------------------------------------------------
    # Build synthetic BhavCopy data
    # -----------------------------------------------------------------------
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-02", periods=30, freq="B")

    # Simulate 10 stocks with varying liquidity profiles
    stock_profiles = {
        "500325": {"base_price": 500.0,  "base_vol": 200_000},  # Reliance
        "532540": {"base_price": 3_500.0, "base_vol": 150_000},  # TCS
        "500180": {"base_price": 1_600.0, "base_vol": 300_000},  # HDFC Bank
        "532978": {"base_price": 12.0,   "base_vol": 5_000},    # low price
        "590111": {"base_price": 450.0,  "base_vol": 800},      # illiquid
        "543396": {"base_price": 85.0,   "base_vol": 50_000},   # mid-cap
        "511288": {"base_price": 8.0,    "base_vol": 200},      # penny/illiquid
        "530591": {"base_price": 22.0,   "base_vol": 12_000},   # borderline
        "524091": {"base_price": 300.0,  "base_vol": 80_000},   # small-cap ok
        "526612": {"base_price": 60.0,   "base_vol": 9_500},    # just below vol
    }

    rows = []
    for code, prof in stock_profiles.items():
        prices = prof["base_price"] * (
            1 + np.cumsum(rng.normal(0, 0.005, len(dates)))
        )
        vols = prof["base_vol"] * (
            1 + rng.normal(0, 0.1, len(dates))
        ).clip(0.5)
        for i, d in enumerate(dates):
            c = max(prices[i], 0.01)
            v = max(int(vols[i]), 0)
            rows.append(
                {
                    "SC_CODE": code,
                    "DATE": d,
                    "Open": round(c * 0.995, 2),
                    "High": round(c * 1.008, 2),
                    "Low": round(c * 0.992, 2),
                    "Close": round(c, 2),
                    "Volume": v,
                    # Value (Close * Volume) intentionally omitted to test
                    # the derived-column fallback
                }
            )

    bhav = pd.DataFrame(rows)
    REFERENCE_DATE = "2024-02-14"  # Last date in our synthetic window

    logger.info(
        "Synthetic BhavCopy: %d rows, %d stocks, date range %s – %s",
        len(bhav),
        bhav["SC_CODE"].nunique(),
        bhav["DATE"].min().date(),
        bhav["DATE"].max().date(),
    )

    # -----------------------------------------------------------------------
    # 1. Full gate via convenience function
    # -----------------------------------------------------------------------
    logger.info("=== apply_tradability_gate (default params) ===")
    tradable = apply_tradability_gate(bhav, REFERENCE_DATE)
    logger.info(
        "Tradable stocks: %s", sorted(tradable["SC_CODE"].unique())
    )

    # -----------------------------------------------------------------------
    # 2. Individual filters
    # -----------------------------------------------------------------------
    gate = TradabilityGate(
        min_value_crore=2.0,
        min_price=20.0,
        min_avg_volume=10_000,
        lookback_days=20,
        max_adv_participation=0.01,
    )

    logger.info("=== Filter report ===")
    report = gate.get_filter_report(bhav, REFERENCE_DATE)
    for k, v in report.items():
        logger.info("  %-25s: %d", k, v)

    # -----------------------------------------------------------------------
    # 3. ADV and max tradeable shares for a specific stock
    # -----------------------------------------------------------------------
    sample_code = "500325"
    adv = gate.compute_adv(bhav, sample_code)
    max_shares = gate.get_max_tradeable_shares(bhav, sample_code)
    logger.info(
        "SC_CODE=%s  ADV=%.0f shares  max_tradeable=%.0f shares",
        sample_code, adv, max_shares,
    )

    # -----------------------------------------------------------------------
    # 4. Tighter thresholds
    # -----------------------------------------------------------------------
    logger.info("=== Tighter thresholds (min_price=100, min_value=10Cr) ===")
    tight = apply_tradability_gate(
        bhav,
        REFERENCE_DATE,
        min_price=100.0,
        min_value_crore=10.0,
        min_avg_volume=50_000,
    )
    logger.info(
        "Tradable under tight thresholds: %s",
        sorted(tight["SC_CODE"].unique()),
    )

    # -----------------------------------------------------------------------
    # 5. Demonstrate individual filter steps
    # -----------------------------------------------------------------------
    logger.info("=== Step-by-step filter walkthrough ===")
    after_price = gate.filter_by_price(bhav, REFERENCE_DATE)
    logger.info(
        "After price filter (>= %.0f): %d stocks",
        gate.min_price, after_price["SC_CODE"].nunique(),
    )

    after_volume = gate.filter_by_volume(after_price, REFERENCE_DATE)
    logger.info(
        "After volume filter (>= %d): %d stocks",
        gate.min_avg_volume, after_volume["SC_CODE"].nunique(),
    )

    after_value = gate.filter_by_value_traded(after_volume, REFERENCE_DATE)
    logger.info(
        "After value filter (>= %.1f Cr): %d stocks",
        gate.min_value_crore, after_value["SC_CODE"].nunique(),
    )
