"""
data_integrity.py
=================
Gap 1 — Data Validation & Integrity for BSE/NSE production trading systems.

Covers:
  - Corporate-action adjustment (splits, bonuses, rights)
  - Survivorship-bias correction
  - Data-quality checks and anomaly detection

All public methods return copies; originals are never mutated in-place.
"""

from __future__ import annotations

import json
import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BSE_CORPORATE_ACTION_URL = (
    "https://api.bseindia.com/BseIndiaAPI/api/CorporateAction/w"
)
BSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bseindia.com/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.bseindia.com",
}

OHLC_COLS = ["Open", "High", "Low", "Close"]
VOL_COL = "Volume"

# Maximum single-day price-change fraction before a tick is considered bad
MAX_DAILY_CHANGE = 0.50


# ===========================================================================
# 1. CorporateActionAdjuster
# ===========================================================================
class CorporateActionAdjuster:
    """
    Adjust OHLCV data for historical corporate actions (splits, bonuses,
    rights issues) so that price series are comparable across time.

    Parameters
    ----------
    None – use ``load_corporate_actions`` to populate the internal table.

    Internal schema of ``self.actions`` (DataFrame):
        SC_CODE      : str   – BSE scrip code
        DATE         : date  – announcement / record date
        ACTION_TYPE  : str   – SPLIT | BONUS | RIGHTS
        RATIO        : float – primary ratio number (see per-method docs)
        EX_DATE      : date  – ex-date; adjustments apply to prices BEFORE this
    """

    def __init__(self) -> None:
        self.actions: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    def load_corporate_actions(self, csv_path: str | Path) -> None:
        """
        Load corporate actions from a CSV file.

        Expected columns:
            SC_CODE, DATE, ACTION_TYPE (SPLIT/BONUS/RIGHTS),
            RATIO, EX_DATE

        For a SPLIT the RATIO column encodes the *split ratio* N where the
        stock splits N-for-1 (e.g. 2 for a 2:1 split).

        For a BONUS the RATIO column encodes M where the bonus is M:N and
        the denominator N is stored in the optional BONUS_DENOM column
        (defaults to 1 if absent).

        Parameters
        ----------
        csv_path : str or Path
            Path to the CSV file.
        """
        path = Path(csv_path)
        if not path.exists():
            warnings.warn(
                f"Corporate actions file not found: {path}. "
                "No adjustments will be applied.",
                UserWarning,
                stacklevel=2,
            )
            return

        df = pd.read_csv(path, dtype={"SC_CODE": str})

        required = {"SC_CODE", "DATE", "ACTION_TYPE", "RATIO", "EX_DATE"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Corporate actions CSV is missing columns: {missing}"
            )

        df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True)
        df["EX_DATE"] = pd.to_datetime(df["EX_DATE"], dayfirst=True)
        df["ACTION_TYPE"] = df["ACTION_TYPE"].str.upper().str.strip()
        df["RATIO"] = pd.to_numeric(df["RATIO"], errors="coerce").fillna(1.0)

        if "BONUS_DENOM" not in df.columns:
            df["BONUS_DENOM"] = 1.0
        else:
            df["BONUS_DENOM"] = pd.to_numeric(
                df["BONUS_DENOM"], errors="coerce"
            ).fillna(1.0)

        self.actions = df.reset_index(drop=True)
        logger.info(
            "Loaded %d corporate action records from %s",
            len(self.actions),
            path,
        )

    # ------------------------------------------------------------------
    def _get_actions_for_code(self, sc_code: str) -> pd.DataFrame:
        """Return corporate actions for a single SC_CODE, sorted by EX_DATE."""
        if self.actions.empty:
            return pd.DataFrame()
        mask = self.actions["SC_CODE"].astype(str) == str(sc_code)
        return (
            self.actions[mask]
            .sort_values("EX_DATE")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    def adjust_prices(self, df: pd.DataFrame, sc_code: str) -> pd.DataFrame:
        """
        Adjust OHLCV data for a single stock's corporate actions.

        Adjustment rules
        ----------------
        SPLIT (ratio N, i.e. N-for-1):
            For rows where DATE < EX_DATE:
                price_col  *= 1/N
                Volume     *= N          (more shares, same value)

        BONUS (M:N, stored as RATIO=M, BONUS_DENOM=N, meaning M bonus
        shares per N existing shares):
            For rows where DATE < EX_DATE:
                price_col  *= N / (N + M)
                Volume stays unchanged (bonus shares already issued)

        RIGHTS are flagged but not price-adjusted here (complex; left to
        a downstream model).

        Parameters
        ----------
        df : pd.DataFrame
            Must have a ``DATE`` column (datetime) and OHLC + Volume.
        sc_code : str
            BSE scrip code to look up corporate actions.

        Returns
        -------
        pd.DataFrame
            Adjusted copy of *df*.
        """
        actions = self._get_actions_for_code(sc_code)
        if actions.empty:
            return df.copy()

        out = df.copy()
        out["DATE"] = pd.to_datetime(out["DATE"])

        for _, row in actions.iterrows():
            ex_date = row["EX_DATE"]
            action = row["ACTION_TYPE"]
            ratio = float(row["RATIO"])

            pre_mask = out["DATE"] < ex_date
            if not pre_mask.any():
                continue

            if action == "SPLIT":
                if ratio <= 0:
                    logger.warning(
                        "SC_CODE=%s: invalid split ratio %s – skipping",
                        sc_code, ratio,
                    )
                    continue
                factor = 1.0 / ratio
                for col in OHLC_COLS:
                    if col in out.columns:
                        out.loc[pre_mask, col] = (
                            out.loc[pre_mask, col] * factor
                        )
                if VOL_COL in out.columns:
                    out.loc[pre_mask, VOL_COL] = (
                        out.loc[pre_mask, VOL_COL] * ratio
                    )
                logger.debug(
                    "SC_CODE=%s SPLIT %.2f applied before %s",
                    sc_code, ratio, ex_date.date(),
                )

            elif action == "BONUS":
                denom = float(row.get("BONUS_DENOM", 1.0)) or 1.0
                # bonus M:N  => price factor = N / (N + M)
                factor = denom / (denom + ratio)
                for col in OHLC_COLS:
                    if col in out.columns:
                        out.loc[pre_mask, col] = (
                            out.loc[pre_mask, col] * factor
                        )
                logger.debug(
                    "SC_CODE=%s BONUS %g:%g applied before %s",
                    sc_code, ratio, denom, ex_date.date(),
                )

            elif action == "RIGHTS":
                logger.info(
                    "SC_CODE=%s RIGHTS issue on %s – flagged, not adjusted",
                    sc_code, ex_date.date(),
                )

            else:
                logger.warning(
                    "SC_CODE=%s: unknown action type '%s' – skipping",
                    sc_code, action,
                )

        return out

    # ------------------------------------------------------------------
    def adjust_all(self, bhav_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply corporate-action adjustments to an entire BhavCopy DataFrame
        grouped by SC_CODE.

        Parameters
        ----------
        bhav_df : pd.DataFrame
            Must contain SC_CODE, DATE, Open, High, Low, Close, Volume.

        Returns
        -------
        pd.DataFrame
            Adjusted DataFrame (concatenation of per-stock adjusted frames).
        """
        if self.actions.empty:
            logger.warning(
                "No corporate actions loaded – returning data unchanged."
            )
            return bhav_df.copy()

        codes = bhav_df["SC_CODE"].unique()
        adjusted_parts: List[pd.DataFrame] = []

        for code in codes:
            subset = bhav_df[bhav_df["SC_CODE"] == code].copy()
            adjusted_parts.append(self.adjust_prices(subset, code))

        result = pd.concat(adjusted_parts, ignore_index=True)
        logger.info(
            "adjust_all: processed %d stocks, %d rows total",
            len(codes), len(result),
        )
        return result

    # ------------------------------------------------------------------
    def download_bse_corporate_actions(
        self,
        start_date: str,
        end_date: str,
        cache_dir: str | Path,
    ) -> pd.DataFrame:
        """
        Fetch BSE corporate actions from the BSE India API and cache locally.

        Parameters
        ----------
        start_date : str
            Format ``YYYY-MM-DD``.
        end_date : str
            Format ``YYYY-MM-DD``.
        cache_dir : str or Path
            Directory to save/read cached CSV files.

        Returns
        -------
        pd.DataFrame
            Corporate actions in the internal schema.
        """
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / f"ca_{start_date}_{end_date}.csv"
        if cache_file.exists():
            logger.info("Loading cached corporate actions from %s", cache_file)
            df = pd.read_csv(cache_file, dtype={"SC_CODE": str})
            df["DATE"] = pd.to_datetime(df["DATE"])
            df["EX_DATE"] = pd.to_datetime(df["EX_DATE"])
            return df

        params = {
            "strdate": datetime.strptime(start_date, "%Y-%m-%d").strftime(
                "%Y%m%d"
            ),
            "enddate": datetime.strptime(end_date, "%Y-%m-%d").strftime(
                "%Y%m%d"
            ),
            "scripcode": "",
            "segment": "Equity",
        }

        try:
            logger.info(
                "Downloading BSE corporate actions %s to %s …",
                start_date, end_date,
            )
            resp = requests.get(
                BSE_CORPORATE_ACTION_URL,
                params=params,
                headers=BSE_HEADERS,
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Failed to download BSE corporate actions: %s", exc)
            return pd.DataFrame()

        try:
            data = resp.json()
        except ValueError as exc:
            logger.error("Could not parse JSON response: %s", exc)
            return pd.DataFrame()

        # BSE API returns a list of dicts under various keys depending on
        # the endpoint version; try common patterns.
        records = (
            data
            if isinstance(data, list)
            else data.get("Table", data.get("data", []))
        )
        if not records:
            logger.warning("No corporate action records returned by API.")
            return pd.DataFrame()

        raw = pd.DataFrame(records)

        # Map BSE field names to our schema (best-effort; BSE may change keys)
        col_map = {
            "SCRIP_CD": "SC_CODE",
            "SC_CODE": "SC_CODE",
            "Ex_Date": "EX_DATE",
            "EX_DATE": "EX_DATE",
            "Record_Date": "DATE",
            "DATE": "DATE",
            "Purpose": "ACTION_TYPE",
            "ACTION_TYPE": "ACTION_TYPE",
            "Ratio": "RATIO",
            "RATIO": "RATIO",
        }
        raw.rename(
            columns={k: v for k, v in col_map.items() if k in raw.columns},
            inplace=True,
        )

        for needed in ["SC_CODE", "EX_DATE", "DATE", "ACTION_TYPE", "RATIO"]:
            if needed not in raw.columns:
                raw[needed] = np.nan

        raw["SC_CODE"] = raw["SC_CODE"].astype(str).str.strip()
        raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce", dayfirst=True)
        raw["EX_DATE"] = pd.to_datetime(
            raw["EX_DATE"], errors="coerce", dayfirst=True
        )
        raw["ACTION_TYPE"] = (
            raw["ACTION_TYPE"].astype(str).str.upper().str.strip()
        )
        raw["RATIO"] = pd.to_numeric(raw["RATIO"], errors="coerce").fillna(1.0)

        raw.to_csv(cache_file, index=False)
        logger.info(
            "Cached %d corporate action records to %s", len(raw), cache_file
        )
        return raw


# ===========================================================================
# 2. SurvivorshipBiasCorrector
# ===========================================================================
class SurvivorshipBiasCorrector:
    """
    Correct for survivorship bias by maintaining a point-in-time universe of
    listed stocks.

    The universe history maps each date to the set of SC_CODEs that were
    actively listed on that date.  Delisted or suspended stocks that appear
    in today's universe are excluded when constructing historical datasets,
    preventing look-ahead bias.

    Internal schema of ``self.universe`` (DataFrame):
        DATE    : datetime
        SC_CODE : str
        STATUS  : str  – LISTED | DELISTED | SUSPENDED
    """

    def __init__(self) -> None:
        self.universe: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    def load_point_in_time_universe(
        self, history_file: str | Path
    ) -> None:
        """
        Load a point-in-time listing history file.

        Supported formats:
        - CSV  – columns: DATE, SC_CODE, STATUS (LISTED/DELISTED/SUSPENDED)
        - JSON – ``{date_str: [sc_code, ...]}`` (all entries assumed LISTED)

        Parameters
        ----------
        history_file : str or Path
            Path to the CSV or JSON file.
        """
        path = Path(history_file)
        if not path.exists():
            warnings.warn(
                f"Universe history file not found: {path}. "
                "Survivorship correction will be skipped.",
                UserWarning,
                stacklevel=2,
            )
            return

        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, encoding="utf-8") as fh:
                raw: Dict[str, List[str]] = json.load(fh)
            records = [
                {"DATE": date_str, "SC_CODE": code, "STATUS": "LISTED"}
                for date_str, codes in raw.items()
                for code in codes
            ]
            df = pd.DataFrame(records)
        elif suffix in {".csv", ".txt"}:
            df = pd.read_csv(path, dtype={"SC_CODE": str})
        else:
            raise ValueError(
                f"Unsupported universe history format: {suffix}"
            )

        required = {"DATE", "SC_CODE", "STATUS"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Universe history file is missing columns: {missing}"
            )

        df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True)
        df["SC_CODE"] = df["SC_CODE"].astype(str).str.strip()
        df["STATUS"] = df["STATUS"].str.upper().str.strip()

        self.universe = df.reset_index(drop=True)
        logger.info(
            "Loaded point-in-time universe: %d records, %d unique codes",
            len(self.universe),
            self.universe["SC_CODE"].nunique(),
        )

    # ------------------------------------------------------------------
    def get_universe_at_date(self, date: str | datetime) -> Set[str]:
        """
        Return the set of SC_CODEs that were actively listed on *date*.

        The method looks at the most recent STATUS record for each stock on
        or before *date*.  A stock is considered active if its latest
        status is LISTED.

        Parameters
        ----------
        date : str or datetime

        Returns
        -------
        set of str
        """
        if self.universe.empty:
            logger.warning(
                "Universe history is empty – returning empty set."
            )
            return set()

        query_date = pd.to_datetime(date)
        snap = self.universe[self.universe["DATE"] <= query_date]
        if snap.empty:
            return set()

        # Keep the most recent record per SC_CODE
        latest = (
            snap.sort_values("DATE")
            .groupby("SC_CODE", sort=False)
            .last()
            .reset_index()
        )
        active = latest[latest["STATUS"] == "LISTED"]["SC_CODE"]
        return set(active.tolist())

    # ------------------------------------------------------------------
    def filter_bhav_survivorship_free(
        self,
        bhav_df: pd.DataFrame,
        reference_date: str | datetime,
    ) -> pd.DataFrame:
        """
        Filter a BhavCopy DataFrame to only stocks that were listed on
        *reference_date* according to the point-in-time universe.

        Parameters
        ----------
        bhav_df : pd.DataFrame
            Must contain SC_CODE column.
        reference_date : str or datetime
            The date for which to retrieve the active universe.

        Returns
        -------
        pd.DataFrame
            Filtered copy of *bhav_df*.
        """
        active = self.get_universe_at_date(reference_date)
        if not active:
            logger.warning(
                "No active universe found for %s – returning full bhav_df.",
                reference_date,
            )
            return bhav_df.copy()

        before = len(bhav_df)
        mask = bhav_df["SC_CODE"].astype(str).isin(active)
        result = bhav_df[mask].copy()
        removed = before - len(result)
        logger.info(
            "filter_bhav_survivorship_free(%s): %d -> %d rows (%d removed, "
            "likely delisted/suspended)",
            reference_date, before, len(result), removed,
        )
        return result

    # ------------------------------------------------------------------
    def build_universe_history_from_bhav(
        self, bhav_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Infer listing history from BhavCopy data.

        A stock that appears in BhavCopy on a given date is assumed to have
        been LISTED on that date.  This is a conservative estimate and
        should be supplemented with official delisting records.

        Parameters
        ----------
        bhav_df : pd.DataFrame
            Must contain SC_CODE and DATE columns.

        Returns
        -------
        pd.DataFrame
            Universe history in the internal schema (DATE, SC_CODE, STATUS).
        """
        df = bhav_df[["SC_CODE", "DATE"]].copy()
        df["SC_CODE"] = df["SC_CODE"].astype(str).str.strip()
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.drop_duplicates()
        df["STATUS"] = "LISTED"

        self.universe = df.reset_index(drop=True)
        logger.info(
            "Built universe history: %d records, %d unique stocks, "
            "%d unique dates",
            len(self.universe),
            self.universe["SC_CODE"].nunique(),
            self.universe["DATE"].nunique(),
        )
        return self.universe.copy()


# ===========================================================================
# 3. DataQualityChecker
# ===========================================================================
class DataQualityChecker:
    """
    Validate and clean BhavCopy OHLCV data.

    Checks performed:
    - Remove rows with impossible OHLC relationships (High < Low, Close <= 0)
    - Remove single-day price changes > MAX_DAILY_CHANGE (50 %)
    - Forward-fill prices for missing trading dates
    - Detect statistical anomalies via rolling z-score on Close
    """

    def __init__(
        self,
        trading_dates: Optional[pd.DatetimeIndex] = None,
        anomaly_window: int = 20,
        anomaly_z_threshold: float = 3.0,
    ) -> None:
        """
        Parameters
        ----------
        trading_dates : pd.DatetimeIndex, optional
            Reference set of trading dates (market calendar).  If not
            supplied, dates are inferred from the data itself.
        anomaly_window : int
            Rolling window (days) for z-score calculation.
        anomaly_z_threshold : float
            Number of standard deviations above which a price is flagged.
        """
        self.trading_dates = trading_dates
        self.anomaly_window = anomaly_window
        self.anomaly_z_threshold = anomaly_z_threshold

    # ------------------------------------------------------------------
    def remove_bad_ticks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with clearly erroneous tick data.

        Removes:
        - Rows where High < Low
        - Rows where Close <= 0
        - Rows where the absolute day-over-day price change > 50 %

        Parameters
        ----------
        df : pd.DataFrame
            Per-stock OHLCV data with DATE, Open, High, Low, Close, Volume.

        Returns
        -------
        pd.DataFrame
            Cleaned copy.
        """
        out = df.copy()
        out["DATE"] = pd.to_datetime(out["DATE"])
        out = out.sort_values("DATE").reset_index(drop=True)

        n_before = len(out)

        # High < Low
        bad_hl = out["High"] < out["Low"]
        # Non-positive close
        bad_close = out["Close"] <= 0
        # Single-day price spike
        pct_chg = out["Close"].pct_change().abs()
        bad_spike = pct_chg > MAX_DAILY_CHANGE

        bad_mask = bad_hl | bad_close | bad_spike
        out = out[~bad_mask].reset_index(drop=True)
        logger.debug(
            "remove_bad_ticks: removed %d/%d rows",
            bad_mask.sum(), n_before,
        )
        return out

    # ------------------------------------------------------------------
    def fill_missing_dates(
        self, df: pd.DataFrame, sc_code: str
    ) -> pd.DataFrame:
        """
        Forward-fill OHLCV data for missing trading dates.

        If ``self.trading_dates`` is set, the output is reindexed to that
        calendar.  Otherwise the gaps between existing dates are filled.

        Parameters
        ----------
        df : pd.DataFrame
            Must have DATE, Open, High, Low, Close, Volume (per stock).
        sc_code : str
            Used for logging only.

        Returns
        -------
        pd.DataFrame
            DataFrame with missing trading dates forward-filled.
        """
        out = df.copy()
        out["DATE"] = pd.to_datetime(out["DATE"])
        out = out.set_index("DATE").sort_index()

        if self.trading_dates is not None:
            ref_dates = self.trading_dates[
                (self.trading_dates >= out.index.min())
                & (self.trading_dates <= out.index.max())
            ]
        else:
            ref_dates = pd.date_range(
                start=out.index.min(), end=out.index.max(), freq="B"
            )

        missing = ref_dates.difference(out.index)
        if len(missing) > 0:
            empty = pd.DataFrame(index=missing, columns=out.columns)
            out = pd.concat([out, empty]).sort_index()
            out = out.ffill()
            logger.debug(
                "fill_missing_dates: SC_CODE=%s filled %d missing dates",
                sc_code, len(missing),
            )

        out = out.reset_index().rename(columns={"index": "DATE"})
        return out

    # ------------------------------------------------------------------
    def detect_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag rows where Close deviates more than ``anomaly_z_threshold``
        standard deviations from a rolling ``anomaly_window``-day mean.

        Parameters
        ----------
        df : pd.DataFrame
            Must have DATE and Close columns, sorted chronologically.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with an additional boolean column ``IS_ANOMALY``.
        """
        out = df.copy()
        out["DATE"] = pd.to_datetime(out["DATE"])
        out = out.sort_values("DATE").reset_index(drop=True)

        roll = out["Close"].rolling(
            window=self.anomaly_window, min_periods=max(1, self.anomaly_window // 2)
        )
        roll_mean = roll.mean()
        roll_std = roll.std()

        # Avoid division by zero
        z_score = (out["Close"] - roll_mean) / roll_std.replace(0, np.nan)
        out["IS_ANOMALY"] = z_score.abs() > self.anomaly_z_threshold

        n_flagged = out["IS_ANOMALY"].sum()
        logger.debug(
            "detect_price_anomalies: flagged %d/%d rows", n_flagged, len(out)
        )
        return out

    # ------------------------------------------------------------------
    def validate_bhav_data(
        self, bhav_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Run all data-quality checks on a BhavCopy DataFrame.

        Steps (applied per stock):
        1. Remove bad ticks (impossible OHLC, large spikes)
        2. Detect price anomalies (rolling z-score)

        Missing-date filling is NOT applied here because BhavCopy data is
        naturally sparse across stocks on the same date.  Call
        ``fill_missing_dates`` per-stock if needed.

        Parameters
        ----------
        bhav_df : pd.DataFrame
            Full BhavCopy DataFrame with SC_CODE, DATE, Open, High, Low,
            Close, Volume.

        Returns
        -------
        clean_df : pd.DataFrame
            Cleaned data (bad ticks removed, anomalies flagged).
        report : dict
            Statistics on removed / flagged rows.
        """
        total_rows = len(bhav_df)
        all_parts: List[pd.DataFrame] = []
        total_bad_ticks = 0
        total_anomalies = 0

        for code, grp in bhav_df.groupby("SC_CODE"):
            cleaned = self.remove_bad_ticks(grp)
            total_bad_ticks += len(grp) - len(cleaned)

            flagged = self.detect_price_anomalies(cleaned)
            total_anomalies += int(flagged["IS_ANOMALY"].sum())
            all_parts.append(flagged)

        if not all_parts:
            clean_df = bhav_df.copy()
            clean_df["IS_ANOMALY"] = False
        else:
            clean_df = pd.concat(all_parts, ignore_index=True)

        report = {
            "total_input_rows": total_rows,
            "rows_after_bad_tick_removal": total_rows - total_bad_ticks,
            "bad_ticks_removed": total_bad_ticks,
            "anomalies_flagged": total_anomalies,
            "final_row_count": len(clean_df),
            "stocks_processed": bhav_df["SC_CODE"].nunique(),
        }

        logger.info(
            "validate_bhav_data: in=%d, bad_ticks=%d, anomalies=%d, out=%d",
            report["total_input_rows"],
            report["bad_ticks_removed"],
            report["anomalies_flagged"],
            report["final_row_count"],
        )
        return clean_df, report


# ===========================================================================
# Quick smoke-test
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # -- synthetic BhavCopy data -------------------------------------------
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    codes = ["500325", "532540", "500180"]

    rows = []
    for code in codes:
        prices = 100.0 + np.cumsum(rng.normal(0, 1, len(dates)))
        for i, d in enumerate(dates):
            c = max(prices[i], 1.0)
            rows.append(
                {
                    "SC_CODE": code,
                    "DATE": d,
                    "Open": c * 0.99,
                    "High": c * 1.01,
                    "Low": c * 0.98,
                    "Close": c,
                    "Volume": int(rng.integers(50_000, 500_000)),
                }
            )

    bhav = pd.DataFrame(rows)

    # Inject a bad tick (High < Low)
    bhav.loc[0, "High"] = bhav.loc[0, "Low"] - 1

    # -- DataQualityChecker ------------------------------------------------
    checker = DataQualityChecker()
    clean, report = checker.validate_bhav_data(bhav)
    logger.info("Report: %s", report)

    # -- SurvivorshipBiasCorrector -----------------------------------------
    corrector = SurvivorshipBiasCorrector()
    corrector.build_universe_history_from_bhav(bhav)
    active_set = corrector.get_universe_at_date("2024-03-01")
    logger.info("Active codes at 2024-03-01: %s", active_set)

    # -- CorporateActionAdjuster -------------------------------------------
    adjuster = CorporateActionAdjuster()
    # No CSV – just shows graceful handling
    adjuster.load_corporate_actions("/nonexistent/path.csv")
    adjusted = adjuster.adjust_all(bhav)
    logger.info(
        "adjust_all (no actions): input rows=%d output rows=%d",
        len(bhav), len(adjusted),
    )
