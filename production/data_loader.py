"""
DataLoader — BhavCopy fetch + standardise adapter.

Thin wrapper around BSEDataFetcher that:
  1. Fetches the raw BhavCopy for a date range
  2. Normalises column names to the production schema
  3. Enforces correct dtypes
  4. Handles missing ValueTraded (derives from Close × Volume)

Expected output schema (all production modules depend on this):
    SC_CODE      object   — BSE script code / ISIN
    SC_NAME      object   — Company name / ticker symbol
    DATE         date     — Trading date (Python date, not datetime)
    Open         float64
    High         float64
    Low          float64
    Close        float64
    Volume       float64  — shares traded
    ValueTraded  float64  — rupee turnover
"""

import sys
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# Canonical column names every production module expects
SCHEMA = {
    "SC_CODE": "object",
    "SC_NAME": "object",
    "DATE": "date",
    "Open": "float64",
    "High": "float64",
    "Low": "float64",
    "Close": "float64",
    "Volume": "float64",
    "ValueTraded": "float64",
}

# All possible aliases from BSE new/old formats → canonical name
_ALIASES = {
    # New BSE BhavCopy (CSV format)
    "FinInstrmId": "SC_CODE", "TckrSymb": "SC_NAME",
    "OpnPric": "Open", "HghPric": "High", "LwPric": "Low", "ClsPric": "Close",
    "TtlTradgVol": "Volume", "TtlTrfVal": "ValueTraded",
    # Old BSE BhavCopy (ZIP format)
    "OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close",
    "NO_OF_SHRS": "Volume", "NET_TURNOV": "ValueTraded",
    "ISIN": "SC_CODE",
    # Already canonical — pass through
    "SC_CODE": "SC_CODE", "SC_NAME": "SC_NAME",
    "Open": "Open", "High": "High", "Low": "Low", "Close": "Close",
    "Volume": "Volume", "ValueTraded": "ValueTraded", "DATE": "DATE",
}


class DataLoader:
    """
    Fetch and standardise BSE BhavCopy data for a date range.

    Usage
    -----
        loader = DataLoader()
        bhav = loader.load(lookback_days=730)      # fetch 2 years to today
        bhav = loader.load(start="2024-01-01", end="2026-04-23")
        bhav = loader.standardise(raw_df)          # normalise an already-fetched df
    """

    def __init__(self, cache_dir: str = "stock_picker_data/cache/bse"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fetcher = None   # lazy

    # ------------------------------------------------------------------
    def _get_fetcher(self):
        if self._fetcher is None:
            from bse_direct_loader import BSEDataFetcher
            self._fetcher = BSEDataFetcher(cache_dir=str(self.cache_dir))
        return self._fetcher

    # ------------------------------------------------------------------
    def load(
        self,
        lookback_days: int = 730,
        start: str = None,
        end: str = None,
        reference_date: str = None,
    ) -> pd.DataFrame:
        """
        Fetch BhavCopy for a date range and return in standard schema.

        Priority: explicit start/end > lookback_days from reference_date.

        Args:
            lookback_days: Calendar days of history to fetch (default 2 years).
            start: 'YYYY-MM-DD' start date (overrides lookback_days).
            end:   'YYYY-MM-DD' end date (defaults to latest trading day).
            reference_date: Anchor for lookback (defaults to today).

        Returns:
            Standardised DataFrame in production schema.
        """
        fetcher = self._get_fetcher()

        if end is None:
            end_dt = fetcher.prev_bday(date.today())
        else:
            end_dt = pd.to_datetime(end).date()

        if start is None:
            anchor = pd.to_datetime(reference_date).date() if reference_date else end_dt
            start_dt = anchor - timedelta(days=lookback_days)
        else:
            start_dt = pd.to_datetime(start).date()

        logger.info("Fetching BhavCopy %s → %s …", start_dt, end_dt)
        raw = fetcher.fetch_bhav_range(start_dt, end_dt)

        if raw is None or raw.empty:
            logger.error("BhavCopy fetch returned empty data.")
            return pd.DataFrame(columns=list(SCHEMA.keys()))

        std = self.standardise(raw)
        logger.info(
            "DataLoader: %d rows, %d stocks, %s → %s",
            len(std), std["SC_CODE"].nunique(),
            std["DATE"].min(), std["DATE"].max(),
        )
        return std

    # ------------------------------------------------------------------
    def standardise(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise any BhavCopy variant to the production schema.

        Handles: new BSE CSV, old BSE ZIP, already-normalised frames.
        """
        df = raw.copy()

        # ── 1. Rename aliases ─────────────────────────────────────────
        rename_map = {c: _ALIASES[c] for c in df.columns if c in _ALIASES}
        df = df.rename(columns=rename_map)

        # ── 2. Ensure required columns exist ──────────────────────────
        missing = [c for c in SCHEMA if c not in df.columns]
        if missing:
            logger.debug("Filling missing columns with NaN: %s", missing)
            for col in missing:
                df[col] = np.nan

        # ── 3. Enforce dtypes ─────────────────────────────────────────
        for col in ["Open", "High", "Low", "Close", "Volume", "ValueTraded"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["SC_CODE"] = df["SC_CODE"].astype(str).str.strip()
        df["SC_NAME"] = df["SC_NAME"].astype(str).str.strip()

        # DATE → Python date (not datetime, not string)
        if df["DATE"].dtype != "object" or not isinstance(df["DATE"].iloc[0], date):
            df["DATE"] = pd.to_datetime(df["DATE"]).dt.date

        # ── 4. Derive ValueTraded if missing ──────────────────────────
        null_vt = df["ValueTraded"].isna() | (df["ValueTraded"] == 0)
        if null_vt.any():
            df.loc[null_vt, "ValueTraded"] = df.loc[null_vt, "Close"] * df.loc[null_vt, "Volume"]

        # ── 5. Drop junk rows ──────────────────────────────────────────
        df = df.dropna(subset=["Close", "DATE"])
        df = df[df["Close"] > 0]

        # Keep only schema columns (drop extras to keep memory small)
        df = df[list(SCHEMA.keys())].copy()
        df = df.sort_values(["SC_CODE", "DATE"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    def load_from_files(self, file_paths: list) -> pd.DataFrame:
        """
        Load and merge multiple BhavCopy CSV/ZIP files from disk
        (useful when fetching from pre-downloaded archives).
        """
        frames = []
        for fp in file_paths:
            try:
                raw = pd.read_csv(fp)
                frames.append(self.standardise(raw))
            except Exception as e:
                logger.warning("Could not load %s: %s", fp, e)
        if not frames:
            return pd.DataFrame(columns=list(SCHEMA.keys()))
        return pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["SC_CODE", "DATE"]
        ).sort_values(["SC_CODE", "DATE"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    def get_available_dates(self, bhav_df: pd.DataFrame = None) -> list:
        """
        Return a sorted list of available trading date strings ('YYYY-MM-DD').

        If bhav_df is provided, dates are derived from it (fast path, preferred).
        Otherwise the BSE cache directory is scanned for date-stamped files.
        """
        if bhav_df is not None and not bhav_df.empty:
            dates = sorted(bhav_df["DATE"].unique())
            return [str(d) for d in dates]

        # Scan cache directory for date-stamped BhavCopy files
        import datetime as _dt
        date_strings: set = set()
        for p in self.cache_dir.rglob("*"):
            if not p.is_file():
                continue
            stem = p.stem
            for fmt in ("%Y%m%d", "%Y-%m-%d"):
                for chunk in (stem, stem[-8:], stem[:8], stem[-10:], stem[:10]):
                    try:
                        d = _dt.datetime.strptime(chunk, fmt).date()
                        date_strings.add(str(d))
                        break
                    except (ValueError, OverflowError):
                        continue
        return sorted(date_strings)

    # ------------------------------------------------------------------
    def get_latest_prices(self, bhav_df: pd.DataFrame) -> pd.DataFrame:
        """
        Return one row per SC_CODE with the most recent available OHLCV data.
        """
        return (
            bhav_df.sort_values("DATE")
            .groupby("SC_CODE")
            .last()
            .reset_index()
        )

    # ------------------------------------------------------------------
    def validate(self, bhav_df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Quick sanity check on a standardised DataFrame.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []
        for col in SCHEMA:
            if col not in bhav_df.columns:
                issues.append(f"Missing column: {col}")

        if bhav_df.empty:
            issues.append("DataFrame is empty.")
            return False, issues

        n_stocks = bhav_df["SC_CODE"].nunique()
        if n_stocks < 100:
            issues.append(f"Very few stocks: {n_stocks} (expected ≥ 1000)")

        null_pct = bhav_df["Close"].isna().mean()
        if null_pct > 0.1:
            issues.append(f"Close column has {null_pct:.1%} nulls.")

        neg_close = (bhav_df["Close"] <= 0).mean()
        if neg_close > 0.01:
            issues.append(f"{neg_close:.1%} rows have Close ≤ 0.")

        return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def load_bhav(lookback_days: int = 730, reference_date: str = None) -> pd.DataFrame:
    """One-liner: fetch and standardise BhavCopy."""
    return DataLoader().load(lookback_days=lookback_days, reference_date=reference_date)


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("DataLoader smoke test — fetching last 30 days …")
    loader = DataLoader()
    bhav = loader.load(lookback_days=30)
    print(bhav.dtypes)
    print(bhav.tail())
    ok, issues = loader.validate(bhav)
    print("Valid:", ok, "|", issues or "no issues")
