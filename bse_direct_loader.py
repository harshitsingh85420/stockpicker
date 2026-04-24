"""
Direct BSE BhavCopy data loader - Proven working code
This is the exact code that works from the user's previous project
With per-day caching to avoid re-downloading already-fetched dates.

Cache strategy: one pickle file per trading day  (bhav_YYYYMMDD.pkl)
Loading a range = reading individual day files + fetching only missing days.
This means re-running with a new end date only downloads the new day(s).
"""

import io
import os
import pickle
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


class BSEDataFetcher:
    """Direct BSE BhavCopy data fetcher with per-day caching"""

    # BSE BhavCopy URLs
    UDIFF_URL = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{ymd}_F_0000.CSV"
    LEGACY_URL = "https://www.bseindia.com/download/BhavCopy/Equity/EQ{ddmmyy}_CSV.ZIP"

    def __init__(self, cache_dir: str = "./stock_picker_data/cache/bse"):
        """Initialize with cache directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[BSE] Cache directory: {self.cache_dir}")

    # Column mapping
    CANON = {
        "TckrSymb": "SC_NAME", "ISIN": "ISIN", "FinInstrmId": "SC_CODE",
        "OpnPric": "Open", "HghPric": "High", "LwPric": "Low", "ClsPric": "Close",
        "TtlTradgVol": "Volume", "TtlTrfVal": "ValueTraded", "DATE": "DATE",
        "SC_CODE": "SC_CODE", "SC_NAME": "SC_NAME",
        "OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close",
        "NO_OF_SHRS": "Volume", "NET_TURNOV": "ValueTraded"
    }
    REQUIRED = ["SC_CODE", "SC_NAME", "Open", "High", "Low", "Close", "Volume", "ValueTraded", "DATE"]

    @staticmethod
    def ymd(d: date) -> str:
        return d.strftime("%Y%m%d")

    @staticmethod
    def ddmmyy(d: date) -> str:
        return d.strftime("%d%m%y")

    @staticmethod
    def is_weekend(d: date) -> bool:
        return d.weekday() >= 5

    @staticmethod
    def prev_bday(d: date) -> date:
        """Return the most recent weekday strictly before d."""
        d = d - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        return d

    @staticmethod
    def safe_get(url: str, timeout: int = 25) -> Optional[requests.Response]:
        try:
            return requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=timeout)
        except Exception as e:
            print(f"[http] {e} :: {url}")
            return None

    def normalize_bhav(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize BhavCopy columns to standard format"""
        out = pd.DataFrame()
        for src, tgt in self.CANON.items():
            if src in df.columns:
                out[tgt] = df[src].copy()

        if "SC_CODE" not in out and "ISIN" in out:
            out["SC_CODE"] = out["ISIN"]
        if "SC_NAME" not in out and "TckrSymb" in df.columns:
            out["SC_NAME"] = df["TckrSymb"]

        for c in ["Open", "High", "Low", "Close", "Volume", "ValueTraded"]:
            if c in out:
                out[c] = pd.to_numeric(out[c], errors="coerce")

        if "DATE" in out:
            out["DATE"] = pd.to_datetime(out["DATE"]).dt.date

        missing = [c for c in self.REQUIRED if c not in out.columns]
        if missing:
            return pd.DataFrame()

        out = out.dropna(subset=["Close", "High", "Low", "Open"])
        out["SC_CODE"] = out["SC_CODE"].astype(str)
        return out[self.REQUIRED]

    # ------------------------------------------------------------------
    # Per-day cache helpers
    # ------------------------------------------------------------------

    def _day_cache_path(self, d: date) -> Path:
        return self.cache_dir / f"bhav_{self.ymd(d)}.pkl"

    def _load_day_cache(self, d: date) -> Optional[pd.DataFrame]:
        p = self._day_cache_path(d)
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    def _save_day_cache(self, d: date, df: pd.DataFrame):
        p = self._day_cache_path(d)
        with open(p, "wb") as f:
            pickle.dump(df, f)

    # ------------------------------------------------------------------
    def fetch_bhav_for(self, d: date) -> Optional[pd.DataFrame]:
        """Fetch BhavCopy for a single date (checks per-day cache first)."""
        cached = self._load_day_cache(d)
        if cached is not None:
            return cached

        # Try UDiFF CSV
        url = self.UDIFF_URL.format(ymd=self.ymd(d))
        r = self.safe_get(url)
        if r and r.ok:
            try:
                df = pd.read_csv(io.BytesIO(r.content))
                df["DATE"] = pd.to_datetime(d).date()
                norm = self.normalize_bhav(df)
                if not norm.empty:
                    self._save_day_cache(d, norm)
                    return norm
            except Exception:
                pass

        # Fallback to legacy ZIP
        url = self.LEGACY_URL.format(ddmmyy=self.ddmmyy(d))
        r = self.safe_get(url)
        if r and r.ok:
            try:
                with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                    name = [n for n in zf.namelist() if n.lower().endswith(".csv")][0]
                    with zf.open(name) as f:
                        df = pd.read_csv(f)
                df["DATE"] = pd.to_datetime(d).date()
                norm = self.normalize_bhav(df)
                if not norm.empty:
                    self._save_day_cache(d, norm)
                    return norm
            except Exception:
                pass

        return None

    def fetch_bhav_range(self, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetch BhavCopy data for a date range.

        Uses per-day cache: already-fetched days load instantly from disk,
        only missing days are downloaded from BSE.  Re-running with a new
        end date downloads just the new day(s) — not the full history.
        """
        got = []
        downloaded = []
        cur = start_date
        while cur <= end_date:
            if not self.is_weekend(cur):
                cached = self._load_day_cache(cur)
                if cached is not None:
                    got.append(cached)
                else:
                    df = self.fetch_bhav_for(cur)
                    if df is not None and len(df):
                        print(f"  [downloaded] {cur} -> {len(df)} stocks")
                        downloaded.append(cur)
                        got.append(df)
            cur += timedelta(days=1)

        if downloaded:
            print(f"[BSE] Downloaded {len(downloaded)} new day(s); "
                  f"{(cur - start_date).days - len(downloaded) - (sum(1 for d in _weekdays(start_date, end_date) if self.is_weekend(d)))} day(s) loaded from cache.")
        else:
            print(f"[BSE] All data loaded from cache ({len(got)} trading days).")

        if not got:
            return pd.DataFrame()

        return pd.concat(got, ignore_index=True)


def _weekdays(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)
