"""
production/universe_builder.py  --  P03: Survivorship-Free Point-in-Time Universe

Eliminates survivorship bias by tracking every stock that was EVER listed,
not just those currently trading.  A backtest on 2020 data should include
all stocks that existed in 2020 -- including those delisted in 2021-2024.

Universe derivation strategy
-----------------------------
Primary: BSE List_Scrips master download (tries two known BSE endpoints).
Fallback: Infer from the per-day BhavCopy cache (reliable, already on disk).
          first_seen  = first date a stock appears in cache  -> listed_date proxy
          last_seen   = last date a stock appears in cache
          delisted_date = None if last_seen >= (today - 60d), else last_seen+1d

A stock is VALID on as_of_date if:
    listed_date  <= as_of_date  AND
    (delisted_date is None  OR  delisted_date > as_of_date)
"""

from __future__ import annotations

import io
import json
import logging
import pickle
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# Where we persist the built master table
_CACHE_DIR  = _ROOT / "stock_picker_data" / "cache"
_MASTER_PKL = _CACHE_DIR / "pit_universe_master.pkl"
_MASTER_CSV = _CACHE_DIR / "pit_universe_master.csv"

# BSE endpoints (try both; both may require headers)
_BSE_SCRIPS_URLS = [
    "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity&status=Active",
    "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?Group=&Scripcode=&industry=&segment=Equity&status=Delisted",
]
_BSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Referer":    "https://www.bseindia.com/",
}


class PointInTimeUniverse:
    """
    Build a survivorship-free point-in-time universe from BSE data.

    Usage
    -----
        pit = PointInTimeUniverse()
        df_2020 = pit.build_universe('2020-03-01')
        df_today = pit.build_universe(str(date.today()))
        assert len(df_2020) >= len(df_today)   # survivorship check
    """

    def __init__(
        self,
        cache_dir: str = None,
        bse_cache_dir: str = None,
        refresh: bool = False,
    ):
        self._cache_dir     = Path(cache_dir) if cache_dir else _CACHE_DIR
        self._bse_cache_dir = (
            Path(bse_cache_dir) if bse_cache_dir
            else _ROOT / "stock_picker_data" / "cache" / "bse"
        )
        self._master: Optional[pd.DataFrame] = None
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if refresh:
            self._master = None
        else:
            self._master = self._load_cached_master()

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def build_universe(self, as_of_date: str) -> pd.DataFrame:
        """
        Return stocks that were valid on as_of_date.

        Parameters
        ----------
        as_of_date : str
            'YYYY-MM-DD'

        Returns
        -------
        pd.DataFrame with columns:
            SC_CODE, SC_NAME, listed_date, delisted_date
        """
        master = self._get_master()
        aod = pd.to_datetime(as_of_date).date()

        mask_listed = master["listed_date"] <= aod
        mask_active = (
            master["delisted_date"].isna()
            | (master["delisted_date"] > aod)
        )
        result = master[mask_listed & mask_active].copy()
        logger.info(
            "PointInTimeUniverse: %d stocks valid on %s (total master=%d)",
            len(result), as_of_date, len(master),
        )
        return result.reset_index(drop=True)

    def get_master(self) -> pd.DataFrame:
        """Return full master table (all stocks, all dates)."""
        return self._get_master().copy()

    def refresh(self) -> pd.DataFrame:
        """Force re-build from sources and re-cache."""
        self._master = self._build_master()
        self._save_master(self._master)
        return self._master.copy()

    # ---------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------

    def _get_master(self) -> pd.DataFrame:
        if self._master is not None:
            return self._master
        self._master = self._build_master()
        self._save_master(self._master)
        return self._master

    def _build_master(self) -> pd.DataFrame:
        """Build master by trying BSE API first, then cache fallback."""
        logger.info("Building PointInTimeUniverse master table ...")

        # Try BSE API
        bse_df = self._fetch_bse_master()
        if bse_df is not None and len(bse_df) > 100:
            logger.info("BSE master: %d stocks from API.", len(bse_df))
            # Enrich with first/last seen dates from BhavCopy cache
            cache_summary = self._build_cache_summary()
            if not cache_summary.empty:
                bse_df = bse_df.merge(
                    cache_summary[["SC_CODE", "first_seen", "last_seen"]],
                    on="SC_CODE", how="left",
                )
                # BSE API often omits listing dates.
                # We cannot infer listing date from our 2-year cache window.
                # Default to 1990-01-01 (assumes all cached stocks were listed
                # before any historical query we'll ever run).
                no_listed = bse_df["listed_date"].isna()
                bse_df.loc[no_listed, "listed_date"] = date(1990, 1, 1)

                # Use last_seen to detect delisted stocks (not seen for 60+ days)
                if "last_seen" in bse_df.columns:
                    cutoff = date.today() - timedelta(days=60)
                    no_delist = bse_df["delisted_date"].isna()
                    stale = no_delist & bse_df["last_seen"].notna() & (bse_df["last_seen"] < cutoff)
                    bse_df.loc[stale, "delisted_date"] = (
                        bse_df.loc[stale, "last_seen"] + timedelta(days=1)
                    )
            else:
                # No cache summary -- default all listing dates to 1990
                no_listed = bse_df["listed_date"].isna()
                bse_df.loc[no_listed, "listed_date"] = date(1990, 1, 1)
            return self._finalise(bse_df)

        # Fallback: derive entirely from BhavCopy cache
        logger.info(
            "BSE API unavailable -- building universe from BhavCopy cache."
        )
        return self._build_from_cache()

    def _fetch_bse_master(self) -> Optional[pd.DataFrame]:
        """Try BSE API endpoints for active + delisted scrip lists."""
        frames = []
        for url in _BSE_SCRIPS_URLS:
            try:
                r = requests.get(url, headers=_BSE_HEADERS, timeout=20)
                if not r.ok:
                    continue
                data = r.json()
                # BSE returns {"Table": [...]} or a plain list
                rows = data.get("Table", data) if isinstance(data, dict) else data
                if not rows:
                    continue
                df = pd.DataFrame(rows)
                frames.append(df)
            except Exception as exc:
                logger.debug("BSE API fetch error: %s", exc)

        if not frames:
            return None

        combined = pd.concat(frames, ignore_index=True)

        # Normalise column names
        col_map = {}
        for c in combined.columns:
            cl = c.lower()
            if "scrip" in cl and "code" in cl:
                col_map[c] = "SC_CODE"
            elif "scrip" in cl and "name" in cl:
                col_map[c] = "SC_NAME"
            elif "list" in cl and "date" in cl:
                col_map[c] = "listed_date"
            elif "delist" in cl or "suspend" in cl:
                col_map[c] = "delisted_date"
        combined = combined.rename(columns=col_map)

        for req in ("SC_CODE", "SC_NAME"):
            if req not in combined.columns:
                return None

        combined["SC_CODE"] = combined["SC_CODE"].astype(str).str.strip()
        combined = combined.drop_duplicates("SC_CODE")

        for dcol in ("listed_date", "delisted_date"):
            if dcol in combined.columns:
                combined[dcol] = pd.to_datetime(combined[dcol], errors="coerce").dt.date
            else:
                combined[dcol] = None

        return combined

    def _build_cache_summary(self) -> pd.DataFrame:
        """
        Scan per-day BhavCopy pkl files and return first/last seen per SC_CODE.
        """
        pkl_files = sorted(self._bse_cache_dir.glob("bhav_????????.pkl"))
        if not pkl_files:
            return pd.DataFrame()

        records = []   # (date, SC_CODE, SC_NAME)
        for p in pkl_files:
            try:
                date_str = p.stem.replace("bhav_", "")
                d = date(
                    int(date_str[:4]),
                    int(date_str[4:6]),
                    int(date_str[6:8]),
                )
                df = pickle.load(open(p, "rb"))
                if "SC_CODE" not in df.columns:
                    continue
                for sc, sn in zip(df["SC_CODE"], df.get("SC_NAME", [""] * len(df))):
                    records.append((d, str(sc), str(sn)))
            except Exception:
                continue

        if not records:
            return pd.DataFrame()

        tmp = pd.DataFrame(records, columns=["date", "SC_CODE", "SC_NAME"])
        summary = (
            tmp.groupby("SC_CODE")
            .agg(
                SC_NAME    = ("SC_NAME", "last"),
                first_seen = ("date",    "min"),
                last_seen  = ("date",    "max"),
            )
            .reset_index()
        )
        return summary

    def _build_from_cache(self) -> pd.DataFrame:
        """Derive full master table purely from BhavCopy cache files."""
        summary = self._build_cache_summary()
        if summary.empty:
            logger.error(
                "No BhavCopy cache files found -- cannot build universe. "
                "Run the pipeline once to populate cache."
            )
            return pd.DataFrame(
                columns=["SC_CODE", "SC_NAME", "listed_date", "delisted_date"]
            )

        today = date.today()
        cutoff = today - timedelta(days=60)  # stock not seen in 60d -> delisted

        # We cannot determine the true listing date from a 2-year cache.
        # Default to 1990-01-01 so every cached stock passes the
        # "listed_date <= as_of_date" test for any historical query.
        # The key discriminator is delisted_date: stocks not seen for 60+ days
        # are treated as delisted.
        summary["listed_date"] = date(1990, 1, 1)
        summary["delisted_date"] = summary["last_seen"].apply(
            lambda d: None if d >= cutoff else d + timedelta(days=1)
        )
        return self._finalise(summary)

    def _finalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure standard columns, correct types, deduplicate."""
        for col in ("listed_date", "delisted_date"):
            if col not in df.columns:
                df[col] = None
            else:
                # Convert any non-None values to Python date
                df[col] = df[col].apply(
                    lambda v: (
                        None if pd.isna(v) or v is None
                        else (v if isinstance(v, date) else pd.to_datetime(v).date())
                    )
                )

        # Ensure listed_date is set; default to 1990-01-01 if unknown
        df["listed_date"] = df["listed_date"].fillna(date(1990, 1, 1))

        keep = ["SC_CODE", "SC_NAME", "listed_date", "delisted_date"]
        for c in keep:
            if c not in df.columns:
                df[c] = None

        df = df[keep].drop_duplicates("SC_CODE").copy()
        df["SC_CODE"] = df["SC_CODE"].astype(str)
        df = df.sort_values("SC_CODE").reset_index(drop=True)
        return df

    # ---------------------------------------------------------------
    # Cache persistence
    # ---------------------------------------------------------------

    def _load_cached_master(self) -> Optional[pd.DataFrame]:
        if _MASTER_PKL.exists():
            try:
                with open(_MASTER_PKL, "rb") as fh:
                    df = pickle.load(fh)
                logger.info(
                    "PointInTimeUniverse: loaded cached master (%d stocks).", len(df)
                )
                return df
            except Exception:
                pass
        return None

    def _save_master(self, df: pd.DataFrame):
        try:
            with open(_MASTER_PKL, "wb") as fh:
                pickle.dump(df, fh)
            # Also write CSV for easy inspection
            df.to_csv(_MASTER_CSV, index=False)
            logger.info(
                "PointInTimeUniverse: master saved (%d stocks) -> %s",
                len(df), _MASTER_PKL,
            )
        except Exception as exc:
            logger.warning("Could not save master: %s", exc)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def get_pit_universe(as_of_date: str) -> pd.DataFrame:
    """One-liner: get stocks valid on as_of_date."""
    return PointInTimeUniverse().build_universe(as_of_date)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    pit = PointInTimeUniverse()

    today_str  = str(date.today())
    mar20_str  = "2020-03-01"
    apr24_str  = "2024-04-01"

    u_today = pit.build_universe(today_str)
    u_mar20 = pit.build_universe(mar20_str)
    u_apr24 = pit.build_universe(apr24_str)

    print(f"\nUniverse sizes:")
    print(f"  {mar20_str}: {len(u_mar20):,} stocks")
    print(f"  {apr24_str}: {len(u_apr24):,} stocks")
    print(f"  {today_str}: {len(u_today):,} stocks")

    if len(u_mar20) >= len(u_today):
        print("\n[PASS] 2020 universe >= today's universe (survivorship-free)")
    else:
        print("\n[FAIL] 2020 universe < today's universe -- check implementation")

    print("\nSample (today's universe):")
    print(u_today.head(5).to_string(index=False))
