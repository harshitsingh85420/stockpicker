"""
CorporateActionsFetcher — free NSE/BSE corporate actions data.

Sources used (all free, no login required):
  1. NSE JSON API  — https://www.nseindia.com/api/corporates-corporateActions
  2. BSE CSV download — https://www.bseindia.com/corporates/download/...
  3. Local CSV fallback — stock_picker_data/corporate_actions.csv

Output schema (CSV / DataFrame):
    SC_CODE      str   — BSE script code or NSE symbol
    SC_NAME      str   — company name
    EX_DATE      date  — ex-date (when the adjustment takes effect)
    ACTION_TYPE  str   — SPLIT / BONUS / DIVIDEND / RIGHTS / AGM
    RATIO        float — e.g. 2.0 for a 2:1 split, 1.0 for 1:1 bonus
    BONUS_DENOM  float — for bonus M:N, RATIO=M, BONUS_DENOM=N (N+M shares after)
    SOURCE       str   — NSE / BSE / LOCAL
"""

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests

# BSE Python package (PyPI: bse) — direct BSE India API, no scraping
try:
    from bse import BSE as _BSEClient
    BSE_PKG_AVAILABLE = True
except ImportError:
    BSE_PKG_AVAILABLE = False
    logger_init = logging.getLogger(__name__)
    logger_init.warning("bse package not installed — run: pip install bse")

logger = logging.getLogger(__name__)

_CACHE_FILE = Path("stock_picker_data/corporate_actions.csv")
_CACHE_MAX_AGE_HOURS = 12   # re-fetch if cache is older than this


class CorporateActionsFetcher:
    """
    Fetch and cache corporate actions from free NSE/BSE endpoints.

    Usage
    -----
        fetcher = CorporateActionsFetcher()
        df = fetcher.get(start_date="2024-01-01", end_date="2026-04-23")
        fetcher.save_to_cache(df)            # write to disk
    """

    NSE_API = "https://www.nseindia.com/api/corporates-corporateActions"
    NSE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions",
        "Connection": "keep-alive",
    }
    NSE_BASE = "https://www.nseindia.com"

    # Action type normalisation map
    _ACTION_MAP = {
        "split": "SPLIT",
        "stock split": "SPLIT",
        "sub-division": "SPLIT",
        "bonus": "BONUS",
        "bonus shares": "BONUS",
        "dividend": "DIVIDEND",
        "interim dividend": "DIVIDEND",
        "final dividend": "DIVIDEND",
        "rights": "RIGHTS",
        "rights issue": "RIGHTS",
        "agm": "AGM",
        "annual general meeting": "AGM",
    }

    def __init__(
        self,
        cache_file: str = None,
        cache_max_age_hours: int = _CACHE_MAX_AGE_HOURS,
    ):
        self.cache_file = Path(cache_file) if cache_file else _CACHE_FILE
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_max_age_hours = cache_max_age_hours
        self._session = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self,
        start_date: str = None,
        end_date: str = None,
        action_types: list = None,
        use_cache: bool = True,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return corporate actions DataFrame for the given date range.

        Falls back gracefully: NSE API -> BSE CSV -> local cache -> empty.

        Args:
            start_date: 'YYYY-MM-DD'; defaults to 2 years ago.
            end_date:   'YYYY-MM-DD'; defaults to today + 30 days (upcoming events).
            action_types: list of 'SPLIT', 'BONUS', 'DIVIDEND', 'RIGHTS'. None = all.
            use_cache:   Load from local cache if fresh.
            force_refresh: Skip cache and always fetch from network.

        Returns:
            DataFrame with columns [SC_CODE, SC_NAME, EX_DATE, ACTION_TYPE, RATIO, BONUS_DENOM, SOURCE].
        """
        end_dt   = pd.to_datetime(end_date).date()   if end_date   else date.today() + timedelta(days=30)
        start_dt = pd.to_datetime(start_date).date() if start_date else date.today() - timedelta(days=730)

        # Try cache first
        if use_cache and not force_refresh and self._cache_is_fresh():
            df = self._load_cache()
            if df is not None and not df.empty:
                logger.info("Corporate actions loaded from cache (%d records).", len(df))
                return self._filter(df, start_dt, end_dt, action_types)

        # P28: Try BSE Python package first (most reliable — direct BSE API)
        df = None
        if BSE_PKG_AVAILABLE:
            logger.info("P28: fetching corporate actions via bse Python package …")
            df = None   # Per-stock fetch happens in fetch_for_universe(); here use legacy flow
            # Fall through to NSE/BSE HTTP as general fetcher; universe-specific
            # batch is in fetch_for_universe() called by the orchestrator.

        # Try NSE API
        if df is None or (df is not None and len(df) < 10):
            df = self._fetch_nse(start_dt, end_dt)

        # Fallback to BSE HTTP if NSE failed or returned little data
        if df is None or len(df) < 10:
            logger.info("NSE returned little data; trying BSE HTTP fallback …")
            bse_df = self._fetch_bse(start_dt, end_dt)
            if bse_df is not None and not bse_df.empty:
                df = pd.concat([df, bse_df], ignore_index=True) if df is not None else bse_df

        # Last resort: local cache even if stale
        if df is None or df.empty:
            logger.warning("Network fetch failed; loading stale cache.")
            df = self._load_cache()

        if df is None or df.empty:
            logger.warning("No corporate actions data available — skipping adjustments.")
            return pd.DataFrame(columns=["SC_CODE", "SC_NAME", "EX_DATE",
                                         "ACTION_TYPE", "RATIO", "BONUS_DENOM", "SOURCE"])

        df = self._deduplicate(df)
        self._save_cache(df)
        return self._filter(df, start_dt, end_dt, action_types)

    # ------------------------------------------------------------------
    # NSE fetch
    # ------------------------------------------------------------------

    def _nse_session(self) -> requests.Session:
        """Create a session with NSE cookies (required to bypass bot detection)."""
        if self._session is not None:
            return self._session
        s = requests.Session()
        s.headers.update(self.NSE_HEADERS)
        try:
            # Prime cookies by visiting the main page
            r = s.get(self.NSE_BASE, timeout=15)
            r.raise_for_status()
            time.sleep(0.5)
        except Exception as e:
            logger.debug("NSE session priming failed (harmless): %s", e)
        self._session = s
        return s

    def _fetch_nse(self, start_dt: date, end_dt: date) -> Optional[pd.DataFrame]:
        """Fetch from NSE JSON API."""
        params = {
            "index": "equities",
            "from_date": start_dt.strftime("%d-%m-%Y"),
            "to_date": end_dt.strftime("%d-%m-%Y"),
        }
        try:
            session = self._nse_session()
            resp = session.get(self.NSE_API, params=params, timeout=20)
            resp.raise_for_status()
            raw = resp.json()

            if not isinstance(raw, list) or len(raw) == 0:
                logger.debug("NSE API returned no records.")
                return None

            rows = []
            for item in raw:
                ex_date_str = item.get("exDate") or item.get("ex_date") or ""
                symbol      = item.get("symbol") or item.get("sym") or ""
                company     = item.get("companyName") or item.get("company") or symbol
                purpose     = (item.get("purpose") or item.get("action") or "").lower()
                series      = item.get("series", "EQ")

                if series not in ("EQ", "BE", ""):
                    continue  # skip F&O-only series

                try:
                    ex_date = datetime.strptime(ex_date_str, "%d-%b-%Y").date()
                except Exception:
                    try:
                        ex_date = pd.to_datetime(ex_date_str).date()
                    except Exception:
                        continue

                action_type, ratio, bonus_denom = self._parse_purpose(purpose)
                if action_type is None:
                    continue

                rows.append({
                    "SC_CODE":     symbol,
                    "SC_NAME":     company,
                    "EX_DATE":     ex_date,
                    "ACTION_TYPE": action_type,
                    "RATIO":       ratio,
                    "BONUS_DENOM": bonus_denom,
                    "SOURCE":      "NSE",
                })

            if not rows:
                return None

            df = pd.DataFrame(rows)
            logger.info("NSE API: fetched %d corporate actions.", len(df))
            return df

        except Exception as e:
            logger.warning("NSE corporate actions fetch failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # BSE Python package fetch (P28 — primary source)
    # ------------------------------------------------------------------

    def _fetch_bse_package(
        self, sc_codes: Optional[List[str]] = None, start_dt: Optional[date] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch corporate actions per stock using the `bse` PyPI package.
        Direct BSE India API — no URL scraping, no NSE dependency.

        Parameters
        ----------
        sc_codes : list of BSE scrip codes; if None fetches for an empty set
                   (returns empty).
        start_dt : Filter out actions before this date.

        Returns
        -------
        DataFrame or None
        """
        if not BSE_PKG_AVAILABLE:
            logger.warning("P28: bse package unavailable — cannot use direct BSE API.")
            return None
        if not sc_codes:
            return None

        bse_cache_dir = Path("tmp/bse_cache")
        bse_cache_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for sc_code in sc_codes:
            try:
                with _BSEClient(download_folder=str(bse_cache_dir)) as bse_client:
                    data = bse_client.actions(scripcode=str(sc_code))

                if not data or "Table" not in data:
                    continue

                for row in data["Table"]:
                    ex_date_str = str(row.get("ExDate") or "").strip()
                    purpose     = str(row.get("Purpose") or "").strip()
                    try:
                        ex_date = pd.to_datetime(ex_date_str).date()
                    except Exception:
                        continue

                    if start_dt and ex_date < start_dt:
                        continue

                    purpose_upper = purpose.upper()
                    action_type, ratio, bonus_denom = self._parse_purpose(purpose.lower())
                    if action_type is None:
                        continue

                    rows.append({
                        "SC_CODE":     str(sc_code),
                        "SC_NAME":     str(row.get("ShortName") or sc_code),
                        "EX_DATE":     ex_date,
                        "ACTION_TYPE": action_type,
                        "RATIO":       ratio,
                        "BONUS_DENOM": bonus_denom,
                        "SOURCE":      "BSE_PKG",
                    })

            except Exception as exc:
                logger.warning(
                    "P28: BSE package fetch failed for %s: %s — skipping.", sc_code, exc
                )
                continue

        if not rows:
            return None

        df = pd.DataFrame(rows)
        logger.info("P28 BSE package: fetched %d corporate actions for %d stocks.",
                    len(df), len(sc_codes))
        return df

    def fetch_for_universe(
        self,
        sc_codes: List[str],
        cache_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Batch-fetch corporate actions for a full tradable universe using the
        BSE Python package.  Results are cached to a date-stamped JSON file
        (valid for 1 calendar day).

        Parameters
        ----------
        sc_codes : List of BSE scrip codes for all tradable stocks.
        cache_date : 'YYYY-MM-DD'; defaults to today.

        Returns
        -------
        DataFrame with corporate actions for the universe.
        """
        today = cache_date or date.today().isoformat()
        cache_path = self.cache_file.parent / f"corporate_actions_{today}.json"

        # Load daily cache if it exists (valid for 1 calendar day)
        if cache_path.exists():
            try:
                with open(cache_path, encoding="utf-8") as fh:
                    cached = json.load(fh)
                df = pd.DataFrame(cached)
                df["EX_DATE"] = pd.to_datetime(df["EX_DATE"]).dt.date
                logger.info(
                    "P28: corporate actions cache hit — %d records (%s)", len(df), today
                )
                return df
            except Exception as exc:
                logger.warning("P28: daily cache load failed (%s) — re-fetching.", exc)

        start_dt = date.today() - timedelta(days=730)
        df = self._fetch_bse_package(sc_codes, start_dt=start_dt)

        if df is None or df.empty:
            logger.warning(
                "P28: BSE package returned no data for %d stocks.", len(sc_codes)
            )
            return pd.DataFrame(
                columns=["SC_CODE", "SC_NAME", "EX_DATE", "ACTION_TYPE",
                         "RATIO", "BONUS_DENOM", "SOURCE"]
            )

        # Persist daily cache
        try:
            records = df.copy()
            records["EX_DATE"] = records["EX_DATE"].astype(str)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(records.to_dict("records"), fh)
            logger.info(
                "P28: cached %d corporate actions -> %s", len(df), cache_path
            )
        except Exception as exc:
            logger.warning("P28: daily cache save failed: %s", exc)

        return df

    # ------------------------------------------------------------------
    # BSE HTTP fallback fetch (legacy)
    # ------------------------------------------------------------------

    def _fetch_bse(self, start_dt: date, end_dt: date) -> Optional[pd.DataFrame]:
        """Fetch from BSE corporate actions page (CSV download)."""
        # BSE publishes dividend / corporate actions in a structured endpoint
        url = (
            "https://api.bseindia.com/BseIndiaAPI/api/CorporateAction/w"
            "?scripcode=&Group=&strSearch=P"
            f"&from_date={start_dt.strftime('%Y%m%d')}"
            f"&to_date={end_dt.strftime('%Y%m%d')}"
            "&type=div"
        )
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.bseindia.com/",
            "Accept": "application/json",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            if not data or not isinstance(data, dict):
                return None

            table = data.get("Table") or data.get("data") or []
            if not table:
                return None

            rows = []
            for item in table:
                sc_code = str(item.get("SCRIP_CD") or item.get("scripcode") or "")
                sc_name = str(item.get("LONG_NAME") or item.get("companyname") or "")
                ex_date_str = str(item.get("EX_DATE") or item.get("ex_date") or "")
                purpose = str(item.get("PURPOSE") or item.get("purpose") or "").lower()

                try:
                    ex_date = pd.to_datetime(ex_date_str).date()
                except Exception:
                    continue

                action_type, ratio, bonus_denom = self._parse_purpose(purpose)
                if action_type is None:
                    continue

                rows.append({
                    "SC_CODE":     sc_code,
                    "SC_NAME":     sc_name,
                    "EX_DATE":     ex_date,
                    "ACTION_TYPE": action_type,
                    "RATIO":       ratio,
                    "BONUS_DENOM": bonus_denom,
                    "SOURCE":      "BSE",
                })

            df = pd.DataFrame(rows) if rows else None
            if df is not None:
                logger.info("BSE API: fetched %d corporate actions.", len(df))
            return df

        except Exception as e:
            logger.debug("BSE corporate actions fetch failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Parse purpose string -> (action_type, ratio, bonus_denom)
    # ------------------------------------------------------------------

    def _parse_purpose(self, purpose: str):
        """
        Parse a free-text purpose string into structured fields.

        Returns:
            (action_type, ratio, bonus_denom) or (None, None, None) if not a
            price-adjusting event.

        Examples:
            "face value split from rs 10 to rs 5"      -> (SPLIT, 2.0, None)
            "bonus 1:1"                                 -> (BONUS, 1.0, 1.0)
            "bonus issue in the ratio of 2 equity shares for every 3" -> (BONUS, 2.0, 3.0)
            "interim dividend rs 3 per share"           -> (DIVIDEND, 3.0, None)
        """
        import re

        p = purpose.lower().strip()

        # ── SPLIT ─────────────────────────────────────────────────────
        if "split" in p or "sub-division" in p:
            # "from rs 10 to rs 5" -> ratio 2
            m = re.search(r"from\s+(?:rs\.?\s*)?(\d+\.?\d*)\s+to\s+(?:rs\.?\s*)?(\d+\.?\d*)", p)
            if m:
                old_fv, new_fv = float(m.group(1)), float(m.group(2))
                ratio = old_fv / new_fv if new_fv > 0 else 1.0
                return "SPLIT", round(ratio, 4), None
            # "2:1" or "1:2"
            m = re.search(r"(\d+)\s*:\s*(\d+)", p)
            if m:
                a, b = int(m.group(1)), int(m.group(2))
                ratio = b / a if a > 0 else 1.0   # new shares per old share
                return "SPLIT", round(ratio, 4), None
            return "SPLIT", 1.0, None   # unknown ratio

        # ── BONUS ─────────────────────────────────────────────────────
        if "bonus" in p:
            # "bonus 1:1" or "bonus 2:3"
            m = re.search(r"(\d+)\s*:\s*(\d+)", p)
            if m:
                bonus_shares, held = int(m.group(1)), int(m.group(2))
                return "BONUS", float(bonus_shares), float(held)
            # "1 bonus share for every 2"
            m = re.search(r"(\d+)\s+bonus.*?(?:for every|per)\s+(\d+)", p)
            if m:
                return "BONUS", float(m.group(1)), float(m.group(2))
            return "BONUS", 1.0, 1.0   # assume 1:1

        # ── DIVIDEND ──────────────────────────────────────────────────
        if "dividend" in p:
            m = re.search(r"rs\.?\s*(\d+\.?\d*)", p)
            ratio = float(m.group(1)) if m else 0.0
            return "DIVIDEND", ratio, None

        # ── RIGHTS ────────────────────────────────────────────────────
        if "rights" in p:
            m = re.search(r"(\d+)\s*:\s*(\d+)", p)
            if m:
                return "RIGHTS", float(m.group(1)), float(m.group(2))
            return "RIGHTS", 1.0, None

        return None, None, None   # not a price-adjusting event

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_is_fresh(self) -> bool:
        if not self.cache_file.exists():
            return False
        age_hours = (time.time() - self.cache_file.stat().st_mtime) / 3600
        return age_hours <= self.cache_max_age_hours

    def _load_cache(self) -> Optional[pd.DataFrame]:
        if not self.cache_file.exists():
            return None
        try:
            df = pd.read_csv(self.cache_file, parse_dates=["EX_DATE"])
            df["EX_DATE"] = pd.to_datetime(df["EX_DATE"]).dt.date
            return df
        except Exception as e:
            logger.warning("Cache load failed: %s", e)
            return None

    def _save_cache(self, df: pd.DataFrame):
        try:
            df.to_csv(self.cache_file, index=False)
            logger.info("Corporate actions cached -> %s (%d records).", self.cache_file, len(df))
        except Exception as e:
            logger.warning("Cache save failed: %s", e)

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(
            subset=["SC_CODE", "EX_DATE", "ACTION_TYPE"]
        ).reset_index(drop=True)

    def _filter(
        self,
        df: pd.DataFrame,
        start_dt: date,
        end_dt: date,
        action_types: Optional[list],
    ) -> pd.DataFrame:
        df["EX_DATE"] = pd.to_datetime(df["EX_DATE"]).dt.date
        mask = (df["EX_DATE"] >= start_dt) & (df["EX_DATE"] <= end_dt)
        if action_types:
            mask &= df["ACTION_TYPE"].isin([a.upper() for a in action_types])
        return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def fetch_corporate_actions(
    start_date: str = None,
    end_date: str = None,
    cache_file: str = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """One-liner convenience wrapper."""
    return CorporateActionsFetcher(cache_file=cache_file).get(
        start_date=start_date,
        end_date=end_date,
        force_refresh=force_refresh,
    )


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print("Fetching corporate actions for last 90 days …")
    df = fetch_corporate_actions(
        start_date=(date.today() - timedelta(days=90)).isoformat(),
        end_date=date.today().isoformat(),
    )
    if df.empty:
        print("No data returned (network may be unavailable).")
    else:
        print(df[["SC_CODE", "SC_NAME", "EX_DATE", "ACTION_TYPE", "RATIO"]].head(20).to_string(index=False))
        print(f"\nTotal records: {len(df)}")
        print("Action type distribution:\n", df["ACTION_TYPE"].value_counts().to_string())
