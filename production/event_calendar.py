"""
event_calendar.py
=================
Gap 4 — Event Risk Management / Corporate Calendar Blackout

Prevents trading around high-impact corporate events that cause abnormal
price behaviour (earnings surprises, dividend ex-dates, bonus / split
announcements, etc.).

Architecture
------------
1. ``EventType``              — enum of all tracked event categories
2. ``CorporateEvent``         — lightweight dataclass for a single event
3. ``EventCalendarBlackout``  — main workhorse:
     - fetches events from BSE IndiaAPI (with 24-hour file-cache)
     - accepts a manually maintained CSV as a supplement
     - blackout check and pick-filtering methods
4. ``apply_event_blackout()`` — module-level convenience function

BSE API endpoint used
---------------------
  https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w
    ?scripcode={sc_code}&Category=Company%20Action&MktCap=&YearFil=

NSE corporate actions endpoint (fallback)
  https://www.nseindia.com/api/corporates-corporateActions?index=equities&symbol={symbol}

Usage
-----
>>> from production.event_calendar import apply_event_blackout
>>> import pandas as pd
>>> picks = pd.DataFrame({"sc_code": ["500325", "532540"], ...})
>>> filtered, excluded = apply_event_blackout(picks, reference_date="2024-03-15")
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional: requests for live API fetching
# ---------------------------------------------------------------------------
try:
    import requests  # type: ignore
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning(
        "requests not installed — live BSE/NSE fetching disabled. "
        "Install with: pip install requests"
    )


# ===========================================================================
# 1.  EventType enum
# ===========================================================================

class EventType(Enum):
    """Categories of corporate events tracked by this module."""
    EARNINGS = auto()
    BOARD_MEETING = auto()
    DIVIDEND_EX_DATE = auto()
    DIVIDEND_RECORD = auto()
    AGM = auto()
    BONUS = auto()
    RIGHTS = auto()
    SPLIT = auto()
    RESULTS = auto()

    @classmethod
    def from_string(cls, s: str) -> "EventType":
        """
        Parse a string (case-insensitive) to an EventType.

        Falls back to ``RESULTS`` when the string is unrecognised.
        """
        _ALIASES: Dict[str, "EventType"] = {
            "earnings": cls.EARNINGS,
            "board meeting": cls.BOARD_MEETING,
            "boardmeeting": cls.BOARD_MEETING,
            "board_meeting": cls.BOARD_MEETING,
            "dividend ex-date": cls.DIVIDEND_EX_DATE,
            "dividend ex date": cls.DIVIDEND_EX_DATE,
            "ex-date": cls.DIVIDEND_EX_DATE,
            "ex date": cls.DIVIDEND_EX_DATE,
            "dividend_ex_date": cls.DIVIDEND_EX_DATE,
            "dividend record": cls.DIVIDEND_RECORD,
            "record date": cls.DIVIDEND_RECORD,
            "dividend_record": cls.DIVIDEND_RECORD,
            "agm": cls.AGM,
            "annual general meeting": cls.AGM,
            "bonus": cls.BONUS,
            "rights": cls.RIGHTS,
            "rights issue": cls.RIGHTS,
            "split": cls.SPLIT,
            "stock split": cls.SPLIT,
            "results": cls.RESULTS,
            "quarterly results": cls.RESULTS,
            "financial results": cls.RESULTS,
        }
        key = s.strip().lower()
        return _ALIASES.get(key, cls.RESULTS)


# ===========================================================================
# 2.  CorporateEvent dataclass
# ===========================================================================

@dataclass
class CorporateEvent:
    """
    A single corporate event for one security.

    Attributes
    ----------
    sc_code : str
        BSE security code (e.g. "500325" for Reliance).
    sc_name : str
        Human-readable company name.
    event_type : EventType
    event_date : datetime
        The date on which the event takes effect (ex-date, board meeting date,
        results announcement date, etc.).
    description : str
        Raw description as received from the data source.
    source : str
        Data source identifier ("BSE_API", "NSE_API", "CSV", "MANUAL").
    """
    sc_code: str
    sc_name: str
    event_type: EventType
    event_date: datetime
    description: str = ""
    source: str = "UNKNOWN"

    def __repr__(self) -> str:
        return (
            f"CorporateEvent(sc_code={self.sc_code!r}, "
            f"event_type={self.event_type.name}, "
            f"event_date={self.event_date.date()}, "
            f"source={self.source!r})"
        )


# ===========================================================================
# 3.  EventCalendarBlackout
# ===========================================================================

# BSE corporate-action categories -> EventType mapping
_BSE_CATEGORY_MAP: Dict[str, EventType] = {
    "Board Meeting": EventType.BOARD_MEETING,
    "AGM": EventType.AGM,
    "Dividend": EventType.DIVIDEND_EX_DATE,
    "Ex-Dividend": EventType.DIVIDEND_EX_DATE,
    "Record Date": EventType.DIVIDEND_RECORD,
    "Bonus": EventType.BONUS,
    "Rights": EventType.RIGHTS,
    "Split": EventType.SPLIT,
    "Quarterly Result": EventType.RESULTS,
    "Financial Result": EventType.RESULTS,
}

_CACHE_TTL_SECONDS = 86_400  # 24 hours


class EventCalendarBlackout:
    """
    Manages a corporate-event calendar and applies blackout windows around
    high-impact events.

    A stock is in *blackout* if ``reference_date`` falls within
    [event_date - blackout_days_before, event_date + blackout_days_after]
    for any event of a tracked type.

    Parameters
    ----------
    blackout_days_before : int
        Calendar days before an event to begin the blackout (default 2).
    blackout_days_after : int
        Calendar days after an event during which the blackout continues
        (default 2).
    event_types_to_avoid : list[EventType], optional
        Event categories that trigger a blackout.
        Defaults to [EARNINGS, BOARD_MEETING, DIVIDEND_EX_DATE].
    """

    _DEFAULT_AVOID = [
        EventType.EARNINGS,
        EventType.BOARD_MEETING,
        EventType.DIVIDEND_EX_DATE,
    ]

    def __init__(
        self,
        blackout_days_before: int = 2,
        blackout_days_after: int = 2,
        event_types_to_avoid: Optional[List[EventType]] = None,
    ) -> None:
        self.blackout_days_before = blackout_days_before
        self.blackout_days_after = blackout_days_after
        self.event_types_to_avoid: List[EventType] = (
            event_types_to_avoid
            if event_types_to_avoid is not None
            else list(self._DEFAULT_AVOID)
        )
        logger.info(
            "EventCalendarBlackout init: before=%d, after=%d, avoid=%s",
            blackout_days_before,
            blackout_days_after,
            [e.name for e in self.event_types_to_avoid],
        )

    # ------------------------------------------------------------------
    # Data acquisition — BSE API
    # ------------------------------------------------------------------

    def fetch_bse_announcements(
        self,
        sc_codes: List[str],
        start_date: Any,
        end_date: Any,
        cache_dir: str = "stock_picker_data/cache/events",
    ) -> List[CorporateEvent]:
        """
        Fetch corporate-action announcements from BSE IndiaAPI.

        Results are cached as JSON files (one per sc_code) and reused for
        ``_CACHE_TTL_SECONDS`` (24 hours) to avoid hammering the endpoint.

        Parameters
        ----------
        sc_codes : list[str]
            BSE security codes to query.
        start_date, end_date : str or datetime
            Date range filter applied client-side.
        cache_dir : str
            Directory for caching raw API responses.

        Returns
        -------
        list[CorporateEvent]
        """
        start_dt = pd.Timestamp(start_date).to_pydatetime()
        end_dt = pd.Timestamp(end_date).to_pydatetime()
        os.makedirs(cache_dir, exist_ok=True)

        all_events: List[CorporateEvent] = []

        for sc_code in sc_codes:
            events = self._fetch_single_bse(sc_code, cache_dir)
            # Filter to date range
            for evt in events:
                if start_dt <= evt.event_date <= end_dt:
                    all_events.append(evt)

        logger.info(
            "fetch_bse_announcements: %d sc_codes -> %d events in range [%s, %s]",
            len(sc_codes), len(all_events),
            start_dt.date(), end_dt.date(),
        )
        return all_events

    def _fetch_single_bse(
        self, sc_code: str, cache_dir: str
    ) -> List[CorporateEvent]:
        """
        Fetch (or load from cache) BSE announcements for one sc_code.
        """
        cache_file = os.path.join(cache_dir, f"bse_{sc_code}.json")

        # Serve from cache if fresh
        if os.path.exists(cache_file):
            age = time.time() - os.path.getmtime(cache_file)
            if age < _CACHE_TTL_SECONDS:
                logger.debug(
                    "Cache hit for sc_code=%s (age=%.0fs).", sc_code, age
                )
                with open(cache_file, "r", encoding="utf-8") as fh:
                    raw = json.load(fh)
                return self._parse_bse_response(sc_code, raw)

        if not REQUESTS_AVAILABLE:
            logger.warning(
                "requests not installed — cannot fetch BSE data for sc_code=%s.",
                sc_code,
            )
            return []

        url = (
            "https://api.bseindia.com/BseIndiaAPI/api/AnnualReport/w"
            f"?scripcode={sc_code}&Category=Company%20Action&MktCap=&YearFil="
        )
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://www.bseindia.com/",
        }

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            raw = resp.json()
            # Persist to cache
            with open(cache_file, "w", encoding="utf-8") as fh:
                json.dump(raw, fh)
            logger.debug("Fetched BSE announcements for sc_code=%s.", sc_code)
            return self._parse_bse_response(sc_code, raw)

        except Exception as exc:
            logger.warning(
                "BSE API fetch failed for sc_code=%s: %s", sc_code, exc
            )
            return []

    def _parse_bse_response(
        self, sc_code: str, raw: Any
    ) -> List[CorporateEvent]:
        """
        Parse a BSE API response JSON into a list of CorporateEvent objects.

        BSE response structure (observed):
        {
          "Table": [
            {
              "short_name": "RELIANCE",
              "Desc": "Board Meeting-To consider dividend",
              "Ex_Date": "20240315",
              "CATEGORY": "Board Meeting"
            }, ...
          ]
        }
        """
        events: List[CorporateEvent] = []

        if not isinstance(raw, dict):
            return events

        rows = raw.get("Table") or raw.get("table") or []
        if not isinstance(rows, list):
            return events

        for row in rows:
            try:
                # Date field can be 'Ex_Date', 'Dt', 'date', etc.
                date_str = (
                    row.get("Ex_Date")
                    or row.get("Dt")
                    or row.get("date")
                    or row.get("Date")
                    or ""
                )
                if not date_str:
                    continue

                event_date = pd.to_datetime(date_str, dayfirst=False, errors="coerce")
                if pd.isna(event_date):
                    continue

                category_str = (
                    row.get("CATEGORY") or row.get("category") or "Results"
                )
                event_type = _BSE_CATEGORY_MAP.get(
                    category_str, EventType.from_string(category_str)
                )

                sc_name = (
                    row.get("short_name")
                    or row.get("ShortName")
                    or row.get("sc_name")
                    or sc_code
                )
                description = row.get("Desc") or row.get("desc") or ""

                events.append(
                    CorporateEvent(
                        sc_code=str(sc_code),
                        sc_name=str(sc_name),
                        event_type=event_type,
                        event_date=event_date.to_pydatetime(),
                        description=str(description),
                        source="BSE_API",
                    )
                )
            except Exception as exc:
                logger.debug("Skipping row (parse error): %s — %s", row, exc)

        return events

    # ------------------------------------------------------------------
    # Data acquisition — NSE earnings calendar
    # ------------------------------------------------------------------

    def _fetch_nse_results_calendar(
        self, symbols: List[str], cache_dir: str
    ) -> List[CorporateEvent]:
        """
        Attempt to fetch upcoming earnings dates from NSE corporate actions.

        NSE endpoint:
          GET https://www.nseindia.com/api/corporates-corporateActions
               ?index=equities&symbol={symbol}

        Returns an empty list on failure (NSE blocks non-browser sessions).
        """
        if not REQUESTS_AVAILABLE:
            return []

        events: List[CorporateEvent] = []
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.nseindia.com/",
            }
        )

        # Prime session cookies
        try:
            session.get("https://www.nseindia.com", timeout=8)
        except Exception:
            return []

        for symbol in symbols:
            cache_file = os.path.join(cache_dir, f"nse_{symbol}.json")
            raw = None

            if os.path.exists(cache_file):
                age = time.time() - os.path.getmtime(cache_file)
                if age < _CACHE_TTL_SECONDS:
                    with open(cache_file, "r", encoding="utf-8") as fh:
                        raw = json.load(fh)

            if raw is None:
                url = (
                    "https://www.nseindia.com/api/corporates-corporateActions"
                    f"?index=equities&symbol={symbol}"
                )
                try:
                    resp = session.get(url, timeout=10)
                    resp.raise_for_status()
                    raw = resp.json()
                    with open(cache_file, "w", encoding="utf-8") as fh:
                        json.dump(raw, fh)
                except Exception as exc:
                    logger.debug(
                        "NSE fetch failed for symbol=%s: %s", symbol, exc
                    )
                    continue

            # Parse NSE response
            rows = raw if isinstance(raw, list) else (raw.get("data") or [])
            for row in rows:
                try:
                    date_str = row.get("exDate") or row.get("recordDate") or ""
                    if not date_str:
                        continue
                    event_date = pd.to_datetime(
                        date_str, dayfirst=True, errors="coerce"
                    )
                    if pd.isna(event_date):
                        continue

                    purpose = str(row.get("subject") or row.get("purpose") or "")
                    event_type = EventType.from_string(purpose)

                    events.append(
                        CorporateEvent(
                            sc_code=symbol,
                            sc_name=str(row.get("comp") or symbol),
                            event_type=event_type,
                            event_date=event_date.to_pydatetime(),
                            description=purpose,
                            source="NSE_API",
                        )
                    )
                except Exception as exc:
                    logger.debug(
                        "NSE row parse error (symbol=%s): %s", symbol, exc
                    )

        return events

    # ------------------------------------------------------------------
    # Data acquisition — Manual CSV
    # ------------------------------------------------------------------

    def load_from_csv(self, csv_path: str) -> List[CorporateEvent]:
        """
        Load a manually maintained event calendar from a CSV file.

        Expected columns (case-insensitive):
        sc_code, sc_name, event_type, event_date, description [, source]

        The ``event_date`` column is parsed with pandas to_datetime.
        The ``event_type`` column is parsed by ``EventType.from_string``.

        Parameters
        ----------
        csv_path : str
            Path to the CSV file.

        Returns
        -------
        list[CorporateEvent]
        """
        if not os.path.exists(csv_path):
            logger.warning("CSV event file not found: %s", csv_path)
            return []

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            logger.error("Failed to read CSV %s: %s", csv_path, exc)
            return []

        # Normalise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        required = {"sc_code", "event_type", "event_date"}
        missing = required - set(df.columns)
        if missing:
            logger.error(
                "CSV %s is missing required columns: %s", csv_path, missing
            )
            return []

        events: List[CorporateEvent] = []
        for _, row in df.iterrows():
            try:
                event_date = pd.to_datetime(
                    row["event_date"], dayfirst=True, errors="coerce"
                )
                if pd.isna(event_date):
                    continue
                events.append(
                    CorporateEvent(
                        sc_code=str(row["sc_code"]).strip(),
                        sc_name=str(row.get("sc_name", row["sc_code"])).strip(),
                        event_type=EventType.from_string(str(row["event_type"])),
                        event_date=event_date.to_pydatetime(),
                        description=str(row.get("description", "")),
                        source=str(row.get("source", "CSV")),
                    )
                )
            except Exception as exc:
                logger.debug("Skipping CSV row (error): %s — %s", row, exc)

        logger.info(
            "load_from_csv(%s): loaded %d events.", csv_path, len(events)
        )
        return events

    # ------------------------------------------------------------------
    # Blackout logic
    # ------------------------------------------------------------------

    def is_in_blackout(
        self,
        sc_code: str,
        reference_date: Any,
        events: Optional[List[CorporateEvent]] = None,
    ) -> bool:
        """
        Return True if *reference_date* falls within the blackout window of
        any relevant event for *sc_code*.

        Parameters
        ----------
        sc_code : str
        reference_date : str or datetime
        events : list[CorporateEvent], optional
            Pre-fetched event list.  If None, returns False (no events known).

        Returns
        -------
        bool
        """
        if not events:
            return False

        ref_dt = pd.Timestamp(reference_date).to_pydatetime().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        before_window = timedelta(days=self.blackout_days_before)
        after_window = timedelta(days=self.blackout_days_after)

        for evt in events:
            if str(evt.sc_code) != str(sc_code):
                continue
            if evt.event_type not in self.event_types_to_avoid:
                continue
            event_dt = evt.event_date.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            window_start = event_dt - before_window
            window_end = event_dt + after_window
            if window_start <= ref_dt <= window_end:
                logger.info(
                    "BLACKOUT: sc_code=%s is within %d/%d days of %s (%s) on %s.",
                    sc_code, self.blackout_days_before, self.blackout_days_after,
                    evt.event_type.name, evt.description, event_dt.date(),
                )
                return True
        return False

    # ------------------------------------------------------------------
    # Pick filtering
    # ------------------------------------------------------------------

    def filter_picks(
        self,
        picks_df: pd.DataFrame,
        reference_date: Any,
        events: Optional[List[CorporateEvent]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove stocks that are in an event blackout window.

        Parameters
        ----------
        picks_df : pd.DataFrame
            Stock picks.  Must contain a ``sc_code`` column (or ``SC_CODE``).
        reference_date : str or datetime
        events : list[CorporateEvent], optional
            If None, returns picks_df unchanged (no events known).

        Returns
        -------
        (filtered_df, excluded_df) : tuple[pd.DataFrame, pd.DataFrame]
            ``filtered_df``  — picks that are NOT in blackout.
            ``excluded_df``  — picks that ARE in blackout (with
                               ``blackout_reason`` column appended).
        """
        if picks_df.empty:
            logger.info("filter_picks: empty picks_df — nothing to filter.")
            return picks_df.copy(), pd.DataFrame(columns=picks_df.columns)

        # Normalise sc_code column name
        col = "sc_code"
        if col not in picks_df.columns:
            if "SC_CODE" in picks_df.columns:
                picks_df = picks_df.rename(columns={"SC_CODE": "sc_code"})
            else:
                logger.warning(
                    "filter_picks: no sc_code / SC_CODE column found. "
                    "Returning picks unchanged."
                )
                return picks_df.copy(), pd.DataFrame(columns=picks_df.columns)

        if not events:
            logger.info("filter_picks: no events provided — returning all picks.")
            return picks_df.copy(), pd.DataFrame(columns=picks_df.columns)

        in_blackout_mask = picks_df["sc_code"].astype(str).apply(
            lambda code: self.is_in_blackout(code, reference_date, events)
        )

        filtered_df = picks_df[~in_blackout_mask].copy()
        excluded_df = picks_df[in_blackout_mask].copy()
        if not excluded_df.empty:
            excluded_df["blackout_reason"] = "event_blackout"

        logger.info(
            "filter_picks(%s): %d picks -> %d kept, %d excluded.",
            reference_date,
            len(picks_df),
            len(filtered_df),
            len(excluded_df),
        )
        return filtered_df, excluded_df

    # ------------------------------------------------------------------
    # Upcoming event lookup
    # ------------------------------------------------------------------

    def get_upcoming_events(
        self,
        sc_codes: List[str],
        reference_date: Any,
        days_ahead: int = 5,
        events: Optional[List[CorporateEvent]] = None,
    ) -> List[CorporateEvent]:
        """
        Return all events for *sc_codes* that fall within the next *days_ahead*
        calendar days from *reference_date*.

        Parameters
        ----------
        sc_codes : list[str]
        reference_date : str or datetime
        days_ahead : int
            Number of calendar days to look forward (default 5).
        events : list[CorporateEvent], optional
            If None, returns empty list.

        Returns
        -------
        list[CorporateEvent]
            Sorted by event_date ascending.
        """
        if not events:
            return []

        ref_dt = pd.Timestamp(reference_date).to_pydatetime().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        horizon = ref_dt + timedelta(days=days_ahead)
        sc_set = {str(s) for s in sc_codes}

        upcoming = [
            evt
            for evt in events
            if str(evt.sc_code) in sc_set
            and ref_dt <= evt.event_date.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) <= horizon
        ]
        upcoming.sort(key=lambda e: e.event_date)
        logger.info(
            "get_upcoming_events(%s, +%dd): %d events found.",
            reference_date, days_ahead, len(upcoming),
        )
        return upcoming

    # ------------------------------------------------------------------
    # Calendar builder
    # ------------------------------------------------------------------

    def build_event_calendar(
        self,
        sc_codes: List[str],
        start_date: Any,
        end_date: Any,
        cache_dir: str = "stock_picker_data/cache/events",
    ) -> pd.DataFrame:
        """
        Build a consolidated event calendar DataFrame for the given codes and
        date range, merging BSE API data.

        Parameters
        ----------
        sc_codes : list[str]
        start_date, end_date : str or datetime
        cache_dir : str

        Returns
        -------
        pd.DataFrame
            Columns: sc_code, sc_name, event_type, event_date, description, source.
            Sorted by event_date ascending.
        """
        events = self.fetch_bse_announcements(
            sc_codes, start_date, end_date, cache_dir=cache_dir
        )

        if not events:
            logger.warning(
                "build_event_calendar: no events fetched — returning empty DataFrame."
            )
            return pd.DataFrame(
                columns=[
                    "sc_code", "sc_name", "event_type",
                    "event_date", "description", "source",
                ]
            )

        rows = [
            {
                "sc_code": e.sc_code,
                "sc_name": e.sc_name,
                "event_type": e.event_type.name,
                "event_date": e.event_date,
                "description": e.description,
                "source": e.source,
            }
            for e in events
        ]
        df = pd.DataFrame(rows).sort_values("event_date").reset_index(drop=True)
        logger.info(
            "build_event_calendar: %d total events for %d codes.",
            len(df), len(sc_codes),
        )
        return df


# ===========================================================================
# 4.  Module-level convenience function
# ===========================================================================

def apply_event_blackout(
    picks_df: pd.DataFrame,
    reference_date: Any,
    events: Optional[List[CorporateEvent]] = None,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper — instantiate ``EventCalendarBlackout`` and filter.

    Parameters
    ----------
    picks_df : pd.DataFrame
        Stock picks with a ``sc_code`` (or ``SC_CODE``) column.
    reference_date : str or datetime
    events : list[CorporateEvent], optional
        Pre-fetched events.  If None, the function has no events to act on
        and returns picks_df unchanged.
    **kwargs
        Passed directly to ``EventCalendarBlackout.__init__``.
        Valid keys: blackout_days_before, blackout_days_after,
        event_types_to_avoid.

    Returns
    -------
    (filtered_df, excluded_df) : tuple[pd.DataFrame, pd.DataFrame]

    Examples
    --------
    >>> events = cal.fetch_bse_announcements(sc_codes, "2024-01-01", "2024-06-30")
    >>> filtered, excluded = apply_event_blackout(picks, "2024-03-15", events=events)
    """
    blackout = EventCalendarBlackout(**kwargs)
    return blackout.filter_picks(picks_df, reference_date, events=events)


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    print("=" * 70)
    print("event_calendar.py — self-test")
    print("=" * 70)

    # --- Build synthetic events ------------------------------------------
    from datetime import date

    ref_date = date.today()
    synthetic_events = [
        CorporateEvent(
            sc_code="500325",
            sc_name="Reliance Industries",
            event_type=EventType.BOARD_MEETING,
            event_date=datetime.combine(
                ref_date + timedelta(days=1), datetime.min.time()
            ),
            description="Board meeting to consider Q4 results",
            source="TEST",
        ),
        CorporateEvent(
            sc_code="532540",
            sc_name="TCS",
            event_type=EventType.DIVIDEND_EX_DATE,
            event_date=datetime.combine(
                ref_date + timedelta(days=3), datetime.min.time()
            ),
            description="Ex-date for interim dividend",
            source="TEST",
        ),
        CorporateEvent(
            sc_code="500180",
            sc_name="HDFC Bank",
            event_type=EventType.EARNINGS,
            event_date=datetime.combine(
                ref_date + timedelta(days=7), datetime.min.time()
            ),
            description="Q4 FY24 results",
            source="TEST",
        ),
    ]

    print(f"\nReference date: {ref_date}")
    print(f"Synthetic events: {len(synthetic_events)}")

    # --- is_in_blackout --------------------------------------------------
    cal = EventCalendarBlackout(blackout_days_before=2, blackout_days_after=2)
    for sc in ["500325", "532540", "500180", "543396"]:
        flag = cal.is_in_blackout(sc, ref_date, events=synthetic_events)
        print(f"  sc_code={sc} in blackout: {flag}")

    # --- filter_picks ----------------------------------------------------
    picks = pd.DataFrame(
        {
            "sc_code": ["500325", "532540", "500180", "543396"],
            "Probability": [0.72, 0.68, 0.75, 0.65],
        }
    )
    filtered, excluded = cal.filter_picks(picks, ref_date, events=synthetic_events)
    print(f"\nFiltered picks ({len(filtered)}):")
    print(filtered.to_string(index=False))
    print(f"\nExcluded picks ({len(excluded)}):")
    print(excluded.to_string(index=False))

    # --- get_upcoming_events ---------------------------------------------
    upcoming = cal.get_upcoming_events(
        ["500325", "532540", "500180"],
        ref_date,
        days_ahead=5,
        events=synthetic_events,
    )
    print(f"\nUpcoming events (next 5 days): {len(upcoming)}")
    for e in upcoming:
        print(f"  {e}")

    # --- apply_event_blackout convenience --------------------------------
    f2, e2 = apply_event_blackout(picks, ref_date, events=synthetic_events)
    print(f"\napply_event_blackout: kept={len(f2)}, excluded={len(e2)}")

    print("\nSelf-test complete.")
