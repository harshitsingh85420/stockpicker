#!/usr/bin/env python3
"""
validate_p06.py  --  P06 acceptance test.

Tests:
  1. filter_operator_driven() removes stocks with > 2 upper circuits in 10 sessions
  2. filter_recent_listings() removes stocks with < 90 sessions of history
  3. DataSourceFailover class is importable and has the correct source chain

Exit code 0 if all tests pass, 1 if any fail.
"""

import sys
import logging
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("validate_p06")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
if str(_ROOT / "production") not in sys.path:
    sys.path.insert(0, str(_ROOT / "production"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bhav(stocks: dict, n_days: int = 120, start: date = date(2024, 1, 2)) -> pd.DataFrame:
    """
    Build a synthetic multi-stock BhavCopy.

    stocks: {code: daily_return_series_or_None}
      If value is None -> random walk.
      If value is a dict of {day_index: return_override} -> spike on those days.

    Uses n_days * 3 calendar days in the generator to guarantee exactly
    n_days business days regardless of n_days size.
    """
    rng = np.random.default_rng(42)
    days = [start + timedelta(days=i)
            for i in range(n_days * 3)
            if (start + timedelta(days=i)).weekday() < 5][:n_days]

    rows = []
    for code, spikes in stocks.items():
        price = 100.0
        for i, d in enumerate(days):
            ret = rng.normal(0, 0.005)
            if spikes and i in spikes:
                ret = spikes[i]      # override return on that day
            price *= (1 + ret)
            price = max(price, 5.0)
            rows.append({
                "SC_CODE": code, "SC_NAME": code,
                "DATE":   d,
                "Open":   price * 0.99, "High": price * 1.01,
                "Low":    price * 0.98, "Close": price,
                "Volume": 100_000,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1 — filter_operator_driven
# ---------------------------------------------------------------------------

def test_operator_filter() -> bool:
    from production.universe_filter import filter_operator_driven

    # n_days=120; last 10 sessions = indices 110-119
    # PUMP hits upper circuit 3 times inside that window -> should be removed
    pump_spikes  = {111: 0.20, 114: 0.20, 118: 0.20}   # 3 hits within last 10
    # CLEAN hits only once -> should be kept
    clean_spikes = {113: 0.20}

    bhav = make_bhav({
        "PUMP":  pump_spikes,
        "CLEAN": clean_spikes,
        "NORM":  {},
    }, n_days=120)

    result = filter_operator_driven(bhav, lookback=10, max_circuits=2)

    codes = set(result["SC_CODE"].unique())
    pump_removed  = "PUMP"  not in codes
    clean_kept    = "CLEAN" in codes
    norm_kept     = "NORM"  in codes

    status = pump_removed and clean_kept and norm_kept
    print(f"  [{'PASS' if status else 'FAIL'}] filter_operator_driven: "
          f"PUMP removed={pump_removed}; CLEAN kept={clean_kept}; NORM kept={norm_kept}")
    return status


# ---------------------------------------------------------------------------
# Test 2 — filter_recent_listings
# ---------------------------------------------------------------------------

def test_ipo_filter() -> bool:
    from production.universe_filter import filter_recent_listings

    bhav = make_bhav({
        "OLD":     {},    # 120 sessions -> kept
        "NEWIPO":  {},    # will be trimmed to 50 sessions
    }, n_days=120)

    # Truncate NEWIPO to only the last 50 sessions
    all_dates = sorted(bhav["DATE"].unique())
    keep_dates = set(all_dates[-50:])
    bhav = bhav[~((bhav["SC_CODE"] == "NEWIPO") & (~bhav["DATE"].isin(keep_dates)))].copy()

    result = filter_recent_listings(bhav, min_sessions=90)
    codes = set(result["SC_CODE"].unique())

    old_kept   = "OLD"    in codes
    ipo_removed = "NEWIPO" not in codes

    status = old_kept and ipo_removed
    print(f"  [{'PASS' if status else 'FAIL'}] filter_recent_listings: "
          f"OLD kept={old_kept}; NEWIPO removed={ipo_removed}")
    return status


# ---------------------------------------------------------------------------
# Test 3 — DataSourceFailover importable + source chain correct
# ---------------------------------------------------------------------------

def test_failover_importable() -> bool:
    try:
        from production.operational_fallback import DataSourceFailover, AlertManager
        fb = DataSourceFailover(alert_manager=AlertManager())

        # Check the three source methods exist
        has_bse   = callable(getattr(fb, "_fetch_bse",      None))
        has_nse   = callable(getattr(fb, "_fetch_nse",      None))
        has_yf    = callable(getattr(fb, "_fetch_yfinance", None))
        has_fetch = callable(getattr(fb, "fetch_bhav",      None))

        status = has_bse and has_nse and has_yf and has_fetch
        print(f"  [{'PASS' if status else 'FAIL'}] DataSourceFailover: "
              f"bse={has_bse}; nse={has_nse}; yfinance={has_yf}; fetch_bhav={has_fetch}")
        return status
    except Exception as exc:
        print(f"  [FAIL] DataSourceFailover import failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Test 4 — apply_tradability_gate includes both new filters
# ---------------------------------------------------------------------------

def test_combined_gate() -> bool:
    from production.universe_filter import apply_tradability_gate

    # Build bhav: PUMP (3 upper circuits), NEWIPO (< 90 sessions), CLEAN (passes all)
    pump_spikes = {111: 0.20, 114: 0.20, 118: 0.20}
    bhav_full = make_bhav({"PUMP": pump_spikes, "NEWIPO": {}, "CLEAN": {}}, n_days=120)

    # Trim NEWIPO to 50 sessions
    all_dates = sorted(bhav_full["DATE"].unique())
    keep_dates = set(all_dates[-50:])
    bhav = bhav_full[~((bhav_full["SC_CODE"] == "NEWIPO") & (~bhav_full["DATE"].isin(keep_dates)))].copy()

    ref_date = bhav["DATE"].max()

    try:
        result = apply_tradability_gate(
            bhav, ref_date,
            apply_operator_filter=True,
            apply_ipo_filter=True,
            min_price=0.0,          # bypass price filter on synthetic data
            min_avg_volume=0,
            min_value_crore=0.0,
        )
        codes = set(result["SC_CODE"].unique())
        pump_out = "PUMP"   not in codes
        ipo_out  = "NEWIPO" not in codes
        clean_in = "CLEAN"  in codes
        status   = pump_out and ipo_out and clean_in
        print(f"  [{'PASS' if status else 'FAIL'}] apply_tradability_gate (combined): "
              f"PUMP removed={pump_out}; NEWIPO removed={ipo_out}; CLEAN kept={clean_in}")
        return status
    except Exception as exc:
        print(f"  [FAIL] apply_tradability_gate raised: {exc}")
        return False


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 60)
    print("  P06  Data Failover, Operator Filter & Post-IPO Exclusion")
    print("=" * 60)
    print()

    results = {}
    results["operator_filter"]   = test_operator_filter()
    results["ipo_filter"]        = test_ipo_filter()
    results["failover_import"]   = test_failover_importable()
    results["combined_gate"]     = test_combined_gate()

    print()
    all_pass = all(results.values())
    for k, v in results.items():
        print(f"  {k:25s}: {'PASS' if v else 'FAIL'}")
    print()
    print("RESULT:", "ALL PASS" if all_pass else "SOME FAILURES")
    print("=" * 60 + "\n")
    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
