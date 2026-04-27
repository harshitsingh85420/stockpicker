#!/usr/bin/env python3
"""
validate_adjustments.py  --  P04 corporate-action adjustment validation.

Tests the CorporateActionAdjuster with synthetic data for 3 scenarios:
  1. 2:1 stock split
  2. 1:1 bonus issue
  3. Cash dividend (Rs 5)
  4. CHAINED: split + dividend on the same stock in the same year

Saves chart to: stock_picker_data/results/adjustment_validation.png

Exit code 0 if all tests pass, 1 if any fail.
"""

import sys
import logging
from datetime import date, timedelta
from pathlib import Path
import tempfile
import csv

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
log = logging.getLogger("validate_adjustments")

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

RESULTS_DIR = _ROOT / "stock_picker_data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_price_series(
    base_price: float = 100.0,
    n_days: int = 60,
    start: date = date(2023, 1, 2),
    slope: float = 0.2,
    noise: float = 0.5,
    seed: int = 42,
    events: dict = None,   # {date: price_multiplier}  -- simulate ex-date drops
) -> pd.DataFrame:
    """
    Build a deterministic synthetic OHLCV series.

    events: dict mapping {event_date: multiplier} -- on event_date the close
    price is multiplied by `multiplier` (e.g. 0.5 for a 2:1 split).
    This simulates the actual market price drop on ex-date.
    """
    rng = np.random.default_rng(seed)
    days = [start + timedelta(days=i) for i in range(n_days)
            if (start + timedelta(days=i)).weekday() < 5]
    raw_changes = slope + rng.normal(0, noise, len(days))
    closes = []
    price = base_price
    for i, d in enumerate(days):
        price += raw_changes[i]
        price = max(price, 5.0)
        if events and d in events:
            price *= events[d]
        closes.append(price)

    rows = []
    for d, c in zip(days, closes):
        rows.append({
            "SC_CODE": "TEST",
            "SC_NAME": "Test Stock",
            "DATE": d,
            "Open":  c * 0.99,
            "High":  c * 1.01,
            "Low":   c * 0.98,
            "Close": c,
            "Volume": 100_000,
        })
    return pd.DataFrame(rows)


def write_ca_csv(tmp_path: str, records: list) -> str:
    """Write a temporary corporate actions CSV."""
    fieldnames = ["SC_CODE", "DATE", "ACTION_TYPE", "RATIO", "BONUS_DENOM", "EX_DATE"]
    with open(tmp_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)
    return tmp_path


def no_jump_on_exdate(adj_df: pd.DataFrame, ex_date: date, tol: float = 0.30) -> bool:
    """
    Check that the price jump on ex_date is within tolerance.
    A good adjustment means the discontinuity is removed (or greatly reduced).
    """
    adj_df = adj_df.copy()
    adj_df["DATE"] = pd.to_datetime(adj_df["DATE"]).dt.date
    before = adj_df[adj_df["DATE"] < ex_date]["Close"]
    on_or_after = adj_df[adj_df["DATE"] >= ex_date]["Close"]
    if before.empty or on_or_after.empty:
        return True
    last_before = before.iloc[-1]
    first_after = on_or_after.iloc[0]
    if last_before <= 0:
        return True
    jump = abs(first_after / last_before - 1.0)
    log.debug("  Jump on %s: %.2f%%", ex_date, jump * 100)
    return jump < tol


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_tests() -> bool:
    from production.data_integrity import CorporateActionAdjuster

    adjuster = CorporateActionAdjuster()
    results = {}

    # ── Test 1: 2:1 SPLIT ────────────────────────────────────────────────
    ex_date1 = date(2023, 2, 15)
    # Simulate actual market: price halves on ex-date (2:1 split)
    df1 = make_price_series(base_price=100.0, n_days=80, seed=1,
                            events={ex_date1: 0.5})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path1 = f.name
    write_ca_csv(csv_path1, [{
        "SC_CODE": "TEST", "DATE": "2023-02-01",
        "ACTION_TYPE": "SPLIT", "RATIO": 2.0, "BONUS_DENOM": 1.0,
        "EX_DATE": str(ex_date1),
    }])
    adjuster.load_corporate_actions(csv_path1)
    adj1 = adjuster.adjust_prices(df1, "TEST")

    # Before split: price should be halved, volume doubled
    before_mask = pd.to_datetime(adj1["DATE"]).dt.date < ex_date1
    after_mask  = pd.to_datetime(adj1["DATE"]).dt.date >= ex_date1
    orig_before  = df1[before_mask]["Close"].mean()
    adj_before   = adj1[before_mask]["Close"].mean()
    orig_vol_bef = df1[before_mask]["Volume"].mean()
    adj_vol_bef  = adj1[before_mask]["Volume"].mean()

    split_price_ok  = abs(adj_before / orig_before - 0.5)  < 0.01
    split_vol_ok    = abs(adj_vol_bef / orig_vol_bef - 2.0) < 0.01
    split_jump_ok   = no_jump_on_exdate(adj1, ex_date1)

    results["SPLIT"] = split_price_ok and split_vol_ok and split_jump_ok
    print(f"  [{'PASS' if results['SPLIT'] else 'FAIL'}] 2:1 Split  "
          f"(price factor={adj_before/orig_before:.3f}, expected=0.500; "
          f"vol factor={adj_vol_bef/orig_vol_bef:.3f}, expected=2.000; "
          f"jump_ok={split_jump_ok})")

    # ── Test 2: 1:1 BONUS ─────────────────────────────────────────────────
    ex_date2 = date(2023, 2, 20)
    # Simulate actual market: price halves on ex-date (1:1 bonus)
    df2 = make_price_series(base_price=200.0, n_days=80, seed=2,
                            events={ex_date2: 0.5})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path2 = f.name
    write_ca_csv(csv_path2, [{
        "SC_CODE": "TEST", "DATE": "2023-02-05",
        "ACTION_TYPE": "BONUS", "RATIO": 1.0, "BONUS_DENOM": 1.0,
        "EX_DATE": str(ex_date2),
    }])
    adjuster.load_corporate_actions(csv_path2)
    adj2 = adjuster.adjust_prices(df2, "TEST")

    before_mask2 = pd.to_datetime(adj2["DATE"]).dt.date < ex_date2
    orig_before2 = df2[before_mask2]["Close"].mean()
    adj_before2  = adj2[before_mask2]["Close"].mean()
    bonus_factor = adj_before2 / orig_before2
    bonus_price_ok = abs(bonus_factor - 0.5) < 0.01   # 1:1 bonus -> price halved
    bonus_jump_ok  = no_jump_on_exdate(adj2, ex_date2)

    results["BONUS"] = bonus_price_ok and bonus_jump_ok
    print(f"  [{'PASS' if results['BONUS'] else 'FAIL'}] 1:1 Bonus  "
          f"(price factor={bonus_factor:.3f}, expected=0.500; "
          f"jump_ok={bonus_jump_ok})")

    # ── Test 3: Cash DIVIDEND Rs 5 ────────────────────────────────────────
    df3 = make_price_series(base_price=150.0, n_days=80, seed=3)
    ex_date3 = date(2023, 2, 25)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path3 = f.name
    write_ca_csv(csv_path3, [{
        "SC_CODE": "TEST", "DATE": "2023-02-10",
        "ACTION_TYPE": "DIVIDEND", "RATIO": 5.0, "BONUS_DENOM": 1.0,
        "EX_DATE": str(ex_date3),
    }])
    adjuster.load_corporate_actions(csv_path3)
    adj3 = adjuster.adjust_prices(df3, "TEST")

    before_mask3 = pd.to_datetime(adj3["DATE"]).dt.date < ex_date3
    adj_df3_pre = adj3[before_mask3]
    last_adj_close = float(adj_df3_pre.iloc[-1]["Close"])
    last_raw_close = float(df3[before_mask3].iloc[-1]["Close"])
    # Expected factor: (close_prev - 5) / close_prev
    expected_factor = (last_raw_close - 5.0) / last_raw_close
    actual_factor   = last_adj_close / last_raw_close
    div_price_ok    = abs(actual_factor - expected_factor) < 0.005
    div_jump_ok     = no_jump_on_exdate(adj3, ex_date3)

    results["DIVIDEND"] = div_price_ok and div_jump_ok
    print(f"  [{'PASS' if results['DIVIDEND'] else 'FAIL'}] Dividend Rs5  "
          f"(factor={actual_factor:.4f}, expected={expected_factor:.4f}; "
          f"jump_ok={div_jump_ok})")

    # ── Test 4: CHAINED (split + dividend same year) ───────────────────────
    ex_split = date(2023, 2, 10)
    ex_div   = date(2023, 3, 20)
    # Market: price halves on split, then drops ~Rs3/price on dividend
    df4 = make_price_series(base_price=500.0, n_days=100, seed=4,
                            events={ex_split: 0.5, ex_div: 0.994})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        csv_path4 = f.name
    write_ca_csv(csv_path4, [
        {   # 2:1 split
            "SC_CODE": "TEST", "DATE": "2023-01-25",
            "ACTION_TYPE": "SPLIT", "RATIO": 2.0, "BONUS_DENOM": 1.0,
            "EX_DATE": str(ex_split),
        },
        {   # Rs 3 dividend
            "SC_CODE": "TEST", "DATE": "2023-03-05",
            "ACTION_TYPE": "DIVIDEND", "RATIO": 3.0, "BONUS_DENOM": 1.0,
            "EX_DATE": str(ex_div),
        },
    ])
    adjuster.load_corporate_actions(csv_path4)
    adj4 = adjuster.adjust_prices(df4, "TEST")

    # Check both splits are smooth
    chain_split_ok = no_jump_on_exdate(adj4, ex_split)
    chain_div_ok   = no_jump_on_exdate(adj4, ex_div)

    # Verify factors multiplied (not added).
    # Rows before split should have: split factor * dividend factor applied
    very_early = pd.to_datetime(adj4["DATE"]).dt.date < ex_split
    split_factor = 0.5  # 2:1 split
    raw_early_close = float(df4[very_early].iloc[-1]["Close"])
    adj_early_close = float(adj4[very_early].iloc[-1]["Close"])

    # Dividend factor depends on close price after split, before div
    mid_mask = (pd.to_datetime(adj4["DATE"]).dt.date >= ex_split) & \
               (pd.to_datetime(adj4["DATE"]).dt.date < ex_div)
    mid_rows_adj = adj4[mid_mask]
    if not mid_rows_adj.empty:
        close_prev_div = float(mid_rows_adj.iloc[-1]["Close"])
        div_factor     = (close_prev_div - 3.0) / close_prev_div
        expected_cum   = split_factor * div_factor
        actual_cum     = adj_early_close / raw_early_close
        chain_factor_ok = abs(actual_cum - expected_cum) < 0.01
    else:
        chain_factor_ok = True  # can't verify, skip

    results["CHAINED"] = chain_split_ok and chain_div_ok and chain_factor_ok
    print(f"  [{'PASS' if results['CHAINED'] else 'FAIL'}] Chained Split+Dividend  "
          f"(split_jump_ok={chain_split_ok}; div_jump_ok={chain_div_ok}; "
          f"cum_factor_ok={chain_factor_ok})")

    # ── Plot all 4 tests ───────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Corporate Action Adjustment Validation (P04)", fontsize=13)

        tests = [
            (df1, adj1, ex_date1, "2:1 Split",         axes[0, 0]),
            (df2, adj2, ex_date2, "1:1 Bonus",         axes[0, 1]),
            (df3, adj3, ex_date3, "Dividend Rs5",      axes[1, 0]),
            (df4, adj4, ex_split, "Chained Split+Div", axes[1, 1]),
        ]

        for raw, adj, ex_d, title, ax in tests:
            raw_dates = pd.to_datetime(raw["DATE"])
            adj_dates = pd.to_datetime(adj["DATE"])
            ax.plot(raw_dates, raw["Close"], "r--", alpha=0.6, linewidth=1, label="Raw")
            ax.plot(adj_dates, adj["Close"], "b-", linewidth=1.5, label="Adjusted")
            ax.axvline(pd.Timestamp(ex_d), color="orange", linestyle="--",
                       linewidth=1, label=f"Ex-date {ex_d}")
            ax.set_title(title)
            ax.set_ylabel("Price (Rs)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = RESULTS_DIR / "adjustment_validation.png"
        plt.savefig(out_path, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"\n  Chart saved -> {out_path}")
    except ImportError:
        print("\n  (matplotlib not available -- chart skipped)")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    all_pass = all(results.values())
    for k, v in results.items():
        print(f"  {k:12s}: {'PASS' if v else 'FAIL'}")
    print()
    print("RESULT:", "ALL PASS" if all_pass else "SOME FAILURES")
    return all_pass


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  P04  Corporate Action Adjustment Validation")
    print("=" * 60)
    ok = run_tests()
    sys.exit(0 if ok else 1)
