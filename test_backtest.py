#!/usr/bin/env python
"""
Quick test backtest - runs on a few October 2024 dates
"""

from datetime import date
from backtest_5session import run_backtest

# Test on 3 October dates
signal_dates = [
    date(2024, 10, 1),
    date(2024, 10, 7),
    date(2024, 10, 14),
]

print("Running quick backtest on 3 October 2024 dates...")
print("This will show if picks actually closed positive in 5 sessions")
print()

# Run with small number of stocks for speed
results = run_backtest(signal_dates, n_stocks=50)

if results is not None and len(results) > 0:
    print("\n✅ Test backtest complete!")
    print(f"Total picks tested: {len(results)}")
else:
    print("\n⚠️ No results - may need different dates")
