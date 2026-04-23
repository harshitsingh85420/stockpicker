#!/usr/bin/env python
"""
Run Backtesting for 5-Session Stock Picker

Tests the system on historical dates to see if picks actually closed positive in 5 sessions

Usage:
    # Backtest on August 2024
    python run_backtest.py

    # Backtest on custom dates
    python run_backtest.py --start 2024-07-01 --end 2024-09-30 --interval 5

    # More stocks for training
    python run_backtest.py --stocks 500
"""

import argparse
from datetime import date, timedelta
from typing import List

from backtest_5session import run_backtest
from bse_loader import BSEDataFetcher


def generate_signal_dates(start_date: date, end_date: date, interval_days: int = 3) -> List[date]:
    """
    Generate signal dates between start and end with specified interval
    Skips weekends
    """
    fetcher = BSEDataFetcher()
    dates = []
    current = start_date

    while current <= end_date:
        if not fetcher.is_weekend(current):
            dates.append(current)
            # Jump by interval
            current += timedelta(days=interval_days)
        else:
            current += timedelta(days=1)

    return dates


def main():
    parser = argparse.ArgumentParser(
        description="Backtest 5-Session Stock Picker on Historical Dates\n\n"
                    "Tests: Run on past date → get picks → check if closed positive in 5 sessions\n"
                    "Reports: Win rate, avg return, per-date breakdown\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--start',
        type=str,
        default='2024-08-01',
        help='Start date for backtest (YYYY-MM-DD), default: 2024-08-01'
    )

    parser.add_argument(
        '--end',
        type=str,
        default='2024-08-31',
        help='End date for backtest (YYYY-MM-DD), default: 2024-08-31'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=3,
        help='Days between signal dates (default: 3)'
    )

    parser.add_argument(
        '--stocks',
        type=int,
        default=200,
        help='Number of stocks for training (default: 200)'
    )

    parser.add_argument(
        '--dates',
        type=str,
        nargs='+',
        help='Specific dates to test (YYYY-MM-DD format), e.g., --dates 2024-08-01 2024-08-15'
    )

    args = parser.parse_args()

    # Generate or use provided dates
    if args.dates:
        signal_dates = [date.fromisoformat(d) for d in args.dates]
    else:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
        signal_dates = generate_signal_dates(start, end, args.interval)

    print("\n" + "=" * 80)
    print("🎯 5-SESSION STOCK PICKER - BACKTESTING")
    print("=" * 80)
    print(f"\n📅 Signal dates: {len(signal_dates)} dates")
    print(f"   Range: {min(signal_dates)} → {max(signal_dates)}")
    print(f"\n📊 Settings:")
    print(f"   Stocks for training: {args.stocks}")
    print(f"   Forward period: 5 sessions")
    print("\n💡 What this does:")
    print("   • Simulates running system on each historical date")
    print("   • Gets picks (using only data available up to that date)")
    print("   • Checks actual 5-session outcome for each pick")
    print("   • Calculates win rate, average return, per-date stats")
    print("\n🕐 This may take 10-20 minutes...")
    print("=" * 80)

    # Run backtest
    results = run_backtest(signal_dates, n_stocks=args.stocks)

    if results is not None and len(results) > 0:
        # Save detailed results to CSV
        output_file = f"backtest_results_{min(signal_dates)}_{max(signal_dates)}.csv"
        results.to_csv(output_file, index=False)

        print("\n✅ Backtesting complete!")
        print(f"📊 Total picks analyzed: {len(results)}")
        print(f"💾 Detailed results saved to: {output_file}")
        print(f"\n📋 CSV contains:")
        print(f"   • Stock codes and names")
        print(f"   • Entry prices and probabilities")
        print(f"   • Actual 5-session returns")
        print(f"   • Outcome (Positive/Negative/Flat)")
        print(f"   • Signal dates")
        print(f"\n💡 Open {output_file} in Excel to see all picks!")
    else:
        print("\n⚠️ No backtest results generated")


if __name__ == "__main__":
    main()
