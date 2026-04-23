#!/usr/bin/env python
"""
View Stock Picks for Specific Dates

Quick tool to inspect backtest results and see which stocks were picked

Usage:
    # View all picks from backtest
    python view_picks.py backtest_results_2024-08-01_2024-08-31.csv

    # View picks for a specific date
    python view_picks.py backtest_results_2024-08-01_2024-08-31.csv --date 2024-08-15

    # View only positive outcomes
    python view_picks.py backtest_results_2024-08-01_2024-08-31.csv --outcome Positive

    # Export filtered picks to new CSV
    python view_picks.py backtest_results_2024-08-01_2024-08-31.csv --date 2024-08-15 --export picks_2024-08-15.csv
"""

import argparse
import pandas as pd
from datetime import date


def main():
    parser = argparse.ArgumentParser(
        description="View and filter stock picks from backtest results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'csv_file',
        help='Backtest results CSV file'
    )

    parser.add_argument(
        '--date',
        type=str,
        help='Filter by specific signal date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--outcome',
        choices=['Positive', 'Negative', 'Flat'],
        help='Filter by outcome'
    )

    parser.add_argument(
        '--top',
        type=int,
        help='Show only top N picks (by probability)'
    )

    parser.add_argument(
        '--export',
        type=str,
        help='Export filtered results to CSV file'
    )

    args = parser.parse_args()

    # Load results
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {args.csv_file}")
        print(f"\n💡 First run backtest to generate results:")
        print(f"   python run_backtest.py --start 2024-08-01 --end 2024-08-31")
        return

    print(f"\n📊 Loaded {len(df)} picks from {args.csv_file}")

    # Apply filters
    filtered_df = df.copy()

    if args.date:
        # Convert date string to match format in CSV
        filter_date = pd.to_datetime(args.date).date()
        filtered_df['SignalDate'] = pd.to_datetime(filtered_df['SignalDate']).dt.date
        filtered_df = filtered_df[filtered_df['SignalDate'] == filter_date]
        print(f"🔍 Filtered by date: {args.date}")

    if args.outcome:
        filtered_df = filtered_df[filtered_df['Outcome'] == args.outcome]
        print(f"🔍 Filtered by outcome: {args.outcome}")

    if args.top:
        filtered_df = filtered_df.nlargest(args.top, 'Probability')
        print(f"🔍 Showing top {args.top} picks")

    if len(filtered_df) == 0:
        print("\n⚠️ No picks match your filters")
        return

    # Display results
    print(f"\n📋 Found {len(filtered_df)} picks")
    print("=" * 100)

    # Summary statistics
    if 'Return_fwd5' in filtered_df.columns:
        valid = filtered_df.dropna(subset=['Return_fwd5'])
        if len(valid) > 0:
            win_rate = (valid['Outcome'] == 'Positive').sum() / len(valid) * 100
            avg_return = valid['Return_fwd5'].mean()
            print(f"\n📈 Performance:")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Avg Return: {avg_return:.2f}%")
            print(f"   Positive: {(valid['Outcome'] == 'Positive').sum()}")
            print(f"   Negative: {(valid['Outcome'] == 'Negative').sum()}")

    # Display picks
    print(f"\n📋 Stock Picks:")
    print("=" * 100)

    # Select key columns for display
    display_cols = ['SignalDate', 'SC_CODE', 'SC_NAME', 'Close', 'Probability',
                   'Return_fwd5', 'Outcome']
    display_cols = [col for col in display_cols if col in filtered_df.columns]

    # Format for better readability
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(filtered_df[display_cols].to_string(index=False))

    # Export if requested
    if args.export:
        filtered_df.to_csv(args.export, index=False)
        print(f"\n💾 Exported {len(filtered_df)} picks to: {args.export}")

    # Show available dates
    if not args.date:
        print(f"\n📅 Available signal dates:")
        dates = pd.to_datetime(df['SignalDate']).dt.date.unique()
        for d in sorted(dates):
            count = len(df[pd.to_datetime(df['SignalDate']).dt.date == d])
            print(f"   {d}: {count} picks")
        print(f"\n💡 Use --date YYYY-MM-DD to see picks for a specific date")


if __name__ == "__main__":
    main()
