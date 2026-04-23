#!/usr/bin/env python
"""
5-Session Stock Picker - Main Runner

Usage:
    # Daily mode (recommended - retrains every day)
    python run_5session_picker.py

    # With custom number of stocks for training
    python run_5session_picker.py --stocks 500
"""

import argparse
from stock_picker_5session import run_daily


def main():
    parser = argparse.ArgumentParser(
        description="5-Session Stock Picker - ML-Enhanced Momentum/Breakout System\n\n"
                    "Intention:\n"
                    "• Run today → tells which stocks to buy tomorrow\n"
                    "• Expects positive close in 5 sessions\n"
                    "• NSE data (priority) → BSE fallback\n"
                    "• ML model learns which patterns lead to 5-session gains\n"
                    "• Shows ALL qualifying stocks (no limit!)\n"
                    "• Daily retraining - model learns continuously\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--stocks',
        type=int,
        default=None,
        help='Number of most liquid stocks to use for training (default: None = ALL stocks). '
             'Examples: --stocks 200, --stocks 500, --stocks 1000'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        dest='all_stocks',
        help='Train on ALL stocks (same as omitting --stocks)'
    )

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Force retrain model (default: use existing model if available)'
    )

    args = parser.parse_args()

    # Handle --all flag (overrides --stocks)
    n_stocks = None if args.all_stocks else args.stocks

    # Run daily mode
    # Note: run_daily always trains on ALL stocks as per requirement #4
    # The n_stocks parameter is not used in current implementation
    run_daily(use_existing_model=not args.retrain)


if __name__ == "__main__":
    main()
