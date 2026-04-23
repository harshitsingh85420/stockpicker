#!/usr/bin/env python
"""
Get Stock Picks for a Specific Date

Quick tool to run the stock picker for a historical or future date and see the picks

Usage:
    # Get picks for a specific past date (with actual outcomes)
    python get_picks.py 2024-08-15

    # Get picks for today (no outcomes yet, just predictions)
    python get_picks.py 2024-11-14

    # Save to CSV
    python get_picks.py 2024-08-15 --export picks_2024-08-15.csv

    # Use more stocks for training (default: 200)
    python get_picks.py 2024-08-15 --stocks 500
"""

import argparse
import sys
from datetime import date, timedelta
import pandas as pd

from stock_picker_5session import StockPicker5Session
from bse_loader import BSEDataFetcher


def get_picks_for_date(signal_date: date, n_stocks: int = 200, show_outcomes: bool = True):
    """
    Run stock picker for a specific date and display picks

    Args:
        signal_date: Date to get picks for
        n_stocks: Number of stocks for training
        show_outcomes: Whether to show actual outcomes (only works for past dates)
    """
    print("\n" + "=" * 80)
    print("🎯 STOCK PICKER - GET PICKS FOR SPECIFIC DATE")
    print("=" * 80)
    print(f"\n📅 Signal Date: {signal_date}")
    print(f"📊 Training stocks: {n_stocks}")
    print("=" * 80)

    # Initialize picker
    picker = StockPicker5Session()
    fetcher = BSEDataFetcher()

    # Calculate date range needed
    # Need 730 days (2 years) of history for training
    start_date = signal_date - timedelta(days=730)

    # If we want outcomes, need data 5 sessions after signal date
    if show_outcomes:
        end_date = signal_date + timedelta(days=15)  # 5 sessions + buffer
    else:
        end_date = signal_date

    print(f"\n📥 Fetching data: {start_date} → {end_date}")

    # Fetch data
    bhav = fetcher.fetch_bhav_range(start_date, end_date)

    if bhav.empty:
        print("❌ No data available for this date range")
        return None

    # Ensure DATE column is datetime64[ns] for all comparisons
    bhav = bhav.copy()
    bhav['DATE'] = pd.to_datetime(bhav['DATE'])

    print(f"✅ Fetched {len(bhav):,} rows | {bhav['SC_CODE'].nunique()} unique stocks")

    # Filter to stocks with enough history
    stock_counts = bhav.groupby('SC_CODE')['DATE'].count()
    qualified_stocks = stock_counts[stock_counts >= 200].index.tolist()

    print(f"📊 Stock universe: {len(qualified_stocks)} stocks with ≥200 days of data")

    if len(qualified_stocks) == 0:
        print("❌ No stocks with sufficient history")
        return None

    bhav_qualified = bhav[bhav['SC_CODE'].isin(qualified_stocks)].copy()

    # Limit to top N by liquidity for training
    if n_stocks and len(qualified_stocks) > n_stocks:
        liquidity = bhav_qualified.groupby('SC_CODE')['ValueTraded'].mean().sort_values(ascending=False)
        top_stocks = liquidity.head(n_stocks).index.tolist()
        bhav_train = bhav_qualified[bhav_qualified['SC_CODE'].isin(top_stocks)].copy()
        print(f"📊 Training on top {n_stocks} most liquid stocks")
    else:
        bhav_train = bhav_qualified.copy()
        print(f"📊 Training on ALL {len(qualified_stocks)} qualified stocks")

    # Run the picker
    print("\n🔧 Computing features and training model...")

    # Import required functions
    from momentum_features import prepare_features_all, add_forward_returns
    import lightgbm as lgb

    # Set feature columns (same as in backtest)
    picker.feature_cols = [
        'EMA20', 'EMA50', 'EMA200', 'EMA20_Slope5', 'EMA200_Slope', 'MA_Health', 'OverEMA20',
        'ATR14', 'ATRpct', 'BBWidth', 'BBWidthPctl',
        'DistTo20', 'DistTo63', 'DistTo52W',
        'Break20_Today', 'Break63_Today', 'Hit52WH_Today',
        'RangePos20',
        'VolMult', 'UD_Vol_Ratio10',
        'RET21D', 'RET63D', 'RS_Composite',
        'RSI14', 'ADX14', '+DI14', '-DI14', 'ADX14_chg3',
        'W_BBWidth', 'W_TrendOK', 'W_BBWidthPctl'
    ]

    # Compute features
    features = prepare_features_all(bhav_train)

    # Add forward returns for training
    features_with_labels = add_forward_returns(features, periods=[picker.FORWARD_PERIOD])

    # Convert signal_date to pd.Timestamp for comparison
    signal_date_ts = pd.Timestamp(signal_date)

    # Prepare training data (only use data up to signal date - no lookahead!)
    label_col = f"Label_fwd{picker.FORWARD_PERIOD}_positive"
    df_train = features_with_labels[features_with_labels['DATE'] < signal_date_ts].copy()
    df_train = df_train.dropna(subset=[label_col])
    df_train = df_train.dropna(subset=picker.feature_cols)

    if len(df_train) < 100:
        print("⚠️  Not enough training data")
        return None

    print(f"   Training samples: {len(df_train):,}")

    # Train model
    X_train = df_train[picker.feature_cols]
    y_train = df_train[label_col]

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'verbose': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=200, callbacks=[lgb.log_evaluation(0)])

    # Get predictions for signal date
    df_signal = features_with_labels[features_with_labels['DATE'] == signal_date_ts].copy()

    if df_signal.empty:
        print(f"⚠️  No data available for {signal_date}")
        return None

    df_predict = df_signal.dropna(subset=picker.feature_cols).copy()
    X_pred = df_predict[picker.feature_cols]

    probabilities = model.predict(X_pred)

    # Get picks above threshold
    threshold = picker.INITIAL_THRESHOLD
    picks_mask = probabilities >= threshold

    # If no picks at initial threshold, lower it
    while picks_mask.sum() == 0 and threshold >= picker.MIN_THRESHOLD:
        threshold -= picker.THRESHOLD_STEP
        picks_mask = probabilities >= threshold

    if picks_mask.sum() == 0:
        print("⚠️  No picks generated (all probabilities too low)")
        return None

    # Create picks DataFrame
    picks = pd.DataFrame({
        'SC_CODE': df_predict.loc[picks_mask, 'SC_CODE'].values,
        'SC_NAME': df_predict.loc[picks_mask, 'SC_NAME'].values,
        'Close': df_predict.loc[picks_mask, 'Close'].values,
        'Probability': probabilities[picks_mask],
        'Threshold': threshold
    })

    picks = picks.sort_values('Probability', ascending=False).reset_index(drop=True)

    print(f"✅ Generated {len(picks)} picks @ threshold {threshold:.2f}")

    # Add actual outcomes if available and requested
    if show_outcomes:
        signal_date_ts = pd.Timestamp(signal_date)
        future_date = signal_date + timedelta(days=10)  # Look for close 5 sessions ahead

        # Get future prices for the picked stocks
        future_data = bhav[
            (bhav['SC_CODE'].isin(picks['SC_CODE'])) &
            (bhav['DATE'] > signal_date_ts) &
            (bhav['DATE'] <= pd.Timestamp(future_date))
        ].copy()

        if not future_data.empty:
            # For each stock, get the close price 5 sessions after signal date
            outcomes = []
            for _, pick in picks.iterrows():
                stock_future = future_data[future_data['SC_CODE'] == pick['SC_CODE']].sort_values('DATE')

                if len(stock_future) >= 5:
                    # Get close price 5 sessions later
                    close_fwd5 = stock_future.iloc[4]['Close']
                    return_fwd5 = (close_fwd5 / pick['Close'] - 1) * 100
                    outcome = 'Positive' if return_fwd5 > 0 else ('Negative' if return_fwd5 < 0 else 'Flat')

                    outcomes.append({
                        'SC_CODE': pick['SC_CODE'],
                        'Close_fwd5': close_fwd5,
                        'Return_fwd5': return_fwd5,
                        'Outcome': outcome
                    })

            if outcomes:
                outcomes_df = pd.DataFrame(outcomes)
                picks = picks.merge(outcomes_df, on='SC_CODE', how='left')

    return picks


def display_picks(picks: pd.DataFrame, signal_date: date):
    """Display picks in a nice format"""
    print("\n" + "=" * 80)
    print(f"📋 STOCK PICKS FOR {signal_date}")
    print("=" * 80)
    print(f"\n🎯 Total picks: {len(picks)}")

    # Show statistics if outcomes available
    if 'Return_fwd5' in picks.columns:
        valid = picks.dropna(subset=['Return_fwd5'])
        if len(valid) > 0:
            win_rate = (valid['Return_fwd5'] > 0).sum() / len(valid) * 100
            avg_return = valid['Return_fwd5'].mean()

            print(f"\n📈 Actual Performance:")
            print(f"   Win Rate: {win_rate:.1f}%")
            print(f"   Avg Return: {avg_return:.2f}%")
            print(f"   Winners: {(valid['Return_fwd5'] > 0).sum()}")
            print(f"   Losers: {(valid['Return_fwd5'] < 0).sum()}")

    # Display picks
    print(f"\n📊 Picks (sorted by probability):")
    print("=" * 100)

    # Select columns to display
    display_cols = ['SC_CODE', 'SC_NAME', 'Close', 'Probability']

    # Add outcome columns if available
    if 'Return_fwd5' in picks.columns:
        display_cols.extend(['Return_fwd5', 'Outcome'])

    # Format DataFrame for display
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.float_format', lambda x: f'{x:.2f}')

    display_df = picks[display_cols].copy()

    # Format probability as percentage
    display_df['Probability'] = display_df['Probability'].apply(lambda x: f'{x*100:.1f}%')

    # Format return if available
    if 'Return_fwd5' in display_df.columns:
        display_df['Return_fwd5'] = display_df['Return_fwd5'].apply(
            lambda x: f'{x:+.2f}%' if pd.notna(x) else 'N/A'
        )

    print(display_df.to_string(index=False))
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Get stock picks for a specific date",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'date',
        type=str,
        help='Date to get picks for (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--stocks',
        type=int,
        default=200,
        help='Number of stocks for training (default: 200)'
    )

    parser.add_argument(
        '--export',
        type=str,
        help='Export picks to CSV file'
    )

    parser.add_argument(
        '--no-outcomes',
        action='store_true',
        help='Skip showing actual outcomes (faster for recent dates)'
    )

    args = parser.parse_args()

    # Parse date
    try:
        signal_date = date.fromisoformat(args.date)
    except ValueError:
        print(f"❌ Invalid date format: {args.date}")
        print("   Use YYYY-MM-DD format, e.g., 2024-08-15")
        return

    # Get picks
    picks = get_picks_for_date(
        signal_date,
        n_stocks=args.stocks,
        show_outcomes=not args.no_outcomes
    )

    if picks is None:
        return

    # Display picks
    display_picks(picks, signal_date)

    # Export if requested
    if args.export:
        picks.to_csv(args.export, index=False)
        print(f"\n💾 Exported {len(picks)} picks to: {args.export}")


if __name__ == "__main__":
    main()
