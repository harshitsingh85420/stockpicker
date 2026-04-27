"""
Backtesting Module for 5-Session Stock Picker

Tests the system on historical dates:
- Run on past date → get picks → check if they closed positive in 5 sessions
- Calculates win rate, average return, per-date breakdown
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import List, Tuple
from pathlib import Path

from bse_loader import BSEDataFetcher
from momentum_features import prepare_features_all, add_forward_returns
from stock_picker_5session import StockPicker5Session


class Backtester5Session:
    """
    Backtest the 5-session stock picker on historical dates
    """

    def __init__(self, picker: StockPicker5Session):
        self.picker = picker
        self.fetcher = BSEDataFetcher()

    def backtest_on_dates(self, signal_dates: List[date], n_stocks: int = 200) -> pd.DataFrame:
        """
        Backtest on multiple historical dates

        Args:
            signal_dates: List of dates to simulate running the system
            n_stocks: Number of stocks for training

        Returns:
            DataFrame with all picks and their 5-session outcomes
        """
        print("\n" + "=" * 80)
        print("📊 BACKTESTING 5-SESSION STOCK PICKER")
        print("=" * 80)
        print(f"Signal dates: {min(signal_dates)} → {max(signal_dates)}")
        print(f"Total signal dates: {len(signal_dates)}")
        print(f"Forward period: {self.picker.FORWARD_PERIOD} sessions")
        print("=" * 80)

        # Fetch data covering all signal dates + forward period
        # Need extra days for the forward returns
        lookback_start = min(signal_dates) - timedelta(days=self.picker.LOOKBACK_DAYS)
        lookback_end = max(signal_dates) + timedelta(days=30)  # Extra buffer for 5-session forward

        print(f"\n📥 Fetching data: {lookback_start} → {lookback_end}")
        bhav = self.fetcher.fetch_bhav_range(lookback_start, lookback_end)

        # Get stock universe
        qualified_stocks = self.fetcher.get_stock_universe(bhav, self.picker.MIN_DATA_POINTS)

        # Limit to top N most liquid for training
        if n_stocks and n_stocks < len(qualified_stocks):
            liquidity = bhav.groupby('SC_CODE')['ValueTraded'].mean().sort_values(ascending=False)
            top_stocks = liquidity.head(n_stocks).index.tolist()
            bhav_train = bhav[bhav['SC_CODE'].isin(top_stocks)].copy()
        else:
            bhav_train = bhav.copy()

        # Compute features
        print(f"\n🔧 Computing features...")
        features = prepare_features_all(bhav_train)
        features_with_labels = add_forward_returns(features, periods=[self.picker.FORWARD_PERIOD])

        print(f"✅ Features computed: {len(features):,} rows")

        # For each signal date, get picks and outcomes
        all_results = []

        for signal_date in signal_dates:
            print(f"\n{'=' * 80}")
            print(f"📅 Signal Date: {signal_date}")
            print(f"{'=' * 80}")

            # Simulate running on this date
            result = self._backtest_single_date(
                features_with_labels,
                signal_date,
                train_until_date=signal_date
            )

            if result is not None and len(result) > 0:
                result['SignalDate'] = signal_date
                all_results.append(result)
                print(f"✅ {len(result)} picks for {signal_date}")
            else:
                print(f"⚠️ No picks for {signal_date}")

        if not all_results:
            print("\n❌ No backtest results generated!")
            return pd.DataFrame()

        # Combine all results
        backtest_df = pd.concat(all_results, ignore_index=True)

        return backtest_df

    def _backtest_single_date(self, features_df: pd.DataFrame, signal_date: date,
                              train_until_date: date) -> pd.DataFrame:
        """
        Simulate running the system on a specific historical date
        """
        # Convert date to pd.Timestamp for comparison with datetime64[ns] column
        signal_date_ts = pd.Timestamp(signal_date)

        # Filter to data available up to signal date (no lookahead!)
        df_until_date = features_df[features_df['DATE'] <= signal_date_ts].copy()

        if df_until_date.empty:
            return None

        # Train model on data up to this date
        label_col = f"Label_fwd{self.picker.FORWARD_PERIOD}_positive"
        df_train = df_until_date.dropna(subset=[label_col]).copy()

        if len(df_train) < 100:  # Need minimum training data
            print(f"   ⚠️ Insufficient training data ({len(df_train)} rows)")
            return None

        # Prepare training data
        df_train = df_train.dropna(subset=self.picker.feature_cols)
        X_train = df_train[self.picker.feature_cols]
        y_train = df_train[label_col]

        # Train model (silently)
        import lightgbm as lgb

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42
        }

        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_data, num_boost_round=200, callbacks=[lgb.log_evaluation(0)])

        # Predict on signal date
        df_signal = features_df[features_df['DATE'] == signal_date_ts].copy()

        if df_signal.empty:
            return None

        df_predict = df_signal.dropna(subset=self.picker.feature_cols).copy()
        X_pred = df_predict[self.picker.feature_cols]

        probabilities = model.predict(X_pred)

        # Get picks above threshold
        threshold = self.picker.INITIAL_THRESHOLD
        picks_mask = probabilities >= threshold

        # If no picks at initial threshold, lower it
        while picks_mask.sum() == 0 and threshold >= self.picker.MIN_THRESHOLD:
            threshold -= self.picker.THRESHOLD_STEP
            picks_mask = probabilities >= threshold

        if picks_mask.sum() == 0:
            return None

        # Get picks with their forward returns
        fwd_ret_col = f"Ret_fwd{self.picker.FORWARD_PERIOD}"
        fwd_close_col = f"Close_fwd{self.picker.FORWARD_PERIOD}"

        picks = pd.DataFrame({
            'SC_CODE': df_predict.loc[picks_mask, 'SC_CODE'].values,
            'SC_NAME': df_predict.loc[picks_mask, 'SC_NAME'].values,
            'Close': df_predict.loc[picks_mask, 'Close'].values,
            'Probability': probabilities[picks_mask],
            'VolMult': df_predict.loc[picks_mask, 'VolMult'].values,
            'RS_Composite': df_predict.loc[picks_mask, 'RS_Composite'].values,
            'ADX14': df_predict.loc[picks_mask, 'ADX14'].values,
            'RSI14': df_predict.loc[picks_mask, 'RSI14'].values,
            'Close_fwd5': df_predict.loc[picks_mask, fwd_close_col].values,
            'Return_fwd5': df_predict.loc[picks_mask, fwd_ret_col].values,
            'Threshold': threshold
        })

        # Add outcome label
        picks['Outcome'] = picks['Return_fwd5'].apply(
            lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Flat')
        )

        print(f"   📊 {len(picks)} picks @ threshold {threshold:.2f}")
        print(f"   📈 Positive: {(picks['Outcome'] == 'Positive').sum()}")
        print(f"   📉 Negative: {(picks['Outcome'] == 'Negative').sum()}")

        return picks.sort_values('Probability', ascending=False).reset_index(drop=True)

    def display_backtest_results(self, backtest_df: pd.DataFrame):
        """
        Display comprehensive backtest results
        """
        if backtest_df.empty:
            print("No backtest results to display!")
            return

        print("\n" + "=" * 80)
        print("📊 BACKTEST RESULTS SUMMARY")
        print("=" * 80)

        # Overall statistics
        total_picks = len(backtest_df)
        valid_outcomes = backtest_df.dropna(subset=['Return_fwd5'])

        pos = (valid_outcomes['Outcome'] == 'Positive').sum()
        neg = (valid_outcomes['Outcome'] == 'Negative').sum()
        flat = (valid_outcomes['Outcome'] == 'Flat').sum()
        total_valid = len(valid_outcomes)

        win_rate = (pos / total_valid * 100) if total_valid > 0 else 0
        avg_return = valid_outcomes['Return_fwd5'].mean()
        median_return = valid_outcomes['Return_fwd5'].median()

        print(f"\n📈 Overall Performance:")
        print(f"   Total picks: {total_picks}")
        print(f"   Valid outcomes: {total_valid}")
        print(f"   Positive: {pos} ({pos / total_valid * 100:.1f}%)")
        print(f"   Negative: {neg} ({neg / total_valid * 100:.1f}%)")
        print(f"   Flat: {flat} ({flat / total_valid * 100:.1f}%)")
        print(f"   Win Rate: {win_rate:.2f}%")
        print(f"   Avg Return: {avg_return:.2f}%")
        print(f"   Median Return: {median_return:.2f}%")

        # Per-date breakdown
        print(f"\n📅 Per-Date Breakdown:")
        per_date = valid_outcomes.groupby('SignalDate').agg({
            'Outcome': lambda x: pd.Series({
                'Total': len(x),
                'Positive': (x == 'Positive').sum(),
                'Negative': (x == 'Negative').sum(),
                'WinRate': (x == 'Positive').sum() / len(x) * 100
            })
        })['Outcome'].apply(pd.Series)

        print(per_date.to_string())

        # Top 10 best picks
        print(f"\n🏆 Top 10 Best Picks:")
        top10 = valid_outcomes.nlargest(10, 'Return_fwd5')[
            ['SignalDate', 'SC_NAME', 'Close', 'Close_fwd5', 'Return_fwd5', 'Probability']
        ]
        print(top10.to_string(index=False))

        # Top 10 worst picks
        print(f"\n💔 Top 10 Worst Picks:")
        worst10 = valid_outcomes.nsmallest(10, 'Return_fwd5')[
            ['SignalDate', 'SC_NAME', 'Close', 'Close_fwd5', 'Return_fwd5', 'Probability']
        ]
        print(worst10.to_string(index=False))

        # Probability calibration
        print(f"\n🎯 Probability Calibration:")
        prob_bins = pd.cut(valid_outcomes['Probability'], bins=[0, 0.55, 0.60, 0.65, 0.70, 1.0])
        prob_analysis = valid_outcomes.groupby(prob_bins).agg({
            'Outcome': [
                ('Count', 'count'),
                ('WinRate', lambda x: (x == 'Positive').sum() / len(x) * 100)
            ],
            'Return_fwd5': [('AvgReturn', 'mean')]
        })
        print(prob_analysis.to_string())

        # Save detailed results
        results_dir = Path("./stock_picker_data/backtest_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        csv_path = results_dir / f"backtest_{min(valid_outcomes['SignalDate'])}_{max(valid_outcomes['SignalDate'])}.csv"
        valid_outcomes.to_csv(csv_path, index=False)
        print(f"\n💾 Detailed results saved to: {csv_path}")


def run_backtest(signal_dates: List[date], n_stocks: int = 200):
    """
    Run backtest on specified historical dates
    """
    picker = StockPicker5Session()

    # Set feature columns (needed for backtesting)
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

    backtester = Backtester5Session(picker)

    # Run backtest
    results = backtester.backtest_on_dates(signal_dates, n_stocks=n_stocks)

    # Display results
    backtester.display_backtest_results(results)

    return results


if __name__ == "__main__":
    # Example: Backtest on August 2024 dates
    # You can change these to any historical dates

    signal_dates = [
        date(2024, 8, 1),
        date(2024, 8, 5),
        date(2024, 8, 8),
        date(2024, 8, 12),
        date(2024, 8, 15),
        date(2024, 8, 19),
        date(2024, 8, 22),
        date(2024, 8, 26),
        date(2024, 8, 29),
    ]

    print("=" * 80)
    print("🎯 5-SESSION STOCK PICKER - BACKTESTING")
    print("=" * 80)
    print(f"\nTesting on {len(signal_dates)} historical dates")
    print(f"Date range: {min(signal_dates)} → {max(signal_dates)}")
    print("\nThis will:")
    print("  1. Simulate running the system on each date")
    print("  2. Get the picks (as if running on that date)")
    print("  3. Check if they closed positive in 5 sessions")
    print("  4. Calculate win rate and average returns")
    print("=" * 80)

    results = run_backtest(signal_dates, n_stocks=200)

    print("\n✅ Backtesting complete!")
