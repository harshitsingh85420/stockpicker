#!/usr/bin/env python3
"""
Adaptive Backtesting Module - Requirement 6 Compliance

This module implements backtesting with adaptive retraining:
- Runs backtest on historical data
- Detects when predictions fail (accuracy below threshold)
- Automatically retrains the model with updated data
- Continues backtesting with improved model

Compliance: Addresses Requirement 6 - "in backtesting also..it will train if the outcome does not come right"
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path
import logging

from stock_picker_5session import StockPicker5Session
from bse_loader import BSEDataFetcher

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveBacktester:
    """
    Backtester that retrains the model when predictions are inaccurate.

    Features:
    - Walk-forward backtesting
    - Automatic model retraining on failure
    - Performance tracking
    - Detailed reporting
    """

    def __init__(
        self,
        retrain_threshold: float = 0.55,  # Retrain if accuracy < 55%
        min_retrain_interval: int = 10,   # Min days between retraining
        forward_period: int = 5,          # Sessions to predict ahead
        evaluation_window: int = 20       # Window to evaluate accuracy
    ):
        """
        Initialize adaptive backtester.

        Args:
            retrain_threshold: Accuracy threshold below which to retrain
            min_retrain_interval: Minimum days between retraining sessions
            forward_period: Number of sessions to predict ahead (5 for 5-session)
            evaluation_window: Number of predictions to use for accuracy calculation
        """
        self.retrain_threshold = retrain_threshold
        self.min_retrain_interval = min_retrain_interval
        self.forward_period = forward_period
        self.evaluation_window = evaluation_window

        self.stock_picker = None
        self.fetcher = BSEDataFetcher()

        # Tracking
        self.backtest_results = []
        self.retrain_dates = []
        self.accuracy_history = []

    def initialize_model(self, initial_train_date: date):
        """Initialize and train the model with data up to initial_train_date."""
        logger.info(f"🔄 Initializing model with data up to {initial_train_date}")

        self.stock_picker = StockPicker5Session()

        # Train on data up to initial_train_date
        end_date = self.fetcher.prev_bday(initial_train_date)
        start_date = end_date - timedelta(days=730)  # 2 years lookback

        logger.info(f"📊 Training on data from {start_date} to {end_date}")

        # Load and train
        bhav = self.fetcher.load_bhav_range(start_date, end_date)

        if bhav.empty:
            raise ValueError(f"No data available for training period {start_date} to {end_date}")

        # Train the model
        self.stock_picker.train(bhav, train_end=end_date)

        logger.info("✅ Initial model trained successfully")

    def get_actual_outcomes(
        self,
        predictions: pd.DataFrame,
        prediction_date: date,
        forward_period: int = 5
    ) -> pd.DataFrame:
        """
        Get actual outcomes for predictions after forward_period sessions.

        Args:
            predictions: DataFrame with stock predictions
            prediction_date: Date when predictions were made
            forward_period: Number of sessions ahead

        Returns:
            DataFrame with actual outcomes
        """
        # Calculate target date (forward_period business days ahead)
        target_date = prediction_date
        for _ in range(forward_period):
            target_date = self.fetcher.next_bday(target_date)

        # Load data for prediction_date and target_date
        pred_date_data = self.fetcher.load_bhav_date(prediction_date)
        target_date_data = self.fetcher.load_bhav_date(target_date)

        if pred_date_data.empty or target_date_data.empty:
            return pd.DataFrame()

        # Merge to get both prices
        merged = pred_date_data[['SC_CODE', 'CLOSE']].merge(
            target_date_data[['SC_CODE', 'CLOSE']],
            on='SC_CODE',
            suffixes=('_pred', '_target')
        )

        # Calculate actual returns
        merged['actual_return'] = (
            (merged['CLOSE_target'] - merged['CLOSE_pred']) / merged['CLOSE_pred'] * 100
        )
        merged['actual_gain'] = merged['actual_return'] > 0

        # Merge with predictions
        results = predictions.merge(
            merged[['SC_CODE', 'actual_return', 'actual_gain']],
            on='SC_CODE',
            how='left'
        )

        return results

    def calculate_accuracy(self, results: pd.DataFrame) -> float:
        """Calculate prediction accuracy."""
        if results.empty or 'actual_gain' not in results.columns:
            return 0.0

        # Accuracy = % of stocks that actually gained as predicted
        valid_results = results.dropna(subset=['actual_gain'])
        if len(valid_results) == 0:
            return 0.0

        accuracy = valid_results['actual_gain'].sum() / len(valid_results)
        return accuracy

    def should_retrain(
        self,
        current_date: date,
        recent_accuracies: List[float]
    ) -> bool:
        """
        Determine if model should be retrained.

        Args:
            current_date: Current backtest date
            recent_accuracies: List of recent accuracy scores

        Returns:
            True if should retrain, False otherwise
        """
        # Check if minimum interval has passed since last retrain
        if self.retrain_dates:
            last_retrain = self.retrain_dates[-1]
            days_since_retrain = (current_date - last_retrain).days
            if days_since_retrain < self.min_retrain_interval:
                return False

        # Check if we have enough accuracy data
        if len(recent_accuracies) < 5:  # Need at least 5 data points
            return False

        # Calculate average recent accuracy
        avg_accuracy = np.mean(recent_accuracies[-self.evaluation_window:])

        # Retrain if accuracy below threshold
        if avg_accuracy < self.retrain_threshold:
            logger.warning(f"⚠️ Accuracy {avg_accuracy:.2%} below threshold {self.retrain_threshold:.2%}")
            return True

        return False

    def retrain_model(self, retrain_date: date):
        """Retrain model with data up to retrain_date."""
        logger.info(f"🔄 RETRAINING model with data up to {retrain_date}")

        end_date = self.fetcher.prev_bday(retrain_date)
        start_date = end_date - timedelta(days=730)  # 2 years lookback

        # Load data
        bhav = self.fetcher.load_bhav_range(start_date, end_date)

        if bhav.empty:
            logger.error(f"❌ No data available for retraining period {start_date} to {end_date}")
            return

        # Retrain
        self.stock_picker.train(bhav, train_end=end_date)

        self.retrain_dates.append(retrain_date)
        logger.info(f"✅ Model retrained successfully (retrain #{len(self.retrain_dates)})")

    def run_backtest(
        self,
        start_date: date,
        end_date: date,
        initial_train_date: Optional[date] = None
    ) -> pd.DataFrame:
        """
        Run adaptive backtest over date range.

        Args:
            start_date: Start date for backtesting
            end_date: End date for backtesting
            initial_train_date: Date to train initial model (defaults to start_date)

        Returns:
            DataFrame with backtest results
        """
        if initial_train_date is None:
            initial_train_date = start_date

        logger.info(f"📈 Starting Adaptive Backtest from {start_date} to {end_date}")
        logger.info(f"⚙️ Retrain threshold: {self.retrain_threshold:.2%}")
        logger.info(f"⚙️ Evaluation window: {self.evaluation_window} predictions")

        # Initialize model
        self.initialize_model(initial_train_date)

        # Generate list of trading days
        current_date = start_date
        trading_days = []
        while current_date <= end_date:
            if self.fetcher.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        logger.info(f"📅 Testing {len(trading_days)} trading days")

        # Track recent accuracies for adaptive retraining
        recent_accuracies = []

        # Run backtest day by day
        for i, test_date in enumerate(trading_days):
            # Skip if we don't have enough future data to evaluate
            if i >= len(trading_days) - self.forward_period:
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"📅 Backtesting day {i+1}/{len(trading_days)}: {test_date}")

            # Make predictions for this date
            try:
                predictions = self.stock_picker.predict(as_of_date=test_date)

                if predictions.empty:
                    logger.warning(f"⚠️ No predictions for {test_date}, skipping")
                    continue

                logger.info(f"🎯 Generated {len(predictions)} predictions")

                # Wait for forward_period and get actual outcomes
                # (In real backtest, we already have this data)
                results = self.get_actual_outcomes(predictions, test_date, self.forward_period)

                if results.empty:
                    logger.warning(f"⚠️ No outcome data available for {test_date}")
                    continue

                # Calculate accuracy
                accuracy = self.calculate_accuracy(results)
                recent_accuracies.append(accuracy)
                self.accuracy_history.append({
                    'date': test_date,
                    'accuracy': accuracy,
                    'n_predictions': len(predictions),
                    'n_evaluated': results['actual_gain'].notna().sum()
                })

                logger.info(f"📊 Accuracy: {accuracy:.2%} ({results['actual_gain'].notna().sum()}/{len(predictions)} evaluated)")

                # Store results
                results['backtest_date'] = test_date
                self.backtest_results.append(results)

                # Check if we should retrain
                if self.should_retrain(test_date, recent_accuracies):
                    self.retrain_model(test_date)
                    # Clear recent accuracies after retrain to give model fresh start
                    recent_accuracies = []

            except Exception as e:
                logger.error(f"❌ Error backtesting {test_date}: {e}")
                continue

        # Compile results
        if self.backtest_results:
            all_results = pd.concat(self.backtest_results, ignore_index=True)
        else:
            all_results = pd.DataFrame()

        self.print_summary()

        return all_results

    def print_summary(self):
        """Print backtest summary."""
        logger.info(f"\n{'='*60}")
        logger.info("📊 ADAPTIVE BACKTEST SUMMARY")
        logger.info(f"{'='*60}")

        if not self.accuracy_history:
            logger.info("❌ No backtest results to summarize")
            return

        acc_df = pd.DataFrame(self.accuracy_history)

        logger.info(f"📈 Total days tested: {len(acc_df)}")
        logger.info(f"🔄 Number of retrains: {len(self.retrain_dates)}")
        logger.info(f"📊 Average accuracy: {acc_df['accuracy'].mean():.2%}")
        logger.info(f"📊 Median accuracy: {acc_df['accuracy'].median():.2%}")
        logger.info(f"📊 Best accuracy: {acc_df['accuracy'].max():.2%}")
        logger.info(f"📊 Worst accuracy: {acc_df['accuracy'].min():.2%}")

        if self.retrain_dates:
            logger.info(f"\n🔄 Retrain dates:")
            for i, retrain_date in enumerate(self.retrain_dates, 1):
                logger.info(f"  {i}. {retrain_date}")

        # Show accuracy trend
        if len(acc_df) >= 10:
            logger.info(f"\n📈 Accuracy trend:")
            logger.info(f"  First 10 days avg: {acc_df.head(10)['accuracy'].mean():.2%}")
            logger.info(f"  Last 10 days avg: {acc_df.tail(10)['accuracy'].mean():.2%}")

    def save_results(self, output_path: str = "adaptive_backtest_results.csv"):
        """Save backtest results to CSV."""
        if self.backtest_results:
            all_results = pd.concat(self.backtest_results, ignore_index=True)
            all_results.to_csv(output_path, index=False)
            logger.info(f"💾 Results saved to {output_path}")

        # Save accuracy history
        if self.accuracy_history:
            acc_df = pd.DataFrame(self.accuracy_history)
            acc_path = output_path.replace('.csv', '_accuracy.csv')
            acc_df.to_csv(acc_path, index=False)
            logger.info(f"💾 Accuracy history saved to {acc_path}")


def main():
    """Run adaptive backtest example."""
    import sys

    # Default dates: backtest last 60 days
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=60)

    # Parse command line args if provided
    if len(sys.argv) >= 3:
        start_date = datetime.strptime(sys.argv[1], '%Y-%m-%d').date()
        end_date = datetime.strptime(sys.argv[2], '%Y-%m-%d').date()

    # Initialize backtester
    backtester = AdaptiveBacktester(
        retrain_threshold=0.55,      # Retrain if accuracy < 55%
        min_retrain_interval=10,     # Min 10 days between retrains
        forward_period=5,            # 5-session prediction
        evaluation_window=20         # Evaluate over 20 predictions
    )

    # Run backtest
    results = backtester.run_backtest(
        start_date=start_date,
        end_date=end_date,
        initial_train_date=start_date - timedelta(days=30)  # Train 30 days before backtest
    )

    # Save results
    backtester.save_results()

    print("\n✅ Adaptive backtest complete!")
    print(f"📊 Results saved to adaptive_backtest_results.csv")


if __name__ == "__main__":
    main()
