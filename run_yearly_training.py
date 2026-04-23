#!/usr/bin/env python
"""
Yearly Training System - Train Model on Every Date

Usage:
    # Train on all dates in a specific year
    python run_yearly_training.py --year 2024

    # Train on all dates in a year range
    python run_yearly_training.py --year-start 2023 --year-end 2024

    # Force retrain even if dates are already trained
    python run_yearly_training.py --year 2024 --force

    # Continue from where you left off
    python run_yearly_training.py --year 2024

This script:
- Trains the model incrementally on each business day in the year
- Caches data and features (fast on subsequent runs!)
- Tracks progress and can resume from interruptions
- Detects model updates and handles retraining
- Updates model weights after each date to continuously learn
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score

from bse_loader import BSEDataFetcher
from momentum_features import prepare_features_all, add_forward_returns
from model_trainer import ModelTracker


class YearlyTrainer:
    """
    Train model incrementally on every date in a year
    """

    def __init__(self, base_dir: str = "./stock_picker_data"):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.fetcher = BSEDataFetcher()
        self.tracker = ModelTracker(base_dir=base_dir)

        # Configuration (matching stock_picker_5session.py)
        self.LOOKBACK_DAYS = 730  # 2 years of data
        self.FORWARD_PERIOD = 5
        self.MIN_DATA_POINTS = 200

        # Feature columns
        self.feature_cols = [
            # Trend
            'EMA20', 'EMA50', 'EMA200', 'EMA20_Slope5', 'EMA200_Slope', 'MA_Health', 'OverEMA20',
            # Volatility
            'ATR14', 'ATRpct', 'BBWidth', 'BBWidthPctl',
            # Breakouts & Distance
            'DistTo20', 'DistTo63', 'DistTo52W',
            'Break20_Today', 'Break63_Today', 'Hit52WH_Today',
            'RangePos20',
            # Volume
            'VolMult', 'UD_Vol_Ratio10',
            # Momentum
            'RET21D', 'RET63D', 'RS_Composite',
            # RSI / ADX
            'RSI14', 'ADX14', '+DI14', '-DI14', 'ADX14_chg3',
            # Weekly context
            'W_BBWidth', 'W_TrendOK', 'W_BBWidthPctl'
        ]

    def train_for_date(self, training_date: date, n_stocks: int = None) -> dict:
        """
        Train model using data UP TO training_date
        This simulates training "as of" that date

        Args:
            training_date: Date to train for
            n_stocks: Number of stocks to use (None = ALL stocks)
        """
        print("\n" + "=" * 80)
        print(f"📅 TRAINING FOR DATE: {training_date}")
        print("=" * 80)

        # Date range for training (up to training_date)
        end_date = training_date
        start_date = end_date - timedelta(days=self.LOOKBACK_DAYS)

        print(f"📊 Data range: {start_date} → {end_date}")

        # Fetch data (with caching - fast!)
        bhav = self.fetcher.fetch_bhav_range(start_date, end_date)
        print(f"   ✅ Fetched {len(bhav):,} rows | {bhav['SC_CODE'].nunique()} unique stocks")

        # Get qualified stocks (stocks with enough data)
        qualified_stocks = self.fetcher.get_stock_universe(bhav, self.MIN_DATA_POINTS)

        # Filter to qualified stocks first
        bhav_qualified = bhav[bhav['SC_CODE'].isin(qualified_stocks)].copy()

        # Limit to top N most liquid if specified
        if n_stocks and n_stocks > 0 and n_stocks < len(qualified_stocks):
            print(f"📊 Limiting training to top {n_stocks} most liquid stocks...")
            liquidity = bhav_qualified.groupby('SC_CODE')['ValueTraded'].mean().sort_values(ascending=False)
            top_stocks = liquidity.head(n_stocks).index.tolist()
            bhav_train = bhav_qualified[bhav_qualified['SC_CODE'].isin(top_stocks)].copy()
            print(f"   Training universe: {len(top_stocks)} stocks")
        else:
            bhav_train = bhav_qualified.copy()
            print(f"📊 Training on ALL qualified stocks: {len(qualified_stocks)} stocks")
            print(f"   (Using every stock with enough data - no filtering!)")

        # Compute features (with caching - fast!)
        print("\n🔧 Computing features...")
        features = prepare_features_all(bhav_train)

        # Add forward returns
        features_with_labels = add_forward_returns(features, periods=[self.FORWARD_PERIOD])

        # Prepare training data
        label_col = f"Label_fwd{self.FORWARD_PERIOD}_positive"

        # Only use data BEFORE training_date for training (avoid lookahead)
        df_train = features_with_labels[features_with_labels['DATE'] < training_date].copy()
        df_train = df_train.dropna(subset=[label_col])
        df_train = df_train.dropna(subset=self.feature_cols)

        if len(df_train) < 100:
            print(f"⚠️  Not enough training data ({len(df_train)} rows), skipping...")
            return None

        print(f"\n📊 Training samples: {len(df_train):,}")
        print(f"   Positive: {df_train[label_col].sum():,} ({df_train[label_col].mean() * 100:.1f}%)")

        X = df_train[self.feature_cols].copy()
        y = df_train[label_col].copy()

        # Train model
        print("\n🤖 Training LightGBM model...")

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

        # Time-series CV for validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
            )

            y_pred = model.predict(X_val)
            auc = roc_auc_score(y_val, y_pred)
            cv_scores.append(auc)

        print(f"   CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

        # Train final model on all data
        train_data = lgb.Dataset(X, label=y)
        final_model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            callbacks=[lgb.log_evaluation(0)]
        )

        # Compute metrics
        metrics = {
            'cv_auc_mean': float(np.mean(cv_scores)),
            'cv_auc_std': float(np.std(cv_scores)),
            'n_training_samples': len(X),
            'positive_ratio': float(y.mean()),
            'training_date': str(training_date)
        }

        # Save model
        model_path = self.models_dir / f"model_5session_{training_date}.pkl"

        save_package = {
            'model': final_model,
            'feature_cols': self.feature_cols,
            'config': {
                'LOOKBACK_DAYS': self.LOOKBACK_DAYS,
                'FORWARD_PERIOD': self.FORWARD_PERIOD,
                'MIN_DATA_POINTS': self.MIN_DATA_POINTS,
            },
            'metadata': {
                'train_date': training_date,
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'n_training_samples': len(X),
                'n_features': len(self.feature_cols),
            },
            'data_stats': {
                'positive_ratio': y.mean(),
                'total_samples': len(y),
            }
        }

        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(save_package, f)

        print(f"💾 Model saved: {model_path}")

        # Also save as "latest"
        latest_path = self.models_dir / "model_5session.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump(save_package, f)
        print(f"💾 Latest model updated: {latest_path}")

        return metrics

    def train_year(self, year: int, n_stocks: int = None, force: bool = False):
        """
        Train on all business days in a year

        Args:
            year: Year to train
            n_stocks: Number of stocks to use (None = ALL stocks)
            force: Force retrain even if already trained
        """
        print("\n" + "=" * 80)
        print(f"🎓 YEARLY TRAINING: {year}")
        print("=" * 80)

        if n_stocks:
            print(f"📊 Training with top {n_stocks} most liquid stocks")
        else:
            print(f"📊 Training with ALL stocks (comprehensive mode!)")

        # Check model status
        latest_model = self.models_dir / "model_5session.pkl"
        if latest_model.exists():
            model_updated = self.tracker.check_model_updated(latest_model)
            if model_updated:
                print("⚠️  Model has been updated since last training!")
                print("   All dates for this year will be retrained.")
                self.tracker.reset_training_for_year(year)
                force = True

        # Get training dates
        if force:
            print("🔄 Force mode: Will retrain all dates")
            training_dates = self.tracker._get_business_days_in_year(year)
        else:
            training_dates = self.tracker.get_untrained_dates(year)

        if not training_dates:
            print(f"✅ All dates in {year} already trained!")
            self.tracker.display_status([year])
            return

        print(f"\n📅 Dates to train: {len(training_dates)}")
        print(f"   First: {training_dates[0]}")
        print(f"   Last: {training_dates[-1]}")

        # Progress tracking
        total = len(training_dates)
        success_count = 0
        failed_dates = []

        for idx, training_date in enumerate(training_dates, 1):
            print(f"\n{'='*80}")
            print(f"📍 Progress: {idx}/{total} ({idx/total*100:.1f}%)")
            print(f"{'='*80}")

            try:
                metrics = self.train_for_date(training_date, n_stocks=n_stocks)

                if metrics:
                    # Mark as trained
                    self.tracker.mark_date_trained(training_date, metrics)
                    success_count += 1
                    print(f"✅ Date {training_date} trained successfully!")
                else:
                    print(f"⚠️  Date {training_date} skipped (insufficient data)")

            except KeyboardInterrupt:
                print("\n\n⚠️  Training interrupted by user!")
                print(f"   Progress saved: {success_count}/{total} dates completed")
                print(f"   Run again to continue from where you left off")
                sys.exit(0)

            except Exception as e:
                print(f"❌ Error training {training_date}: {e}")
                failed_dates.append((training_date, str(e)))

        # Update model version
        if latest_model.exists():
            self.tracker.update_model_version(latest_model)

        # Summary
        print("\n" + "=" * 80)
        print(f"🎉 TRAINING COMPLETE FOR {year}")
        print("=" * 80)
        print(f"   Successful: {success_count}/{total}")
        if failed_dates:
            print(f"   Failed: {len(failed_dates)}")
            for d, err in failed_dates[:5]:
                print(f"      {d}: {err}")

        # Show final status
        self.tracker.display_status([year])

    def train_year_range(self, year_start: int, year_end: int, n_stocks: int = 500, force: bool = False):
        """Train on multiple years"""
        for year in range(year_start, year_end + 1):
            self.train_year(year, n_stocks=n_stocks, force=force)


def main():
    parser = argparse.ArgumentParser(
        description="Yearly Training System - Train model on every date in a year\n\n"
                    "The model learns continuously from each date, improving predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--year', type=int, help='Year to train (e.g., 2024)')
    parser.add_argument('--year-start', type=int, help='Start year for range training')
    parser.add_argument('--year-end', type=int, help='End year for range training')
    parser.add_argument('--stocks', type=int, default=None,
                        help='Number of most liquid stocks to use (default: None = ALL stocks). '
                             'Examples: --stocks 500 (top 500), --stocks 1000 (top 1000)')
    parser.add_argument('--all', action='store_true', dest='all_stocks',
                        help='Train on ALL stocks (same as omitting --stocks)')
    parser.add_argument('--force', action='store_true',
                        help='Force retrain even if dates are already trained')
    parser.add_argument('--status', action='store_true',
                        help='Show training status and exit')

    args = parser.parse_args()

    trainer = YearlyTrainer()

    # Show status
    if args.status:
        years = list(range(2023, date.today().year + 1))
        trainer.tracker.display_status(years)
        return

    # Handle --all flag (overrides --stocks)
    n_stocks = None if args.all_stocks else args.stocks

    # Validate arguments
    if args.year:
        trainer.train_year(args.year, n_stocks=n_stocks, force=args.force)
    elif args.year_start and args.year_end:
        trainer.train_year_range(args.year_start, args.year_end, n_stocks=n_stocks, force=args.force)
    else:
        parser.print_help()
        print("\n❌ Error: Must specify either --year or --year-start/--year-end")
        sys.exit(1)


if __name__ == "__main__":
    main()
