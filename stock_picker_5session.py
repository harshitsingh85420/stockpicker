"""
5-Session Stock Picker - ML-Enhanced Momentum/Breakout System

Intention:
- Run today → tells which stocks to buy tomorrow → expects positive close in 5 sessions
- Uses BSE official BhavCopy data
- ML model learns which momentum/breakout patterns lead to 5-session gains
- Shows ALL qualifying stocks (no arbitrary limit)
- Daily retraining - model learns continuously
"""

import os
import pickle
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score

# Import our modules
from bse_loader import BSEDataFetcher
from momentum_features import prepare_features_all, add_forward_returns


class StockPicker5Session:
    """
    5-Session Stock Picker using ML to predict positive closes
    """

    def __init__(self, base_dir: str = "./stock_picker_data", auto_load_model: bool = False):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.results_dir = self.base_dir / "results"

        # Create directories
        for d in [self.base_dir, self.models_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.LOOKBACK_DAYS = 730  # 2 years of data
        self.FORWARD_PERIOD = 5  # Predict 5-session ahead
        self.MIN_DATA_POINTS = 200  # Minimum rows per stock
        self.INITIAL_THRESHOLD = 0.62  # Starting probability threshold
        self.MIN_THRESHOLD = 0.52  # Minimum threshold
        self.THRESHOLD_STEP = 0.02  # Threshold adjustment step

        # Initialize data fetcher
        self.fetcher = BSEDataFetcher()

        # Model will be set during training or loading
        self.model = None
        self.feature_cols = None

        # Auto-load model if requested
        if auto_load_model:
            self.load_model()

    def load_model(self) -> bool:
        """
        Load existing trained model from disk
        Returns True if model was loaded successfully, False otherwise
        """
        model_path = self.models_dir / "model_5session.pkl"

        if not model_path.exists():
            print("⚠️  No trained model found. Please train a model first.")
            return False

        try:
            print(f"📦 Loading model from: {model_path}")
            with open(model_path, 'rb') as f:
                save_package = pickle.load(f)

            self.model = save_package['model']
            self.feature_cols = save_package['feature_cols']

            # Load configuration
            if 'config' in save_package:
                config = save_package['config']
                self.LOOKBACK_DAYS = config.get('LOOKBACK_DAYS', self.LOOKBACK_DAYS)
                self.FORWARD_PERIOD = config.get('FORWARD_PERIOD', self.FORWARD_PERIOD)
                self.MIN_DATA_POINTS = config.get('MIN_DATA_POINTS', self.MIN_DATA_POINTS)

            # Show metadata
            if 'metadata' in save_package:
                metadata = save_package['metadata']
                print(f"   ✅ Model loaded successfully!")
                print(f"   📅 Trained: {metadata.get('train_date', 'Unknown')}")
                print(f"   📊 CV AUC: {metadata.get('cv_mean', 0):.4f}")
                print(f"   🔢 Training samples: {metadata.get('n_training_samples', 0):,}")

            return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def model_exists(self) -> bool:
        """Check if a trained model exists"""
        model_path = self.models_dir / "model_5session.pkl"
        return model_path.exists()

    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch raw BhavCopy data from BSE

        CRITICAL: Trains on ALL 5600+ BSE stocks (no filtering!)
        User requirement: "don't filter based on liquidity..nor price..
                          nor you only train with top 500 or anything..
                          do train with every stock..strictly"

        Returns:
            (raw_bhav, features_with_labels)
        """
        print("\n" + "=" * 80)
        print("📥 STEP 1: FETCH BSE DATA")
        print("=" * 80)

        # Date range
        end_date = self.fetcher.prev_bday(date.today())
        start_date = end_date - timedelta(days=self.LOOKBACK_DAYS)

        print(f"Date range: {start_date} → {end_date}")

        # Fetch bhav data
        bhav = self.fetcher.fetch_bhav_range(start_date, end_date)

        print(f"✅ Fetched {len(bhav):,} rows | {bhav['SC_CODE'].nunique()} unique stocks")

        # Get stock universe (stocks with enough data)
        qualified_stocks = self.fetcher.get_stock_universe(bhav, self.MIN_DATA_POINTS)

        # Filter to qualified stocks only (minimum data requirement)
        bhav_qualified = bhav[bhav['SC_CODE'].isin(qualified_stocks)].copy()

        # TRAIN ON ALL QUALIFIED STOCKS - NO FILTERING!
        bhav_train = bhav_qualified.copy()
        print(f"📊 Training on ALL {len(qualified_stocks)} qualified stocks")
        print(f"   ✅ COMPREHENSIVE MODE: Every stock with ≥{self.MIN_DATA_POINTS} data points")
        print(f"   ✅ NO liquidity filtering")
        print(f"   ✅ NO price filtering")
        print(f"   ✅ NO arbitrary limits")
        print(f"   📈 Expected universe: 5600+ BSE stocks (filtered only by data availability)")

        # Compute features
        print("\n" + "=" * 80)
        print("🔧 STEP 2: COMPUTE MOMENTUM/BREAKOUT FEATURES")
        print("=" * 80)

        features = prepare_features_all(bhav_train)

        # Add forward returns and labels
        features_with_labels = add_forward_returns(features, periods=[self.FORWARD_PERIOD])

        return bhav, features_with_labels

    def prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features (X) and labels (y) for training
        """
        print("\n" + "=" * 80)
        print("🎯 STEP 3: PREPARE TRAINING DATA")
        print("=" * 80)

        # Feature columns (all the momentum/breakout indicators)
        feature_cols = [
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

        # Filter to rows with valid labels
        label_col = f"Label_fwd{self.FORWARD_PERIOD}_positive"
        df_train = features_df.dropna(subset=[label_col]).copy()

        # Remove rows with missing features (keep only complete cases)
        df_train = df_train.dropna(subset=feature_cols)

        print(f"📊 Training samples: {len(df_train):,}")
        print(f"   Positive labels: {df_train[label_col].sum():,} ({df_train[label_col].mean() * 100:.1f}%)")
        print(f"   Negative labels: {(df_train[label_col] == 0).sum():,} ({(1 - df_train[label_col].mean()) * 100:.1f}%)")

        X = df_train[feature_cols].copy()
        y = df_train[label_col].copy()

        # Store feature names
        self.feature_cols = feature_cols

        return X, y

    def train_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Train LightGBM model to predict 5-session positive close
        """
        print("\n" + "=" * 80)
        print("🤖 STEP 4: TRAIN ML MODEL")
        print("=" * 80)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

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

            model = lgb.train(
                params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)]
            )

            y_pred = model.predict(X_val)
            auc = roc_auc_score(y_val, y_pred)
            acc = accuracy_score(y_val, y_pred > 0.5)
            cv_scores.append(auc)

            print(f"   Fold {fold}: AUC = {auc:.4f}, Accuracy = {acc:.4f}")

        print(f"\n✅ CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

        # Train final model on all data
        print("\n📚 Training final model on all data...")
        train_data = lgb.Dataset(X, label=y)

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            callbacks=[lgb.log_evaluation(0)]
        )

        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        print(f"\n📊 Top 10 Most Important Features:")
        for idx, row in importance.head(10).iterrows():
            print(f"   {row['feature']:20s} : {row['importance']:.1f}")

        # Save model with complete metadata
        model_path = self.models_dir / "model_5session.pkl"

        # Prepare comprehensive save package
        save_package = {
            # Model
            'model': self.model,
            'feature_cols': self.feature_cols,

            # Configuration
            'config': {
                'LOOKBACK_DAYS': self.LOOKBACK_DAYS,
                'FORWARD_PERIOD': self.FORWARD_PERIOD,
                'MIN_DATA_POINTS': self.MIN_DATA_POINTS,
                'INITIAL_THRESHOLD': self.INITIAL_THRESHOLD,
                'MIN_THRESHOLD': self.MIN_THRESHOLD,
                'THRESHOLD_STEP': self.THRESHOLD_STEP,
            },

            # Training metadata
            'metadata': {
                'train_date': date.today(),
                'cv_scores': cv_scores,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'n_training_samples': len(X),
                'n_features': len(self.feature_cols),
                'feature_importance': importance.to_dict('records'),
            },

            # Data statistics
            'data_stats': {
                'positive_ratio': y.mean(),
                'negative_ratio': 1 - y.mean(),
                'total_samples': len(y),
            }
        }

        with open(model_path, 'wb') as f:
            pickle.dump(save_package, f)
        print(f"\n💾 Model + Config + Metadata saved to: {model_path}")

    def predict(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict on latest data
        Returns all stocks with probabilities
        """
        print("\n" + "=" * 80)
        print("🔮 STEP 5: PREDICT ON LATEST DATA")
        print("=" * 80)

        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Get latest date
        latest_date = features_df['DATE'].max()
        df_latest = features_df[features_df['DATE'] == latest_date].copy()

        print(f"📅 Prediction date: {latest_date}")
        print(f"📊 Stocks to predict: {len(df_latest)}")

        # Prepare features
        df_predict = df_latest.dropna(subset=self.feature_cols).copy()
        X_pred = df_predict[self.feature_cols]

        # Predict
        probabilities = self.model.predict(X_pred)

        # Create results DataFrame
        results = pd.DataFrame({
            'SC_CODE': df_predict['SC_CODE'],
            'SC_NAME': df_predict['SC_NAME'],
            'Close': df_predict['Close'],
            'Probability': probabilities,
            'VolMult': df_predict['VolMult'],
            'RS_Composite': df_predict['RS_Composite'],
            'ADX14': df_predict['ADX14'],
            'RSI14': df_predict['RSI14'],
            'DistTo52W': df_predict['DistTo52W'],
            'Break63_Today': df_predict['Break63_Today']
        })

        return results.sort_values('Probability', ascending=False).reset_index(drop=True)

    def select_picks(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Select ALL qualifying stocks above threshold
        """
        print("\n" + "=" * 80)
        print("🎯 STEP 6: SELECT ALL QUALIFYING STOCKS")
        print("=" * 80)

        print(f"📊 Probability distribution:")
        print(f"   Max: {predictions['Probability'].max():.4f}")
        print(f"   Mean: {predictions['Probability'].mean():.4f}")
        print(f"   Min: {predictions['Probability'].min():.4f}")

        # Find highest threshold that gives us stocks
        threshold = self.INITIAL_THRESHOLD
        picks = pd.DataFrame()

        print(f"\n🎯 Finding ALL stocks above threshold (starts at {self.INITIAL_THRESHOLD}):")

        while threshold >= self.MIN_THRESHOLD:
            picks_at_threshold = predictions[predictions['Probability'] >= threshold].copy()
            print(f"   Threshold {threshold:.2f}: {len(picks_at_threshold)} stocks")

            if len(picks_at_threshold) > 0 and picks.empty:
                picks = picks_at_threshold
                final_threshold = threshold

            threshold -= self.THRESHOLD_STEP

        if picks.empty:
            picks = predictions[predictions['Probability'] >= self.MIN_THRESHOLD].copy()
            final_threshold = self.MIN_THRESHOLD

        picks = picks.sort_values('Probability', ascending=False).reset_index(drop=True)
        picks['Rank'] = range(1, len(picks) + 1)

        print(f"\n✅ Final threshold: {final_threshold:.2f}")
        print(f"✅ Total qualifying stocks: {len(picks)}")

        return picks, final_threshold

    def display_picks(self, picks: pd.DataFrame, threshold: float):
        """
        Display picks to user
        """
        print("\n" + "=" * 80)
        print(f"🏆 ALL {len(picks)} QUALIFYING STOCKS FOR {date.today()}")
        print("=" * 80)

        print(f"\n📊 Statistics:")
        print(f"   Total picks: {len(picks)}")
        print(f"   Threshold used: {threshold:.2f}")
        print(f"   Avg probability: {picks['Probability'].mean():.4f}")

        print(f"\n📋 Top 20 Picks:")
        print(picks.head(20).to_string(index=False))

        # Save to CSV
        today_str = date.today().strftime("%Y%m%d")
        csv_path = self.results_dir / f"picks_{today_str}.csv"
        picks.to_csv(csv_path, index=False)
        print(f"\n💾 All {len(picks)} picks saved to: {csv_path}")

        return csv_path


def run_daily(use_existing_model: bool = True):
    """
    Daily mode: Fetch data, optionally train model, predict, show ALL qualifying stocks

    CRITICAL: Trains on ALL 5600+ BSE stocks (no filtering by liquidity/price/limit)

    Args:
        use_existing_model: If True, use existing model instead of retraining (faster!)
    """
    print("\n" + "=" * 80)
    print("🎯 5-SESSION STOCK PICKER - DAILY MODE")
    print("=" * 80)
    print("\nIntention:")
    print("   • Run today → tells which stocks to buy tomorrow")
    print("   • Expects positive close in 5 sessions")
    print("   • NSE data (priority) → BSE fallback")
    print("   • Trains on ALL 5600+ BSE stocks (comprehensive mode)")
    print("   • Shows ALL qualifying stocks (no limit!)")
    print("=" * 80)

    picker = StockPicker5Session()

    # Check if we should use existing model
    should_train = True
    if use_existing_model and picker.model_exists():
        print("\n📦 Existing model found!")
        print("   Loading instead of retraining (saves 5-10 minutes)...")
        if picker.load_model():
            should_train = False
        else:
            print("   Load failed, will train new model...")

    # Fetch & prepare data (ALL STOCKS - no filtering!)
    raw_bhav, features_with_labels = picker.fetch_data()

    # Train if needed
    if should_train:
        print("\n🎓 Training new model on ALL stocks...")
        X, y = picker.prepare_training_data(features_with_labels)
        picker.train_model(X, y)
    else:
        print("\n✅ Using existing model (no training needed)")

    # Predict on latest
    predictions = picker.predict(features_with_labels)

    # Select ALL qualifying
    picks, threshold = picker.select_picks(predictions)

    # Display
    picker.display_picks(picks, threshold)

    print("\n✅ DONE! Check the CSV file for all picks.")


if __name__ == "__main__":
    import sys

    # No more n_stocks parameter - always trains on ALL stocks
    # User can pass 'retrain' as argument to force retraining
    force_retrain = len(sys.argv) > 1 and sys.argv[1].lower() == 'retrain'
    use_existing = not force_retrain

    run_daily(use_existing_model=use_existing)
