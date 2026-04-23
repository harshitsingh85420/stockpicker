"""
5-Session Stock Picker - Core Pipeline Module

This module contains all the core functionality from the notebook
converted into reusable functions for standalone operation.
"""

import os
import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score

from joblib import Memory, Parallel, delayed
from tqdm import tqdm

# Optional imports
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not available, will use BSE data only")

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# Import our proven modules
try:
    from indian_trading_system.indicators.technical import TechnicalIndicators
    from indian_trading_system.indicators.volatility import VolatilityEstimators
    from indian_trading_system.indicators.patterns import CandlestickPatterns
    from indian_trading_system.models.features import FeatureEngineer
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("⚠️ Warning: indian_trading_system modules not available, using fallbacks")


class StockPickerConfig:
    """Configuration for the stock picker system"""

    def __init__(self, base_dir: str = './stock_picker_data'):
        self.BASE_DIR = base_dir
        self.DATA_DIR = os.path.join(base_dir, 'data')
        self.STOCK_HISTORIES_DIR = os.path.join(base_dir, 'data', 'stock_histories')
        self.MODELS_DIR = os.path.join(base_dir, 'models')
        self.CACHE_DIR = os.path.join(base_dir, 'cache')
        self.RESULTS_DIR = os.path.join(base_dir, 'results')

        # Prediction parameters
        self.TARGET_GAIN = 1.5  # Minimum gain % over 5 sessions
        self.HOLDING_PERIOD = 5  # Trading sessions
        # No TARGET_PICKS - show ALL stocks that pass criteria
        self.INITIAL_THRESHOLD = 0.62
        self.MIN_THRESHOLD = 0.52
        self.THRESHOLD_STEP = 0.02

        # Risk filters - minimal filtering (F&O ban and ASM/GSM only)
        # No price or liquidity filters - include all stocks from penny to expensive

        # Data parameters
        self.LOOKBACK_DAYS = 730  # 2 years
        self.MIN_DATA_POINTS = 200

        # Model parameters
        self.RANDOM_STATE = 42
        self.N_CV_SPLITS = 5

        # Create directories
        for dir_path in [self.BASE_DIR, self.DATA_DIR, self.STOCK_HISTORIES_DIR,
                         self.MODELS_DIR, self.CACHE_DIR, self.RESULTS_DIR]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def log(message: str, level: str = 'INFO'):
    """Simple logging function"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {level}: {message}")


class DataLoader:
    """Download and cache stock data using BSE official data"""

    def __init__(self, config: StockPickerConfig):
        self.config = config
        self.memory = Memory(config.CACHE_DIR, verbose=0)

        # Import the proven working BSE fetcher with caching
        try:
            from bse_direct_loader import BSEDataFetcher
            cache_path = str(config.CACHE_DIR / 'bse')
            self.bse_fetcher = BSEDataFetcher(cache_dir=cache_path)
            self.use_bse = True
            log("✅ Using BSE official BhavCopy data with caching (proven working code)")
        except Exception as e:
            self.use_bse = False
            log(f"⚠️ BSE loader import failed: {e}, will try yfinance")

    def download_stock(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Download single stock data"""
        try:
            if self.use_bse:
                # BSE uses stock codes, try to get data by name
                from datetime import datetime
                start = datetime.strptime(start_date, '%Y-%m-%d').date()
                end = datetime.strptime(end_date, '%Y-%m-%d').date()

                # Fetch bhav data for date range
                df = self.bse_loader.fetch_bhav_range(start, end)

                # Clean symbol (remove .NS suffix)
                clean_symbol = symbol.replace('.NS', '').replace('.BO', '')

                # Find stock by name
                stock_df = df[df['SC_NAME'].str.contains(clean_symbol, case=False, na=False)]

                if stock_df.empty:
                    return None

                # Get the most common SC_CODE for this symbol
                sc_code = stock_df['SC_CODE'].mode()[0] if len(stock_df) > 0 else None
                if sc_code is None:
                    return None

                # Filter for this specific stock code
                stock_df = df[df['SC_CODE'] == sc_code].copy()

                if len(stock_df) < self.config.MIN_DATA_POINTS:
                    return None

                # Rename columns to lowercase
                stock_df = stock_df.rename(columns={
                    'DATE': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })

                # Ensure required columns
                stock_df = stock_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                stock_df = stock_df.sort_values('date').reset_index(drop=True)
                stock_df = stock_df.dropna()

                return stock_df

            else:
                # Fallback to yfinance
                if not YFINANCE_AVAILABLE:
                    log(f"Cannot download {symbol}: yfinance not available", 'ERROR')
                    return None

                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, auto_adjust=True)

                if df.empty or len(df) < self.config.MIN_DATA_POINTS:
                    return None

                df.columns = [col.lower() for col in df.columns]
                df = df.reset_index()
                df['date'] = pd.to_datetime(df['date'])
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                df = df.dropna()

                return df

        except Exception as e:
            log(f"Error downloading {symbol}: {str(e)}", 'ERROR')
            return None

    def download_multiple(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download multiple stocks - BSE fetches all at once for efficiency"""
        log(f"Downloading {len(symbols)} stocks...")

        if self.use_bse:
            try:
                from datetime import datetime as dt
                start = dt.strptime(start_date, '%Y-%m-%d').date()
                end = dt.strptime(end_date, '%Y-%m-%d').date()

                # Fetch all BSE data at once (much faster!) using proven code
                log("Fetching BSE BhavCopy data (official source)...")
                all_data = self.bse_fetcher.fetch_bhav_range(start, end)

                if all_data.empty:
                    log("❌ No BSE data fetched, falling back", 'WARNING')
                    raise Exception("Empty BSE data")

                log(f"✅ Fetched {len(all_data)} total records from BSE")

                # Extract data for each symbol
                stock_data = {}
                for symbol in tqdm(symbols, desc="Processing stocks"):
                    clean_symbol = symbol.replace('.NS', '').replace('.BO', '')

                    # Find stock by name
                    stock_df = all_data[all_data['SC_NAME'].str.contains(clean_symbol, case=False, na=False)]

                    if stock_df.empty:
                        continue

                    # Get most common SC_CODE
                    sc_code = stock_df['SC_CODE'].mode()[0] if len(stock_df) > 0 else None
                    if sc_code is None:
                        continue

                    # Filter for this stock code
                    stock_df = all_data[all_data['SC_CODE'] == sc_code].copy()

                    if len(stock_df) < self.config.MIN_DATA_POINTS:
                        continue

                    # Rename columns
                    stock_df = stock_df.rename(columns={
                        'DATE': 'date',
                        'Open': 'open',
                        'High': 'high',
                        'Low': 'low',
                        'Close': 'close',
                        'Volume': 'volume'
                    })

                    stock_df = stock_df[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
                    stock_df = stock_df.sort_values('date').reset_index(drop=True)
                    stock_df = stock_df.dropna()

                    stock_data[symbol] = stock_df

                log(f"✅ Downloaded {len(stock_data)}/{len(symbols)} stocks from BSE")
                return stock_data

            except Exception as e:
                log(f"BSE bulk download failed: {e}, falling back to yfinance", 'WARNING')
                # Fall through to individual downloads

        # Fallback: individual downloads with yfinance
        if not YFINANCE_AVAILABLE:
            log("❌ Cannot download stocks: neither BSE loader nor yfinance available", 'ERROR')
            log("💡 Install with: pip install yfinance", 'INFO')
            log("💡 Or ensure BSE data fetcher is working", 'INFO')
            return {}

        log("Using yfinance fallback...")
        results = Parallel(n_jobs=4)(
            delayed(self.download_stock)(symbol, start_date, end_date)
            for symbol in tqdm(symbols, desc="Downloading")
        )

        stock_data = {
            symbol: df
            for symbol, df in zip(symbols, results)
            if df is not None
        }

        log(f"Downloaded {len(stock_data)}/{len(symbols)} stocks")
        return stock_data


class RiskFilters:
    """Apply risk filters to stock universe"""

    @staticmethod
    def fetch_fno_ban_list() -> List[str]:
        """Fetch F&O ban list"""
        try:
            url = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                banned = df.iloc[:, 0].tolist() if not df.empty else []
                log(f"F&O ban list: {len(banned)} stocks")
                return [f"{s}.NS" for s in banned]
        except Exception as e:
            log(f"Error fetching F&O ban: {e}", 'WARNING')
        return []

    @staticmethod
    def apply_liquidity_filter(stock_data: Dict, min_turnover: float, lookback: int = 20):
        """Filter by minimum liquidity"""
        filtered = {}
        for symbol, df in stock_data.items():
            if len(df) < lookback:
                continue
            recent = df.tail(lookback).copy()
            recent['turnover'] = recent['close'] * recent['volume']
            if recent['turnover'].mean() >= min_turnover:
                filtered[symbol] = df
        log(f"Liquidity filter: {len(filtered)}/{len(stock_data)} passed")
        return filtered

    @staticmethod
    def apply_price_filter(stock_data: Dict, min_price: float, max_price: float):
        """Filter by price range"""
        filtered = {}
        for symbol, df in stock_data.items():
            price = df['close'].iloc[-1]
            if min_price <= price <= max_price:
                filtered[symbol] = df
        log(f"Price filter: {len(filtered)}/{len(stock_data)} passed")
        return filtered

    @staticmethod
    def fetch_asm_gsm_list() -> List[str]:
        """Fetch ASM/GSM surveillance stocks from NSE"""
        try:
            # ASM - Additional Surveillance Measure
            asm_url = "https://nsearchives.nseindia.com/surveillance/ASM.csv"
            # GSM - Graded Surveillance Measure
            gsm_url = "https://nsearchives.nseindia.com/surveillance/GSM.csv"

            headers = {'User-Agent': 'Mozilla/5.0'}
            surveillance_stocks = []

            for url in [asm_url, gsm_url]:
                try:
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        from io import StringIO
                        df = pd.read_csv(StringIO(response.text))
                        if not df.empty:
                            # Get symbol column (might be named differently)
                            symbol_col = [c for c in df.columns if 'symbol' in c.lower() or 'name' in c.lower()]
                            if symbol_col:
                                symbols = df[symbol_col[0]].tolist()
                                surveillance_stocks.extend(symbols)
                except Exception as e:
                    log(f"Could not fetch surveillance list from {url}: {e}", 'WARNING')

            surveillance_stocks = list(set(surveillance_stocks))
            log(f"ASM/GSM surveillance list: {len(surveillance_stocks)} stocks")
            return [f"{s}.NS" for s in surveillance_stocks]
        except Exception as e:
            log(f"Error fetching ASM/GSM: {e}", 'WARNING')
        return []

    @classmethod
    def apply_all(cls, stock_data: Dict, config: StockPickerConfig):
        """Apply minimal risk filters (F&O ban and ASM/GSM only)"""
        log("Applying risk filters (F&O ban, ASM/GSM)...")
        initial_count = len(stock_data)

        # F&O ban
        banned = cls.fetch_fno_ban_list()
        if banned:
            stock_data = {s: df for s, df in stock_data.items() if s not in banned}
            log(f"F&O ban filter: Excluded {initial_count - len(stock_data)} stocks")

        # ASM/GSM surveillance
        surveillance = cls.fetch_asm_gsm_list()
        if surveillance:
            before = len(stock_data)
            stock_data = {s: df for s, df in stock_data.items() if s not in surveillance}
            log(f"ASM/GSM filter: Excluded {before - len(stock_data)} stocks")

        # Note: No liquidity or price filters - include ALL stocks
        log(f"✅ Final universe: {len(stock_data)} stocks (all prices, all volumes)")
        return stock_data


class FeatureComputer:
    """Compute features for stocks with intelligent caching"""

    def __init__(self, cache_dir: str = './stock_picker_data/cache/features'):
        if MODULES_AVAILABLE:
            self.technical = TechnicalIndicators()
            self.volatility = VolatilityEstimators()
            self.patterns = CandlestickPatterns()
            self.engineer = FeatureEngineer()

        # Setup feature cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = True

    def _get_cache_key(self, symbol: str, df: pd.DataFrame) -> str:
        """Generate cache key based on symbol and data hash"""
        # Use first and last date + row count as cache key
        if len(df) == 0:
            return None
        first_date = df['date'].iloc[0] if 'date' in df.columns else str(df.index[0])
        last_date = df['date'].iloc[-1] if 'date' in df.columns else str(df.index[-1])
        return f"{symbol}_{first_date}_{last_date}_{len(df)}"

    def compute_features(self, df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """Compute all features with caching"""

        # Try to load from cache if symbol provided
        if self.cache_enabled and symbol:
            cache_key = self._get_cache_key(symbol, df)
            if cache_key:
                cache_file = self.cache_dir / f"{cache_key}.pkl"

                if cache_file.exists():
                    try:
                        cached_df = pd.read_pickle(cache_file)
                        # Verify cache is valid (same shape)
                        if len(cached_df) == len(df):
                            log(f"✅ Loaded cached features for {symbol}", 'DEBUG')
                            return cached_df
                    except Exception as e:
                        log(f"⚠️ Cache load failed for {symbol}: {e}", 'WARNING')

        # Compute features (cache miss or disabled)
        df_features = df.copy()

        if MODULES_AVAILABLE:
            df_features = self.technical.calculate_all(df_features)
            df_features = self.volatility.calculate_all(df_features)
            df_features = self.patterns.detect_all_patterns(df_features)
            df_features = self.patterns.calculate_pattern_strength(df_features)
            df_features = self.engineer.create_all_features(df_features)
        else:
            # Basic fallback features
            df_features = self._compute_basic_features(df_features)

        # 5-session specific features
        df_features = self._add_5session_features(df_features)

        # Save to cache if enabled
        if self.cache_enabled and symbol:
            cache_key = self._get_cache_key(symbol, df)
            if cache_key:
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                try:
                    df_features.to_pickle(cache_file)
                    log(f"💾 Cached features for {symbol}", 'DEBUG')
                except Exception as e:
                    log(f"⚠️ Cache save failed for {symbol}: {e}", 'WARNING')

        return df_features

    def _compute_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic features using pandas"""
        # Returns
        for period in [1, 3, 5, 10]:
            df[f'return_{period}d'] = df['close'].pct_change(period) * 100

        # Moving averages
        for period in [5, 10, 20]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # Volatility
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_10d'] = df['return_1d'].rolling(10).std()

        # Volume
        df['volume_ratio_5d'] = df['volume'] / df['volume'].rolling(5).mean()
        df['volume_ratio_20d'] = df['volume'] / df['volume'].rolling(20).mean()

        return df

    def _add_5session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 5-session specific features"""
        df['return_5d'] = df['close'].pct_change(5) * 100
        df['high_5d'] = df['high'].rolling(5).max()
        df['low_5d'] = df['low'].rolling(5).min()
        df['range_5d'] = (df['high_5d'] - df['low_5d']) / df['close'] * 100
        df['volume_5d_avg'] = df['volume'].rolling(5).mean()
        df['volume_ratio_5d'] = df['volume'] / df['volume_5d_avg']
        df['volatility_5d'] = df['return_1d'].rolling(5).std() if 'return_1d' in df.columns else 0
        return df


def generate_labels(df: pd.DataFrame, holding_period: int, target_gain: float) -> pd.DataFrame:
    """Generate binary labels"""
    df['forward_return'] = (df['close'].shift(-holding_period) / df['close'] - 1) * 100
    df['target'] = (df['forward_return'] >= target_gain).astype(int)
    df = df[:-holding_period].copy()
    df = df.dropna(subset=['forward_return', 'target'])
    return df


class LightGBMPredictor:
    """LightGBM model with proper feature tracking"""

    def __init__(self, config: StockPickerConfig):
        self.config = config
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'max_depth': 6,
            'random_state': config.RANDOM_STATE,
            'n_jobs': -1,
            'verbose': -1
        }
        self.model = None
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]):
        """Train model and store feature names"""
        log("Training LightGBM model...")
        self.feature_names = feature_names

        train_data = lgb.Dataset(X, label=y, feature_name=feature_names)
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=500,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        log(f"Model trained! Features: {len(self.feature_names)}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict with proper feature handling"""
        if self.feature_names is None:
            raise ValueError("Model not trained!")

        # Add missing features
        for feat in self.feature_names:
            if feat not in X.columns:
                X[feat] = 0

        # Select in correct order
        X_ordered = X[self.feature_names].fillna(0)
        return self.model.predict(X_ordered)

    def save(self, path: str):
        """Save model and features"""
        self.model.save_model(path)
        with open(path.replace('.txt', '_features.json'), 'w') as f:
            json.dump(self.feature_names, f)
        log(f"Model saved: {path}")

    def load(self, path: str):
        """Load model and features"""
        self.model = lgb.Booster(model_file=path)
        with open(path.replace('.txt', '_features.json'), 'r') as f:
            self.feature_names = json.load(f)
        log(f"Model loaded: {path}")


def build_stock_universe(max_stocks: int = None, from_bse_data: pd.DataFrame = None) -> List[str]:
    """
    Build comprehensive NSE/BSE stock universe (3000+ stocks)

    Args:
        max_stocks: Limit number of stocks (None = all 3000+ stocks)
        from_bse_data: If provided, extract all unique stock names from BSE data

    Returns:
        List of stock symbols with .NS suffix
    """
    log("Building comprehensive NSE/BSE stock universe (3000+ stocks)...")

    all_symbols = set()

    # Method 1: Extract from BSE data if available (gets ALL 3000+ stocks!)
    if from_bse_data is not None and not from_bse_data.empty:
        log("Extracting stocks from BSE BhavCopy data...")
        unique_stocks = from_bse_data['SC_NAME'].unique() if 'SC_NAME' in from_bse_data.columns else []
        for stock_name in unique_stocks:
            # Clean and add stock name
            if stock_name and isinstance(stock_name, str):
                clean_name = stock_name.strip().replace(' ', '').replace('-', '').upper()
                if len(clean_name) > 2:  # Valid symbol
                    all_symbols.add(clean_name)
        log(f"✅ Extracted {len(all_symbols)} stocks from BSE data")

    # Method 2: Fetch from NSE indices
    indices = [
        'NIFTY 50', 'NIFTY NEXT 50', 'NIFTY 100', 'NIFTY 200',
        'NIFTY 500', 'NIFTY MIDCAP 50', 'NIFTY MIDCAP 100', 'NIFTY MIDCAP 150',
        'NIFTY SMALLCAP 50', 'NIFTY SMALLCAP 100', 'NIFTY SMALLCAP 250',
        'NIFTY MICROCAP 250'
    ]

    for index in indices:
        try:
            # Try to fetch from NSE (may not work due to restrictions)
            url = f"https://www.nseindia.com/api/equity-stockIndices?index={index.replace(' ', '%20')}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for stock in data.get('data', []):
                    symbol = stock.get('symbol', '').strip()
                    if symbol and symbol not in ['NIFTY', 'BANKNIFTY']:
                        all_symbols.add(symbol)
                log(f"✅ Fetched {len(data.get('data', []))} from {index}")
        except Exception as e:
            log(f"⚠️ Could not fetch {index}: {str(e)}", 'WARNING')

    # Method 2: Add comprehensive fallback list (Top 500+ stocks)
    # This ensures we have good coverage even if API fails
    fallback_symbols = [
        # Nifty 50
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'ICICIBANK', 'SBIN',
        'BHARTIARTL', 'ITC', 'KOTAKBANK', 'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI',
        'BAJFINANCE', 'HCLTECH', 'WIPRO', 'ULTRACEMCO', 'TITAN', 'SUNPHARMA',
        'NESTLEIND', 'ONGC', 'TATAMOTORS', 'NTPC', 'POWERGRID', 'M&M', 'TECHM',
        'ADANIPORTS', 'COALINDIA', 'BAJAJFINSV', 'DRREDDY', 'INDUSINDBK', 'DIVISLAB',
        'SHREECEM', 'CIPLA', 'EICHERMOT', 'BRITANNIA', 'GRASIM', 'BPCL', 'HINDALCO',
        'TATASTEEL', 'APOLLOHOSP', 'UPL', 'TATACONSUM', 'HEROMOTOCO', 'JSWSTEEL',
        'BAJAJ-AUTO', 'SBILIFE', 'HDFCLIFE', 'ADANIENT',

        # Nifty Next 50
        'ADANIGREEN', 'ADANIPORTS', 'AMBUJACEM', 'APOLLOTYRE', 'ASHOKLEY', 'AUROPHARMA',
        'BANDHANBNK', 'BERGEPAINT', 'BEL', 'BOSCHLTD', 'COLPAL', 'CONCOR', 'COFORGE',
        'DABUR', 'DLF', 'DMART', 'GAIL', 'GODREJCP', 'HAVELLS', 'ICICIGI', 'ICICIPRULI',
        'IDEA', 'INDIGO', 'IOC', 'IRCTC', 'JINDALSTEL', 'LICHSGFIN', 'LUPIN', 'MARICO',
        'MCDOWELL-N', 'MUTHOOTFIN', 'NMDC', 'NYKAA', 'OFSS', 'PAGEIND', 'PETRONET',
        'PIDILITIND', 'PNB', 'RECLTD', 'SBICARD', 'SHRIRAMFIN', 'SIEMENS', 'TATAPOWER',
        'TORNTPHARM', 'TRENT', 'VEDL', 'ZOMATO', 'ZYDUSLIFE',

        # Mid Cap 100 (sample)
        'ABCAPITAL', 'ABFRL', 'ACC', 'ALKEM', 'ARE&M', 'ASTRAL', 'ATGL', 'AUROPHARMA',
        'BALKRISIND', 'BATAINDIA', 'BHARATFORG', 'BHEL', 'BIOCON', 'CANBK', 'CANFINHOME',
        'CHAMBLFERT', 'CHOLAFIN', 'CUMMINSIND', 'DEEPAKNTR', 'ESCORTS', 'EXIDEIND',
        'FEDERALBNK', 'GLENMARK', 'GMRINFRA', 'GODREJPROP', 'GSPL', 'HDFCAMC',
        'HINDPETRO', 'HONAUT', 'IDFCFIRSTB', 'INDUSTOWER', 'INTELLECT', 'IRFC',
        'JUBLFOOD', 'L&TFH', 'LALPATHLAB', 'LAURUSLABS', 'LTTS', 'MANAPPURAM',
        'MFSL', 'MGL', 'MOTHERSON', 'MPHASIS', 'MRF', 'NAM-INDIA', 'NATIONALUM',
        'NAUKRI', 'NAVINFLUOR', 'OBEROIRLTY', 'PERSISTENT', 'POLYCAB', 'PVRINOX',
        'RBLBANK', 'SAIL', 'SRTRANSFIN', 'SUPREMEIND', 'SYNGENE', 'TATACHEM',
        'TATACOMM', 'TATAELXSI', 'TIINDIA', 'TORNTPOWER', 'TVSMOTOR', 'UBL',
        'UNIONBANK', 'UPL', 'VOLTAS', 'WHIRLPOOL', 'YESBANK',

        # Small Cap (sample - add more for full coverage)
        'AARTIIND', 'AJANTPHARM', 'APLLTD', 'ASHOKLEY', 'ASTRAZEN', 'BAYERCROP',
        'BRIGADE', 'CGPOWER', 'COROMANDEL', 'CROMPTON', 'CUMMINSIND', 'DELTACORP',
        'DIXON', 'EMAMILTD', 'ENDURANCE', 'FORTIS', 'GILLETTE', 'GLAND', 'GRAPHITE',
        'GUJGASLTD', 'HFCL', 'HINDCOPPER', 'HINDZINC', 'HUDCO', 'IIFL', 'INDHOTEL',
        'INDIACEM', 'INDIGOPNTS', 'JKCEMENT', 'JKTYRE', 'JUBLPHARMA', 'JUSTDIAL',
        'KANSAINER', 'KEI', 'KPITTECH', 'LODHA', 'M&MFIN', 'MASTEK', 'MAXHEALTH',
        'MCX', 'METROPOLIS', 'MOTILALOFS', 'NATCOPHARM', 'NLCINDIA', 'PAYTM',
        'PGHH', 'PHOENIXLTD', 'POLYMED', 'PVR', 'RAJESHEXPO', 'RAMCOCEM',
        'RITES', 'ROUTE', 'SKFINDIA', 'SONACOMS', 'STAR', 'SUNDARMFIN',
        'SUNDRMFAST', 'SUPREMEPET', 'SWANENERGY', 'SYMPHONY', 'TATAINVEST',
        'THERMAX', 'TIMKEN', 'TRITURBINE', 'TTML', 'TV18BRDCST', 'UCOBANK',
        'UJJIVAN', 'ULTRACEMCO', 'VGUARD', 'VBL', 'VINATIORGA', 'VSTIND',
        'WELCORP', 'WELSPUNIND', 'WESTLIFE', 'ZEEL', 'ZENSARTECH'
    ]

    all_symbols.update(fallback_symbols)

    # Convert to list with .NS suffix
    symbols = sorted([f"{s}.NS" for s in all_symbols])

    # Apply limit if specified
    if max_stocks and len(symbols) > max_stocks:
        symbols = symbols[:max_stocks]
        log(f"Limited to {max_stocks} stocks")

    log(f"✅ Built universe: {len(symbols)} stocks")
    return symbols
