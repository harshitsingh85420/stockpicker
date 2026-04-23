"""
Momentum/Breakout Feature Engineering
Based on original rule-based screening logic
Features focus on trend strength, breakouts, volume surges, and relative strength
"""

import os
import pickle
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple
import warnings
warnings.filterwarnings("ignore")


def ema(s: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average"""
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average True Range"""
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n // 2).mean()


def bbands(close: pd.Series, n: int = 20, k: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands"""
    ma = close.rolling(n, min_periods=n // 2).mean()
    sd = close.rolling(n, min_periods=n // 2).std()
    upper = ma + k * sd
    lower = ma - k * sd
    return lower, ma, upper


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out


def adx_block(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    """ADX (Average Directional Index) + Directional Indicators"""
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move.clip(lower=0)
    minus_dm = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move.clip(lower=0)

    tr1 = (high - low)
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / n, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / n, adjust=False).mean() / atr_.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / n, adjust=False).mean()

    return plus_di, minus_di, adx


def compute_per_symbol_features(g: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all momentum/breakout features for a single symbol
    This is the core feature engineering from the original code
    """
    g = g.sort_values("DATE").copy()

    # ========== TREND BASELINES ==========
    g["EMA20"] = ema(g["Close"], 20)
    g["EMA50"] = ema(g["Close"], 50)
    g["EMA200"] = ema(g["Close"], 200)

    # ========== VOLATILITY ==========
    g["ATR14"] = atr(g, 14)
    g["ATRpct"] = g["ATR14"] / g["Close"] * 100.0

    # ========== HIGH RAILS (for breakouts) ==========
    # Previous day's rolling highs (shifted by 1 to avoid lookahead)
    g["High20_prev"] = g["High"].shift(1).rolling(20, min_periods=10).max()
    g["High63_prev"] = g["High"].shift(1).rolling(63, min_periods=30).max()
    g["High252_prev"] = g["High"].shift(1).rolling(252, min_periods=60).max()

    # ========== DISTANCES TO HIGHS ==========
    # Positive = below line (distance to reach high)
    g["DistTo20"] = (g["High20_prev"] - g["Close"]) / g["High20_prev"]
    g["DistTo63"] = (g["High63_prev"] - g["Close"]) / g["High63_prev"]
    g["DistTo52W"] = (g["High252_prev"] - g["Close"]) / g["High252_prev"]

    # ========== BREAKOUT FLAGS ==========
    g["Break20_Today"] = (g["Close"] >= g["High20_prev"]).astype(int)
    g["Break63_Today"] = (g["Close"] >= g["High63_prev"]).astype(int)
    g["Hit52WH_Today"] = (g["Close"] >= g["High252_prev"]).astype(int)

    # ========== POSITION IN RANGE ==========
    # Where is close within the 20-day range? (0 = low, 1 = high)
    low20 = g["Low"].rolling(20, min_periods=10).min()
    high20 = g["High"].rolling(20, min_periods=10).max()
    rng20 = (high20 - low20).replace(0, np.nan)
    g["RangePos20"] = ((g["Close"] - low20) / rng20).clip(0, 1).fillna(0.5)

    # ========== BOLLINGER BANDS ==========
    lb, mb, ub = bbands(g["Close"], n=20, k=2.0)
    g["BBLower"], g["BBMid"], g["BBUpper"] = lb, mb, ub
    g["BBWidth"] = (ub - lb) / mb.replace(0, np.nan)

    # ========== VOLUME ==========
    g["VolAvg20"] = g["Volume"].rolling(20, min_periods=10).mean()
    g["VolMult"] = g["Volume"] / g["VolAvg20"]

    # Up/Down volume ratio (10d) - proxy for buying/selling pressure
    up = (g["Close"] > g["Close"].shift(1)).astype(int)
    dn = (g["Close"] < g["Close"].shift(1)).astype(int)
    upv = (up * g["Volume"]).rolling(10, min_periods=5).sum()
    dnv = (dn * g["Volume"]).rolling(10, min_periods=5).sum()
    g["UD_Vol_Ratio10"] = (upv / dnv.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    # ========== MOMENTUM ==========
    g["RET21D"] = g["Close"].pct_change(21)  # ~1 month
    g["RET63D"] = g["Close"].pct_change(63)  # ~3 months

    # ========== SLOPES & EXTENSIONS ==========
    g["EMA20_Slope5"] = (g["EMA20"] - g["EMA20"].shift(5)) / g["EMA20"].shift(5)
    g["MA_Health"] = ((g["Close"] > g["EMA20"]) & (g["Close"] > g["EMA50"])).astype(int)
    g["OverEMA20"] = g["Close"] / g["EMA20"]

    # 200DMA slope (rising or falling)
    g["EMA200_Slope"] = (g["EMA200"] - g["EMA200"].shift(5)) / g["EMA200"].shift(5)

    # ========== RSI / ADX ==========
    g["RSI14"] = rsi(g["Close"], 14)
    pdi, mdi, adx_val = adx_block(g["High"], g["Low"], g["Close"], 14)
    g["+DI14"] = pdi
    g["-DI14"] = mdi
    g["ADX14"] = adx_val
    g["ADX14_chg3"] = g["ADX14"] - g["ADX14"].shift(3)  # ADX rising?

    return g


def compute_cross_sectional_ranks(x: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cross-sectional rankings per day (relative strength)
    This identifies leaders vs laggards on each day
    """
    x = x.copy()

    # RS Composite = average of 21d & 63d return percentiles (0..1)
    # Higher = stronger momentum vs peers
    p1 = x["RET21D"].rank(pct=True)
    p2 = x["RET63D"].rank(pct=True)
    x["RS_Composite"] = 0.5 * (p1.fillna(0.5) + p2.fillna(0.5))

    # BB width percentile (where does this stock's BB width rank today?)
    # Lower = tighter (potential squeeze)
    x["BBWidthPctl"] = x["BBWidth"].rank(pct=True)

    return x


def compute_weekly_features(gsym: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weekly context features
    Weekly timeframe for longer-term trend confirmation
    """
    gsym = gsym.set_index(pd.to_datetime(gsym["DATE"]), drop=True)

    # Resample to weekly (Friday close)
    wk = gsym.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum"
    }).dropna(subset=["Close"])

    # Weekly Bollinger Bands
    lb, mb, ub = bbands(wk["Close"], 20, 2.0)
    wk["W_BBWidth"] = (ub - lb) / mb.replace(0, np.nan)

    # Weekly EMA
    wk["W_EMA10"] = ema(wk["Close"], 10)

    # Weekly trend OK? (close > EMA10 and EMA10 rising)
    wk["W_TrendOK"] = ((wk["Close"] > wk["W_EMA10"]) & (wk["W_EMA10"].diff() > 0)).astype(int)

    # Convert index to datetime (normalize to midnight to match daily DATE format)
    wk["W_Date"] = pd.to_datetime(wk.index.date)

    return wk[["W_BBWidth", "W_TrendOK", "W_Date"]]


def prepare_features_all(bhav: pd.DataFrame, cache_dir: str = "./stock_picker_data/cache/features") -> pd.DataFrame:
    """
    Main feature engineering function with caching
    Computes all momentum/breakout features from the original code

    Args:
        bhav: Raw BhavCopy data with columns [SC_CODE, SC_NAME, Open, High, Low, Close, Volume, ValueTraded, DATE]
        cache_dir: Directory for feature cache

    Returns:
        DataFrame with all features computed
    """
    # ========== CHECK CACHE ==========
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Create cache key from data characteristics
    min_date = bhav['DATE'].min()
    max_date = bhav['DATE'].max()
    n_stocks = bhav['SC_CODE'].nunique()
    n_rows = len(bhav)

    cache_key = f"features_{min_date}_{max_date}_{n_stocks}_{n_rows}.pkl"
    cache_file = cache_path / cache_key

    # Try to load from cache
    if cache_file.exists():
        print(f"✅ Loading features from cache: {cache_key}")
        print(f"   (Saves 10-15 minutes of computation!)")
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            print(f"⚠️ Cache file corrupted ({type(e).__name__}), deleting and recomputing...")
            try:
                cache_file.unlink()  # Delete corrupted cache
                print(f"   Deleted corrupted cache: {cache_key}")
            except Exception as delete_error:
                print(f"   Warning: Could not delete cache file: {delete_error}")
            # Continue to recompute features below

    print("🔧 Computing momentum/breakout features (not cached)...")
    print(f"   This will take 10-15 minutes but will be cached for future runs...")
    df = bhav.sort_values(["SC_CODE", "DATE"]).copy()

    # Ensure DATE is datetime type (not string/object)
    if df['DATE'].dtype == 'object':
        df['DATE'] = pd.to_datetime(df['DATE'])

    # ========== STEP 1: Per-symbol features ==========
    print("   • Computing per-symbol features (EMAs, ATR, breakouts, RSI, ADX)...")
    df = df.groupby("SC_CODE", group_keys=False).apply(compute_per_symbol_features)

    # ========== STEP 2: Cross-sectional ranks (per day) ==========
    print("   • Computing cross-sectional ranks (relative strength)...")
    df = df.groupby("DATE", group_keys=False).apply(compute_cross_sectional_ranks)

    # ========== STEP 3: Weekly context ==========
    print("   • Computing weekly context features...")
    wk_list = []
    for sc, gsym in df.groupby("SC_CODE"):
        try:
            wk = compute_weekly_features(gsym)
            wk["SC_CODE"] = sc
            wk_list.append(wk.reset_index(drop=True))
        except Exception as e:
            # Skip stocks with insufficient data
            pass

    if wk_list:
        wk_all = pd.concat(wk_list, ignore_index=True)
        wk_all = wk_all.sort_values(["SC_CODE", "W_Date"])

        # Merge weekly features to daily data
        df = df.merge(
            wk_all.drop_duplicates(["SC_CODE", "W_Date"]),
            left_on=["SC_CODE", "DATE"],
            right_on=["SC_CODE", "W_Date"],
            how="left"
        ).drop(columns=["W_Date"], errors='ignore')

        # Forward-fill within each symbol
        df[["W_BBWidth", "W_TrendOK"]] = (
            df.sort_values(["SC_CODE", "DATE"])
            .groupby("SC_CODE")[["W_BBWidth", "W_TrendOK"]]
            .apply(lambda x: x.ffill())
            .reset_index(drop=True)
        )

        # Weekly BB width percentile (cross-sectional per day)
        def w_pctl(x: pd.DataFrame) -> pd.DataFrame:
            x = x.copy()
            x["W_BBWidthPctl"] = x["W_BBWidth"].rank(pct=True)
            return x

        df = df.groupby("DATE", group_keys=False).apply(w_pctl)

    # ========== STEP 4: Clean numeric columns ==========
    numeric_cols = [
        "VolMult", "RangePos20", "BBWidthPctl", "W_BBWidthPctl", "RS_Composite",
        "EMA20_Slope5", "DistTo20", "DistTo63", "DistTo52W", "UD_Vol_Ratio10",
        "ADX14", "+DI14", "-DI14", "RSI14", "EMA200_Slope", "OverEMA20"
    ]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    print(f"✅ Features computed: {len(df):,} rows | {df['SC_CODE'].nunique()} stocks")

    # ========== SAVE TO CACHE ==========
    result = df.sort_values(["SC_CODE", "DATE"]).reset_index(drop=True)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"💾 Features cached to: {cache_key}")
        print(f"   Next run will be 10-15 minutes faster!")
    except Exception as e:
        print(f"⚠️ Cache save failed: {e}")

    return result


def add_forward_returns(df: pd.DataFrame, periods: list = [5]) -> pd.DataFrame:
    """
    Add forward return labels for model training
    For each stock, compute forward returns after N sessions

    Args:
        df: DataFrame with features
        periods: List of forward periods (e.g., [5] for 5-session forward return)

    Returns:
        DataFrame with forward return columns added
    """
    print(f"📊 Computing forward returns for periods: {periods}")

    def add_fwd(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("DATE").copy()
        for p in periods:
            # Forward close (p sessions ahead)
            g[f"Close_fwd{p}"] = g["Close"].shift(-p)
            # Forward return (percentage)
            g[f"Ret_fwd{p}"] = (g[f"Close_fwd{p}"] / g["Close"] - 1.0) * 100.0
            # Binary label: 1 if positive, 0 if negative/flat
            g[f"Label_fwd{p}_positive"] = (g[f"Ret_fwd{p}"] > 0).astype(int)

        return g

    result = df.groupby("SC_CODE", group_keys=False).apply(add_fwd)
    print(f"✅ Forward returns added")

    return result


# Test
if __name__ == "__main__":
    from bse_loader import BSEDataFetcher
    from datetime import date, timedelta

    print("Testing momentum/breakout feature engineering...")

    # Fetch some data
    fetcher = BSEDataFetcher()
    end_d = fetcher.prev_bday(date.today())
    start_d = end_d - timedelta(days=100)

    bhav = fetcher.fetch_bhav_range(start_d, end_d)
    print(f"\nRaw data: {len(bhav):,} rows")

    # Compute features
    features = prepare_features_all(bhav)
    print(f"\nFeatures shape: {features.shape}")
    print(f"Feature columns: {features.columns.tolist()}")

    # Add forward returns (5-session)
    features_with_labels = add_forward_returns(features, periods=[5])

    # Check a sample
    print(f"\nSample with forward returns:")
    sample = features_with_labels.dropna(subset=["Ret_fwd5"]).tail(10)
    print(sample[["DATE", "SC_NAME", "Close", "Close_fwd5", "Ret_fwd5", "Label_fwd5_positive"]])

    # Check label distribution
    labels = features_with_labels["Label_fwd5_positive"].dropna()
    print(f"\nLabel distribution (5-session positive close):")
    print(f"  Positive: {labels.sum():,} ({labels.mean()*100:.1f}%)")
    print(f"  Negative: {(labels==0).sum():,} ({(1-labels.mean())*100:.1f}%)")
