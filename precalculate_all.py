#!/usr/bin/env python
"""
Pre-Calculate Everything System
================================

BSE "Download Once, Use Forever" Strategy

This script downloads and computes EVERYTHING that can be calculated upfront:
1. BSE BhavCopy data (per-date caching)
2. Market-level data (FII/DII, India VIX, Options IV)
3. ALL features (90+ including advanced techniques)
4. Feature warehouse ready for instant model training

Run this once per period (weekly/monthly), then use cached data forever!

Expected Time:
- First run: 30-45 minutes (downloads + computes everything)
- Future runs: 2-3 minutes (only fetches new dates)

Usage:
    # Pre-calculate last 2 years (default for model training)
    python precalculate_all.py

    # Pre-calculate specific date range
    python precalculate_all.py --start 2023-01-01 --end 2025-01-31

    # Force recompute (clear cache and recalculate)
    python precalculate_all.py --force

    # Advanced mode (include ALL 90+ features)
    python precalculate_all.py --advanced
"""

import os
import pickle
import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Import our modules
from bse_loader import BSEDataFetcher
from momentum_features import prepare_features_all
from cache_manager import CacheManager


class FeatureWarehouse:
    """
    Comprehensive feature pre-calculation and caching system
    Downloads BSE data once, computes all features once, uses forever!
    """

    def __init__(self, base_dir: str = "./stock_picker_data"):
        self.base_dir = Path(base_dir)
        self.cache_dir = self.base_dir / "cache"
        self.bse_cache = self.cache_dir / "bse_data"
        self.features_cache = self.cache_dir / "features"
        self.market_cache = self.cache_dir / "market_data"

        # Create directories
        for d in [self.bse_cache, self.features_cache, self.market_cache]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize fetcher and cache manager
        self.bse_fetcher = BSEDataFetcher(cache_dir=str(self.bse_cache))
        self.cache_manager = CacheManager(base_dir=str(self.base_dir))

    def precalculate_bse_data(self, start_date: date, end_date: date, force: bool = False) -> pd.DataFrame:
        """
        Step 1: Download and cache ALL BSE BhavCopy data
        Uses per-date caching - each date cached separately
        """
        print("\n" + "=" * 80)
        print("📥 STEP 1: DOWNLOAD & CACHE BSE DATA")
        print("=" * 80)
        print(f"Date range: {start_date} → {end_date}")
        print(f"Strategy: Download each date ONCE, cache separately, use FOREVER!")

        if force:
            print("⚠️  FORCE mode: Clearing BSE cache...")
            self.cache_manager.clear_bse_cache(confirm=True)

        # Fetch BSE data (uses per-date caching automatically)
        bhav = self.bse_fetcher.fetch_bhav_range(start_date, end_date)

        print("\n" + "=" * 80)
        print("✅ BSE DATA CACHED")
        print("=" * 80)
        print(f"Total rows: {len(bhav):,}")
        print(f"Unique stocks: {bhav['SC_CODE'].nunique()}")
        print(f"Date range: {bhav['DATE'].min()} → {bhav['DATE'].max()}")
        print(f"Cache location: {self.bse_cache}")
        print(f"\n💡 This data will NEVER be re-downloaded (per-date cache)")
        print(f"   Only NEW dates (future) will be fetched!")

        return bhav

    def precalculate_market_data(self, start_date: date, end_date: date) -> dict:
        """
        Step 2: Pre-fetch and cache market-level data
        - FII/DII institutional flows (NSE)
        - India VIX
        - Sector indices
        """
        print("\n" + "=" * 80)
        print("📊 STEP 2: PRE-FETCH MARKET-LEVEL DATA")
        print("=" * 80)

        market_data = {}

        # Cache key
        cache_key = f"market_data_{start_date}_{end_date}.pkl"
        cache_file = self.market_cache / cache_key

        if cache_file.exists():
            print(f"✅ Loading market data from cache: {cache_key}")
            with open(cache_file, 'rb') as f:
                market_data = pickle.load(f)
            print(f"   Cached components: {list(market_data.keys())}")
            return market_data

        print("📥 Fetching market-level data...")

        # 1. FII/DII data (if available)
        try:
            from advanced_features import fetch_fii_dii_data
            print("   • Fetching FII/DII institutional flows (NSE)...")
            fii_dii = fetch_fii_dii_data(start_date, end_date)
            market_data['fii_dii'] = fii_dii
            print(f"     ✅ FII/DII: {len(fii_dii)} days")
        except Exception as e:
            print(f"     ⚠️ FII/DII fetch failed: {e}")
            market_data['fii_dii'] = None

        # 2. India VIX (if available)
        try:
            from nse_data_fetcher import fetch_india_vix
            print("   • Fetching India VIX...")
            vix = fetch_india_vix(start_date, end_date)
            market_data['india_vix'] = vix
            print(f"     ✅ India VIX: {len(vix)} days")
        except Exception as e:
            print(f"     ⚠️ India VIX fetch failed: {e}")
            market_data['india_vix'] = None

        # Cache market data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(market_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"\n💾 Market data cached: {cache_key}")
        except Exception as e:
            print(f"⚠️ Market data cache save failed: {e}")

        return market_data

    def precalculate_features(self, bhav: pd.DataFrame,
                             advanced: bool = False,
                             force: bool = False) -> pd.DataFrame:
        """
        Step 3: Compute and cache ALL features
        - Basic 50+ features (always computed)
        - Advanced 40+ features (if advanced=True)
        """
        print("\n" + "=" * 80)
        print("🔧 STEP 3: COMPUTE & CACHE ALL FEATURES")
        print("=" * 80)
        print(f"Mode: {'ADVANCED (90+ features)' if advanced else 'BASIC (50+ features)'}")

        if force:
            print("⚠️  FORCE mode: Clearing features cache...")
            self.cache_manager.clear_features_cache(confirm=True)

        # Use enhanced or basic feature computation
        if advanced:
            try:
                from momentum_features_enhanced import prepare_features_enhanced
                print("🚀 Computing ENHANCED features (90+ total)...")
                print("   Includes:")
                print("   • Original 50+ momentum/breakout features")
                print("   • Fractional differentiation (stationarity + memory)")
                print("   • FII/DII institutional flows")
                print("   • Volume-weighted indicators (VWAP, OBV, A/D)")
                print("   • Unconventional indicators (Squeeze Pro, PPO, Ichimoku)")
                print("   • Market regime features")
                print("   • Liquidity risk indicators")

                features = prepare_features_enhanced(
                    bhav,
                    cache_dir=str(self.features_cache),
                    use_advanced=True,
                    use_fii_dii=True
                )

                print("\n" + "=" * 80)
                print("✅ ENHANCED FEATURES COMPUTED & CACHED")
                print("=" * 80)

            except ImportError as e:
                print(f"⚠️ Enhanced features not available: {e}")
                print("   Falling back to basic features...")
                features = prepare_features_all(bhav, cache_dir=str(self.features_cache))

                print("\n" + "=" * 80)
                print("✅ BASIC FEATURES COMPUTED & CACHED")
                print("=" * 80)
        else:
            print("📊 Computing BASIC features (50+ total)...")
            features = prepare_features_all(bhav, cache_dir=str(self.features_cache))

            print("\n" + "=" * 80)
            print("✅ BASIC FEATURES COMPUTED & CACHED")
            print("=" * 80)

        print(f"Total rows: {len(features):,}")
        print(f"Total features: {len(features.columns)}")
        print(f"Unique stocks: {features['SC_CODE'].nunique()}")
        print(f"Cache location: {self.features_cache}")
        print(f"\n💡 These features will NEVER be re-computed for this data!")
        print(f"   Instant load (<1 second) on future runs!")

        return features

    def run_full_precalculation(self, start_date: date, end_date: date,
                               advanced: bool = False, force: bool = False):
        """
        Run complete pre-calculation pipeline:
        1. Download & cache BSE data
        2. Pre-fetch market-level data
        3. Compute & cache all features
        """
        print("\n" + "🎯" * 40)
        print("BSE FEATURE WAREHOUSE - DOWNLOAD ONCE, USE FOREVER")
        print("🎯" * 40)

        start_time = pd.Timestamp.now()

        # Step 1: BSE data
        bhav = self.precalculate_bse_data(start_date, end_date, force=force)

        # Step 2: Market data
        market_data = self.precalculate_market_data(start_date, end_date)

        # Step 3: Features
        features = self.precalculate_features(bhav, advanced=advanced, force=force)

        # Summary
        elapsed = (pd.Timestamp.now() - start_time).total_seconds()

        print("\n" + "=" * 80)
        print("🎉 PRE-CALCULATION COMPLETE!")
        print("=" * 80)
        print(f"Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"\n📦 Cached Components:")
        print(f"   • BSE Data: {len(bhav):,} rows | {bhav['SC_CODE'].nunique()} stocks")
        print(f"   • Features: {len(features):,} rows | {len(features.columns)} columns")
        print(f"   • Market Data: {len(market_data)} components")

        print(f"\n💡 Future Runs:")
        print(f"   • BSE data: Only NEW dates will be downloaded")
        print(f"   • Features: Loaded from cache in <1 second")
        print(f"   • Total time: ~2-3 minutes (vs {elapsed/60:.1f} minutes today)")

        print(f"\n📊 Cache Size:")
        self.cache_manager.get_cache_info()

        return {
            'bhav': bhav,
            'features': features,
            'market_data': market_data,
            'elapsed_seconds': elapsed
        }


def main():
    """Main entry point for pre-calculation"""
    parser = argparse.ArgumentParser(
        description="Pre-calculate and cache BSE data + features (Download Once, Use Forever)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 2 years of data (for model training)
  python precalculate_all.py

  # Specific date range
  python precalculate_all.py --start 2023-01-01 --end 2025-01-31

  # Advanced mode (90+ features including all enhancements)
  python precalculate_all.py --advanced

  # Force recompute (clear cache first)
  python precalculate_all.py --force

  # Check cache status
  python cache_manager.py
"""
    )

    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD). Default: 2 years ago'
    )

    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD). Default: yesterday'
    )

    parser.add_argument(
        '--advanced',
        action='store_true',
        help='Use advanced features (90+ total). Default: basic 50+ features'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recompute (clear cache first)'
    )

    args = parser.parse_args()

    # Parse dates
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date()
    else:
        # Default: yesterday (latest BSE data)
        end_date = BSEDataFetcher.prev_bday(date.today())

    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
    else:
        # Default: 2 years ago (730 days for model training)
        start_date = end_date - timedelta(days=730)

    # Run pre-calculation
    warehouse = FeatureWarehouse()
    results = warehouse.run_full_precalculation(
        start_date=start_date,
        end_date=end_date,
        advanced=args.advanced,
        force=args.force
    )

    print("\n✅ Done! Your feature warehouse is ready.")
    print("   Run stock_picker_5session.py or stock_picker_enhanced.py to use cached data.")


if __name__ == "__main__":
    main()
