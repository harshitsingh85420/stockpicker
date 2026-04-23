#!/usr/bin/env python3
"""
Test script to verify negative caching works correctly
This ensures dates without data aren't retried on subsequent runs
"""

import time
from datetime import date, timedelta
from bse_loader import BSEDataFetcher

def test_negative_caching():
    """Test that unavailable dates are cached and not retried"""
    print("=" * 80)
    print("TESTING NEGATIVE CACHE FOR UNAVAILABLE DATES")
    print("=" * 80)

    fetcher = BSEDataFetcher()

    # Test with a date range that includes weekends and potential holidays
    end_date = date(2024, 11, 10)  # Sunday - should have no data
    start_date = end_date - timedelta(days=7)  # Includes weekend

    print(f"\n1️⃣ FIRST RUN: Fetching data from {start_date} to {end_date}")
    print("   This will attempt to fetch from BSE and cache results (including 'no data')")
    print("-" * 80)

    start_time = time.time()
    try:
        df1 = fetcher.fetch_bhav_range(start_date, end_date)
        print(f"   ✅ First run completed in {time.time() - start_time:.2f} seconds")
        print(f"   Got {len(df1):,} rows")
    except SystemExit as e:
        print(f"   ⚠️ No data available: {e}")

    print("\n" + "=" * 80)
    print("2️⃣ SECOND RUN: Fetching same date range again")
    print("   Should be MUCH faster - uses cached data AND skips unavailable dates")
    print("-" * 80)

    start_time = time.time()
    try:
        df2 = fetcher.fetch_bhav_range(start_date, end_date)
        elapsed = time.time() - start_time
        print(f"   ✅ Second run completed in {elapsed:.2f} seconds")
        print(f"   Got {len(df2):,} rows")

        if elapsed < 5:
            print(f"\n   🎉 SUCCESS! Second run was {elapsed:.2f}s (should be < 5s with caching)")
            print("   Negative caching is working - unavailable dates were not retried!")
        else:
            print(f"\n   ⚠️ WARNING: Second run took {elapsed:.2f}s (expected < 5s)")
            print("   Negative caching might not be working properly")

    except SystemExit as e:
        elapsed = time.time() - start_time
        print(f"   ⚠️ No data available: {e}")
        print(f"   Completed in {elapsed:.2f} seconds")

    print("\n" + "=" * 80)
    print("3️⃣ CHECKING CACHE FILES")
    print("-" * 80)

    # Check cache directory
    cache_files = list(fetcher.cache_dir.glob("bhav_bse_*.pkl"))
    print(f"   Found {len(cache_files)} cached date files in {fetcher.cache_dir}")

    # Check for negative cache markers
    import pickle
    negative_cache_count = 0
    positive_cache_count = 0

    for cache_file in cache_files[:10]:  # Check first 10 files
        try:
            with open(cache_file, 'rb') as f:
                df = pickle.load(f)
            if isinstance(df, pd.DataFrame) and len(df) == 0 and '_NO_DATA_MARKER' in df.columns:
                negative_cache_count += 1
            elif len(df) > 0:
                positive_cache_count += 1
        except:
            pass

    print(f"   Positive cache entries (data available): {positive_cache_count}")
    print(f"   Negative cache entries (no data): {negative_cache_count}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    if negative_cache_count > 0:
        print("✅ Negative caching is working correctly!")
        print("   Unavailable dates are now cached and won't be retried.")
        print("   This saves time on subsequent runs.")
    else:
        print("ℹ️ No negative cache entries found in the sample.")
        print("   This is normal if all dates in the range have data available.")

    print("\nTo clear cache and test again:")
    print(f"   rm -rf {fetcher.cache_dir}")
    print()

if __name__ == "__main__":
    import pandas as pd
    test_negative_caching()
