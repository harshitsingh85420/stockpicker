# BSE Stock Picker - Caching Strategy Guide

**"Download Once, Use Forever" - Complete Guide**

## Overview

This system implements a comprehensive 3-layer caching strategy that ensures:
- **BSE data**: Downloaded once per date, used forever
- **Features**: Computed once per dataset, used forever
- **Market data**: Fetched once per period, used forever

**Result**: First run takes 30-45 minutes, subsequent runs take 2-3 minutes!

---

## 🎯 Quick Start

### Option 1: Use Pre-Calculation Script (Recommended)

```bash
# Pre-calculate everything for the last 2 years (default)
python precalculate_all.py

# Pre-calculate with advanced features (90+ features)
python precalculate_all.py --advanced

# Pre-calculate specific date range
python precalculate_all.py --start 2023-01-01 --end 2025-01-31

# Force recompute (clear cache and recalculate)
python precalculate_all.py --force
```

### Option 2: Use Stock Picker Directly (Auto-Caches)

```bash
# Basic version (50+ features)
python run_5session_picker.py

# Enhanced version (90+ features)
python stock_picker_enhanced.py
```

Both automatically use caching!

---

## 📦 Caching Layers

### Layer 1: BSE BhavCopy Data Cache

**Location**: `./stock_picker_data/cache/bse_data/`

**Files**: `bhav_bse_YYYYMMDD.pkl` (one per date)

**What's Cached**:
- Raw OHLCV data from BSE India
- All 5600+ stocks
- Official BhavCopy data (UDiFF + Legacy formats)

**Caching Strategy**:
- **Per-date caching**: Each date stored separately
- **Permanent cache**: Historical dates NEVER re-downloaded
- **Only new dates**: Future runs only fetch new dates

**Example**:
```
Day 1:  Download 730 dates (2 years) → 15-20 minutes
Day 2:  Download 1 new date → 10 seconds
Day 30: Download 30 new dates → 5 minutes
```

**Code**: `bse_loader.py` → `fetch_bhav_range()`

### Layer 2: Features Cache

**Location**: `./stock_picker_data/cache/features/`

**Files**:
- Basic: `features_v1_basic_STARTDATE_ENDDATE_NSTOCKS_NROWS.pkl`
- Enhanced: `features_v2_enhanced_STARTDATE_ENDDATE_NSTOCKS_NROWS.pkl`

**What's Cached**:
- **Basic**: 50+ technical indicators (EMAs, RSI, ADX, breakouts, volume)
- **Enhanced**: 90+ indicators (above + fractional diff, FII/DII, VWAP, etc.)

**Caching Strategy**:
- **Hash-based key**: Based on data characteristics (dates, stocks, rows)
- **Automatic invalidation**: New data range = new cache file
- **Same data = instant load**: <1 second vs 15-20 minutes

**Code**:
- Basic: `momentum_features.py` → `prepare_features_all()`
- Enhanced: `momentum_features_enhanced.py` → `prepare_features_enhanced()`

### Layer 3: Market-Level Data Cache

**Location**: `./stock_picker_data/cache/market_data/`

**Files**: `market_data_STARTDATE_ENDDATE.pkl`

**What's Cached**:
- **FII/DII flows**: Daily institutional money flows (NSE)
- **India VIX**: Volatility index
- **Sector indices**: Nifty sector rotation data

**Caching Strategy**:
- Fetched once per date range
- Cached separately from stock data
- Used across all feature computations

**Code**: `nse_data_fetcher.py` → `fetch_all_market_data()`

### Layer 4: Model Cache

**Location**: `./stock_picker_data/models/`

**Files**: `model_5session.pkl`

**What's Cached**:
- Trained LightGBM or ensemble model
- Feature columns list
- Configuration parameters
- Training metadata (date, AUC, samples)

**Caching Strategy**:
- Train once, use for all predictions
- Automatically loaded if exists
- Can be retrained on demand

**Code**: `stock_picker_5session.py` → `load_model()` / `save_model()`

---

## 🔄 Cache Workflow

### First Run (30-45 minutes)

```
1. Download BSE data (730 dates) → 15-20 min
   ├─ Fetch from BSE India
   ├─ Cache per-date: bhav_bse_20230101.pkl, bhav_bse_20230102.pkl, ...
   └─ Total: ~700 files

2. Fetch market data → 2-3 min
   ├─ FII/DII flows (NSE)
   ├─ India VIX
   └─ Cache: market_data_20230101_20250131.pkl

3. Compute features → 10-15 min
   ├─ Process all stocks × all indicators
   ├─ 50+ basic OR 90+ enhanced features
   └─ Cache: features_v2_enhanced_20230101_20250131_5643_987654.pkl

4. Train model → 2-3 min
   ├─ LightGBM or ensemble
   └─ Cache: model_5session.pkl
```

### Subsequent Runs (2-3 minutes)

```
1. Load BSE data → 10 seconds
   ├─ Check cache for all dates
   ├─ Load cached files
   └─ Only fetch NEW dates (e.g., today)

2. Load market data → <1 second
   └─ Load from cache

3. Load features → <1 second
   └─ Load from cache (if same data range)

4. Load model → <1 second
   └─ Load from cache

5. Generate predictions → 1-2 minutes
```

**Speedup**: 15-20x faster!

---

## 📊 Cache Management

### Check Cache Status

```bash
python cache_manager.py
```

**Output**:
```
💾 CACHE INFORMATION
================================================================================

📊 BSE Data Cache:
   Location: ./stock_picker_data/cache/bse_data
   Files: 730
   Size: 245.67 MB
   Contains: Raw BhavCopy data (OHLCV)

🔧 Features Cache:
   Location: ./stock_picker_data/cache/features
   Files: 3
   Size: 512.34 MB
   Contains: Computed technical indicators (50+ features)

🤖 Trained Models:
   Location: ./stock_picker_data/models
   Files: 1
   Size: 45.23 MB
   Contains: LightGBM models + config + metadata

📦 TOTAL:
   Files: 734
   Size: 803.24 MB
```

### Clear Specific Cache

```python
from cache_manager import CacheManager

manager = CacheManager()

# Clear BSE data only
manager.clear_bse_cache()

# Clear features only
manager.clear_features_cache()

# Clear everything
manager.clear_all_cache()
```

### List Cached Files

```bash
python cache_manager.py
# Select option 2 → List cached files
```

---

## 🎯 Pre-Calculation Best Practices

### Weekly Workflow

```bash
# Monday morning: Pre-calculate last week's data
python precalculate_all.py --advanced

# Rest of week: Instant predictions
python stock_picker_enhanced.py
```

### Monthly Workflow

```bash
# First day of month: Full pre-calculation
python precalculate_all.py --start 2023-01-01 --end 2025-01-31 --advanced

# Daily: Only fetch new date, use cached features
python stock_picker_enhanced.py
```

### Backtest Workflow

```bash
# Pre-calculate entire range once
python precalculate_all.py --start 2023-01-01 --end 2024-12-31 --advanced

# Run backtests (instant - uses cache)
python run_backtest.py --start 2024-01-01 --end 2024-12-31
```

---

## 💡 Advanced Tips

### 1. Separate Cache for Different Stock Counts

```python
# Training on top 500 stocks
bhav_500 = bhav[bhav['SC_CODE'].isin(top_500_stocks)]
features_500 = prepare_features_all(bhav_500)  # Creates separate cache

# Training on ALL stocks
features_all = prepare_features_all(bhav)  # Different cache
```

Cache key includes stock count, so these create separate files!

### 2. Force Recompute When Needed

```bash
# Use --force to clear cache and recalculate
python precalculate_all.py --force

# Or manually
python cache_manager.py
# Select option 5 → Clear ALL caches
```

**When to force recompute**:
- Updated feature formulas
- Changed indicator parameters
- Suspected corrupted cache
- Testing new implementations

### 3. Cache Disk Space Management

**Typical sizes**:
- BSE data: 200-500 MB (730 dates × 5600 stocks)
- Features (basic): 300-500 MB
- Features (enhanced): 500-800 MB
- Models: 20-50 MB each

**Total**: ~1-2 GB for full system

**To save space**:
1. Keep only recent BSE data (delete old dates)
2. Use basic features if disk space limited
3. Clear old feature caches periodically

### 4. Distributed Caching (Advanced)

For large-scale operations:

```python
# Cache on fast SSD
cache_manager = CacheManager(base_dir="/mnt/fast_ssd/stock_picker_data")

# Share cache across machines via NFS/network drive
cache_manager = CacheManager(base_dir="/nfs/shared/stock_picker_data")
```

---

## 🔍 Troubleshooting

### Cache Not Being Used

**Symptom**: Every run takes 30+ minutes

**Causes**:
1. Cache directory doesn't exist
2. Permissions issue
3. Cache file corrupted

**Fix**:
```bash
# Check cache directory
ls -lh ./stock_picker_data/cache/

# Verify permissions
chmod -R 755 ./stock_picker_data/

# Test cache
python -c "from cache_manager import CacheManager; CacheManager().get_cache_info()"
```

### Cache Size Too Large

**Symptom**: >5 GB disk usage

**Causes**:
1. Multiple feature caches for different date ranges
2. Old model files

**Fix**:
```bash
# List all cached files
python cache_manager.py
# Option 2 → See all files

# Clear old feature caches
python cache_manager.py
# Option 4 → Clear features cache

# Manually delete old files
rm ./stock_picker_data/cache/features/features_*_old_*.pkl
```

### Out of Memory During Feature Computation

**Symptom**: Process killed during feature computation

**Causes**:
1. Computing features for ALL 5600 stocks at once
2. Insufficient RAM

**Fix**:
```python
# Process in batches
from bse_loader import BSEDataFetcher
from momentum_features import prepare_features_all

# Limit to top 1000 most liquid stocks
top_1000 = bhav.groupby('SC_CODE')['ValueTraded'].mean().nlargest(1000).index
bhav_limited = bhav[bhav['SC_CODE'].isin(top_1000)]

features = prepare_features_all(bhav_limited)
```

---

## 📚 Cache File Formats

All caches use Python pickle format (`.pkl`):

```python
import pickle

# Load any cache manually
with open('./stock_picker_data/cache/bse_data/bhav_bse_20250131.pkl', 'rb') as f:
    bhav_data = pickle.load(f)

print(bhav_data.head())
```

**Columns**:
- BSE data: `SC_CODE, SC_NAME, Open, High, Low, Close, Volume, ValueTraded, DATE`
- Features: Original columns + 50-90 indicator columns
- Market data: `Date, FII_Net_Crore, DII_Net_Crore, India_VIX, ...`

---

## ✅ Summary

### What Gets Cached:
1. ✅ BSE BhavCopy data (per-date, permanent)
2. ✅ Technical indicators (50+ basic or 90+ enhanced)
3. ✅ Market-level data (FII/DII, VIX)
4. ✅ Trained models

### What Doesn't Get Cached:
1. ❌ Today's predictions (generated fresh)
2. ❌ Backtest results (computed on demand)
3. ❌ Temporary computations during model training

### Performance Impact:
- **First run**: 30-45 minutes
- **Cached runs**: 2-3 minutes
- **Speedup**: 15-20x

### Disk Usage:
- **Minimal**: ~1-2 GB for complete system
- **Scales linearly**: More dates = more cache

---

## 🚀 Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Pre-calculate everything** (one-time):
   ```bash
   python precalculate_all.py --advanced
   ```

3. **Use cached data** (daily):
   ```bash
   python stock_picker_enhanced.py
   ```

4. **Check cache status** (anytime):
   ```bash
   python cache_manager.py
   ```

---

**That's it!** Your BSE stock picker now has a complete "Download Once, Use Forever" caching system. Enjoy the 15-20x speedup! 🚀
