# Cache Clearing Instructions

## When to Clear Cache

If you encounter errors during training after code fixes, especially DATE-related ambiguity errors, you should clear the feature cache to force recomputation with the fixed code.

## Corrupted Cache Symptoms

- `pickle data was truncated` errors
- `'DATE' is both an index level and a column label, which is ambiguous` errors
- Features loading from cache but training still failing

## How to Clear Feature Cache

The feature cache is stored in: `./stock_picker_data/cache/features/`

To clear all cached features:

```bash
rm -rf ./stock_picker_data/cache/features/*.pkl
```

Or to completely remove the cache directory:

```bash
rm -rf ./stock_picker_data/cache/features/
```

## What Gets Cleared

Feature cache files follow the naming pattern:
```
features_{start_date}_{end_date}_{n_stocks}_{n_rows}.pkl
```

Example: `features_2023-01-02_2023-10-23_2544_508800.pkl`

## After Clearing

After clearing the cache:
1. Next training run will recompute features (takes 10-15 minutes per unique date range)
2. New cache files will be created with the fixed code
3. Subsequent runs will be fast again

## Note

BSE data cache (in `./stock_picker_data/cache/bse/`) does NOT need to be cleared - it only contains raw OHLCV data, not computed features.
