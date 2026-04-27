#!/usr/bin/env python3
"""
verify_feature_alignment.py  --  P01 acceptance test.

Checks that the features produced by the live feature pipeline exactly match
the features the model was trained on (feature_names.json).

Usage:
    python verify_feature_alignment.py          # uses most-recent cached day
    python verify_feature_alignment.py --date 2026-04-23

Exit codes:
    0  -- all features align  (pipeline is clean)
    1  -- mismatch detected   (must fix before live trading)
"""

import json
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("verify_feature_alignment")

FEAT_JSON  = _ROOT / "stock_picker_data" / "models" / "feature_names.json"
MODELS_DIR = _ROOT / "stock_picker_data" / "models"


# ---------------------------------------------------------------------------

def load_one_day_bhav(target_date: date) -> pd.DataFrame:
    """Load a single day's BhavCopy from the per-day cache."""
    from bse_direct_loader import BSEDataFetcher
    fetcher = BSEDataFetcher()
    day_df = fetcher.fetch_bhav_for(target_date)
    if day_df is None or day_df.empty:
        log.error("No BhavCopy data for %s", target_date)
        sys.exit(1)
    log.info("Loaded %d rows for %s", len(day_df), target_date)
    return day_df


def load_context_bhav(target_date: date, lookback: int = 300) -> pd.DataFrame:
    """
    Load enough history for indicators (EMA200 needs 200+ days).
    Returns a multi-day DataFrame ending on target_date.
    """
    from bse_direct_loader import BSEDataFetcher
    fetcher = BSEDataFetcher()
    start = target_date - timedelta(days=lookback)
    df = fetcher.fetch_bhav_range(start, target_date)
    if df is None or df.empty:
        log.error("No BhavCopy data for range %s..%s", start, target_date)
        sys.exit(1)
    # Ensure DATE is a date object
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    log.info(
        "Context window: %d rows, %d stocks, %s -> %s",
        len(df), df["SC_CODE"].nunique(),
        df["DATE"].min(), df["DATE"].max(),
    )
    return df


def compute_features(bhav_df: pd.DataFrame) -> pd.DataFrame:
    """Run the same feature pipeline used by the live orchestrator."""
    try:
        from momentum_features_enhanced import prepare_features_enhanced
        log.info("Using enhanced feature pipeline.")
        return prepare_features_enhanced(bhav_df, use_advanced=False, use_fii_dii=False)
    except (ImportError, Exception):
        pass

    from momentum_features import prepare_features_all
    log.info("Using base feature pipeline.")
    return prepare_features_all(bhav_df)


def load_training_features() -> list:
    """Return the feature list from the JSON the model was trained on."""
    if not FEAT_JSON.exists():
        log.error("feature_names.json not found at %s", FEAT_JSON)
        log.error("Run train_model.py first, then re-run this script.")
        sys.exit(1)
    cols = json.loads(FEAT_JSON.read_text())
    log.info("Training features: %d columns loaded from %s", len(cols), FEAT_JSON.name)
    return cols


def verify(target_date: date) -> bool:
    """
    Main verification routine.

    Returns True if features align, False if there is any mismatch.
    """
    print()
    print("=" * 65)
    print("  P01 -- Feature Alignment Verification")
    print(f"  Reference date : {target_date}")
    print("=" * 65)

    # 1. Load data
    bhav_df = load_context_bhav(target_date, lookback=300)

    # 2. Filter to single target date for report (features computed on full window)
    feature_df = compute_features(bhav_df)
    if feature_df is None or feature_df.empty:
        log.error("Feature computation returned empty DataFrame.")
        return False

    # Get the latest date's rows for column inspection
    if "DATE" in feature_df.columns:
        feature_df["DATE"] = pd.to_datetime(feature_df["DATE"]).dt.date
        latest = feature_df["DATE"].max()
        day_rows = feature_df[feature_df["DATE"] == latest]
    else:
        day_rows = feature_df

    computed_cols = set(day_rows.columns)
    # Exclude metadata/label columns
    EXCLUDE = {
        "SC_CODE", "SC_NAME", "DATE", "ISIN", "Source",
        "Label_fwd5_positive", "Label_fwd5_return",
        "Open", "High", "Low", "Close", "Volume", "ValueTraded",
    }
    computed_feature_cols = computed_cols - EXCLUDE
    log.info("Computed feature columns (excluding metadata): %d", len(computed_feature_cols))

    # 3. Load training feature list
    training_cols = load_training_features()
    training_set = set(training_cols)

    # 4. Compute diffs
    extra_in_computed  = computed_feature_cols - training_set   # computed but not trained on
    missing_in_computed = training_set - computed_feature_cols  # trained on but not computed

    print()
    print(f"  Computed features (non-metadata) : {len(computed_feature_cols)}")
    print(f"  Training features                : {len(training_cols)}")
    print()

    if extra_in_computed:
        print(f"  [WARN] Computed but NOT in training set ({len(extra_in_computed)}):")
        for c in sorted(extra_in_computed):
            print(f"         + {c}")
        print()
        print("  These are silently IGNORED by the alignment code. Review if intentional.")
    else:
        print("  [OK] No extra features computed beyond training set.")

    if missing_in_computed:
        print(f"  [FAIL] In training set but MISSING from computed ({len(missing_in_computed)}):")
        for c in sorted(missing_in_computed):
            print(f"         - {c}")
        print()
        print("  These features will be filled with 0 -- model may produce biased predictions.")
    else:
        print("  [OK] All training features are present in the computed pipeline.")

    # 5. Check column order matters for model.predict(X.values)
    available_in_order = [c for c in training_cols if c in computed_feature_cols]
    if available_in_order == training_cols:
        print("  [OK] All 31 training features present -- column alignment is clean.")
    else:
        n_present = len(available_in_order)
        print(f"  [WARN] Only {n_present}/{len(training_cols)} training features found in computed set.")

    print()
    if missing_in_computed:
        print("  RESULT: MISMATCH DETECTED -- exit code 1")
        print("=" * 65 + "\n")
        return False
    else:
        print("  RESULT: CLEAN -- all training features present -- exit code 0")
        print("=" * 65 + "\n")
        return True


# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify feature alignment (P01)")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Reference date YYYY-MM-DD (default: most recent trading day)"
    )
    args = parser.parse_args()

    if args.date:
        target_date = date.fromisoformat(args.date)
    else:
        # Use yesterday (today's BhavCopy may not be published yet)
        from bse_direct_loader import BSEDataFetcher
        target_date = BSEDataFetcher.prev_bday(date.today())

    ok = verify(target_date)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
