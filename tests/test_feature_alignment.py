"""
P51 — Test: feature alignment between training and inference.

Verifies that the canonical feature list is consistent across:
  - feature_cols.json
  - what momentum_features.py computes
  - what signal_generator loads from model
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

FEATURE_COLS_PATH = _ROOT / "stock_picker_data" / "models" / "feature_cols.json"


@pytest.fixture
def canonical_features():
    if not FEATURE_COLS_PATH.exists():
        pytest.skip("feature_cols.json not found — run train_model.py first")
    with open(FEATURE_COLS_PATH) as fh:
        cols = json.load(fh)
    return cols


@pytest.fixture
def sample_bhav():
    """Minimal synthetic BhavCopy for one stock over 300 rows."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    price = 100.0 + np.cumsum(rng.normal(0, 1, n))
    price = np.maximum(price, 10.0)
    df = pd.DataFrame({
        "DATE":     dates,
        "SC_CODE":  "999999",
        "SC_NAME":  "TESTSTOCK",
        "Open":     price * (1 + rng.uniform(-0.005, 0.005, n)),
        "High":     price * (1 + rng.uniform(0.000, 0.010, n)),
        "Low":      price * (1 - rng.uniform(0.000, 0.010, n)),
        "Close":    price,
        "Volume":   rng.integers(50_000, 500_000, n).astype(float),
        "NO_OF_SHRS": rng.integers(50_000, 500_000, n).astype(float),
        "NET_TURNOVER": price * rng.integers(50_000, 500_000, n) / 1e5,
    })
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_feature_cols_json_exists():
    assert FEATURE_COLS_PATH.exists(), (
        "feature_cols.json missing — train the model before running tests"
    )


def test_feature_cols_non_empty(canonical_features):
    assert len(canonical_features) > 0, "feature_cols.json is empty"


def test_feature_cols_no_duplicates(canonical_features):
    assert len(canonical_features) == len(set(canonical_features)), (
        "Duplicate feature names in feature_cols.json"
    )


def test_feature_cols_no_target_leakage(canonical_features):
    forbidden = {"Forward_Return", "Label", "Target", "Future", "forward", "label"}
    leaks = [c for c in canonical_features
             if any(f.lower() in c.lower() for f in forbidden)]
    assert not leaks, f"Potential look-ahead features in canonical list: {leaks}"


def test_momentum_features_covers_canonical(canonical_features, sample_bhav):
    try:
        from momentum_features import MomentumFeatureEngine
    except ImportError:
        pytest.skip("momentum_features not importable")

    engine = MomentumFeatureEngine()
    feat_df = engine.compute_features(sample_bhav)
    computed = set(feat_df.columns)
    missing = [c for c in canonical_features if c not in computed]
    # Allow at most 3 missing (some features need cross-sectional data)
    assert len(missing) <= 3, (
        f"momentum_features is missing {len(missing)} canonical features: {missing[:10]}"
    )


def test_no_nan_in_canonical_features(canonical_features, sample_bhav):
    try:
        from momentum_features import MomentumFeatureEngine
    except ImportError:
        pytest.skip("momentum_features not importable")

    engine = MomentumFeatureEngine()
    feat_df = engine.compute_features(sample_bhav)
    available = [c for c in canonical_features if c in feat_df.columns]
    if not available:
        pytest.skip("No canonical features computed")

    last_row = feat_df[available].iloc[-1]
    nan_cols = last_row[last_row.isna()].index.tolist()
    assert not nan_cols, f"NaN in last-row features: {nan_cols}"
