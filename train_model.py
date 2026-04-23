"""
train_model.py — Periodic LightGBM retraining script.

Usage:
    python train_model.py                          # train on last 2 years
    python train_model.py --lookback 365           # 1 year
    python train_model.py --start 2023-01-01       # explicit range
    python train_model.py --no-calibrate           # skip calibration step
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "production") not in sys.path:
    sys.path.insert(0, str(_ROOT / "production"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_DIR = _ROOT / "stock_picker_data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_data(args) -> pd.DataFrame:
    from production.data_loader import DataLoader
    loader = DataLoader()
    if args.start:
        bhav = loader.load(start=args.start, end=args.end)
    else:
        bhav = loader.load(lookback_days=args.lookback)
    ok, issues = loader.validate(bhav)
    if not ok:
        logger.warning("Data quality issues: %s", issues)
    logger.info("Loaded %d rows, %d stocks", len(bhav), bhav["SC_CODE"].nunique())
    return bhav


def _apply_corporate_actions(bhav: pd.DataFrame) -> pd.DataFrame:
    try:
        from production.corporate_actions_fetcher import CorporateActionsFetcher
        from production.data_integrity import CorporateActionAdjuster
        fetcher = CorporateActionsFetcher()
        min_date = str(bhav["DATE"].min())
        max_date = str(bhav["DATE"].max())
        ca_df = fetcher.get(min_date, max_date)
        if not ca_df.empty:
            adjuster = CorporateActionAdjuster()
            bhav = adjuster.adjust_all(bhav, ca_df)
            logger.info("Applied %d corporate action records", len(ca_df))
        else:
            logger.info("No corporate actions found for the period — skipping adjustment")
    except Exception as exc:
        logger.warning("Corporate action adjustment skipped: %s", exc)
    return bhav


def _filter_universe(bhav: pd.DataFrame) -> pd.DataFrame:
    from production.universe_filter import TradabilityGate
    gate = TradabilityGate(min_value_crore=2.0, min_price=20.0, min_avg_volume=10_000)
    tradable_codes = gate.apply(bhav)
    filtered = bhav[bhav["SC_CODE"].isin(tradable_codes)].copy()
    logger.info(
        "Universe filter: %d → %d stocks",
        bhav["SC_CODE"].nunique(), filtered["SC_CODE"].nunique(),
    )
    return filtered


def _build_features(bhav: pd.DataFrame) -> pd.DataFrame:
    try:
        from momentum_features import prepare_features_all, add_forward_returns
    except ImportError:
        from production.signal_generator import SignalGenerator
        sg = SignalGenerator()
        try:
            from momentum_features import prepare_features_all, add_forward_returns
        except ImportError:
            raise RuntimeError(
                "momentum_features.py not found — cannot build features"
            )

    logger.info("Building features …")
    feat_df = prepare_features_all(bhav)
    feat_df = add_forward_returns(feat_df, periods=[5])
    label_col = "Label_fwd5_positive"
    if label_col not in feat_df.columns:
        raise RuntimeError(f"add_forward_returns did not produce '{label_col}'")

    # Drop rows without a label (last 5 sessions can't have forward returns)
    before = len(feat_df)
    feat_df = feat_df.dropna(subset=[label_col])
    logger.info(
        "Features: %d rows → %d after dropping unlabelled tail",
        before, len(feat_df),
    )
    return feat_df


def _train_lgbm(feat_df: pd.DataFrame, feature_cols: list, n_cv_splits: int, n_boost_rounds: int):
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score

    label_col = "Label_fwd5_positive"
    X = feat_df[feature_cols].values
    y = feat_df[label_col].values

    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    cv_aucs = []

    logger.info(
        "TimeSeriesSplit CV (%d folds) on %d samples × %d features …",
        n_cv_splits, len(X), len(feature_cols),
    )
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
        }
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_boost_rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )
        val_preds = model.predict(X_val)
        auc = roc_auc_score(y_val, val_preds)
        cv_aucs.append(auc)
        logger.info("  Fold %d: AUC=%.4f  trees=%d", fold, auc, model.num_trees())

    logger.info("CV AUC: %.4f ± %.4f", np.mean(cv_aucs), np.std(cv_aucs))

    # Final model on full dataset
    logger.info("Training final model on full dataset …")
    dtrain_full = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params,
        dtrain_full,
        num_boost_round=int(np.mean([m.num_trees() for m in [model]])),
        callbacks=[lgb.log_evaluation(period=-1)],
    )
    return final_model, {"cv_aucs": cv_aucs, "mean_auc": float(np.mean(cv_aucs)), "std_auc": float(np.std(cv_aucs))}


def _save_model(model, feature_cols: list, metrics: dict):
    model_path = MODEL_DIR / "lgbm_model.txt"
    feat_path = MODEL_DIR / "feature_cols.json"
    metrics_path = MODEL_DIR / "training_metrics.json"

    model.save_model(str(model_path))
    feat_path.write_text(json.dumps(feature_cols, indent=2))

    metrics["trained_on"] = str(date.today())
    metrics["n_features"] = len(feature_cols)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    logger.info("Model saved → %s", model_path)
    logger.info("Features saved → %s", feat_path)
    logger.info("Metrics saved → %s", metrics_path)


def _fit_calibrator(model, feat_df: pd.DataFrame, feature_cols: list):
    try:
        from production.probability_calibration import ProbabilityCalibrator, CalibrationMethod
        import pickle

        label_col = "Label_fwd5_positive"
        X = feat_df[feature_cols].values
        y = feat_df[label_col].values
        raw_probs = model.predict(X)

        calibrator = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC)
        calibrator.fit(raw_probs, y)

        cal_path = MODEL_DIR / "calibrator.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump(calibrator, f)
        logger.info("Calibrator saved → %s", cal_path)
    except Exception as exc:
        logger.warning("Calibrator fitting skipped: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Retrain LightGBM swing-trade model")
    parser.add_argument("--lookback", type=int, default=730, help="Calendar days of history (default 730)")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--cv-splits", type=int, default=5, help="TimeSeriesSplit folds (default 5)")
    parser.add_argument("--boost-rounds", type=int, default=500, help="Max boosting rounds (default 500)")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip probability calibration step")
    parser.add_argument("--no-ca", action="store_true", help="Skip corporate action adjustment")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Stockpicker model retraining — %s", date.today())
    logger.info("=" * 60)

    # 1. Load data
    bhav = _load_data(args)
    if bhav.empty:
        logger.error("No data loaded — aborting.")
        sys.exit(1)

    # 2. Corporate action adjustment
    if not args.no_ca:
        bhav = _apply_corporate_actions(bhav)

    # 3. Universe filter (training universe should match live universe)
    bhav = _filter_universe(bhav)
    if bhav.empty:
        logger.error("Universe filter removed all rows — aborting.")
        sys.exit(1)

    # 4. Feature engineering + labels
    feat_df = _build_features(bhav)
    if feat_df.empty:
        logger.error("No labelled rows — aborting.")
        sys.exit(1)

    # 5. Determine feature columns
    try:
        from production.signal_generator import BASE_FEATURE_COLS
        feature_cols = [c for c in BASE_FEATURE_COLS if c in feat_df.columns]
    except ImportError:
        from momentum_features import BASE_FEATURE_COLS
        feature_cols = [c for c in BASE_FEATURE_COLS if c in feat_df.columns]

    if len(feature_cols) < 10:
        logger.error("Too few feature columns available (%d) — check momentum_features.py", len(feature_cols))
        sys.exit(1)
    logger.info("Using %d feature columns", len(feature_cols))

    # 6. Train
    model, metrics = _train_lgbm(feat_df, feature_cols, args.cv_splits, args.boost_rounds)

    # 7. Save model + metadata
    _save_model(model, feature_cols, metrics)

    # 8. Calibration
    if not args.no_calibrate:
        _fit_calibrator(model, feat_df, feature_cols)

    logger.info("=" * 60)
    logger.info("Retraining complete.  CV AUC = %.4f", metrics["mean_auc"])
    logger.info("Model ready at: %s", MODEL_DIR / "lgbm_model.txt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
