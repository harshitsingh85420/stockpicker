"""
train_model.py — Periodic LightGBM retraining script.

CLI usage:
    python train_model.py                          # train on last 2 years
    python train_model.py --lookback 365           # 1 year
    python train_model.py --start 2023-01-01       # explicit range
    python train_model.py --no-calibrate           # skip calibration step

Programmatic usage (from run_full_cycle.py etc.):
    from train_model import run_training
    metrics = run_training(lookback_days=730, end_date="2026-04-24")
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

logger = logging.getLogger(__name__)

MODEL_DIR = _ROOT / "stock_picker_data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Internal helpers (all private — prefixed with _)
# ---------------------------------------------------------------------------

def _load_data(start=None, end=None, lookback=730) -> pd.DataFrame:
    from production.data_loader import DataLoader
    loader = DataLoader()
    if start:
        bhav = loader.load(start=start, end=end)
    else:
        bhav = loader.load(lookback_days=lookback, end=end)
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
            logger.info("No corporate actions found — skipping adjustment")
    except Exception as exc:
        logger.warning("Corporate action adjustment skipped: %s", exc)
    return bhav


def _filter_universe(bhav: pd.DataFrame) -> pd.DataFrame:
    from production.universe_filter import TradabilityGate
    gate = TradabilityGate(min_value_crore=2.0, min_price=20.0, min_avg_volume=10_000)
    reference_date = bhav["DATE"].max()
    result = gate.apply(bhav, reference_date)
    # apply() returns a filtered DataFrame; extract tradable codes and re-filter full bhav
    if isinstance(result, pd.DataFrame) and "SC_CODE" in result.columns:
        tradable_codes = set(result["SC_CODE"].unique())
    else:
        tradable_codes = set(result)          # fallback if it returns a set/list
    before = bhav["SC_CODE"].nunique()
    filtered = bhav[bhav["SC_CODE"].isin(tradable_codes)].copy()
    logger.info("Universe filter: %d -> %d stocks", before, filtered["SC_CODE"].nunique())
    return filtered


def _build_features(bhav: pd.DataFrame, use_triple_barrier: bool = False) -> pd.DataFrame:
    from momentum_features import prepare_features_all, add_forward_returns, add_fracdiff_features

    logger.info("Building features ...")
    feat_df = prepare_features_all(bhav)
    feat_df = add_fracdiff_features(feat_df, d=0.4)   # P14: fractional differentiation
    # P47: FII/DII market-wide features (no-op when API unavailable)
    try:
        from momentum_features import FIIDIIFeatures
        start_str = str(bhav["DATE"].min())[:10]
        end_str   = str(bhav["DATE"].max())[:10]
        feat_df = FIIDIIFeatures().merge_into_features(feat_df, start_str, end_str)
    except Exception as _fe:
        logger.debug("FII/DII features skipped in training: %s", _fe)
    feat_df = add_forward_returns(feat_df, periods=[5])

    # P10 -- optionally use triple-barrier labels
    if use_triple_barrier:
        try:
            from momentum_features import add_triple_barrier_labels
            feat_df = add_triple_barrier_labels(feat_df)
            label_col = "Label_tb_positive"
            if label_col not in feat_df.columns:
                logger.warning("Triple-barrier label missing -- falling back to binary label.")
                label_col = "Label_fwd5_positive"
            else:
                logger.info("P10: Using triple-barrier labels (Label_tb_positive).")
        except Exception as e:
            logger.warning("Triple-barrier labels failed (%s) -- using binary label.", e)
            label_col = "Label_fwd5_positive"
    else:
        label_col = "Label_fwd5_positive"

    if label_col not in feat_df.columns:
        raise RuntimeError(f"Label column '{label_col}' not found after feature build.")

    feat_df.attrs["label_col"] = label_col

    before = len(feat_df)
    feat_df = feat_df.dropna(subset=[label_col])
    logger.info(
        "Features: %d rows -> %d after dropping unlabelled tail  (label=%s)",
        before, len(feat_df), label_col,
    )
    return feat_df


def _train_lgbm(feat_df: pd.DataFrame, feature_cols: list, n_cv_splits: int, n_boost_rounds: int):
    import lightgbm as lgb
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score

    # P10: use triple-barrier label if _build_features set it
    label_col = feat_df.attrs.get("label_col", "Label_fwd5_positive")

    # ── P05 safeguards: assert no target leakage into features ──────────
    FORBIDDEN_FEATURES = {
        "Label_fwd5_positive", "Label_fwd5_return",
        "forward_return_5d", "_fwd_close",
    }
    leaked = [c for c in feature_cols if c in FORBIDDEN_FEATURES]
    assert not leaked, f"P05: Target column(s) leaked into feature set! {leaked}"

    FORWARD_SHIFT_PATTERN = [c for c in feature_cols if "fwd" in c.lower() or "forward" in c.lower()]
    if FORWARD_SHIFT_PATTERN:
        logger.warning(
            "P05: Suspicious forward-looking column names in features: %s",
            FORWARD_SHIFT_PATTERN,
        )
    # ─────────────────────────────────────────────────────────────────────

    X = feat_df[feature_cols].fillna(0).values
    y = feat_df[label_col].values

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
        "n_jobs": -1,
        "verbose": -1,
    }

    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    cv_aucs = []
    best_rounds = []

    logger.info(
        "TimeSeriesSplit CV (%d folds) on %d samples × %d features …",
        n_cv_splits, len(X), len(feature_cols),
    )
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval   = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        m = lgb.train(
            params,
            dtrain,
            num_boost_round=n_boost_rounds,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=0)],
        )
        auc = roc_auc_score(y_val, m.predict(X_val))
        cv_aucs.append(auc)
        best_rounds.append(m.best_iteration or m.num_trees())
        logger.info("  Fold %d: AUC=%.4f  trees=%d", fold, auc, best_rounds[-1])

    logger.info("CV AUC: %.4f ± %.4f", np.mean(cv_aucs), np.std(cv_aucs))

    # Final model on all data — use median of CV best-round counts
    final_rounds = max(50, int(np.median(best_rounds)))
    logger.info("Training final model — %d boost rounds …", final_rounds)
    dtrain_full = lgb.Dataset(X, label=y)
    final_model = lgb.train(
        params, dtrain_full,
        num_boost_round=final_rounds,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    metrics = {
        "cv_aucs":  [round(a, 6) for a in cv_aucs],
        "mean_auc": float(np.mean(cv_aucs)),
        "std_auc":  float(np.std(cv_aucs)),
        "final_boost_rounds": final_rounds,
    }
    return final_model, metrics


def _save_model(model, feature_cols: list, metrics: dict):
    model_path   = MODEL_DIR / "lgbm_model.txt"
    feat_path    = MODEL_DIR / "feature_cols.json"
    metrics_path = MODEL_DIR / "training_metrics.json"

    model.save_model(str(model_path))
    feat_path.write_text(json.dumps(feature_cols, indent=2))
    metrics["trained_on"] = str(date.today())
    metrics["n_features"]  = len(feature_cols)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Also write feature_names.json (path expected by SignalGenerator)
    (MODEL_DIR / "feature_names.json").write_text(json.dumps(feature_cols, indent=2))

    logger.info("Model saved      → %s", model_path)
    logger.info("Features saved   → %s", feat_path)
    logger.info("Metrics saved    → %s", metrics_path)
    return model_path


def _fit_calibrator(model, feat_df: pd.DataFrame, feature_cols: list):
    """
    P43 (fix): Fit calibrator using out-of-fold predictions so isotonic
    regression doesn't overfit to the same data the model was trained on.
    Falls back to global fit if OOF fails.
    """
    try:
        import pickle
        from production.probability_calibration import fit_oof_calibrator

        label_col = "Label_fwd5_positive"
        X = feat_df[feature_cols].fillna(0).values
        y = feat_df[label_col].values

        # OOF calibration — train-set ECE is no longer the target metric
        calibrator = fit_oof_calibrator(model, X, y, n_splits=5, method="isotonic")

        cal_path = MODEL_DIR / "calibrator.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump(calibrator, f)
        logger.info("P43 OOF calibrator saved → %s", cal_path)
    except Exception as exc:
        logger.warning("Calibrator fitting skipped: %s", exc)


def _train_xgb(feat_df: pd.DataFrame, feature_cols: list, n_cv_splits: int, n_boost_rounds: int):
    """
    P12 -- Train XGBoost model on same features/labels used by LightGBM.
    Returns (model, metrics) or (None, {}) if xgboost is not installed.
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("P12: xgboost not installed — skipping XGBoost ensemble component.")
        return None, {}

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    import pickle

    label_col = feat_df.attrs.get("label_col", "Label_fwd5_positive")
    X = feat_df[feature_cols].fillna(0).values
    y = feat_df[label_col].values

    params = {
        "objective":        "binary:logistic",
        "eval_metric":      "auc",
        "learning_rate":    0.05,
        "max_depth":        6,
        "min_child_weight": 50,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "reg_alpha":        0.1,
        "reg_lambda":       0.1,
        "n_jobs":           -1,
        "verbosity":        0,
        "seed":             42,
    }

    tscv = TimeSeriesSplit(n_splits=n_cv_splits)
    cv_aucs = []
    best_rounds = []

    logger.info("P12 XGBoost TimeSeriesSplit CV (%d folds) ...", n_cv_splits)
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dval   = xgb.DMatrix(X_val, label=y_val)

        m = xgb.train(
            params,
            dtrain,
            num_boost_round=n_boost_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        auc = roc_auc_score(y_val, m.predict(dval))
        cv_aucs.append(auc)
        best_rounds.append(m.best_iteration + 1)
        logger.info("  XGB Fold %d: AUC=%.4f  trees=%d", fold, auc, best_rounds[-1])

    logger.info("P12 XGBoost CV AUC: %.4f +/- %.4f", np.mean(cv_aucs), np.std(cv_aucs))

    final_rounds = max(50, int(np.median(best_rounds)))
    dtrain_full  = xgb.DMatrix(X, label=y)
    final_model  = xgb.train(
        params, dtrain_full,
        num_boost_round=final_rounds,
        verbose_eval=False,
    )

    xgb_path = MODEL_DIR / "xgb_model.pkl"
    with open(xgb_path, "wb") as fh:
        pickle.dump(final_model, fh)
    logger.info("P12 XGBoost model saved -> %s", xgb_path)

    metrics = {
        "xgb_cv_aucs":  [round(a, 6) for a in cv_aucs],
        "xgb_mean_auc": float(np.mean(cv_aucs)),
        "xgb_std_auc":  float(np.std(cv_aucs)),
        "xgb_final_boost_rounds": final_rounds,
    }
    return final_model, metrics


# ---------------------------------------------------------------------------
# Public callable API
# ---------------------------------------------------------------------------

def run_training(
    lookback_days: int = 730,
    start: str = None,
    end: str = None,
    cv_splits: int = 5,
    boost_rounds: int = 500,
    calibrate: bool = True,
    apply_ca: bool = True,
    use_triple_barrier: bool = False,
) -> dict:
    """
    Train the LightGBM swing-trade model and save to disk.

    Can be called from other scripts (e.g. run_full_cycle.py).

    Args:
        lookback_days:       Calendar days of history to use (if start not given).
        start:               Explicit start date 'YYYY-MM-DD'.
        end:                 Explicit end date 'YYYY-MM-DD' (default: today).
        cv_splits:           TimeSeriesSplit folds.
        boost_rounds:        Max LightGBM boosting rounds per fold.
        calibrate:           Fit and save ProbabilityCalibrator after training.
        apply_ca:            Apply corporate action price adjustments.
        use_triple_barrier:  P10: Use triple-barrier labels instead of binary
                             close-to-close labels (default False).

    Returns:
        dict with keys: mean_auc, std_auc, feature_count, n_training_samples,
                        model_path, trained_on, label_col.
    """
    logger.info("=" * 60)
    logger.info("Model retraining — %s", date.today())
    logger.info("=" * 60)

    # 1. Load data
    bhav = _load_data(start=start, end=end, lookback=lookback_days)
    if bhav.empty:
        raise RuntimeError("No BhavCopy data loaded — aborting training.")

    # 2. Corporate actions
    if apply_ca:
        bhav = _apply_corporate_actions(bhav)

    # 3. Universe filter
    bhav = _filter_universe(bhav)
    if bhav.empty:
        raise RuntimeError("Universe filter removed all rows — aborting training.")

    # 4. Feature engineering + labels
    feat_df = _build_features(bhav, use_triple_barrier=use_triple_barrier)
    if feat_df.empty:
        raise RuntimeError("No labelled feature rows — aborting training.")

    # 5. Select feature columns (P29: canonical list enforced everywhere)
    # Priority: existing feature_cols.json > BASE_FEATURE_COLS > numeric fallback
    canonical_path = MODEL_DIR / "feature_cols.json"
    if canonical_path.exists():
        try:
            canonical = json.load(open(canonical_path))
            # Only keep features that exist in this training data
            feature_cols = [c for c in canonical if c in feat_df.columns]
            missing = [c for c in canonical if c not in feat_df.columns]
            if missing:
                logger.warning("P29: %d canonical features missing from data: %s",
                               len(missing), missing)
        except Exception:
            feature_cols = []
    else:
        feature_cols = []

    if len(feature_cols) < 10:
        try:
            from production.signal_generator import BASE_FEATURE_COLS
        except ImportError:
            BASE_FEATURE_COLS = []
        feature_cols = [c for c in BASE_FEATURE_COLS if c in feat_df.columns]

    if len(feature_cols) < 10:
        exclude = {
            "SC_CODE", "SC_NAME", "DATE",
            "Label_fwd5_positive", "Label_fwd5_return",
            "Label_tb_positive", "Label_tb_pct",
        }
        feature_cols = [
            c for c in feat_df.select_dtypes(include=[np.number]).columns
            if c not in exclude
        ]

    # P29: Hard subset + assertion — never use X.columns directly
    feat_df_cols_available = [c for c in feature_cols if c in feat_df.columns]
    feature_cols = feat_df_cols_available
    logger.info("P29 MAIN TRAIN COLS (%d): %s", len(feature_cols), sorted(feature_cols))
    assert len(feature_cols) > 0, "P29: zero feature columns — aborting"

    # 6. Train
    model, metrics = _train_lgbm(feat_df, feature_cols, cv_splits, boost_rounds)

    # 7. Save
    model_path = _save_model(model, feature_cols, metrics)

    # 7b. Train XGBoost ensemble member (P12)
    xgb_model, xgb_metrics = _train_xgb(feat_df, feature_cols, cv_splits, boost_rounds)
    if xgb_metrics:
        metrics.update(xgb_metrics)

    # 8. Calibration + calibration curve report (P16)
    if calibrate:
        _fit_calibrator(model, feat_df, feature_cols)
        try:
            from production.probability_calibration import calibration_curve_report
            import pickle
            label_col = feat_df.attrs.get("label_col", "Label_fwd5_positive")
            y = feat_df[label_col].dropna().values
            X_cal = feat_df.loc[feat_df[label_col].notna(), feature_cols].fillna(0).values
            raw_probs = model.predict(X_cal)
            cal_path = MODEL_DIR / "calibrator.pkl"
            if cal_path.exists():
                with open(cal_path, "rb") as fh:
                    cal_obj = pickle.load(fh)
                cal_probs = cal_obj.calibrate(raw_probs)
            else:
                cal_probs = None
            cal_report = calibration_curve_report(
                y, raw_probs, cal_probs,
                chart_path=str(MODEL_DIR / "calibration_curve.png"),
            )
            metrics["raw_ece"]   = cal_report["raw_ece"]
            metrics["raw_brier"] = cal_report["raw_brier"]
            if "cal_ece" in cal_report:
                metrics["cal_ece"]   = cal_report["cal_ece"]
                metrics["cal_brier"] = cal_report["cal_brier"]
            logger.info(
                "P16 Calibration: raw_ece=%.4f  cal_ece=%s",
                cal_report["raw_ece"],
                f"{cal_report.get('cal_ece', 'N/A'):.4f}" if "cal_ece" in cal_report else "N/A",
            )
        except Exception as exc:
            logger.warning("P16 calibration curve report skipped: %s", exc)

    logger.info("=" * 60)
    logger.info("Training complete.  CV AUC = %.4f", metrics["mean_auc"])
    logger.info("Model ready at: %s", model_path)
    logger.info("=" * 60)

    return {
        "mean_auc":            metrics["mean_auc"],
        "std_auc":             metrics["std_auc"],
        "feature_count":       len(feature_cols),
        "n_training_samples":  len(feat_df),
        "model_path":          str(model_path),
        "trained_on":          str(date.today()),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Retrain LightGBM swing-trade model")
    parser.add_argument("--lookback",      type=int,   default=730,  help="Calendar days of history (default 730)")
    parser.add_argument("--start",         type=str,   default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",           type=str,   default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--cv-splits",     type=int,   default=5,    help="TimeSeriesSplit folds (default 5)")
    parser.add_argument("--boost-rounds",  type=int,   default=500,  help="Max boosting rounds (default 500)")
    parser.add_argument("--no-calibrate",  action="store_true",      help="Skip probability calibration")
    parser.add_argument("--no-ca",         action="store_true",      help="Skip corporate action adjustment")
    args = parser.parse_args()

    try:
        metrics = run_training(
            lookback_days=args.lookback,
            start=args.start,
            end=args.end,
            cv_splits=args.cv_splits,
            boost_rounds=args.boost_rounds,
            calibrate=not args.no_calibrate,
            apply_ca=not args.no_ca,
        )
        print(f"\nTraining complete. CV AUC = {metrics['mean_auc']:.4f}")
    except Exception as e:
        logger.error("Training failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
