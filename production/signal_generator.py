"""
SignalGenerator — the heart of the production pipeline.

Bridges the existing feature-engineering pipeline (momentum_features.py)
and the LightGBM model (stock_picker_enhanced.py / saved pkl) to the
production layer, outputting a ranked DataFrame of trade candidates.

Flow
----
  1. Accepts standardised BhavCopy (from DataLoader)
  2. Calls prepare_features_all() or prepare_features_enhanced() for indicators
  3. Loads or trains a LightGBM model
  4. Predicts on latest-date rows only
  5. Applies optional calibration (ProbabilityCalibrator)
  6. Returns picks DataFrame + full feature_df for SHAP / calibration
"""

import json
import logging
import pickle
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# -- Feature columns used by the base model -----------------------------------
BASE_FEATURE_COLS = [
    "EMA20", "EMA50", "EMA200", "EMA20_Slope5", "EMA200_Slope", "MA_Health", "OverEMA20",
    "ATR14", "ATRpct", "BBWidth", "BBWidthPctl",
    "DistTo20", "DistTo63", "DistTo52W",
    "Break20_Today", "Break63_Today", "Hit52WH_Today", "RangePos20",
    "VolMult", "UD_Vol_Ratio10",
    "RET21D", "RET63D", "RS_Composite",
    "RSI14", "ADX14", "+DI14", "-DI14", "ADX14_chg3",
    "W_BBWidth", "W_TrendOK", "W_BBWidthPctl",
    # P14: fractional differentiation features
    "FracDiff_Close", "FracDiff_LogClose",
]

# Label column produced by add_forward_returns()
_LABEL_COL_TEMPLATE = "Label_fwd{period}_positive"


class SignalGenerator:
    """
    Generate trade signals from standardised BhavCopy data.

    Attributes
    ----------
    model_path       : Path to saved LightGBM model (txt) or pkl.
    feature_path     : Path to saved feature names JSON.
    calibrator_path  : Path to saved ProbabilityCalibrator pkl.
    base_dir         : Root data directory.
    forward_period   : Sessions ahead used for the label (default 5).
    min_data_points  : Minimum rows per stock for training (default 200).
    """

    def __init__(
        self,
        base_dir: str = "stock_picker_data",
        forward_period: int = 5,
        min_data_points: int = 200,
    ):
        self.base_dir         = Path(base_dir)
        self.models_dir       = self.base_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.forward_period   = forward_period
        self.min_data_points  = min_data_points

        self.model_path       = self.models_dir / "lgbm_model.txt"
        self.feature_path     = self.models_dir / "feature_names.json"
        self.calibrator_path  = self.models_dir / "calibrator.pkl"
        self.xgb_model_path   = self.models_dir / "xgb_model.pkl"

        self._model           = None
        self._feature_cols    = None
        self._calibrator      = None
        self._xgb_model       = None   # P12: XGBoost ensemble member

    # ====================================================================
    # Public API
    # ====================================================================

    def generate(
        self,
        bhav_df: pd.DataFrame,
        threshold: float = 0.62,
        calibrate: bool = True,
        regime_threshold_override: float = None,
        apply_meta_filter: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main entry point: compute features, predict, return picks.

        Args:
            bhav_df:                  Standardised BhavCopy (from DataLoader).
            threshold:                Minimum calibrated probability to include.
            calibrate:                Apply saved calibrator if available.
            regime_threshold_override: Override threshold (e.g. from RegimeFilter).
            apply_meta_filter:        Apply P13 meta-labeling filter if model exists.

        Returns:
            (picks_df, feature_df)
            picks_df   -- sorted by probability desc, with all signal columns.
            feature_df -- full feature DataFrame for SHAP / drift monitoring.
        """
        # P31: temporary safeguard while pick-count explosion root cause is investigated
        MAX_DAILY_PICKS = 20

        effective_threshold = regime_threshold_override if regime_threshold_override else threshold

        # -- 1. Feature engineering -------------------------------------
        logger.info("Computing features …")
        feature_df = self._compute_features(bhav_df)
        if feature_df is None or feature_df.empty:
            logger.error("Feature computation returned empty DataFrame.")
            return pd.DataFrame(), pd.DataFrame()

        # -- 2. Load / train model --------------------------------------
        model = self._ensure_model(feature_df)
        if model is None:
            logger.error("No model available — cannot generate signals.")
            return pd.DataFrame(), feature_df

        # -- 3. Predict on latest date ----------------------------------
        raw_picks = self._predict_latest(feature_df, model)
        if raw_picks.empty:
            logger.warning("No predictions generated.")
            return pd.DataFrame(), feature_df

        # -- 4. Calibrate -----------------------------------------------
        if calibrate:
            raw_picks = self._apply_calibration(raw_picks)

        # -- P31: daily pick diagnostics (pre-threshold) ----------------
        universe_size = len(raw_picks)
        n_above = (raw_picks["Probability"] >= effective_threshold).sum()
        mean_score   = float(raw_picks["Probability"].mean())
        median_score = float(raw_picks["Probability"].median())
        pct_above    = n_above / universe_size if universe_size > 0 else 0.0
        logger.info(
            "P31 pick diagnostics: universe=%d  above_thresh=%d (%.1f%%)  "
            "mean_prob=%.4f  median_prob=%.4f  threshold=%.4f",
            universe_size, n_above, pct_above * 100,
            mean_score, median_score, effective_threshold,
        )

        # -- 5. Apply threshold -----------------------------------------
        picks = raw_picks[raw_picks["Probability"] >= effective_threshold].copy()
        picks = picks.sort_values("Probability", ascending=False).reset_index(drop=True)
        picks["Signal_Threshold"] = round(effective_threshold, 4)

        # -- 6. Meta-labeling filter (P13) ------------------------------
        if apply_meta_filter and not picks.empty:
            try:
                from production.meta_labeler import MetaLabeler
                meta = MetaLabeler()
                picks, _ = meta.filter_picks(picks, primary_prob_col="Probability_Raw")
            except Exception as exc:
                logger.debug("P13 meta-filter skipped: %s", exc)

        picks = picks.sort_values("Probability", ascending=False).reset_index(drop=True)

        # -- P31: enforce MAX_DAILY_PICKS cap ---------------------------
        # Root cause: P31 pick explosion investigation — cap is a temporary safeguard.
        # Remove once score-distribution drift is fully diagnosed and fixed.
        if len(picks) > MAX_DAILY_PICKS:
            logger.warning(
                "P31 MAX_DAILY_PICKS cap: %d picks -> %d (top by probability). "
                "Root cause of explosion under investigation.",
                len(picks), MAX_DAILY_PICKS,
            )
            picks = picks.head(MAX_DAILY_PICKS)

        picks["Rank"] = range(1, len(picks) + 1)

        logger.info(
            "SignalGenerator: %d picks above threshold %.2f (from %d candidates).",
            len(picks), effective_threshold, len(raw_picks),
        )
        return picks, feature_df

    # ------------------------------------------------------------------
    def train(
        self,
        bhav_df: pd.DataFrame,
        save: bool = True,
        n_boost_rounds: int = 500,
        n_cv_splits: int = 5,
    ) -> dict:
        """
        Train a LightGBM model on bhav_df and optionally save to disk.

        Returns:
            dict with cv_auc, feature_count, n_training_samples, model_path.
        """
        import lightgbm as lgb
        from sklearn.model_selection import TimeSeriesSplit

        logger.info("Starting model training …")

        # -- Feature engineering ----------------------------------------
        feature_df = self._compute_features(bhav_df)
        if feature_df is None or feature_df.empty:
            raise ValueError("Feature computation failed — cannot train.")

        # -- Add labels -------------------------------------------------
        label_col = _LABEL_COL_TEMPLATE.format(period=self.forward_period)
        feature_df = self._add_labels(feature_df)
        df_train = feature_df.dropna(subset=[label_col]).copy()

        # -- Select feature columns -------------------------------------
        feat_cols = self._select_feature_cols(df_train)

        # Drop features where >50% of training rows are NaN (e.g. EMA200 on short windows)
        feat_cols = [
            c for c in feat_cols
            if df_train[c].isna().mean() <= 0.50
        ]
        if not feat_cols:
            raise ValueError(
                "All feature columns have >50% NaN — dataset too short for chosen indicators. "
                "Use lookback_days >= 250."
            )

        df_train  = df_train.dropna(subset=feat_cols)
        X = df_train[feat_cols].fillna(0)
        y = df_train[label_col].astype(int)

        logger.info("Training set: %d rows, %d features, %.1f%% positive.",
                    len(X), len(feat_cols), y.mean() * 100)

        # -- Time-series CV ---------------------------------------------
        tscv   = TimeSeriesSplit(n_splits=n_cv_splits)
        aucs   = []

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval   = lgb.Dataset(X_va, label=y_va, reference=dtrain)

            m = lgb.train(
                self._lgb_params(),
                dtrain,
                num_boost_round=n_boost_rounds,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False),
                           lgb.log_evaluation(0)],
            )
            preds = m.predict(X_va)
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_va, preds)
            aucs.append(auc)
            logger.info("  Fold %d AUC = %.4f", fold + 1, auc)

        cv_auc = float(np.mean(aucs))
        logger.info("CV AUC = %.4f ± %.4f", cv_auc, float(np.std(aucs)))

        # -- Final model on all data ------------------------------------
        dtrain_full = lgb.Dataset(X, label=y)
        final_model = lgb.train(
            self._lgb_params(),
            dtrain_full,
            num_boost_round=int(n_boost_rounds * 0.8),
            callbacks=[lgb.log_evaluation(0)],
        )

        self._model       = final_model
        self._feature_cols = feat_cols

        if save:
            self._save_model(final_model, feat_cols)
            self._save_reference_distribution(X, final_model.predict(X.values))

        return {
            "cv_auc":            cv_auc,
            "feature_count":     len(feat_cols),
            "n_training_samples": len(X),
            "model_path":        str(self.model_path),
        }

    # ------------------------------------------------------------------
    def is_model_available(self) -> bool:
        return self.model_path.exists()

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance if model is loaded."""
        model = self._ensure_model(None)
        if model is None or self._feature_cols is None:
            return pd.DataFrame()
        imp = model.feature_importance(importance_type="gain")
        return (
            pd.DataFrame({"feature": self._feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ====================================================================
    # Internal helpers
    # ====================================================================

    def _compute_features(self, bhav_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Call the existing feature-engineering pipeline."""
        try:
            # Try enhanced features first
            try:
                from momentum_features_enhanced import prepare_features_enhanced
                logger.info("Using enhanced feature pipeline (90+ features).")
                return prepare_features_enhanced(bhav_df, use_advanced=False, use_fii_dii=False)
            except (ImportError, Exception):
                pass

            from momentum_features import prepare_features_all
            logger.info("Using base feature pipeline (50+ features).")
            return prepare_features_all(bhav_df)

        except Exception as e:
            logger.error("Feature computation error: %s", e)
            return None

    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add forward-return labels (uses momentum_features helper if available)."""
        try:
            from momentum_features import add_forward_returns
            return add_forward_returns(df, periods=[self.forward_period])
        except Exception:
            # Fallback: compute manually
            label_col = _LABEL_COL_TEMPLATE.format(period=self.forward_period)
            df = df.sort_values(["SC_CODE", "DATE"]).copy()
            df["_fwd_close"] = df.groupby("SC_CODE")["Close"].shift(-self.forward_period)
            df[label_col] = ((df["_fwd_close"] / df["Close"] - 1) > 0).astype(int)
            df = df.drop(columns=["_fwd_close"])
            return df

    def _predict_latest(self, feature_df: pd.DataFrame, model) -> pd.DataFrame:
        """Predict on the most recent trading date's rows."""
        date_col = "DATE"
        feature_df[date_col] = pd.to_datetime(feature_df[date_col])
        latest = feature_df[date_col].max()
        today_df = feature_df[feature_df[date_col] == latest].copy()

        if today_df.empty:
            return pd.DataFrame()

        feat_cols = self._feature_cols or self._select_feature_cols(feature_df)
        available = [c for c in feat_cols if c in today_df.columns]
        if not available:
            logger.error("No feature columns found in today's data.")
            return pd.DataFrame()

        # Fill any missing feature cols with 0
        X = today_df.reindex(columns=feat_cols, fill_value=0).fillna(0)

        try:
            probs = model.predict(X.values)
        except Exception:
            try:
                probs = model.predict_proba(X.values)[:, 1]
            except Exception as e:
                logger.error("Model predict failed: %s", e)
                return pd.DataFrame()

        # P12: XGBoost ensemble — average with LightGBM if available
        xgb_model = self._load_xgb_model()
        if xgb_model is not None:
            try:
                import xgboost as xgb
                dmat = xgb.DMatrix(X.values)
                xgb_probs = xgb_model.predict(dmat)
                probs = 0.5 * probs + 0.5 * xgb_probs
                logger.info("P12: LightGBM + XGBoost ensemble applied.")
            except Exception as exc:
                logger.debug("P12: XGBoost ensemble skipped: %s", exc)

        today_df = today_df.copy()
        today_df["Probability_Raw"] = probs
        today_df["Probability"]     = probs   # overwritten by calibration

        # Build picks dataframe with all useful signal columns
        keep_cols = ["SC_CODE", "SC_NAME", "Close", "Probability_Raw", "Probability"]
        signal_cols = [
            "ATR14", "ATRpct", "VolMult", "RS_Composite",
            "ADX14", "RSI14", "DistTo52W", "Break63_Today",
            "EMA20", "EMA50", "EMA200", "BBWidth",
        ]
        for c in signal_cols:
            if c in today_df.columns:
                keep_cols.append(c)

        result = today_df[keep_cols].copy()
        result["Prediction_Date"] = latest.date() if hasattr(latest, "date") else latest
        return result.sort_values("Probability", ascending=False).reset_index(drop=True)

    def _apply_calibration(self, picks_df: pd.DataFrame) -> pd.DataFrame:
        """Apply probability calibration if calibrator is available."""
        cal = self._ensure_calibrator()
        if cal is None:
            return picks_df
        try:
            raw = picks_df["Probability_Raw"].values
            calibrated = cal.calibrate(raw)
            picks_df = picks_df.copy()
            picks_df["Probability"] = calibrated
            logger.info("Probability calibration applied.")
        except Exception as e:
            logger.debug("Calibration skipped: %s", e)
        return picks_df

    def _select_feature_cols(self, df: pd.DataFrame) -> list:
        """Select available feature columns from the preferred list."""
        if self._feature_cols:
            return self._feature_cols

        # Try saved list first
        if self.feature_path.exists():
            try:
                with open(self.feature_path) as fh:
                    cols = json.load(fh)
                available = [c for c in cols if c in df.columns]
                if available:
                    self._feature_cols = available
                    return available
            except Exception:
                pass

        # Fallback: use BASE_FEATURE_COLS that exist in df
        available = [c for c in BASE_FEATURE_COLS if c in df.columns]
        if not available:
            # Last resort: all numeric columns except metadata
            exclude = {"SC_CODE", "SC_NAME", "DATE", "ISIN", "Source",
                       "Label_fwd5_positive", "Label_fwd5_return"}
            available = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]
        self._feature_cols = available
        return available

    def _ensure_model(self, feature_df: Optional[pd.DataFrame]):
        """Load model from disk or train if not available."""
        if self._model is not None:
            return self._model

        # Try loading saved LightGBM txt model
        if self.model_path.exists():
            try:
                import lightgbm as lgb
                self._model = lgb.Booster(model_file=str(self.model_path))
                if self.feature_path.exists():
                    with open(self.feature_path) as fh:
                        self._feature_cols = json.load(fh)
                logger.info("Model loaded from %s.", self.model_path)
                return self._model
            except Exception as e:
                logger.warning("LightGBM model load failed: %s", e)

        # Try loading pkl model
        pkl_path = self.models_dir / "lgbm_model.pkl"
        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as fh:
                    pkg = pickle.load(fh)
                self._model = pkg.get("model", pkg)
                self._feature_cols = pkg.get("feature_names")
                logger.info("Pkl model loaded from %s.", pkl_path)
                return self._model
            except Exception as e:
                logger.warning("Pkl model load failed: %s", e)

        # No saved model — train on provided data
        if feature_df is not None and not feature_df.empty:
            logger.warning("No saved model found — training now (this will take a few minutes).")
            try:
                self.train(feature_df, save=True)
                return self._model
            except Exception as e:
                logger.error("Auto-training failed: %s", e)

        return None

    def _ensure_calibrator(self):
        """Load calibrator from disk."""
        if self._calibrator is not None:
            return self._calibrator
        if self.calibrator_path.exists():
            try:
                from production.probability_calibration import ProbabilityCalibrator
                cal = ProbabilityCalibrator()
                cal.load(str(self.calibrator_path))
                self._calibrator = cal
            except Exception as e:
                logger.debug("Calibrator load failed: %s", e)
        return self._calibrator

    def _load_xgb_model(self):
        """P12: Load XGBoost ensemble member from disk (returns None if absent)."""
        if self._xgb_model is not None:
            return self._xgb_model
        if not self.xgb_model_path.exists():
            return None
        try:
            import pickle
            import xgboost  # noqa: F401  -- verify it is installed
            with open(self.xgb_model_path, "rb") as fh:
                self._xgb_model = pickle.load(fh)
            logger.info("P12: XGBoost model loaded from %s.", self.xgb_model_path)
            return self._xgb_model
        except Exception as exc:
            logger.debug("P12: XGBoost model load skipped: %s", exc)
            return None

    def _save_model(self, model, feat_cols: list):
        """Persist model and feature list to disk."""
        try:
            model.save_model(str(self.model_path))
            with open(self.feature_path, "w") as fh:
                json.dump(feat_cols, fh)
            logger.info("Model saved -> %s", self.model_path)
        except Exception as e:
            logger.error("Model save failed: %s", e)

    def compute_shap_attributions(
        self,
        picks_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        top_n: int = 3,
    ) -> pd.DataFrame:
        """
        P15 -- Compute TreeSHAP values for each pick and attach top-N features.

        Adds columns to picks_df:
          SHAP_top1_feature, SHAP_top1_value
          SHAP_top2_feature, SHAP_top2_value
          SHAP_top3_feature, SHAP_top3_value

        Falls back silently if shap is not installed or the model is not loaded.

        Parameters
        ----------
        picks_df   : Output of generate() -- rows are individual picks.
        feature_df : Full feature DataFrame from generate() -- used to get
                     feature values for the picks (matched on SC_CODE + DATE).
        top_n      : How many top SHAP features to attach (default 3).

        Returns
        -------
        picks_df with SHAP columns added (original if shap unavailable).
        """
        try:
            import shap
        except ImportError:
            logger.debug("P15: shap not installed -- skipping SHAP attribution.")
            return picks_df

        model = self._ensure_model(None)
        if model is None or self._feature_cols is None:
            return picks_df

        feat_cols = self._feature_cols
        try:
            # Match picks to their feature rows
            feature_df = feature_df.copy()
            feature_df["DATE"] = pd.to_datetime(feature_df["DATE"])
            picks_copy = picks_df.copy()

            if "Prediction_Date" in picks_copy.columns:
                pred_date = pd.to_datetime(picks_copy["Prediction_Date"].iloc[0])
                feat_rows = feature_df[feature_df["DATE"] == pred_date]
            else:
                feat_rows = feature_df[feature_df["DATE"] == feature_df["DATE"].max()]

            feat_rows = feat_rows.set_index("SC_CODE")
            X_picks = (
                picks_copy["SC_CODE"]
                .map(lambda sc: feat_rows.loc[sc] if sc in feat_rows.index else None)
            )
            # Build matrix in feature_cols order
            X_mat = pd.DataFrame(
                [feat_rows.loc[sc][feat_cols].fillna(0).values
                 if sc in feat_rows.index else np.zeros(len(feat_cols))
                 for sc in picks_copy["SC_CODE"]],
                columns=feat_cols,
            )

            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_mat)   # shape (n_picks, n_features)

            # For each pick, record top-N features by |SHAP|
            for rank in range(1, top_n + 1):
                picks_copy[f"SHAP_top{rank}_feature"] = ""
                picks_copy[f"SHAP_top{rank}_value"]   = np.nan

            for i, (_, row) in enumerate(picks_copy.iterrows()):
                sv = shap_vals[i]
                top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
                for rank, idx in enumerate(top_idx, 1):
                    picks_copy.at[row.name, f"SHAP_top{rank}_feature"] = feat_cols[idx]
                    picks_copy.at[row.name, f"SHAP_top{rank}_value"]   = round(float(sv[idx]), 6)

            logger.info(
                "P15: SHAP attributions computed for %d picks (%d features).",
                len(picks_copy), len(feat_cols),
            )
            return picks_copy

        except Exception as exc:
            logger.debug("P15: SHAP computation failed: %s", exc)
            return picks_df

    def _save_reference_distribution(self, X: pd.DataFrame, preds: np.ndarray):
        """Save reference feature distribution for drift monitoring."""
        try:
            ref_path = self.models_dir / "reference_distribution.pkl"
            with open(ref_path, "wb") as fh:
                pickle.dump({"X": X, "preds": preds}, fh)
        except Exception:
            pass

    @staticmethod
    def _lgb_params() -> dict:
        return {
            "objective":        "binary",
            "metric":           "auc",
            "boosting_type":    "gbdt",
            "num_leaves":       31,
            "learning_rate":    0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq":     5,
            "max_depth":        6,
            "min_child_samples": 20,
            "lambda_l1":        0.1,
            "lambda_l2":        0.1,
            "n_jobs":           -1,
            "verbose":          -1,
            "random_state":     42,
        }


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    sg = SignalGenerator()
    print("Model available:", sg.is_model_available())
    print("SignalGenerator initialised. Run via TradeOrchestrator.run_daily().")
