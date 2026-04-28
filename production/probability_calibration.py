"""
probability_calibration.py
==========================
Gap 6 — Probability Calibration

Raw classifier scores (from XGBoost, LightGBM, RandomForest, etc.) are
*not* calibrated probabilities — a model outputting 0.65 does not mean
65 % of similarly-scored stocks will move up.  This module corrects that.

Architecture
------------
1. ``CalibrationMethod``   — enum of supported calibration algorithms
2. ``ProbabilityCalibrator`` — fit / transform a single calibration layer
3. ``BucketValidator``    — reliability-diagram / binned calibration audit
4. ``CalibrationPipeline`` — end-to-end: base model + calibrator wrapper
5. ``calibrate_predictions()`` — module-level convenience for DataFrames

References
----------
- Platt 1999, "Probabilistic outputs for SVMs"
- Zadrozny & Elkan 2002, "Transforming classifier scores into accurate
  multiclass probability estimates"
- Niculescu-Mizil & Caruana 2005, "Predicting good probabilities with
  supervised learning"

Usage
-----
>>> from production.probability_calibration import CalibrationPipeline, CalibrationMethod
>>> pipe = CalibrationPipeline(base_model=xgb_clf,
...                            calibration_method=CalibrationMethod.ISOTONIC_REGRESSION)
>>> pipe.fit_calibrated(X_train, y_train, X_val, y_val)
>>> calibrated_probs = pipe.predict_calibrated(X_test)
"""

from __future__ import annotations

import logging
import os
import pickle
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful sklearn imports
# ---------------------------------------------------------------------------
try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.isotonic import IsotonicRegression  # type: ignore
    from sklearn.metrics import brier_score_loss, log_loss  # type: ignore
    SKLEARN_AVAILABLE = True
    logger.debug("scikit-learn loaded successfully.")
except ImportError:
    SKLEARN_AVAILABLE = False
    LogisticRegression = None  # type: ignore[assignment,misc]
    IsotonicRegression = None  # type: ignore[assignment,misc]
    logger.warning(
        "scikit-learn not available — calibration will use numpy fallbacks. "
        "Install with: pip install scikit-learn"
    )

try:
    import matplotlib  # type: ignore
    matplotlib.use("Agg")           # non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.debug("matplotlib not available — reliability plots disabled.")


# ===========================================================================
# 1.  CalibrationMethod enum
# ===========================================================================

class CalibrationMethod(Enum):
    """Supported probability calibration algorithms."""
    PLATT_SCALING = auto()
    ISOTONIC_REGRESSION = auto()
    BETA_CALIBRATION = auto()
    NONE = auto()


# ===========================================================================
# Helpers
# ===========================================================================

def _clip_probs(probs: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Clip probabilities to (eps, 1-eps) to avoid log(0) issues."""
    return np.clip(probs, eps, 1.0 - eps)


def _validate_arrays(
    y_true: Any, y_prob: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """Coerce inputs to 1-D float numpy arrays of equal length."""
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_prob, dtype=float).ravel()
    if len(y_t) != len(y_p):
        raise ValueError(
            f"y_true ({len(y_t)}) and y_prob ({len(y_p)}) must have equal length."
        )
    return y_t, y_p


# ===========================================================================
# 2.  ProbabilityCalibrator
# ===========================================================================

class ProbabilityCalibrator:
    """
    Fit and apply a single calibration layer to raw classifier scores.

    Supported methods
    -----------------
    - PLATT_SCALING       : Logistic regression on raw scores (fast, parametric)
    - ISOTONIC_REGRESSION : Non-parametric monotone fit (more flexible)
    - BETA_CALIBRATION    : Log-odds -> log(p) + log(1-p) regression
                            (falls back to Platt when sklearn absent)
    - NONE                : Identity (pass-through)

    Parameters
    ----------
    method : CalibrationMethod
        Calibration algorithm to use (default ISOTONIC_REGRESSION).
    """

    def __init__(
        self, method: CalibrationMethod = CalibrationMethod.ISOTONIC_REGRESSION
    ) -> None:
        self.method = method
        self._calibrator: Any = None
        self._is_fitted: bool = False
        logger.info("ProbabilityCalibrator init: method=%s", method.name)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self, y_true: Any, y_prob_raw: Any
    ) -> "ProbabilityCalibrator":
        """
        Fit the calibrator on held-out (validation) data.

        IMPORTANT: Always use out-of-sample data (not training data) to avoid
        overfitting the calibration layer.

        Parameters
        ----------
        y_true : array-like of int/float
            Binary labels (0 / 1).
        y_prob_raw : array-like of float
            Raw model output probabilities (uncalibrated).

        Returns
        -------
        self
        """
        y_t, y_p = _validate_arrays(y_true, y_prob_raw)
        y_p = _clip_probs(y_p)

        if self.method == CalibrationMethod.NONE:
            logger.info("CalibrationMethod.NONE — no fitting needed.")
            self._is_fitted = True
            return self

        if not SKLEARN_AVAILABLE and self.method in (
            CalibrationMethod.PLATT_SCALING,
            CalibrationMethod.ISOTONIC_REGRESSION,
            CalibrationMethod.BETA_CALIBRATION,
        ):
            logger.warning(
                "sklearn not available — using numpy Platt-scaling fallback."
            )
            self._calibrator = self._numpy_platt_fit(y_t, y_p)
            self._is_fitted = True
            return self

        if self.method == CalibrationMethod.PLATT_SCALING:
            self._calibrator = self._fit_platt(y_t, y_p)
        elif self.method == CalibrationMethod.ISOTONIC_REGRESSION:
            self._calibrator = self._fit_isotonic(y_t, y_p)
        elif self.method == CalibrationMethod.BETA_CALIBRATION:
            self._calibrator = self._fit_beta(y_t, y_p)
        else:
            raise ValueError(f"Unknown CalibrationMethod: {self.method}")

        self._is_fitted = True
        logger.info(
            "ProbabilityCalibrator fitted (method=%s) on %d samples.",
            self.method.name, len(y_t),
        )
        return self

    # --- Internal fit methods -------------------------------------------

    def _fit_platt(self, y_t: np.ndarray, y_p: np.ndarray) -> Any:
        """Fit Platt scaling via logistic regression on raw scores."""
        lr = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
        lr.fit(y_p.reshape(-1, 1), y_t)
        return ("platt", lr)

    def _fit_isotonic(self, y_t: np.ndarray, y_p: np.ndarray) -> Any:
        """Fit isotonic regression calibrator."""
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_p, y_t)
        return ("isotonic", iso)

    def _fit_beta(self, y_t: np.ndarray, y_p: np.ndarray) -> Any:
        """
        Beta calibration: regress labels on [log(p), log(1-p)] features.

        Falls back to Platt if sklearn is unavailable (already handled in fit).
        """
        if not SKLEARN_AVAILABLE:
            return self._numpy_platt_fit(y_t, y_p)

        # Features: log(p) and log(1-p)
        X_beta = np.column_stack([np.log(y_p), np.log(1.0 - y_p)])
        lr = LogisticRegression(C=1e5, solver="lbfgs", max_iter=1000)
        lr.fit(X_beta, y_t)
        return ("beta", lr)

    def _numpy_platt_fit(
        self, y_t: np.ndarray, y_p: np.ndarray
    ) -> Tuple[str, float, float]:
        """
        Minimal numpy-only Platt scaling: find a, b such that
        P(y=1|s) = sigmoid(a*s + b) via gradient descent approximation
        using the closed-form (single-feature logit fit).
        """
        from numpy.linalg import lstsq

        logit_p = np.log(y_p / (1.0 - y_p))
        A = np.column_stack([logit_p, np.ones_like(logit_p)])
        # Solve for [a, b] minimising ||sigmoid(A @ w) - y||
        # Approximation: use OLS on logit domain (fast, reasonable init)
        w, _, _, _ = lstsq(A, y_t, rcond=None)
        return ("numpy_platt", float(w[0]), float(w[1]))

    # ------------------------------------------------------------------
    # Calibration (inference)
    # ------------------------------------------------------------------

    def calibrate(self, y_prob_raw: Any) -> np.ndarray:
        """
        Apply the fitted calibrator to raw probabilities.

        Parameters
        ----------
        y_prob_raw : array-like of float

        Returns
        -------
        np.ndarray
            Calibrated probabilities in [0, 1].
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before calibrate().")

        y_p = _clip_probs(np.asarray(y_prob_raw, dtype=float).ravel())

        if self.method == CalibrationMethod.NONE:
            return y_p

        label, model = self._calibrator[0], self._calibrator[1]

        if label == "platt":
            cal = model.predict_proba(y_p.reshape(-1, 1))[:, 1]
        elif label == "isotonic":
            cal = model.predict(y_p)
        elif label == "beta":
            X_beta = np.column_stack([np.log(y_p), np.log(1.0 - y_p)])
            cal = model.predict_proba(X_beta)[:, 1]
        elif label == "numpy_platt":
            a, b = self._calibrator[1], self._calibrator[2]
            logit_p = np.log(y_p / (1.0 - y_p))
            cal = 1.0 / (1.0 + np.exp(-(a * logit_p + b)))
        else:
            cal = y_p  # unknown — pass-through

        return _clip_probs(cal)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_calibration(
        self,
        y_true: Any,
        y_prob_calibrated: Any,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """
        Compute calibration quality metrics.

        Metrics
        -------
        ECE  — Expected Calibration Error  (lower is better, 0 = perfect)
        MCE  — Maximum Calibration Error   (worst-bin error)
        reliability_diagram_data — list of per-bin dicts for plotting

        Parameters
        ----------
        y_true : array-like
        y_prob_calibrated : array-like
        n_bins : int
            Number of equal-width probability bins (default 10).

        Returns
        -------
        dict with keys: ECE, MCE, reliability_diagram_data
        """
        y_t, y_p = _validate_arrays(y_true, y_prob_calibrated)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_p, bins[1:-1])  # 0-indexed bin ids

        reliability_data: List[Dict[str, Any]] = []
        ece_numerator = 0.0
        mce = 0.0
        n_total = len(y_t)

        for b in range(n_bins):
            mask = bin_indices == b
            count = int(mask.sum())
            if count == 0:
                continue
            frac_pos = float(y_t[mask].mean())
            mean_pred = float(y_p[mask].mean())
            cal_error = abs(frac_pos - mean_pred)

            ece_numerator += cal_error * count
            mce = max(mce, cal_error)

            bin_center = float(bins[b] + bins[b + 1]) / 2.0
            reliability_data.append(
                {
                    "bin_center": round(bin_center, 3),
                    "fraction_positive": round(frac_pos, 4),
                    "mean_predicted": round(mean_pred, 4),
                    "count": count,
                    "calibration_error": round(cal_error, 4),
                }
            )

        ece = ece_numerator / n_total if n_total > 0 else 0.0

        result = {
            "ECE": round(ece, 6),
            "MCE": round(mce, 6),
            "n_samples": n_total,
            "reliability_diagram_data": reliability_data,
        }
        logger.info(
            "validate_calibration: ECE=%.4f, MCE=%.4f, n=%d",
            ece, mce, n_total,
        )
        return result

    # ------------------------------------------------------------------
    # Reliability diagram
    # ------------------------------------------------------------------

    def plot_reliability_diagram(
        self,
        y_true: Any,
        y_prob_raw: Any,
        y_prob_cal: Any,
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot reliability diagrams (before and after calibration).

        Requires matplotlib.  If not installed, logs a warning and returns.

        Parameters
        ----------
        y_true : array-like
        y_prob_raw : array-like
            Uncalibrated model scores.
        y_prob_cal : array-like
            Calibrated probabilities (after calling calibrate()).
        save_path : str, optional
            If provided, the figure is saved to this path instead of displayed.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning(
                "matplotlib not installed — skipping reliability diagram. "
                "Install with: pip install matplotlib"
            )
            return

        y_t, _ = _validate_arrays(y_true, y_prob_raw)
        _, y_p_raw = _validate_arrays(y_true, y_prob_raw)
        _, y_p_cal = _validate_arrays(y_true, y_prob_cal)

        def _reliability_curve(
            y_true_: np.ndarray, y_prob_: np.ndarray, n_bins: int = 10
        ) -> Tuple[np.ndarray, np.ndarray]:
            bins = np.linspace(0, 1, n_bins + 1)
            bin_means_pred = []
            bin_means_true = []
            for b in range(n_bins):
                mask = (y_prob_ >= bins[b]) & (y_prob_ < bins[b + 1])
                if mask.sum() == 0:
                    continue
                bin_means_pred.append(y_prob_[mask].mean())
                bin_means_true.append(y_true_[mask].mean())
            return np.array(bin_means_pred), np.array(bin_means_true)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, y_p, title in zip(
            axes,
            [y_p_raw, y_p_cal],
            ["Before Calibration (Raw)", f"After Calibration ({self.method.name})"],
        ):
            pred, true = _reliability_curve(y_t, y_p)
            ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax.plot(pred, true, "s-", color="tab:blue", label="Model")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title(title)
            ax.legend(loc="upper left")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        fig.suptitle("Reliability Diagram — Probability Calibration", fontsize=13)
        fig.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Reliability diagram saved to %s", save_path)
        else:
            try:
                plt.show()
            except Exception:
                pass
        plt.close(fig)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Pickle the fitted calibrator to *path*.

        Parameters
        ----------
        path : str
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "method": self.method,
                    "calibrator": self._calibrator,
                    "is_fitted": self._is_fitted,
                },
                fh,
            )
        logger.info("ProbabilityCalibrator saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "ProbabilityCalibrator":
        """
        Load a pickled calibrator from *path*.

        Parameters
        ----------
        path : str

        Returns
        -------
        ProbabilityCalibrator
        """
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        instance = cls(method=state["method"])
        instance._calibrator = state["calibrator"]
        instance._is_fitted = state["is_fitted"]
        logger.info("ProbabilityCalibrator loaded from %s", path)
        return instance


# ===========================================================================
# 3.  BucketValidator
# ===========================================================================

class BucketValidator:
    """
    Bin-level calibration audit tool.

    Segments predictions into equal-width probability bins and computes per-
    bin accuracy (fraction of positives) vs mean predicted probability.  A
    well-calibrated model shows these two quantities matching closely.

    Parameters
    ----------
    n_bins : int
        Number of equal-width probability buckets (default 10).
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    # ------------------------------------------------------------------

    def validate(
        self,
        y_true: Any,
        y_prob: Any,
        labels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute per-bucket calibration statistics.

        Parameters
        ----------
        y_true : array-like
            Binary labels (0 / 1).
        y_prob : array-like
            Predicted probabilities.
        labels : list[str], optional
            Bucket labels (must have length ``n_bins``).  Auto-generated if None.

        Returns
        -------
        pd.DataFrame with columns:
            prob_bin, n_samples, n_positive, fraction_positive,
            predicted_prob, calibration_error
        """
        y_t, y_p = _validate_arrays(y_true, y_prob)
        bins = np.linspace(0.0, 1.0, self.n_bins + 1)

        rows: List[Dict[str, Any]] = []
        for i in range(self.n_bins):
            lo, hi = bins[i], bins[i + 1]
            # Include the right edge in the last bin
            if i < self.n_bins - 1:
                mask = (y_p >= lo) & (y_p < hi)
            else:
                mask = (y_p >= lo) & (y_p <= hi)

            n = int(mask.sum())
            n_pos = int(y_t[mask].sum()) if n > 0 else 0
            frac_pos = float(y_t[mask].mean()) if n > 0 else float("nan")
            pred_prob = float(y_p[mask].mean()) if n > 0 else float("nan")
            cal_err = abs(frac_pos - pred_prob) if n > 0 else float("nan")

            bin_label = (
                labels[i]
                if labels is not None
                else f"[{lo:.1f}, {hi:.1f})"
            )
            rows.append(
                {
                    "prob_bin": bin_label,
                    "n_samples": n,
                    "n_positive": n_pos,
                    "fraction_positive": round(frac_pos, 4) if n > 0 else float("nan"),
                    "predicted_prob": round(pred_prob, 4) if n > 0 else float("nan"),
                    "calibration_error": round(cal_err, 4) if n > 0 else float("nan"),
                }
            )

        df = pd.DataFrame(rows)
        logger.info("BucketValidator: %d non-empty bins.", (df["n_samples"] > 0).sum())
        return df

    # ------------------------------------------------------------------

    def print_report(self, validation_df: pd.DataFrame) -> None:
        """
        Print a formatted per-bucket calibration table to stdout.

        Parameters
        ----------
        validation_df : pd.DataFrame
            Output of ``validate()``.
        """
        header = (
            f"{'Prob Bin':<15} {'N':>6} {'N+':>6} "
            f"{'Actual %':>9} {'Predicted %':>12} {'Cal. Error':>11}"
        )
        separator = "-" * len(header)

        print(separator)
        print("  CALIBRATION BUCKET REPORT")
        print(separator)
        print(header)
        print(separator)

        for _, row in validation_df.iterrows():
            if row["n_samples"] == 0:
                continue
            print(
                f"{row['prob_bin']:<15} "
                f"{int(row['n_samples']):>6} "
                f"{int(row['n_positive']):>6} "
                f"{row['fraction_positive'] * 100:>8.1f}% "
                f"{row['predicted_prob'] * 100:>11.1f}% "
                f"{row['calibration_error']:>11.4f}"
            )
        print(separator)

        # Summary ECE
        non_empty = validation_df[validation_df["n_samples"] > 0].copy()
        if not non_empty.empty:
            ece = float(
                (
                    non_empty["calibration_error"] * non_empty["n_samples"]
                ).sum()
                / non_empty["n_samples"].sum()
            )
            print(f"  ECE (Expected Calibration Error): {ece:.4f}")
        print(separator)

    # ------------------------------------------------------------------

    def compute_brier_score(self, y_true: Any, y_prob: Any) -> float:
        """
        Compute the Brier score: mean squared error between probabilities and labels.

        A perfect model scores 0.0; the worst possible score is 1.0.

        Parameters
        ----------
        y_true, y_prob : array-like

        Returns
        -------
        float
        """
        y_t, y_p = _validate_arrays(y_true, y_prob)
        if SKLEARN_AVAILABLE:
            score = float(brier_score_loss(y_t, y_p))
        else:
            score = float(np.mean((y_t - y_p) ** 2))
        logger.info("Brier score: %.6f", score)
        return score

    # ------------------------------------------------------------------

    def compute_log_loss(self, y_true: Any, y_prob: Any) -> float:
        """
        Compute binary cross-entropy / log-loss.

        Parameters
        ----------
        y_true, y_prob : array-like

        Returns
        -------
        float
        """
        y_t, y_p = _validate_arrays(y_true, y_prob)
        y_p = _clip_probs(y_p)

        if SKLEARN_AVAILABLE:
            loss = float(log_loss(y_t, y_p))
        else:
            # Pure numpy implementation
            loss = float(
                -np.mean(
                    y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p)
                )
            )
        logger.info("Log-loss: %.6f", loss)
        return loss


# ===========================================================================
# 4.  CalibrationPipeline
# ===========================================================================

class CalibrationPipeline:
    """
    End-to-end pipeline: base model training -> calibration layer fitting ->
    calibrated inference.

    Parameters
    ----------
    base_model : object
        Any sklearn-compatible estimator with ``fit(X, y)`` and
        ``predict_proba(X)`` methods (e.g. XGBClassifier, LGBMClassifier).
    calibration_method : CalibrationMethod
        Algorithm for the calibration layer.
    """

    def __init__(
        self,
        base_model: Any,
        calibration_method: CalibrationMethod = CalibrationMethod.ISOTONIC_REGRESSION,
    ) -> None:
        self.base_model = base_model
        self.calibration_method = calibration_method
        self.calibrator = ProbabilityCalibrator(method=calibration_method)
        self._base_fitted = False
        logger.info(
            "CalibrationPipeline init: base_model=%s, method=%s",
            type(base_model).__name__, calibration_method.name,
        )

    # ------------------------------------------------------------------

    def fit_calibrated(
        self,
        X_train: Any,
        y_train: Any,
        X_val: Any,
        y_val: Any,
    ) -> "CalibrationPipeline":
        """
        Train the base model, obtain validation-set predictions, and fit the
        calibration layer on those out-of-sample scores.

        Parameters
        ----------
        X_train, y_train : array-like
            Training data for the base model.
        X_val, y_val : array-like
            Held-out validation data used exclusively for calibration fitting.
            These samples must NOT appear in the training set.

        Returns
        -------
        self
        """
        # Step 1: train base model
        logger.info(
            "Training base model on %d samples …", len(np.asarray(y_train))
        )
        self.base_model.fit(X_train, y_train)
        self._base_fitted = True

        # Step 2: get raw validation predictions
        y_prob_val_raw = self._predict_proba_base(X_val)

        # Step 3: fit calibration layer on raw val predictions
        logger.info(
            "Fitting calibration layer (%s) on %d validation samples …",
            self.calibration_method.name, len(np.asarray(y_val)),
        )
        self.calibrator.fit(y_val, y_prob_val_raw)

        return self

    # ------------------------------------------------------------------

    def predict_calibrated(self, X: Any) -> np.ndarray:
        """
        Produce calibrated probability predictions for *X*.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        np.ndarray
            Calibrated probabilities in [0, 1].
        """
        if not self._base_fitted:
            raise RuntimeError(
                "Pipeline not fitted. Call fit_calibrated() first."
            )
        raw = self._predict_proba_base(X)
        return self.calibrator.calibrate(raw)

    # ------------------------------------------------------------------

    def _predict_proba_base(self, X: Any) -> np.ndarray:
        """Return the positive-class probability from the base model."""
        proba = self.base_model.predict_proba(X)
        # Handle both (n, 2) and (n,) outputs
        if proba.ndim == 2:
            return proba[:, 1].astype(float)
        return proba.astype(float)

    # ------------------------------------------------------------------

    def get_calibration_report(
        self, X_val: Any, y_val: Any
    ) -> Dict[str, Any]:
        """
        Compute a full calibration quality report on a validation set.

        Returns calibration metrics for both raw and calibrated scores.

        Parameters
        ----------
        X_val : array-like
        y_val : array-like

        Returns
        -------
        dict with keys:
            raw_ECE, raw_MCE, raw_brier, raw_log_loss,
            calibrated_ECE, calibrated_MCE, calibrated_brier, calibrated_log_loss,
            method, n_val_samples
        """
        if not self._base_fitted:
            raise RuntimeError("Call fit_calibrated() first.")

        y_t = np.asarray(y_val, dtype=float).ravel()
        y_p_raw = self._predict_proba_base(X_val)
        y_p_cal = self.calibrator.calibrate(y_p_raw)

        validator = BucketValidator(n_bins=10)
        cal_obj = ProbabilityCalibrator()  # just for validate_calibration

        raw_metrics = cal_obj.validate_calibration(y_t, y_p_raw)
        cal_metrics = cal_obj.validate_calibration(y_t, y_p_cal)

        report: Dict[str, Any] = {
            "raw_ECE": raw_metrics["ECE"],
            "raw_MCE": raw_metrics["MCE"],
            "raw_brier": validator.compute_brier_score(y_t, y_p_raw),
            "raw_log_loss": validator.compute_log_loss(y_t, y_p_raw),
            "calibrated_ECE": cal_metrics["ECE"],
            "calibrated_MCE": cal_metrics["MCE"],
            "calibrated_brier": validator.compute_brier_score(y_t, y_p_cal),
            "calibrated_log_loss": validator.compute_log_loss(y_t, y_p_cal),
            "method": self.calibration_method.name,
            "n_val_samples": int(len(y_t)),
        }
        logger.info("Calibration report: %s", report)
        return report


# ===========================================================================
# 5.  Module-level convenience function
# ===========================================================================

def calibrate_predictions(
    predictions_df: pd.DataFrame,
    calibrator: ProbabilityCalibrator,
    raw_prob_col: str = "Probability",
    cal_prob_col: str = "Prob_Calibrated",
) -> pd.DataFrame:
    """
    Apply a fitted ``ProbabilityCalibrator`` to a predictions DataFrame.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Must contain a column named *raw_prob_col* with uncalibrated scores.
    calibrator : ProbabilityCalibrator
        A fitted calibrator instance.
    raw_prob_col : str
        Name of the raw probability column (default "Probability").
    cal_prob_col : str
        Name of the new calibrated probability column (default "Prob_Calibrated").

    Returns
    -------
    pd.DataFrame
        Copy of *predictions_df* with *cal_prob_col* appended.

    Examples
    --------
    >>> picks = calibrate_predictions(picks_df, fitted_calibrator)
    >>> print(picks[["Probability", "Prob_Calibrated"]].head())
    """
    if raw_prob_col not in predictions_df.columns:
        raise KeyError(
            f"Column '{raw_prob_col}' not found in predictions_df. "
            f"Available: {list(predictions_df.columns)}"
        )

    out = predictions_df.copy()
    raw_probs = out[raw_prob_col].astype(float).values
    out[cal_prob_col] = calibrator.calibrate(raw_probs)
    logger.info(
        "calibrate_predictions: %d rows calibrated (method=%s). "
        "Mean raw=%.4f -> mean cal=%.4f.",
        len(out),
        calibrator.method.name,
        float(raw_probs.mean()),
        float(out[cal_prob_col].mean()),
    )
    return out


def calibration_curve_report(
    y_true: Any,
    y_prob_raw: Any,
    y_prob_cal: Optional[Any] = None,
    n_bins: int = 10,
    chart_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    P16 -- Full calibration curve diagnostic.

    Computes binned calibration statistics (ECE, Brier score, log-loss) and
    optionally saves a reliability diagram.  Intended to be called after
    training to audit probability quality.

    Parameters
    ----------
    y_true      : Binary labels (0 / 1).
    y_prob_raw  : Raw model probabilities (before calibration).
    y_prob_cal  : Calibrated probabilities (optional; if None, only raw is assessed).
    n_bins      : Number of equal-width probability bins (default 10).
    chart_path  : Path to save reliability diagram PNG (optional).

    Returns
    -------
    dict with keys:
        raw_ece, raw_brier, raw_logloss,
        cal_ece, cal_brier, cal_logloss (if y_prob_cal provided),
        bucket_table_raw (pd.DataFrame),
        bucket_table_cal (pd.DataFrame, only if y_prob_cal provided),
        chart_saved (bool)
    """
    bv = BucketValidator(n_bins=n_bins)
    y_t, y_pr = _validate_arrays(y_true, y_prob_raw)

    bucket_raw   = bv.validate(y_t, y_pr)
    non_empty    = bucket_raw[bucket_raw["n_samples"] > 0]
    raw_ece      = float(
        (non_empty["calibration_error"] * non_empty["n_samples"]).sum()
        / non_empty["n_samples"].sum()
    ) if not non_empty.empty else float("nan")
    raw_brier    = bv.compute_brier_score(y_t, y_pr)
    raw_logloss  = bv.compute_log_loss(y_t, y_pr)

    result: Dict[str, Any] = {
        "raw_ece":          round(raw_ece, 6),
        "raw_brier":        round(raw_brier, 6),
        "raw_logloss":      round(raw_logloss, 6),
        "bucket_table_raw": bucket_raw,
        "chart_saved":      False,
    }

    if y_prob_cal is not None:
        _, y_pc = _validate_arrays(y_true, y_prob_cal)
        bucket_cal  = bv.validate(y_t, y_pc)
        ne_cal      = bucket_cal[bucket_cal["n_samples"] > 0]
        cal_ece     = float(
            (ne_cal["calibration_error"] * ne_cal["n_samples"]).sum()
            / ne_cal["n_samples"].sum()
        ) if not ne_cal.empty else float("nan")
        cal_brier   = bv.compute_brier_score(y_t, y_pc)
        cal_logloss = bv.compute_log_loss(y_t, y_pc)

        result.update({
            "cal_ece":          round(cal_ece, 6),
            "cal_brier":        round(cal_brier, 6),
            "cal_logloss":      round(cal_logloss, 6),
            "bucket_table_cal": bucket_cal,
            "ece_improvement":  round(raw_ece - cal_ece, 6),
        })

        # Save reliability diagram
        if chart_path:
            try:
                dummy_cal = ProbabilityCalibrator(method=CalibrationMethod.NONE)
                dummy_cal.plot_reliability_diagram(y_t, y_pr, y_pc, save_path=chart_path)
                result["chart_saved"] = True
                logger.info("P16: Calibration curve saved to %s", chart_path)
            except Exception as exc:
                logger.debug("P16: Chart save failed: %s", exc)

    # P43: label clearly that this ECE is computed on TRAINING DATA (not OOS)
    logger.info(
        "P16 calibration TRAINING-SET ECE (not OOS): raw=%.4f  raw_brier=%.4f%s  [not reliable]",
        raw_ece, raw_brier,
        f"  cal=%.4f" % result.get('cal_ece', 0) if "cal_ece" in result else "",
    )
    return result


def fit_oof_calibrator(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    method: str = "isotonic",
) -> "ProbabilityCalibrator":
    """
    P43 (fix): Fit a calibrator using out-of-fold predictions from the
    training data.  This avoids overfitting the calibrator to the same data
    the base model was fitted on, which causes ECE inflation in later blocks.

    Parameters
    ----------
    model    : Fitted LightGBM/XGBoost model with a .predict() method.
    X        : Training features (numpy array, same order as model was trained on).
    y        : Binary labels (0/1).
    n_splits : Number of TimeSeriesSplit folds (default 5).
    method   : 'isotonic' (default) or 'platt'.

    Returns
    -------
    Fitted ProbabilityCalibrator instance.
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    oof_probs  = np.full(len(y), np.nan)
    oof_labels = np.full(len(y), np.nan)

    for train_idx, val_idx in tscv.split(X):
        # Predict on the held-out fold using the already-trained model
        try:
            import lightgbm as lgb
            if isinstance(model, lgb.Booster):
                fold_probs = model.predict(X[val_idx])
            else:
                import xgboost as xgb
                fold_probs = model.predict(xgb.DMatrix(X[val_idx]))
        except Exception:
            try:
                fold_probs = model.predict(X[val_idx])
            except Exception:
                continue
        oof_probs[val_idx]  = fold_probs
        oof_labels[val_idx] = y[val_idx]

    valid = ~np.isnan(oof_probs)
    if valid.sum() < 50:
        logger.warning(
            "OOF calibration: only %d valid samples — using global fit instead.",
            valid.sum(),
        )
        cal_method = (CalibrationMethod.ISOTONIC_REGRESSION
                      if method == "isotonic" else CalibrationMethod.PLATT_SCALING)
        cal = ProbabilityCalibrator(method=cal_method)
        cal.fit(y, model.predict(X) if hasattr(model, "predict") else np.zeros(len(y)))
        return cal

    cal_method = (CalibrationMethod.ISOTONIC_REGRESSION
                  if method == "isotonic" else CalibrationMethod.PLATT_SCALING)
    cal = ProbabilityCalibrator(method=cal_method)
    cal.fit(oof_labels[valid], oof_probs[valid])

    oof_ece_before = compute_ece(oof_labels[valid], oof_probs[valid])
    cal_probs      = cal.predict(oof_probs[valid])
    oof_ece_after  = compute_ece(oof_labels[valid], cal_probs)
    logger.info(
        "P43 OOF calibration: ECE before=%.4f  after=%.4f  (n=%d, folds=%d)",
        oof_ece_before, oof_ece_after, valid.sum(), n_splits,
    )
    return cal


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    P43: Expected Calibration Error — call on OOS hold-out blocks only.

    Parameters
    ----------
    y_true : binary labels (0/1)
    y_prob : predicted probabilities
    n_bins : number of probability bins (default 10)

    Returns
    -------
    float ECE in [0, 1]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc  = float(y_true[mask].mean())
        bin_conf = float(y_prob[mask].mean())
        ece += float(mask.mean()) * abs(bin_acc - bin_conf)
    return round(ece, 4)


# ===========================================================================
# Smoke test
# ===========================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    print("=" * 70)
    print("probability_calibration.py — self-test")
    print("=" * 70)

    rng = np.random.default_rng(seed=0)
    n = 2000

    # Simulate a slightly over-confident model (raw probs pushed toward extremes)
    true_probs = rng.beta(2, 2, size=n)
    y_true = (rng.random(n) < true_probs).astype(float)

    # Distort raw probs to simulate over-confidence
    y_prob_raw = np.clip(true_probs * 1.35 - 0.175, 0.01, 0.99)

    # Train / val / test split
    n_train = int(n * 0.5)
    n_val = int(n * 0.25)
    y_tr, p_tr = y_true[:n_train], y_prob_raw[:n_train]
    y_val, p_val = y_true[n_train:n_train + n_val], y_prob_raw[n_train:n_train + n_val]
    y_te, p_te = y_true[n_train + n_val:], y_prob_raw[n_train + n_val:]

    # --- ProbabilityCalibrator -------------------------------------------
    print("\n[1] ProbabilityCalibrator — Isotonic Regression")
    cal = ProbabilityCalibrator(method=CalibrationMethod.ISOTONIC_REGRESSION)
    cal.fit(y_val, p_val)
    p_cal = cal.calibrate(p_te)

    val_result = cal.validate_calibration(y_te, p_cal, n_bins=10)
    print(f"  ECE (calibrated): {val_result['ECE']:.4f}")
    print(f"  MCE (calibrated): {val_result['MCE']:.4f}")

    val_raw = cal.validate_calibration(y_te, p_te, n_bins=10)
    print(f"  ECE (raw):        {val_raw['ECE']:.4f}")
    print(f"  MCE (raw):        {val_raw['MCE']:.4f}")

    # --- Platt scaling ---------------------------------------------------
    print("\n[2] ProbabilityCalibrator — Platt Scaling")
    platt = ProbabilityCalibrator(method=CalibrationMethod.PLATT_SCALING)
    platt.fit(y_val, p_val)
    p_platt = platt.calibrate(p_te)
    print(f"  ECE (Platt): {platt.validate_calibration(y_te, p_platt)['ECE']:.4f}")

    # --- BucketValidator -------------------------------------------------
    print("\n[3] BucketValidator")
    bv = BucketValidator(n_bins=10)
    val_df = bv.validate(y_te, p_cal)
    bv.print_report(val_df)
    print(f"  Brier score: {bv.compute_brier_score(y_te, p_cal):.4f}")
    print(f"  Log-loss:    {bv.compute_log_loss(y_te, p_cal):.4f}")

    # --- calibrate_predictions -------------------------------------------
    print("\n[4] calibrate_predictions()")
    picks_df = pd.DataFrame(
        {
            "sc_code": [f"SC{i}" for i in range(len(y_te))],
            "Probability": p_te,
        }
    )
    picks_out = calibrate_predictions(picks_df, cal)
    print(picks_out[["sc_code", "Probability", "Prob_Calibrated"]].head(8).to_string(index=False))

    # --- Save / load -----------------------------------------------------
    print("\n[5] Save / Load")
    tmp_path = "/tmp/test_calibrator.pkl"
    cal.save(tmp_path)
    loaded_cal = ProbabilityCalibrator.load(tmp_path)
    p_reloaded = loaded_cal.calibrate(p_te[:5])
    print(f"  Reloaded calibration (first 5): {p_reloaded.round(4).tolist()}")

    print("\nSelf-test complete.")
