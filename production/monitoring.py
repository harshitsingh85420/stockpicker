"""
monitoring.py
=============
Production model monitoring for the Indian market stock picker.

Handles:
  - Gap 13: Model Drift Detection (PSI-based)
  - Rolling performance monitoring and accuracy degradation detection

Usage:
    from production.monitoring import ModelDriftMonitor, PerformanceMonitor
    from production.monitoring import run_daily_monitoring_checks
"""

import json
import logging
import datetime
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PSICalculator
# ---------------------------------------------------------------------------

class PSICalculator:
    """
    Population Stability Index (PSI) calculator.

    PSI measures the shift between a reference distribution and a current
    distribution.  The standard interpretation thresholds are:

        PSI < 0.10  : stable (no significant shift)
        0.10 – 0.20 : slight shift (monitor closely)
        PSI > 0.20  : major shift (consider retraining)

    Parameters
    ----------
    n_bins : int
        Number of equal-frequency bins to use when binning a continuous feature.
    min_count_per_bin : int
        Minimum expected count per bin; bins with fewer counts are merged or
        clipped to avoid log(0).
    psi_warning : float
        PSI threshold for 'WARNING' status.
    psi_critical : float
        PSI threshold for 'CRITICAL' / 'RETRAIN_NEEDED' status.
    """

    def __init__(
        self,
        n_bins: int = 10,
        min_count_per_bin: int = 5,
        psi_warning: float = 0.10,
        psi_critical: float = 0.20,
    ) -> None:
        self.n_bins = n_bins
        self.min_count_per_bin = min_count_per_bin
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical

    # ------------------------------------------------------------------
    # Low-level PSI computation
    # ------------------------------------------------------------------

    def compute_bins(self, reference_data: np.ndarray, n_bins: int) -> np.ndarray:
        """
        Compute equal-frequency bin edges from reference data.

        Returns an array of (n_bins + 1) edge values.
        """
        reference_data = np.asarray(reference_data, dtype=float)
        reference_data = reference_data[np.isfinite(reference_data)]
        if len(reference_data) == 0:
            raise ValueError("reference_data contains no finite values")

        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.unique(np.percentile(reference_data, percentiles))
        # Ensure coverage of all values
        edges[0] = -np.inf
        edges[-1] = np.inf
        return edges

    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute PSI between an expected (reference) and actual (current)
        distribution.

        PSI = sum( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )

        Parameters
        ----------
        expected : array-like
            Reference distribution values.
        actual : array-like
            Current distribution values.
        bins : array-like, optional
            Pre-computed bin edges.  If None, bins are derived from ``expected``.

        Returns
        -------
        float
            PSI value (>= 0).
        """
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)

        # Remove non-finite values
        expected = expected[np.isfinite(expected)]
        actual = actual[np.isfinite(actual)]

        if len(expected) == 0 or len(actual) == 0:
            logger.warning("Empty array passed to compute_psi; returning 0.0")
            return 0.0

        if bins is None:
            bins = self.compute_bins(expected, self.n_bins)

        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)

        # Convert to proportions; add small epsilon to avoid log(0)
        eps = 1e-6
        expected_pct = (expected_counts + eps) / (len(expected) + eps * len(bins))
        actual_pct = (actual_counts + eps) / (len(actual) + eps * len(bins))

        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi)

    def _psi_status(self, psi: float) -> str:
        if psi < self.psi_warning:
            return "STABLE"
        if psi < self.psi_critical:
            return "WARNING"
        return "CRITICAL"

    # ------------------------------------------------------------------
    # Feature-level PSI
    # ------------------------------------------------------------------

    def compute_feature_psi(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> pd.DataFrame:
        """
        Compute PSI for each feature column.

        Returns
        -------
        pd.DataFrame
            Columns: feature, psi, status
        """
        records = []
        for col in feature_cols:
            if col not in reference_df.columns or col not in current_df.columns:
                logger.warning("Feature '%s' missing from one of the DataFrames; skipping.", col)
                continue
            try:
                psi = self.compute_psi(
                    reference_df[col].values,
                    current_df[col].values,
                )
                records.append({"feature": col, "psi": psi, "status": self._psi_status(psi)})
            except Exception as exc:
                logger.warning("PSI computation failed for feature '%s': %s", col, exc)
                records.append({"feature": col, "psi": np.nan, "status": "ERROR"})

        result = pd.DataFrame(records, columns=["feature", "psi", "status"])
        return result.sort_values("psi", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Prediction-level PSI
    # ------------------------------------------------------------------

    def compute_prediction_psi(
        self,
        reference_preds: np.ndarray,
        current_preds: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute PSI on model prediction distributions.

        Returns
        -------
        dict
            psi, status, reference_mean, current_mean
        """
        reference_preds = np.asarray(reference_preds, dtype=float)
        current_preds = np.asarray(current_preds, dtype=float)

        psi = self.compute_psi(reference_preds, current_preds)
        return {
            "psi": psi,
            "status": self._psi_status(psi),
            "reference_mean": float(np.nanmean(reference_preds)),
            "current_mean": float(np.nanmean(current_preds)),
            "mean_shift": float(np.nanmean(current_preds) - np.nanmean(reference_preds)),
        }


# ---------------------------------------------------------------------------
# ModelDriftMonitor
# ---------------------------------------------------------------------------

class ModelDriftMonitor:
    """
    Detects model drift by comparing current feature and prediction
    distributions against a saved reference (training-time) distribution.

    Parameters
    ----------
    reference_data_path : str
        Path to a pickle file where the reference distributions are stored.
    alert_threshold_features : float
        PSI threshold above which a feature is considered drifted.
    alert_threshold_predictions : float
        PSI threshold above which model predictions are considered drifted.
    """

    _DRIFT_LOG_FILENAME = "drift_log.json"

    def __init__(
        self,
        reference_data_path: str = "stock_picker_data/models/reference_distribution.pkl",
        alert_threshold_features: float = 0.20,
        alert_threshold_predictions: float = 0.20,
    ) -> None:
        self.reference_data_path = Path(reference_data_path)
        self.alert_threshold_features = alert_threshold_features
        self.alert_threshold_predictions = alert_threshold_predictions
        self._psi_calc = PSICalculator(psi_critical=alert_threshold_features)
        self._reference: Optional[Dict[str, Any]] = None
        self._drift_log_path = self.reference_data_path.parent / self._DRIFT_LOG_FILENAME

        self._load_reference()

    # ------------------------------------------------------------------
    # Reference management
    # ------------------------------------------------------------------

    def set_reference_distribution(
        self,
        X_reference: pd.DataFrame,
        y_pred_reference: np.ndarray,
    ) -> None:
        """
        Save the training-time (reference) feature and prediction distributions.

        Call this once after training before deploying the model.
        """
        self.reference_data_path.parent.mkdir(parents=True, exist_ok=True)
        self._reference = {
            "feature_data": X_reference.to_dict(orient="list"),
            "feature_cols": list(X_reference.columns),
            "predictions": list(map(float, y_pred_reference)),
            "created_at": datetime.datetime.now().isoformat(),
            "n_samples": len(X_reference),
        }
        try:
            with open(self.reference_data_path, "wb") as fh:
                pickle.dump(self._reference, fh)
            logger.info(
                "Reference distribution saved to %s (%d samples, %d features)",
                self.reference_data_path,
                len(X_reference),
                len(X_reference.columns),
            )
        except OSError as exc:
            logger.error("Failed to save reference distribution: %s", exc)
            raise

    def _load_reference(self) -> None:
        if not self.reference_data_path.exists():
            logger.info(
                "No reference distribution found at %s. "
                "Call set_reference_distribution() after training.",
                self.reference_data_path,
            )
            return
        try:
            with open(self.reference_data_path, "rb") as fh:
                self._reference = pickle.load(fh)
            logger.info("Reference distribution loaded from %s", self.reference_data_path)
        except (OSError, pickle.UnpicklingError) as exc:
            logger.error("Could not load reference distribution: %s", exc)

    # ------------------------------------------------------------------
    # Drift computation
    # ------------------------------------------------------------------

    def compute_drift_report(
        self,
        X_current: pd.DataFrame,
        y_pred_current: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute a full drift report comparing current data against the reference.

        Returns
        -------
        dict with keys:
            feature_drift, prediction_drift, overall_status,
            drifted_features, checked_at
        """
        if self._reference is None:
            raise RuntimeError(
                "No reference distribution loaded. "
                "Call set_reference_distribution() first."
            )

        X_ref = pd.DataFrame(self._reference["feature_data"])
        ref_preds = np.array(self._reference["predictions"])

        # Use at most the top 20 features (by column order)
        feature_cols = [c for c in self._reference["feature_cols"] if c in X_current.columns]
        feature_cols = feature_cols[:20]

        # Feature PSI
        feature_psi_df = self._psi_calc.compute_feature_psi(X_ref, X_current, feature_cols)
        feature_drift: Dict[str, Dict[str, Any]] = {}
        for _, row in feature_psi_df.iterrows():
            feature_drift[row["feature"]] = {
                "psi": row["psi"],
                "status": row["status"],
            }

        # Prediction PSI
        prediction_drift = self._psi_calc.compute_prediction_psi(ref_preds, np.asarray(y_pred_current))

        # Drifted features
        drifted = [
            f for f, v in feature_drift.items()
            if isinstance(v["psi"], float) and v["psi"] > self.alert_threshold_features
        ]

        # Overall status
        pred_drifted = prediction_drift["psi"] > self.alert_threshold_predictions
        if pred_drifted or len(drifted) >= 3:
            overall_status = "RETRAIN_NEEDED"
        elif len(drifted) >= 1:
            overall_status = "WARNING"
        else:
            overall_status = "STABLE"

        report = {
            "feature_drift": feature_drift,
            "prediction_drift": prediction_drift,
            "overall_status": overall_status,
            "drifted_features": drifted,
            "n_drifted_features": len(drifted),
            "checked_at": datetime.datetime.now().isoformat(),
        }

        if overall_status in ("WARNING", "RETRAIN_NEEDED"):
            logger.warning(
                "Model drift detected: status=%s  drifted_features=%s  pred_psi=%.3f",
                overall_status,
                drifted,
                prediction_drift["psi"],
            )

        return report

    def should_retrain(
        self,
        X_current: pd.DataFrame,
        y_pred_current: np.ndarray,
    ) -> bool:
        """
        Convenience method: return True if the drift report recommends retraining.
        """
        try:
            report = self.compute_drift_report(X_current, y_pred_current)
            return report["overall_status"] == "RETRAIN_NEEDED"
        except Exception as exc:
            logger.error("should_retrain() failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Drift event logging
    # ------------------------------------------------------------------

    def log_drift_event(
        self,
        drift_report: Dict[str, Any],
        date: Optional[datetime.date] = None,
    ) -> None:
        """Append a drift report to the JSON log file."""
        date_str = (date or datetime.date.today()).isoformat()
        entry = {
            "date": date_str,
            "overall_status": drift_report.get("overall_status"),
            "n_drifted_features": drift_report.get("n_drifted_features", 0),
            "drifted_features": drift_report.get("drifted_features", []),
            "prediction_psi": drift_report.get("prediction_drift", {}).get("psi"),
            "checked_at": drift_report.get("checked_at"),
        }

        self._drift_log_path.parent.mkdir(parents=True, exist_ok=True)
        existing: List[Dict] = []
        if self._drift_log_path.exists():
            try:
                with open(self._drift_log_path) as fh:
                    existing = json.load(fh)
            except (OSError, json.JSONDecodeError):
                existing = []

        existing.append(entry)
        try:
            with open(self._drift_log_path, "w") as fh:
                json.dump(existing, fh, indent=2)
        except OSError as exc:
            logger.error("Failed to write drift log: %s", exc)

    def get_drift_history(self) -> pd.DataFrame:
        """Return historical drift events as a DataFrame."""
        if not self._drift_log_path.exists():
            return pd.DataFrame(
                columns=["date", "overall_status", "n_drifted_features", "prediction_psi"]
            )
        try:
            with open(self._drift_log_path) as fh:
                data = json.load(fh)
            return pd.DataFrame(data)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.error("Could not read drift history: %s", exc)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# PerformanceMonitor
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """
    Tracks live trade results and detects degradation in model accuracy.

    Parameters
    ----------
    performance_log : str
        Path to JSON file where trade results are persisted.
    """

    def __init__(
        self,
        performance_log: str = "stock_picker_data/models/performance_log.json",
    ) -> None:
        self.performance_log = Path(performance_log)
        self._records: List[Dict[str, Any]] = []
        self._load_log()

    # ------------------------------------------------------------------
    # Log management
    # ------------------------------------------------------------------

    def _load_log(self) -> None:
        if not self.performance_log.exists():
            return
        try:
            with open(self.performance_log) as fh:
                self._records = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not load performance log: %s", exc)
            self._records = []

    def _save_log(self) -> None:
        self.performance_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.performance_log, "w") as fh:
                json.dump(self._records, fh, indent=2)
        except OSError as exc:
            logger.error("Failed to save performance log: %s", exc)

    # ------------------------------------------------------------------
    # Logging trade results
    # ------------------------------------------------------------------

    def log_trade_result(
        self,
        date: Any,
        sc_code: str,
        predicted_prob: float,
        actual_return: float,
        signal_correct: bool,
    ) -> None:
        """
        Record the outcome of a trade signal.

        Parameters
        ----------
        date : date-like
            Trade date.
        sc_code : str
            BSE/NSE stock code.
        predicted_prob : float
            Model probability at signal time (0–1).
        actual_return : float
            Actual 5-day return realised on this trade.
        signal_correct : bool
            True if the direction of the trade matched the actual outcome.
        """
        record = {
            "date": str(date),
            "sc_code": sc_code,
            "predicted_prob": float(predicted_prob),
            "actual_return": float(actual_return),
            "signal_correct": bool(signal_correct),
            "logged_at": datetime.datetime.now().isoformat(),
        }
        self._records.append(record)
        self._save_log()
        logger.debug("Trade result logged: sc_code=%s correct=%s", sc_code, signal_correct)

    # ------------------------------------------------------------------
    # Rolling metrics
    # ------------------------------------------------------------------

    def _recent_df(self, window_days: int) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame()
        df = pd.DataFrame(self._records)
        df["date"] = pd.to_datetime(df["date"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=window_days)
        return df[df["date"] >= cutoff]

    def compute_rolling_accuracy(self, window_days: int = 20) -> float:
        """
        Fraction of signals that were directionally correct over the last
        ``window_days`` calendar days.  Returns NaN if no data.
        """
        df = self._recent_df(window_days)
        if df.empty:
            return float("nan")
        return float(df["signal_correct"].mean())

    def compute_rolling_win_rate(self, window_days: int = 20) -> float:
        """
        Fraction of trades with a positive actual return over the last
        ``window_days`` days.  Returns NaN if no data.
        """
        df = self._recent_df(window_days)
        if df.empty:
            return float("nan")
        return float((df["actual_return"] > 0).mean())

    # ------------------------------------------------------------------
    # Degradation detection
    # ------------------------------------------------------------------

    def detect_accuracy_degradation(
        self, threshold_pct: float = 0.10
    ) -> Tuple[bool, str]:
        """
        Compare recent 20-day accuracy against the long-run baseline.

        A degradation is flagged when recent accuracy drops more than
        ``threshold_pct`` (absolute) below the baseline.

        Returns
        -------
        (is_degraded, message) : (bool, str)
        """
        if len(self._records) < 30:
            return False, "Insufficient data for degradation detection (need >= 30 records)."

        df = pd.DataFrame(self._records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        baseline_acc = float(df["signal_correct"].mean())
        recent_acc = self.compute_rolling_accuracy(window_days=20)

        if np.isnan(recent_acc):
            return False, "No trades in the last 20 days."

        drop = baseline_acc - recent_acc
        message = (
            f"Baseline accuracy: {baseline_acc:.1%}  |  "
            f"Recent (20-day) accuracy: {recent_acc:.1%}  |  "
            f"Drop: {drop:.1%}"
        )

        if drop > threshold_pct:
            logger.warning("ACCURACY DEGRADATION DETECTED: %s", message)
            return True, message

        return False, message

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_performance_summary(self, last_n_days: int = 90) -> Dict[str, Any]:
        """Return a comprehensive performance summary dict."""
        df = self._recent_df(last_n_days)
        if df.empty:
            return {"status": "NO_DATA", "window_days": last_n_days}

        is_degraded, degradation_msg = self.detect_accuracy_degradation()

        return {
            "window_days": last_n_days,
            "n_trades": len(df),
            "accuracy": float(df["signal_correct"].mean()),
            "win_rate": float((df["actual_return"] > 0).mean()),
            "mean_return": float(df["actual_return"].mean()),
            "median_return": float(df["actual_return"].median()),
            "std_return": float(df["actual_return"].std()),
            "is_degraded": is_degraded,
            "degradation_message": degradation_msg,
            "rolling_20d_accuracy": self.compute_rolling_accuracy(20),
            "rolling_20d_win_rate": self.compute_rolling_win_rate(20),
            "computed_at": datetime.datetime.now().isoformat(),
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def run_daily_monitoring_checks(
    model: Any,
    X_current: pd.DataFrame,
    y_pred_current: np.ndarray,
    reference_path: str,
) -> Dict[str, Any]:
    """
    Run all daily monitoring checks and return a consolidated report.

    Parameters
    ----------
    model : sklearn-compatible model (unused directly; reserved for future checks)
    X_current : pd.DataFrame
        Today's feature matrix.
    y_pred_current : np.ndarray
        Today's model predictions / probabilities.
    reference_path : str
        Path to the reference distribution pickle file.

    Returns
    -------
    dict
        drift_report, performance_summary, overall_action
    """
    date_today = datetime.date.today()
    results: Dict[str, Any] = {"date": date_today.isoformat()}

    # --- Drift ---
    try:
        drift_monitor = ModelDriftMonitor(reference_data_path=reference_path)
        drift_report = drift_monitor.compute_drift_report(X_current, y_pred_current)
        drift_monitor.log_drift_event(drift_report, date=date_today)
        results["drift_report"] = drift_report
    except Exception as exc:
        logger.error("Drift check failed: %s", exc)
        results["drift_report"] = {"error": str(exc), "overall_status": "UNKNOWN"}

    # --- Performance ---
    try:
        perf_monitor = PerformanceMonitor()
        perf_summary = perf_monitor.get_performance_summary(last_n_days=90)
        results["performance_summary"] = perf_summary
    except Exception as exc:
        logger.error("Performance check failed: %s", exc)
        results["performance_summary"] = {"error": str(exc)}

    # --- Overall recommendation ---
    drift_status = results.get("drift_report", {}).get("overall_status", "UNKNOWN")
    is_degraded = results.get("performance_summary", {}).get("is_degraded", False)

    if drift_status == "RETRAIN_NEEDED" or is_degraded:
        results["overall_action"] = "RETRAIN_MODEL"
    elif drift_status == "WARNING":
        results["overall_action"] = "MONITOR_CLOSELY"
    else:
        results["overall_action"] = "CONTINUE_TRADING"

    logger.info(
        "Daily monitoring: drift=%s  degraded=%s  action=%s",
        drift_status,
        is_degraded,
        results["overall_action"],
    )
    return results


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    np.random.seed(42)

    print("=" * 60)
    print("PSICalculator demo")
    print("=" * 60)

    calc = PSICalculator(n_bins=10)
    ref = np.random.normal(0, 1, 1000)
    stable_curr = np.random.normal(0, 1, 500)       # same distribution
    drifted_curr = np.random.normal(2, 1.5, 500)    # different distribution

    print(f"  PSI (stable):  {calc.compute_psi(ref, stable_curr):.4f}")
    print(f"  PSI (drifted): {calc.compute_psi(ref, drifted_curr):.4f}")

    print()
    print("=" * 60)
    print("ModelDriftMonitor demo")
    print("=" * 60)

    feature_names = [f"feature_{i}" for i in range(5)]
    X_ref = pd.DataFrame(np.random.normal(0, 1, (500, 5)), columns=feature_names)
    y_ref = np.random.uniform(0, 1, 500)

    monitor = ModelDriftMonitor(
        reference_data_path="stock_picker_data/models/demo_reference.pkl"
    )
    monitor.set_reference_distribution(X_ref, y_ref)

    X_curr = pd.DataFrame(np.random.normal(1, 1.5, (200, 5)), columns=feature_names)
    y_curr = np.random.uniform(0.3, 1, 200)

    report = monitor.compute_drift_report(X_curr, y_curr)
    print(f"  Overall status:      {report['overall_status']}")
    print(f"  Drifted features:    {report['drifted_features']}")
    print(f"  Prediction PSI:      {report['prediction_drift']['psi']:.4f}")
    print(f"  Should retrain:      {monitor.should_retrain(X_curr, y_curr)}")

    print()
    print("=" * 60)
    print("PerformanceMonitor demo")
    print("=" * 60)

    pm = PerformanceMonitor(performance_log="stock_picker_data/models/demo_perf_log.json")
    for i in range(50):
        pm.log_trade_result(
            date=datetime.date.today() - datetime.timedelta(days=50 - i),
            sc_code=f"50{i:04d}",
            predicted_prob=np.random.uniform(0.5, 0.9),
            actual_return=np.random.normal(0.01, 0.03),
            signal_correct=np.random.random() > 0.4,
        )

    summary = pm.get_performance_summary(last_n_days=90)
    print(f"  Accuracy:       {summary['accuracy']:.1%}")
    print(f"  Win rate:       {summary['win_rate']:.1%}")
    print(f"  Mean return:    {summary['mean_return']:.2%}")
    print(f"  Is degraded:    {summary['is_degraded']}")
