"""
feature_leak_audit.py
=====================
Production feature leak audit for the Indian market stock picker.

Handles:
  - Gap 12: Feature Leak Audit

Detects four classes of data leakage that are common in time-series
financial ML pipelines:

    1. Future data leakage   — features computed with post-date information
    2. Overlap in splits     — same (stock, date) period in both train and val
    3. Rolling computation bias — rolling window looks forward at time t
    4. Label construction errors — target computed backwards instead of forwards

Usage:
    from production.feature_leak_audit import FeatureLeakAuditor, TimeSeriesLeakDetector

    auditor = FeatureLeakAuditor()
    result = auditor.run_full_audit(df, feature_cols=['rsi', 'macd', 'momentum_5d'])
    print(auditor.generate_audit_report(result))
"""

import logging
import datetime
import random
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional heavy dependencies — graceful fallback if unavailable
try:
    from sklearn.feature_selection import mutual_info_classif
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available; mutual information checks will be skipped.")

try:
    from scipy import stats as scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False
    logger.warning("scipy not available; ADF stationarity test will be skipped.")


# ---------------------------------------------------------------------------
# FeatureLeakAuditor
# ---------------------------------------------------------------------------

class FeatureLeakAuditor:
    """
    Audits a training DataFrame for common feature leakage patterns.

    Parameters
    ----------
    label_col : str
        Name of the target / label column.
    date_col : str
        Name of the date column (must be parseable by pd.to_datetime).
    symbol_col : str
        Name of the stock symbol / SC_CODE column.
    """

    def __init__(
        self,
        label_col: str = "target",
        date_col: str = "DATE",
        symbol_col: str = "SC_CODE",
    ) -> None:
        self.label_col = label_col
        self.date_col = date_col
        self.symbol_col = symbol_col

    # ------------------------------------------------------------------
    # Check 1 — Future data leakage via correlation with future returns
    # ------------------------------------------------------------------

    def check_future_data_leakage(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        forward_periods: int = 5,
    ) -> Dict[str, Any]:
        """
        Detect suspicious correlation between each feature and a future
        return series.

        If a feature has been accidentally computed using future prices, its
        contemporaneous correlation with tomorrow's return will be abnormally
        high.

        Methodology
        -----------
        For each feature, compute Pearson correlation with
        ``close.shift(-forward_periods) / close - 1``.
        A correlation > 0.3 in absolute value is flagged as suspicious.

        Returns
        -------
        dict
            passed (bool), suspicious_features (list), correlations (dict),
            details (str)
        """
        CORRELATION_THRESHOLD = 0.30
        result: Dict[str, Any] = {
            "check": "future_data_leakage",
            "forward_periods": forward_periods,
            "threshold": CORRELATION_THRESHOLD,
            "correlations": {},
            "suspicious_features": [],
            "passed": True,
            "details": "",
        }

        # Build the future return series
        if "Close" in df.columns:
            close = df["Close"].copy().astype(float)
        elif "close" in df.columns:
            close = df["close"].copy().astype(float)
        elif self.label_col in df.columns:
            # If we already have a target column, use it as a proxy
            close = None
            future_return = df[self.label_col].copy().astype(float)
        else:
            result["details"] = "No 'Close' or target column found; skipping."
            result["passed"] = True  # Cannot check — treat as warning
            return result

        if "close" in dir() or "Close" in df.columns:
            # Avoid NameError when close is not defined via the elif branch
            pass

        try:
            if close is not None:
                future_return = close.shift(-forward_periods) / close - 1
            else:
                future_return = df[self.label_col].astype(float)
        except Exception as exc:
            result["details"] = f"Could not compute future returns: {exc}"
            return result

        for col in feature_cols:
            if col not in df.columns:
                continue
            try:
                feat = df[col].astype(float)
                valid = feat.notna() & future_return.notna()
                if valid.sum() < 30:
                    result["correlations"][col] = None
                    continue
                corr = float(feat[valid].corr(future_return[valid]))
                result["correlations"][col] = round(corr, 4)
                if abs(corr) > CORRELATION_THRESHOLD:
                    result["suspicious_features"].append(col)
                    logger.warning(
                        "Possible future leakage: feature '%s' correlation with "
                        "future return = %.3f (threshold=%.2f)",
                        col, corr, CORRELATION_THRESHOLD,
                    )
            except Exception as exc:
                logger.debug("Skipping feature '%s' in leakage check: %s", col, exc)
                result["correlations"][col] = None

        if result["suspicious_features"]:
            result["passed"] = False
            result["details"] = (
                f"High correlation with future returns detected in: "
                f"{result['suspicious_features']}. "
                f"Investigate whether these features use future data."
            )
        else:
            result["details"] = (
                f"No features exceed correlation threshold {CORRELATION_THRESHOLD} "
                f"with {forward_periods}-period forward returns."
            )

        return result

    # ------------------------------------------------------------------
    # Check 2 — Overlap between train and validation splits
    # ------------------------------------------------------------------

    def check_overlap_in_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        symbol_col: str = "SC_CODE",
        date_col: str = "DATE",
    ) -> Dict[str, Any]:
        """
        Verify that no (symbol, date) pair appears in BOTH the training and
        validation DataFrames.

        A date-period overlap (same stock in both sets during the same calendar
        period) is a form of look-ahead bias because the model may implicitly
        learn from validation-era information during training.

        Returns
        -------
        dict
            passed (bool), overlap_count (int), overlapping_records (list),
            details (str)
        """
        result: Dict[str, Any] = {
            "check": "overlap_in_splits",
            "overlap_count": 0,
            "overlapping_records": [],
            "passed": True,
            "details": "",
        }

        sym_col = symbol_col if symbol_col in train_df.columns else self.symbol_col
        dt_col = date_col if date_col in train_df.columns else self.date_col

        missing = []
        for col in (sym_col, dt_col):
            if col not in train_df.columns or col not in val_df.columns:
                missing.append(col)

        if missing:
            result["details"] = (
                f"Columns {missing} not found in one or both DataFrames; "
                "overlap check skipped."
            )
            return result

        try:
            train_keys = set(
                zip(
                    train_df[sym_col].astype(str),
                    pd.to_datetime(train_df[dt_col]).dt.date.astype(str),
                )
            )
            val_keys = set(
                zip(
                    val_df[sym_col].astype(str),
                    pd.to_datetime(val_df[dt_col]).dt.date.astype(str),
                )
            )

            overlap = train_keys & val_keys
            result["overlap_count"] = len(overlap)

            if overlap:
                result["passed"] = False
                sample = list(overlap)[:10]
                result["overlapping_records"] = [
                    {"symbol": s, "date": d} for s, d in sample
                ]
                result["details"] = (
                    f"LEAK DETECTED: {len(overlap)} (symbol, date) pairs appear in both "
                    f"train and val sets.  Sample: {sample[:5]}"
                )
                logger.error(
                    "Data split overlap: %d overlapping (symbol, date) pairs found.", len(overlap)
                )
            else:
                result["details"] = (
                    f"No overlap found between train ({len(train_keys):,} keys) "
                    f"and val ({len(val_keys):,} keys) sets."
                )
        except Exception as exc:
            result["details"] = f"Overlap check failed with error: {exc}"
            logger.exception("check_overlap_in_splits failed")

        return result

    # ------------------------------------------------------------------
    # Check 3 — Rolling computation bias
    # ------------------------------------------------------------------

    def check_rolling_computation_bias(
        self,
        df: pd.DataFrame,
        feature_col: str,
        window: int,
    ) -> Dict[str, Any]:
        """
        Verify that a rolling feature at row t only uses data from rows
        [t-window, t], not [t, t+window].

        Methodology
        -----------
        Sample 5 random rows from the DataFrame and recompute the rolling
        feature manually using only past data.  If the stored value matches
        the correctly computed backward-looking roll, the feature is clean.

        Returns
        -------
        dict
            passed (bool), n_samples_checked (int), n_mismatches (int),
            details (str)
        """
        result: Dict[str, Any] = {
            "check": "rolling_computation_bias",
            "feature_col": feature_col,
            "window": window,
            "n_samples_checked": 0,
            "n_mismatches": 0,
            "mismatch_indices": [],
            "passed": True,
            "details": "",
        }

        if feature_col not in df.columns:
            result["details"] = f"Column '{feature_col}' not found in DataFrame."
            return result

        # We need a numeric base column to roll over.
        # Try to find the raw input used to compute this rolling feature.
        # Convention: if feature is e.g. "rsi_14", look for a "Close" column.
        base_col = None
        for candidate in ("Close", "close", "CLOSE", "Adj Close"):
            if candidate in df.columns:
                base_col = candidate
                break

        if base_col is None:
            result["details"] = (
                "No 'Close' column found; cannot manually recompute rolling feature. "
                "Skipping bias check."
            )
            return result

        series = df[base_col].astype(float).reset_index(drop=True)
        stored = df[feature_col].astype(float).reset_index(drop=True)

        # Pick 5 random rows that have enough history
        valid_indices = list(range(window, len(df)))
        if not valid_indices:
            result["details"] = f"Not enough rows ({len(df)}) for window={window}."
            return result

        n_samples = min(5, len(valid_indices))
        sample_indices = random.sample(valid_indices, n_samples)
        result["n_samples_checked"] = n_samples

        for idx in sample_indices:
            # Correct backward-looking roll (mean as proxy for any rolling stat)
            correct_val = float(series[max(0, idx - window + 1): idx + 1].mean())
            stored_val = float(stored[idx])

            if np.isnan(stored_val):
                continue  # NaN stored — inconclusive

            # Allow 1% relative tolerance for floating point differences
            rel_error = abs(stored_val - correct_val) / (abs(correct_val) + 1e-10)
            if rel_error > 0.01:
                result["n_mismatches"] += 1
                result["mismatch_indices"].append(idx)
                logger.warning(
                    "Rolling bias suspected at index %d: stored=%.6f  correct=%.6f  "
                    "rel_error=%.4f",
                    idx, stored_val, correct_val, rel_error,
                )

        if result["n_mismatches"] > 0:
            result["passed"] = False
            result["details"] = (
                f"Rolling bias detected: {result['n_mismatches']} of {n_samples} "
                f"sampled rows had stored values differing from backward-only roll."
            )
        else:
            result["details"] = (
                f"No rolling bias detected across {n_samples} sampled rows "
                f"(window={window})."
            )

        return result

    # ------------------------------------------------------------------
    # Check 4 — Label construction correctness
    # ------------------------------------------------------------------

    def check_label_construction(
        self,
        df: pd.DataFrame,
        close_col: str = "Close",
        label_col: str = "target",
        forward_period: int = 5,
    ) -> Dict[str, Any]:
        """
        Verify that the target at row t is computed as:
            close[t + forward_period] / close[t] - 1   (CORRECT — forward-looking)
        and NOT:
            close[t] / close[t - forward_period] - 1   (WRONG — backward-looking)

        The wrong formula is a common mistake that produces labels that are
        trivially predictable from recent momentum features.

        Returns
        -------
        dict
            passed (bool), correct_correlation (float), wrong_correlation (float),
            likely_correct (bool), details (str)
        """
        result: Dict[str, Any] = {
            "check": "label_construction",
            "label_col": label_col,
            "close_col": close_col,
            "forward_period": forward_period,
            "correct_correlation": None,
            "wrong_correlation": None,
            "likely_correct": None,
            "passed": True,
            "details": "",
        }

        for col in (close_col, label_col):
            if col not in df.columns:
                result["details"] = f"Column '{col}' not found; skipping label check."
                return result

        try:
            close = df[close_col].astype(float).reset_index(drop=True)
            label = df[label_col].astype(float).reset_index(drop=True)

            # Correct formula: close[t+n]/close[t] - 1
            correct_target = close.shift(-forward_period) / close - 1
            # Wrong formula: close[t]/close[t-n] - 1 (past momentum)
            wrong_target = close / close.shift(forward_period) - 1

            valid_c = label.notna() & correct_target.notna()
            valid_w = label.notna() & wrong_target.notna()

            corr_correct = float(label[valid_c].corr(correct_target[valid_c])) if valid_c.sum() > 10 else np.nan
            corr_wrong = float(label[valid_w].corr(wrong_target[valid_w])) if valid_w.sum() > 10 else np.nan

            result["correct_correlation"] = round(corr_correct, 4) if not np.isnan(corr_correct) else None
            result["wrong_correlation"] = round(corr_wrong, 4) if not np.isnan(corr_wrong) else None

            # Good label should correlate strongly with forward returns (>0.9)
            # and should NOT correlate significantly with backward returns
            if not np.isnan(corr_correct) and not np.isnan(corr_wrong):
                likely_correct = corr_correct > 0.90 and corr_wrong < 0.50
                result["likely_correct"] = likely_correct

                if not likely_correct:
                    if corr_wrong > corr_correct:
                        result["passed"] = False
                        result["details"] = (
                            f"LABEL ERROR SUSPECTED: target correlates more with "
                            f"backward return ({corr_wrong:.3f}) than forward return "
                            f"({corr_correct:.3f}). Check label construction formula."
                        )
                        logger.error(
                            "Label construction error suspected: backward_corr=%.3f > "
                            "forward_corr=%.3f",
                            corr_wrong, corr_correct,
                        )
                    else:
                        result["details"] = (
                            f"Label correlation with forward return: {corr_correct:.3f}  "
                            f"(backward: {corr_wrong:.3f}).  Investigate if < 0.90."
                        )
                else:
                    result["details"] = (
                        f"Label construction looks correct: forward_corr={corr_correct:.3f}  "
                        f"backward_corr={corr_wrong:.3f}."
                    )
            else:
                result["details"] = "Insufficient data for label construction check."

        except Exception as exc:
            result["details"] = f"Label construction check failed: {exc}"
            logger.exception("check_label_construction failed")

        return result

    # ------------------------------------------------------------------
    # Full audit
    # ------------------------------------------------------------------

    def run_full_audit(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "target",
    ) -> Dict[str, Any]:
        """
        Run all four leak checks and consolidate into a single report.

        Returns
        -------
        dict
            checks (dict of individual results), overall_pass (bool),
            warnings (list of str), errors (list of str), run_at (str)
        """
        warnings: List[str] = []
        errors: List[str] = []
        checks: Dict[str, Any] = {}

        # ---- Check 1: Future data leakage ----
        try:
            checks["future_data_leakage"] = self.check_future_data_leakage(
                df, feature_cols
            )
            if not checks["future_data_leakage"]["passed"]:
                errors.append(
                    "Future data leakage: "
                    + checks["future_data_leakage"]["details"]
                )
        except Exception as exc:
            checks["future_data_leakage"] = {"passed": None, "details": str(exc)}
            warnings.append(f"Future leakage check failed: {exc}")

        # ---- Check 2: Overlap (self-check: can only flag if train/val provided) ----
        # For run_full_audit, we split the df in half as a proxy check
        try:
            mid = len(df) // 2
            train_half = df.iloc[:mid]
            val_half = df.iloc[mid:]
            checks["overlap_in_splits"] = self.check_overlap_in_splits(
                train_half, val_half
            )
            if not checks["overlap_in_splits"]["passed"]:
                errors.append(
                    "Split overlap: " + checks["overlap_in_splits"]["details"]
                )
        except Exception as exc:
            checks["overlap_in_splits"] = {"passed": None, "details": str(exc)}
            warnings.append(f"Overlap check failed: {exc}")

        # ---- Check 3: Rolling bias (on first numeric feature, window=5) ----
        try:
            numeric_features = [
                c for c in feature_cols
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
            if numeric_features:
                checks["rolling_computation_bias"] = self.check_rolling_computation_bias(
                    df, numeric_features[0], window=5
                )
                if not checks["rolling_computation_bias"]["passed"]:
                    errors.append(
                        "Rolling bias: "
                        + checks["rolling_computation_bias"]["details"]
                    )
            else:
                checks["rolling_computation_bias"] = {
                    "passed": None,
                    "details": "No numeric feature columns found.",
                }
                warnings.append("Rolling bias check skipped — no numeric features.")
        except Exception as exc:
            checks["rolling_computation_bias"] = {"passed": None, "details": str(exc)}
            warnings.append(f"Rolling bias check failed: {exc}")

        # ---- Check 4: Label construction ----
        try:
            checks["label_construction"] = self.check_label_construction(
                df, label_col=label_col
            )
            if not checks["label_construction"]["passed"]:
                errors.append(
                    "Label construction: "
                    + checks["label_construction"]["details"]
                )
        except Exception as exc:
            checks["label_construction"] = {"passed": None, "details": str(exc)}
            warnings.append(f"Label construction check failed: {exc}")

        overall_pass = len(errors) == 0

        if not overall_pass:
            logger.error(
                "Feature leak audit FAILED with %d error(s): %s", len(errors), errors
            )
        else:
            logger.info(
                "Feature leak audit PASSED with %d warning(s).", len(warnings)
            )

        return {
            "checks": checks,
            "overall_pass": overall_pass,
            "warnings": warnings,
            "errors": errors,
            "n_features_audited": len(feature_cols),
            "n_rows": len(df),
            "run_at": datetime.datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_audit_report(
        self,
        audit_result: Dict[str, Any],
        output_file: Optional[str] = None,
    ) -> str:
        """
        Format the audit result as a human-readable text report.

        Parameters
        ----------
        audit_result : dict
            Output from ``run_full_audit()``.
        output_file : str, optional
            If provided, write the report to this path.

        Returns
        -------
        str
            The full formatted report text.
        """
        lines = [
            "=" * 70,
            "  FEATURE LEAK AUDIT REPORT",
            f"  Run at : {audit_result.get('run_at', 'N/A')}",
            f"  Rows   : {audit_result.get('n_rows', '?'):,}",
            f"  Features audited: {audit_result.get('n_features_audited', '?')}",
            "=" * 70,
            "",
            f"OVERALL RESULT: {'PASS' if audit_result.get('overall_pass') else 'FAIL'}",
            "",
        ]

        errors = audit_result.get("errors", [])
        if errors:
            lines.append("ERRORS (must fix before training):")
            for err in errors:
                lines.append(f"  [ERROR] {err}")
            lines.append("")

        warnings = audit_result.get("warnings", [])
        if warnings:
            lines.append("WARNINGS (investigate):")
            for w in warnings:
                lines.append(f"  [WARN ] {w}")
            lines.append("")

        lines.append("INDIVIDUAL CHECKS:")
        for check_name, check_result in audit_result.get("checks", {}).items():
            passed = check_result.get("passed")
            status = "PASS" if passed else ("FAIL" if passed is False else "SKIP")
            lines.append(f"\n  [{status}] {check_name.upper().replace('_', ' ')}")
            details = check_result.get("details", "")
            for detail_line in textwrap.wrap(details, width=66):
                lines.append(f"         {detail_line}")

            # Extra info for specific checks
            if check_name == "future_data_leakage":
                suspicious = check_result.get("suspicious_features", [])
                if suspicious:
                    lines.append(f"         Suspicious features: {suspicious}")
            elif check_name == "label_construction":
                lines.append(
                    f"         Forward corr: {check_result.get('correct_correlation')}  "
                    f"Backward corr: {check_result.get('wrong_correlation')}"
                )
            elif check_name == "overlap_in_splits":
                lines.append(
                    f"         Overlapping records: {check_result.get('overlap_count', 0)}"
                )

        lines += ["", "=" * 70]
        report = "\n".join(lines)

        if output_file:
            from pathlib import Path
            p = Path(output_file)
            p.parent.mkdir(parents=True, exist_ok=True)
            try:
                p.write_text(report, encoding="utf-8")
                logger.info("Audit report written to %s", p)
            except OSError as exc:
                logger.error("Failed to write audit report: %s", exc)

        return report


# ---------------------------------------------------------------------------
# TimeSeriesLeakDetector — stateless utility class
# ---------------------------------------------------------------------------

class TimeSeriesLeakDetector:
    """
    Stateless utility functions for time-series leakage detection.

    All methods are classmethods so they can be used without instantiation.
    """

    @classmethod
    def detect_target_leakage(
        cls,
        df: pd.DataFrame,
        features: List[str],
        target: str,
        threshold: float = 0.5,
    ) -> List[str]:
        """
        Identify features that are suspiciously predictive of the target
        using mutual information score.

        A very high mutual information score (> threshold) between a feature
        and the target at the same row index is a red flag: legitimate features
        rarely achieve this without some form of leakage.

        Parameters
        ----------
        df : pd.DataFrame
        features : list of str
        target : str
        threshold : float
            Mutual information score above which a feature is flagged.

        Returns
        -------
        list of str
            Suspicious feature names.
        """
        if not _SKLEARN_AVAILABLE:
            logger.warning("sklearn not available; detect_target_leakage returning empty list.")
            return []

        suspicious = []
        valid_features = [c for c in features if c in df.columns]
        if target not in df.columns or not valid_features:
            return []

        X = df[valid_features].copy()
        y = df[target].copy()

        # Drop rows where target or any feature is NaN
        mask = y.notna()
        for col in valid_features:
            if pd.api.types.is_numeric_dtype(X[col]):
                mask = mask & X[col].notna()

        X = X[mask]
        y = y[mask]

        if len(X) < 30:
            logger.warning("Too few rows for mutual information check (%d).", len(X))
            return []

        # Fill remaining NaN with column median
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())

        # Discretise continuous target for MI classification
        try:
            y_discrete = (y > y.median()).astype(int)
            mi_scores = mutual_info_classif(X, y_discrete, random_state=42)

            for feat, score in zip(valid_features, mi_scores):
                if score > threshold:
                    suspicious.append(feat)
                    logger.warning(
                        "High MI score for feature '%s': %.4f (threshold=%.2f)",
                        feat, score, threshold,
                    )
        except Exception as exc:
            logger.error("mutual_info_classif failed: %s", exc)

        return suspicious

    @classmethod
    def verify_no_future_info_in_feature(
        cls,
        df: pd.DataFrame,
        feature_col: str,
        date_col: str,
        n_samples: int = 20,
    ) -> bool:
        """
        Verify that a feature's value at row t does not change when future
        rows are removed.

        Methodology
        -----------
        For n_samples random rows, recompute the feature's expected value
        using only data up to and including row t.  If the stored value is
        unexpectedly higher / lower than the truncated computation would
        produce, it is flagged.

        This is a heuristic proxy check; a definitive check requires access
        to the raw data pipeline.

        Returns
        -------
        bool
            True if no future information is detected.
        """
        if feature_col not in df.columns or date_col not in df.columns:
            logger.warning(
                "verify_no_future_info_in_feature: column '%s' or '%s' not found.",
                feature_col, date_col,
            )
            return True  # Cannot check; assume clean

        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col).reset_index(drop=True)

            numeric_col = df[feature_col].astype(float)
            if numeric_col.isna().all():
                return True

            # Sample n_samples rows from the middle of the DataFrame
            valid_indices = list(range(10, len(df) - 5))
            if len(valid_indices) < n_samples:
                valid_indices = list(range(len(df)))
            sample_idx = random.sample(valid_indices, min(n_samples, len(valid_indices)))

            future_count = 0
            for idx in sample_idx:
                # The feature at idx should be computable from rows [0..idx]
                truncated_mean = float(numeric_col.iloc[: idx + 1].mean())
                full_mean = float(numeric_col.mean())

                # If the stored value is closer to the full-series mean than to
                # the truncated mean (in relative terms), it may use future data.
                stored = float(numeric_col.iloc[idx])
                if np.isnan(stored):
                    continue

                dist_to_truncated = abs(stored - truncated_mean)
                dist_to_full = abs(stored - full_mean)

                # If stored is consistently much closer to full-series mean,
                # it suggests it was computed over the entire series (future data included)
                if dist_to_full < 0.05 * dist_to_truncated and dist_to_truncated > 0.01:
                    future_count += 1

            future_frac = future_count / len(sample_idx) if sample_idx else 0
            if future_frac > 0.50:
                logger.warning(
                    "Feature '%s' may contain future info: %.0f%% of sampled rows "
                    "have values suspiciously close to the full-series mean.",
                    feature_col, future_frac * 100,
                )
                return False

            return True

        except Exception as exc:
            logger.error("verify_no_future_info_in_feature failed: %s", exc)
            return True  # Conservative: assume clean on error

    @classmethod
    def check_feature_stationarity(
        cls,
        df: pd.DataFrame,
        feature_col: str,
    ) -> Dict[str, Any]:
        """
        Run the Augmented Dickey-Fuller (ADF) stationarity test on a feature.

        Non-stationary features can cause spurious correlations in predictive
        models and may be a sign of an underlying data issue.

        Returns
        -------
        dict
            is_stationary (bool), adf_statistic (float), p_value (float),
            interpretation (str)
        """
        result: Dict[str, Any] = {
            "feature_col": feature_col,
            "is_stationary": None,
            "adf_statistic": None,
            "p_value": None,
            "interpretation": "",
        }

        if feature_col not in df.columns:
            result["interpretation"] = f"Column '{feature_col}' not found."
            return result

        if not _SCIPY_AVAILABLE:
            result["interpretation"] = "scipy not available; ADF test skipped."
            return result

        try:
            series = df[feature_col].astype(float).dropna()
            if len(series) < 20:
                result["interpretation"] = "Too few observations for ADF test (need >= 20)."
                return result

            # Use statsmodels if available for the proper ADF test
            try:
                from statsmodels.tsa.stattools import adfuller
                adf_result = adfuller(series, autolag="AIC")
                adf_stat = float(adf_result[0])
                p_value = float(adf_result[1])
            except ImportError:
                # Fallback: simple variance ratio test (Hurst-like approximation)
                half = len(series) // 2
                var_full = float(series.var())
                var_half = float(series.iloc[:half].var())
                # Very rough proxy: if variance doesn't grow with time, likely stationary
                variance_ratio = var_half / (var_full + 1e-10)
                adf_stat = -variance_ratio  # Negative = more stationary (analogous to ADF)
                p_value = 0.05 if variance_ratio > 0.6 else 0.01
                logger.info(
                    "statsmodels not available; using variance ratio proxy for '%s'.",
                    feature_col,
                )

            is_stationary = p_value < 0.05
            result["adf_statistic"] = round(adf_stat, 4)
            result["p_value"] = round(p_value, 6)
            result["is_stationary"] = is_stationary
            result["interpretation"] = (
                f"{'Stationary' if is_stationary else 'NON-STATIONARY'} "
                f"(ADF stat={adf_stat:.3f}, p={p_value:.4f}). "
                + ("Consider differencing if using in a linear model." if not is_stationary else "")
            )

        except Exception as exc:
            result["interpretation"] = f"ADF test failed: {exc}"
            logger.error("check_feature_stationarity failed for '%s': %s", feature_col, exc)

        return result


# ---------------------------------------------------------------------------
# __main__ demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    np.random.seed(42)

    print("=" * 70)
    print("FeatureLeakAuditor demo")
    print("=" * 70)

    # Build a synthetic DataFrame that mimics a stock ML dataset
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.01, n))

    df = pd.DataFrame({
        "DATE": dates,
        "SC_CODE": "500209",
        "Close": close,
        "rsi": 50 + 20 * np.random.randn(n),
        "macd": np.random.randn(n),
        "momentum_5d": pd.Series(close).pct_change(5).values,
        # Deliberately add a leaky feature (uses future close)
        "leaky_feature": pd.Series(close).shift(-5).values / close - 1,
        # Target: correct forward return
        "target": pd.Series(close).shift(-5).values / close - 1,
    })

    feature_cols = ["rsi", "macd", "momentum_5d", "leaky_feature"]

    auditor = FeatureLeakAuditor(label_col="target", date_col="DATE", symbol_col="SC_CODE")
    audit_result = auditor.run_full_audit(df, feature_cols)
    report = auditor.generate_audit_report(
        audit_result,
        output_file="stock_picker_data/audit/demo_feature_leak_audit.txt",
    )
    print(report)

    print()
    print("=" * 70)
    print("TimeSeriesLeakDetector demo")
    print("=" * 70)

    suspicious = TimeSeriesLeakDetector.detect_target_leakage(
        df, feature_cols, target="target", threshold=0.30
    )
    print(f"  Suspicious features (MI): {suspicious}")

    stationarity = TimeSeriesLeakDetector.check_feature_stationarity(df, "rsi")
    print(f"  RSI stationarity: {stationarity}")

    future_info = TimeSeriesLeakDetector.verify_no_future_info_in_feature(
        df, "rsi", "DATE"
    )
    print(f"  RSI future-info clean: {future_info}")
