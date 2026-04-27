"""
production/drift_monitor.py -- P18: PSI feature drift monitoring + retraining governance.

Population Stability Index (PSI) quantifies how much a feature's distribution
has shifted between a reference period (training) and a monitoring period (current).

PSI thresholds (industry standard):
  PSI < 0.10  -- stable, no action needed
  PSI < 0.20  -- slight shift, monitor closely
  PSI >= 0.20 -- significant shift, trigger retraining

Retraining governance combines:
  - Max PSI across key features
  - Days since last training
  - Recent model AUC vs. training AUC (optional)

Usage
-----
from production.drift_monitor import DriftMonitor, RetrainingGovernance

monitor = DriftMonitor()
report  = monitor.run(reference_df, current_df, feature_cols)
monitor.print_report(report)

gov = RetrainingGovernance()
decision = gov.should_retrain(report, days_since_train=45, current_auc=0.61)
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT     = Path(__file__).parent.parent
MODEL_DIR = _ROOT / "stock_picker_data" / "models"

# PSI thresholds
PSI_STABLE   = 0.10
PSI_WARNING  = 0.20   # >= this triggers WARNING
PSI_CRITICAL = 0.20   # >= this triggers retraining recommendation


def compute_psi(
    expected: np.ndarray,
    actual:   np.ndarray,
    n_bins:   int = 10,
    eps:      float = 1e-6,
) -> float:
    """
    Compute the Population Stability Index between two distributions.

    Parameters
    ----------
    expected : Reference-period feature values (training data).
    actual   : Current-period feature values (live data).
    n_bins   : Number of equal-width bins (default 10).
    eps      : Smoothing constant to avoid log(0).

    Returns
    -------
    float -- PSI value.
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual,   dtype=float)

    # Drop NaN
    expected = expected[np.isfinite(expected)]
    actual   = actual[np.isfinite(actual)]

    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Bin edges from reference distribution
    min_v = min(expected.min(), actual.min())
    max_v = max(expected.max(), actual.max())
    if min_v == max_v:
        return 0.0

    edges = np.linspace(min_v, max_v, n_bins + 1)
    edges[-1] += 1e-9   # include the max value

    exp_counts = np.histogram(expected, bins=edges)[0].astype(float)
    act_counts = np.histogram(actual,   bins=edges)[0].astype(float)

    exp_pct = (exp_counts / exp_counts.sum()).clip(eps)
    act_pct = (act_counts / act_counts.sum()).clip(eps)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return psi


# ===========================================================================
# DriftMonitor
# ===========================================================================

class DriftMonitor:
    """
    Monitor feature distribution drift using PSI.

    Parameters
    ----------
    n_bins         : PSI histogram bins (default 10).
    warning_thresh : PSI threshold for WARNING status (default 0.10).
    critical_thresh: PSI threshold for CRITICAL status (default 0.20).
    """

    def __init__(
        self,
        n_bins:          int   = 10,
        warning_thresh:  float = PSI_STABLE,
        critical_thresh: float = PSI_CRITICAL,
    ) -> None:
        self.n_bins          = n_bins
        self.warning_thresh  = warning_thresh
        self.critical_thresh = critical_thresh

    # ------------------------------------------------------------------

    def run(
        self,
        reference_df: pd.DataFrame,
        current_df:   pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, Any]:
        """
        Compute per-feature PSI and return drift report.

        Parameters
        ----------
        reference_df : Training-time feature DataFrame.
        current_df   : Current (live) feature DataFrame.
        feature_cols : Features to monitor.

        Returns
        -------
        dict with keys:
            feature_psi (dict: feature -> PSI value),
            max_psi, mean_psi, n_warnings, n_critical,
            status ('OK' | 'WARNING' | 'CRITICAL'),
            timestamp
        """
        feature_psi: Dict[str, float] = {}
        n_warn = 0
        n_crit = 0

        for col in feature_cols:
            if col not in reference_df.columns or col not in current_df.columns:
                continue
            psi = compute_psi(
                reference_df[col].values,
                current_df[col].values,
                n_bins=self.n_bins,
            )
            feature_psi[col] = round(psi, 6)
            if psi >= self.critical_thresh:
                n_crit += 1
            elif psi >= self.warning_thresh:
                n_warn += 1

        if not feature_psi:
            return {"error": "No features found in both DataFrames.", "status": "UNKNOWN"}

        max_psi  = float(max(feature_psi.values()))
        mean_psi = float(np.mean(list(feature_psi.values())))

        if n_crit > 0:
            status = "CRITICAL"
        elif n_warn > 0:
            status = "WARNING"
        else:
            status = "OK"

        report = {
            "feature_psi":  feature_psi,
            "max_psi":      round(max_psi, 6),
            "mean_psi":     round(mean_psi, 6),
            "n_features":   len(feature_psi),
            "n_warnings":   n_warn,
            "n_critical":   n_crit,
            "status":       status,
            "timestamp":    datetime.now().isoformat(timespec="seconds"),
        }
        logger.info(
            "P18 DriftMonitor: status=%s  max_psi=%.4f  n_crit=%d  n_warn=%d",
            status, max_psi, n_crit, n_warn,
        )
        return report

    # ------------------------------------------------------------------

    def save_reference(
        self,
        reference_df: pd.DataFrame,
        feature_cols: List[str],
        path: Optional[Path] = None,
    ) -> None:
        """Persist per-feature quantiles as reference distribution to disk."""
        path = path or (MODEL_DIR / "drift_reference.json")
        path.parent.mkdir(parents=True, exist_ok=True)

        ref: Dict[str, Any] = {"saved_on": str(date.today())}
        for col in feature_cols:
            if col not in reference_df.columns:
                continue
            vals = reference_df[col].dropna().values.astype(float)
            if len(vals) == 0:
                continue
            ref[col] = {
                "mean":  float(np.mean(vals)),
                "std":   float(np.std(vals)),
                "p10":   float(np.percentile(vals, 10)),
                "p50":   float(np.percentile(vals, 50)),
                "p90":   float(np.percentile(vals, 90)),
            }
        path.write_text(json.dumps(ref, indent=2))
        logger.info("P18: Reference distribution saved to %s (%d features)", path, len(ref) - 1)

    # ------------------------------------------------------------------

    def print_report(self, report: Dict[str, Any], top_n: int = 10) -> None:
        if "error" in report:
            print(f"Drift monitor error: {report['error']}")
            return

        status_sym = {"OK": "OK ", "WARNING": "!!!", "CRITICAL": "!!!"}
        print("=" * 60)
        print(f"  PSI DRIFT REPORT  [{report['status']}]  (P18)")
        print(f"  max_psi={report['max_psi']:.4f}  mean={report['mean_psi']:.4f}"
              f"  critical={report['n_critical']}  warning={report['n_warnings']}")
        print("=" * 60)
        print(f"  {'Feature':<30} {'PSI':>8}  {'Status'}")
        print("-" * 60)
        sorted_feats = sorted(report["feature_psi"].items(), key=lambda x: -x[1])
        for feat, psi in sorted_feats[:top_n]:
            if psi >= self.critical_thresh:
                sym = "CRITICAL"
            elif psi >= self.warning_thresh:
                sym = "warning"
            else:
                sym = ""
            print(f"  {feat:<30} {psi:>8.4f}  {sym}")
        if len(sorted_feats) > top_n:
            print(f"  ... ({len(sorted_feats) - top_n} more features)")
        print("=" * 60)


# ===========================================================================
# RetrainingGovernance
# ===========================================================================

class RetrainingGovernance:
    """
    P18 -- Decide whether the model should be retrained.

    Combines three signals:
      1. Max feature PSI (from DriftMonitor.run())
      2. Days since last training
      3. Optional: current live AUC vs. training AUC

    Parameters
    ----------
    psi_retrain_threshold  : Max PSI that triggers retraining (default 0.20).
    max_days_without_train : Force retrain after this many calendar days (default 30).
    auc_drop_threshold     : AUC drop (vs training) that triggers retraining (default 0.03).
    """

    def __init__(
        self,
        psi_retrain_threshold:  float = PSI_CRITICAL,
        max_days_without_train: int   = 30,
        auc_drop_threshold:     float = 0.03,
    ) -> None:
        self.psi_thresh       = psi_retrain_threshold
        self.max_days         = max_days_without_train
        self.auc_drop_thresh  = auc_drop_threshold

    # ------------------------------------------------------------------

    def should_retrain(
        self,
        drift_report:      Dict[str, Any],
        days_since_train:  int,
        train_auc:         float = 1.0,
        current_auc:       Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate retraining signals and return a governance decision.

        Parameters
        ----------
        drift_report      : Output of DriftMonitor.run().
        days_since_train  : Calendar days since the model was last trained.
        train_auc         : CV AUC from last training run (default 1.0 = skip check).
        current_auc       : Recent live AUC estimate (None = skip AUC check).

        Returns
        -------
        dict with keys:
            should_retrain (bool),
            reasons (list[str]),
            urgency ('LOW' | 'MEDIUM' | 'HIGH')
        """
        reasons: List[str] = []
        urgency_score = 0

        # Signal 1: PSI drift
        max_psi = drift_report.get("max_psi", 0.0)
        if max_psi >= self.psi_thresh:
            n_crit = drift_report.get("n_critical", 0)
            reasons.append(
                f"Feature drift: max_psi={max_psi:.4f} >= {self.psi_thresh} "
                f"({n_crit} critical features)"
            )
            urgency_score += 2

        # Signal 2: Time since last training
        if days_since_train >= self.max_days:
            reasons.append(
                f"Stale model: {days_since_train} days since last training "
                f"(threshold={self.max_days})"
            )
            urgency_score += 1

        # Signal 3: AUC drop
        if current_auc is not None and train_auc < 1.0:
            auc_drop = train_auc - current_auc
            if auc_drop >= self.auc_drop_thresh:
                reasons.append(
                    f"AUC degradation: {train_auc:.4f} -> {current_auc:.4f} "
                    f"(drop={auc_drop:.4f} >= {self.auc_drop_thresh})"
                )
                urgency_score += 2

        should = len(reasons) > 0
        if urgency_score >= 3:
            urgency = "HIGH"
        elif urgency_score >= 1:
            urgency = "MEDIUM"
        else:
            urgency = "LOW"

        decision = {
            "should_retrain":   should,
            "reasons":          reasons,
            "urgency":          urgency,
            "max_psi":          round(max_psi, 6),
            "days_since_train": days_since_train,
            "current_auc":      current_auc,
        }
        if should:
            logger.warning(
                "P18 RetrainingGovernance: RETRAIN RECOMMENDED (%s). Reasons: %s",
                urgency, "; ".join(reasons),
            )
        else:
            logger.info("P18 RetrainingGovernance: model is HEALTHY -- no retrain needed.")
        return decision

    def print_decision(self, decision: Dict[str, Any]) -> None:
        header = "RETRAIN RECOMMENDED" if decision["should_retrain"] else "Model healthy"
        print("=" * 60)
        print(f"  P18 RETRAINING GOVERNANCE: {header}  [{decision['urgency']}]")
        print("=" * 60)
        if decision["reasons"]:
            for r in decision["reasons"]:
                print(f"  - {r}")
        else:
            print("  No retraining signals triggered.")
        print(f"  max_psi={decision['max_psi']:.4f}  days={decision['days_since_train']}"
              f"  auc={decision['current_auc']}")
        print("=" * 60)
