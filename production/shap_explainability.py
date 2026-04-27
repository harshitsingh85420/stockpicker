"""
SHAP Explainability Module — Gap 15
Provides SHAP values for every prediction and logs why each stock was picked,
including top contributing features.
"""

import os
import json
import pickle
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("shap not installed — install with: pip install shap")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Core explainer
# ---------------------------------------------------------------------------

class SHAPExplainer:
    """
    Wraps a trained LightGBM / sklearn model with SHAP explanation capabilities.
    Computes per-prediction feature attributions and logs human-readable rationale.
    """

    def __init__(
        self,
        model,
        feature_names: list,
        explainer_type: str = "tree",
        log_dir: str = "stock_picker_data/shap_logs",
    ):
        """
        Args:
            model: Trained LightGBM Booster or sklearn-compatible model.
            feature_names: Ordered list matching model input columns.
            explainer_type: 'tree' (LightGBM/XGBoost) or 'linear' or 'kernel'.
            log_dir: Directory for JSON explanation logs.
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._explainer = None
        self._background_data = None

    # ------------------------------------------------------------------
    def _build_explainer(self, background_data: Optional[np.ndarray] = None):
        """Lazily build the SHAP explainer on first use."""
        if not SHAP_AVAILABLE:
            return

        if self.explainer_type == "tree":
            self._explainer = shap.TreeExplainer(
                self.model,
                feature_perturbation="interventional",
            )
        elif self.explainer_type == "linear":
            self._explainer = shap.LinearExplainer(self.model, background_data)
        else:
            if background_data is None:
                raise ValueError("KernelExplainer requires background_data.")
            self._background_data = background_data
            self._explainer = shap.KernelExplainer(
                self.model.predict, background_data[:100]
            )

    # ------------------------------------------------------------------
    def compute_shap_values(
        self, X: pd.DataFrame, background_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute SHAP values for the given feature matrix.

        Returns:
            shap_values: ndarray of shape (n_samples, n_features).
                         Positive = pushes prediction higher (more bullish).
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available — returning zero attributions.")
            return np.zeros((len(X), len(self.feature_names)))

        if self._explainer is None:
            self._build_explainer(background_data)

        X_arr = X[self.feature_names].fillna(0).values

        sv = self._explainer.shap_values(X_arr)
        # For binary classifiers some backends return a list [class0, class1]
        if isinstance(sv, list):
            sv = sv[1]
        return sv  # shape (n_samples, n_features)

    # ------------------------------------------------------------------
    def explain_prediction(
        self,
        X_row: pd.Series,
        predicted_prob: float,
        sc_code: str,
        sc_name: str,
        top_n: int = 8,
        background_data: Optional[np.ndarray] = None,
    ) -> dict:
        """
        Generate a human-readable explanation for a single stock pick.

        Returns:
            dict with keys: sc_code, sc_name, predicted_prob,
                            top_features (list of {feature, value, shap_value, direction}),
                            rationale (string summary)
        """
        X_df = pd.DataFrame([X_row])

        sv = self.compute_shap_values(X_df, background_data)
        shap_row = sv[0]  # (n_features,)

        # Pair feature names with shap values
        pairs = sorted(
            zip(self.feature_names, shap_row),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:top_n]

        top_features = []
        for feat, sv_val in pairs:
            raw_val = X_row.get(feat, float("nan"))
            top_features.append(
                {
                    "feature": feat,
                    "value": round(float(raw_val), 4) if not np.isnan(float(raw_val)) else None,
                    "shap_value": round(float(sv_val), 5),
                    "direction": "bullish" if sv_val > 0 else "bearish",
                }
            )

        # Build text rationale
        bullish = [f["feature"] for f in top_features if f["direction"] == "bullish"]
        bearish = [f["feature"] for f in top_features if f["direction"] == "bearish"]
        rationale = (
            f"{sc_name} ({sc_code}) picked with probability {predicted_prob:.2%}. "
        )
        if bullish:
            rationale += f"Key upside drivers: {', '.join(bullish[:3])}. "
        if bearish:
            rationale += f"Risk factors: {', '.join(bearish[:2])}."

        return {
            "sc_code": sc_code,
            "sc_name": sc_name,
            "predicted_prob": round(predicted_prob, 5),
            "top_features": top_features,
            "rationale": rationale,
        }

    # ------------------------------------------------------------------
    def explain_portfolio(
        self,
        picks_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        top_n: int = 8,
        background_data: Optional[np.ndarray] = None,
    ) -> list:
        """
        Explain every stock in picks_df.

        Args:
            picks_df: DataFrame with SC_CODE, SC_NAME, Probability columns.
            feature_df: Full feature DataFrame indexed or filterable by SC_CODE + latest date.

        Returns:
            List of explanation dicts (one per pick).
        """
        explanations = []
        for _, row in picks_df.iterrows():
            sc_code = row.get("SC_CODE", row.get("sc_code", ""))
            sc_name = row.get("SC_NAME", row.get("sc_name", ""))
            prob = row.get("Probability", row.get("probability", 0.0))

            # Get the feature row for this stock
            mask = feature_df["SC_CODE"] == sc_code if "SC_CODE" in feature_df else (
                feature_df.index == sc_code
            )
            feat_rows = feature_df[mask]
            if feat_rows.empty:
                logger.debug("No feature row found for %s — skipping SHAP.", sc_code)
                continue

            feat_row = feat_rows.iloc[-1]
            exp = self.explain_prediction(
                feat_row, prob, sc_code, sc_name,
                top_n=top_n, background_data=background_data
            )
            explanations.append(exp)

        return explanations

    # ------------------------------------------------------------------
    def log_explanations(self, explanations: list, date: str = None) -> Path:
        """
        Persist explanations to a JSON log file.

        Returns:
            Path to the written log file.
        """
        date_str = date or datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"shap_{date_str}.json"

        payload = {
            "date": date_str,
            "generated_at": datetime.now().isoformat(),
            "n_picks": len(explanations),
            "picks": explanations,
        }

        with open(log_file, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)

        logger.info("SHAP explanations saved -> %s", log_file)
        return log_file

    # ------------------------------------------------------------------
    def get_global_importance(
        self, X: pd.DataFrame, background_data: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Compute mean absolute SHAP value (global feature importance).

        Returns:
            DataFrame with columns: feature, mean_abs_shap — sorted descending.
        """
        sv = self.compute_shap_values(X, background_data)
        mean_abs = np.abs(sv).mean(axis=0)
        result = pd.DataFrame(
            {"feature": self.feature_names, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return result

    # ------------------------------------------------------------------
    def plot_summary(
        self,
        X: pd.DataFrame,
        save_path: str = None,
        background_data: Optional[np.ndarray] = None,
    ):
        """Generate a SHAP beeswarm summary plot (saved to file if path given)."""
        if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            logger.warning("SHAP or matplotlib not available — skipping plot.")
            return

        sv = self.compute_shap_values(X, background_data)
        X_plot = X[self.feature_names].fillna(0)

        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(sv, X_plot, show=False, max_display=20)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=120)
            logger.info("SHAP summary plot saved -> %s", save_path)
        else:
            plt.tight_layout()

        plt.close(fig)

    # ------------------------------------------------------------------
    def save(self, path: str):
        with open(path, "wb") as fh:
            pickle.dump({"explainer": self._explainer, "feature_names": self.feature_names}, fh)
        logger.info("SHAPExplainer saved -> %s", path)

    def load(self, path: str):
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._explainer = data["explainer"]
        self.feature_names = data["feature_names"]
        logger.info("SHAPExplainer loaded from %s", path)


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def explain_picks(
    model,
    feature_names: list,
    picks_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    date: str = None,
    log_dir: str = "stock_picker_data/shap_logs",
    top_n: int = 8,
) -> list:
    """
    One-call convenience: build explainer, explain all picks, log to disk.

    Returns:
        List of explanation dicts.
    """
    explainer = SHAPExplainer(model, feature_names, log_dir=log_dir)
    explanations = explainer.explain_portfolio(picks_df, feature_df, top_n=top_n)
    if explanations:
        explainer.log_explanations(explanations, date)
    return explanations


def explain_batch(
    model,
    feature_names: list,
    picks_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    date: str = None,
    log_dir: str = "stock_picker_data/shap_logs",
    top_n: int = 6,
) -> pd.DataFrame:
    """
    P35: Return a structured DataFrame of SHAP columns for every pick.

    Columns (indexed by SC_CODE):
        shap_top1_feature, shap_top1_value,
        shap_top2_feature, shap_top2_value,
        shap_top3_feature, shap_top3_value,
        shap_dominated   (bool — top feature SHAP > 3x second, concentration flag)

    Falls back to empty strings / False when SHAP is unavailable.
    """
    empty_row = {
        "shap_top1_feature": "", "shap_top1_value": 0.0,
        "shap_top2_feature": "", "shap_top2_value": 0.0,
        "shap_top3_feature": "", "shap_top3_value": 0.0,
        "shap_dominated": False,
    }
    codes = picks_df["SC_CODE"].tolist() if "SC_CODE" in picks_df.columns else []
    if not codes:
        return pd.DataFrame()

    exps = explain_picks(model, feature_names, picks_df, feature_df, date=date,
                         log_dir=log_dir, top_n=top_n)

    rows = []
    exp_map = {e["sc_code"]: e for e in exps}
    for code in codes:
        exp = exp_map.get(code)
        if exp is None:
            rows.append({"SC_CODE": code, **empty_row})
            continue
        feats = exp.get("top_features", [])
        row = {"SC_CODE": code}
        for i in range(1, 4):
            if i - 1 < len(feats):
                row[f"shap_top{i}_feature"] = feats[i - 1]["feature"]
                row[f"shap_top{i}_value"]   = feats[i - 1]["shap_value"]
            else:
                row[f"shap_top{i}_feature"] = ""
                row[f"shap_top{i}_value"]   = 0.0
        sv1 = abs(row["shap_top1_value"])
        sv2 = abs(row["shap_top2_value"])
        row["shap_dominated"] = bool(sv2 > 0 and sv1 > 3 * sv2)
        rows.append(row)

    return pd.DataFrame(rows).set_index("SC_CODE")


def print_explanation(exp: dict):
    """Pretty-print a single stock explanation to stdout."""
    print(f"\n{'='*60}")
    print(f"  {exp['sc_name']} ({exp['sc_code']})")
    print(f"  Predicted probability: {exp['predicted_prob']:.2%}")
    print(f"  {exp['rationale']}")
    print(f"\n  Top contributing features:")
    for f in exp["top_features"]:
        arrow = "▲" if f["direction"] == "bullish" else "▼"
        val_str = f"{f['value']:.4f}" if f["value"] is not None else "N/A"
        print(
            f"    {arrow} {f['feature']:30s}  val={val_str:>10}  "
            f"SHAP={f['shap_value']:+.4f}"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# __main__ smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import lightgbm as lgb

    logging.basicConfig(level=logging.INFO)
    print("Testing SHAPExplainer with synthetic data...")

    np.random.seed(42)
    n, p = 500, 20
    feature_names = [f"feat_{i}" for i in range(p)]
    X = pd.DataFrame(np.random.randn(n, p), columns=feature_names)
    y = (X["feat_0"] + X["feat_1"] > 0).astype(int)

    dtrain = lgb.Dataset(X, label=y)
    params = {"objective": "binary", "verbose": -1, "num_leaves": 15}
    model = lgb.train(params, dtrain, num_boost_round=50)

    # Build explainer
    explainer = SHAPExplainer(model, feature_names)

    # Global importance
    importance = explainer.get_global_importance(X)
    print("\nGlobal feature importance (top 5):")
    print(importance.head(5).to_string(index=False))

    # Single prediction
    probs = model.predict(X.values)
    picks_df = pd.DataFrame({
        "SC_CODE": ["RELIANCE", "TCS", "INFY"],
        "SC_NAME": ["Reliance", "TCS", "Infosys"],
        "Probability": probs[:3],
    })
    # Pretend feature_df has SC_CODE column
    feat_df = X.copy()
    feat_df.insert(0, "SC_CODE", picks_df["SC_CODE"].tolist() + ["OTHER"] * (n - 3))

    explanations = explainer.explain_portfolio(picks_df, feat_df, top_n=5)
    for exp in explanations:
        print_explanation(exp)

    print("\n* SHAPExplainer smoke test passed.")
