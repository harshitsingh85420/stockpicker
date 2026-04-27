"""
production/benchmark.py -- P17: Benchmark & ablation tests.

Two capabilities:

1. BenchmarkComparator
   Compares walk-forward strategy returns against a simple buy-and-hold of
   Nifty 50.  Metrics: CAGR, Sharpe, max-drawdown, win-rate.

2. AblationTester
   Removes one feature group at a time and measures CV AUC impact.  Identifies
   which features drive model performance and which are dead weight.

Usage
-----
from production.benchmark import BenchmarkComparator, AblationTester

# --- Benchmark ---
cmp = BenchmarkComparator()
report = cmp.compare(detail_df, bhav_df, start_date, end_date)
cmp.print_report(report)

# --- Ablation ---
abl = AblationTester()
results = abl.run(feat_df, feature_cols, label_col='Label_fwd5_positive')
abl.print_results(results)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature groups for ablation
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "trend":    ["EMA20", "EMA50", "EMA200", "EMA20_Slope5", "EMA200_Slope", "MA_Health", "OverEMA20"],
    "momentum": ["RET21D", "RET63D", "RS_Composite", "DistTo20", "DistTo63", "DistTo52W"],
    "volatility": ["ATR14", "ATRpct", "BBWidth", "BBWidthPctl", "W_BBWidth", "W_BBWidthPctl"],
    "breakout": ["Break20_Today", "Break63_Today", "Hit52WH_Today", "RangePos20"],
    "volume":   ["VolMult", "UD_Vol_Ratio10"],
    "oscillators": ["RSI14", "ADX14", "+DI14", "-DI14", "ADX14_chg3"],
    "fracdiff": ["FracDiff_Close", "FracDiff_LogClose"],
    "weekly":   ["W_TrendOK"],
}


# ===========================================================================
# 1. BenchmarkComparator
# ===========================================================================

class BenchmarkComparator:
    """
    Compare walk-forward strategy returns against Nifty 50 buy-and-hold.

    Parameters
    ----------
    trading_days_per_year : int
        Used for annualisation (default 252).
    """

    def __init__(self, trading_days_per_year: int = 252) -> None:
        self.tpy = trading_days_per_year

    # ------------------------------------------------------------------

    def compare(
        self,
        detail_df: pd.DataFrame,
        nifty_df: Optional[pd.DataFrame] = None,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Compute strategy vs. benchmark metrics.

        Parameters
        ----------
        detail_df  : Walk-forward detail DataFrame.  Must have columns:
                     pred_date, gross_return (fractional, per-pick).
        nifty_df   : Nifty 50 data with DATE and Close columns.  If None,
                     fetched via yfinance (^NSEI).
        start_date, end_date : Date range to restrict comparison.

        Returns
        -------
        dict with keys:
            strategy_* and benchmark_* metrics + alpha, beta, information_ratio.
        """
        if detail_df.empty:
            return {"error": "Empty detail_df -- no picks to benchmark."}

        detail_df = detail_df.copy()
        detail_df["pred_date"] = pd.to_datetime(detail_df["pred_date"])

        if start_date:
            detail_df = detail_df[detail_df["pred_date"] >= pd.Timestamp(start_date)]
        if end_date:
            detail_df = detail_df[detail_df["pred_date"] <= pd.Timestamp(end_date)]

        if detail_df.empty:
            return {"error": "No picks in date range."}

        # --- Strategy daily returns: mean of all picks each day ----------
        daily_strat = (
            detail_df.groupby("pred_date")["gross_return"]
            .mean()
            .sort_index()
        )

        # --- Nifty daily returns -----------------------------------------
        nifty_df = self._get_nifty(nifty_df, daily_strat.index[0], daily_strat.index[-1])
        if nifty_df is not None and not nifty_df.empty:
            nifty_ret = nifty_df.set_index("DATE")["Close"].pct_change().dropna()
            nifty_ret.index = pd.to_datetime(nifty_ret.index)
            # align on common dates
            common = daily_strat.index.intersection(nifty_ret.index)
            strat_aligned = daily_strat.reindex(common)
            bench_aligned = nifty_ret.reindex(common)
        else:
            strat_aligned = daily_strat
            bench_aligned = pd.Series(np.zeros(len(daily_strat)), index=daily_strat.index)

        strat_metrics  = self._compute_metrics(strat_aligned,  "strategy")
        bench_metrics  = self._compute_metrics(bench_aligned, "benchmark")

        # --- Alpha / Beta -----------------------------------------------
        alpha, beta = self._alpha_beta(strat_aligned.values, bench_aligned.values)
        te = strat_aligned - bench_aligned
        ir = float(te.mean() / te.std()) * np.sqrt(self.tpy) if te.std() > 0 else 0.0

        report = {
            **strat_metrics,
            **bench_metrics,
            "alpha_annualised": round(alpha * self.tpy, 6),
            "beta":             round(beta, 4),
            "information_ratio": round(ir, 4),
            "n_strategy_days":  len(daily_strat),
            "n_common_days":    len(strat_aligned),
        }
        logger.info("P17 BenchmarkComparator: %s", {k: v for k, v in report.items() if isinstance(v, float)})
        return report

    def print_report(self, report: Dict[str, Any]) -> None:
        if "error" in report:
            print(f"Benchmark error: {report['error']}")
            return
        w = 32
        print("=" * 60)
        print("  BENCHMARK COMPARISON REPORT  (P17)")
        print("=" * 60)
        for key in sorted(report):
            if isinstance(report[key], float):
                print(f"  {key:<{w}}: {report[key]:.4f}")
            else:
                print(f"  {key:<{w}}: {report[key]}")
        print("=" * 60)

    # ------------------------------------------------------------------

    def _get_nifty(self, nifty_df, start, end) -> Optional[pd.DataFrame]:
        if nifty_df is not None:
            return nifty_df
        try:
            import yfinance as yf
            raw = yf.Ticker("^NSEI").history(
                start=str(start.date() if hasattr(start, "date") else start),
                end=str((end + pd.Timedelta(days=5)).date() if hasattr(end, "date") else end),
                interval="1d",
            ).reset_index()
            date_col = "Date" if "Date" in raw.columns else "Datetime"
            df = pd.DataFrame({
                "DATE":  pd.to_datetime(raw[date_col]).dt.normalize(),
                "Close": raw["Close"].astype(float),
            })
            return df.dropna().sort_values("DATE").reset_index(drop=True)
        except Exception as exc:
            logger.warning("P17: Nifty fetch failed (%s) -- using zero benchmark.", exc)
            return None

    def _compute_metrics(self, returns: pd.Series, prefix: str) -> Dict[str, Any]:
        r = returns.dropna()
        if len(r) == 0:
            return {f"{prefix}_cagr": 0.0, f"{prefix}_sharpe": 0.0,
                    f"{prefix}_max_drawdown": 0.0, f"{prefix}_win_rate": 0.0}
        cum = (1 + r).cumprod()
        n_years = len(r) / self.tpy
        cagr    = float(cum.iloc[-1] ** (1 / max(n_years, 0.01)) - 1)
        sharpe  = float(r.mean() / r.std()) * np.sqrt(self.tpy) if r.std() > 0 else 0.0
        roll_max = cum.cummax()
        dd      = (cum / roll_max - 1)
        mdd     = float(dd.min())
        win_rate = float((r > 0).mean())
        return {
            f"{prefix}_cagr":         round(cagr, 6),
            f"{prefix}_sharpe":       round(sharpe, 4),
            f"{prefix}_max_drawdown": round(mdd, 4),
            f"{prefix}_win_rate":     round(win_rate, 4),
        }

    @staticmethod
    def _alpha_beta(strat: np.ndarray, bench: np.ndarray) -> Tuple[float, float]:
        if len(strat) < 2 or bench.std() == 0:
            return 0.0, 1.0
        beta  = float(np.cov(strat, bench)[0, 1] / np.var(bench))
        alpha = float(strat.mean() - beta * bench.mean())
        return alpha, beta


# ===========================================================================
# 2. AblationTester
# ===========================================================================

class AblationTester:
    """
    Remove one feature group at a time and measure CV AUC impact.

    Parameters
    ----------
    n_cv_splits : int
        TimeSeriesSplit folds (default 3 -- fast).
    n_boost_rounds : int
        LightGBM boost rounds per fold (default 100 -- fast).
    """

    def __init__(self, n_cv_splits: int = 3, n_boost_rounds: int = 100) -> None:
        self.n_cv_splits    = n_cv_splits
        self.n_boost_rounds = n_boost_rounds

    # ------------------------------------------------------------------

    def run(
        self,
        feat_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "Label_fwd5_positive",
        groups: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run ablation: full model AUC vs. AUC with each feature group dropped.

        Parameters
        ----------
        feat_df      : Feature DataFrame (chronological, labelled).
        feature_cols : All feature columns to use.
        label_col    : Binary target column.
        groups       : Dict of {group_name: [feature_cols_to_drop]}.
                       Defaults to FEATURE_GROUPS.

        Returns
        -------
        List of dicts, each with keys:
            group_name, n_features_dropped, full_auc, ablated_auc, auc_delta.
        """
        if groups is None:
            groups = FEATURE_GROUPS

        df = feat_df.dropna(subset=[label_col]).copy()
        X_full = df[feature_cols].fillna(0).values
        y      = df[label_col].astype(int).values

        full_auc = self._cv_auc(X_full, y, feature_cols)
        logger.info("P17 Ablation: full model CV AUC = %.4f", full_auc)

        results = [{"group_name": "FULL", "n_features_dropped": 0,
                    "full_auc": round(full_auc, 6), "ablated_auc": round(full_auc, 6),
                    "auc_delta": 0.0}]

        for group_name, drop_cols in groups.items():
            ablated_cols = [c for c in feature_cols if c not in drop_cols]
            n_dropped = len([c for c in drop_cols if c in feature_cols])
            if n_dropped == 0:
                logger.debug("P17 Ablation: group '%s' has no columns in feat_df -- skipping.", group_name)
                continue
            if len(ablated_cols) < 3:
                logger.debug("P17 Ablation: group '%s' leaves < 3 features -- skipping.", group_name)
                continue

            X_abl = df[ablated_cols].fillna(0).values
            abl_auc = self._cv_auc(X_abl, y, ablated_cols)
            delta = round(full_auc - abl_auc, 6)

            results.append({
                "group_name":        group_name,
                "n_features_dropped": n_dropped,
                "full_auc":          round(full_auc, 6),
                "ablated_auc":       round(abl_auc, 6),
                "auc_delta":         delta,
            })
            logger.info(
                "P17 Ablation [%s]: full=%.4f  ablated=%.4f  delta=%+.4f",
                group_name, full_auc, abl_auc, delta,
            )

        results.sort(key=lambda x: -x["auc_delta"])
        return results

    def print_results(self, results: List[Dict[str, Any]]) -> None:
        print("=" * 65)
        print("  ABLATION TEST RESULTS  (P17)")
        print(f"  {'Group':<16} {'Dropped':>7} {'Full AUC':>10} {'Ablated AUC':>12} {'Delta':>8}")
        print("=" * 65)
        for r in results:
            flag = "  *" if r["auc_delta"] > 0.005 else ""
            print(
                f"  {r['group_name']:<16} {r['n_features_dropped']:>7} "
                f"{r['full_auc']:>10.4f} {r['ablated_auc']:>12.4f} "
                f"{r['auc_delta']:>+8.4f}{flag}"
            )
        print("=" * 65)
        print("  * = important feature group (delta > 0.005)")

    # ------------------------------------------------------------------

    def _cv_auc(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> float:
        import lightgbm as lgb
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import roc_auc_score

        params = {
            "objective": "binary", "metric": "auc",
            "num_leaves": 31, "learning_rate": 0.05,
            "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 5, "n_jobs": -1, "verbose": -1,
        }
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        aucs = []
        for tr_idx, va_idx in tscv.split(X):
            if len(va_idx) < 5:
                continue
            dtrain = lgb.Dataset(X[tr_idx], label=y[tr_idx])
            dval   = lgb.Dataset(X[va_idx], label=y[va_idx], reference=dtrain)
            m = lgb.train(
                params, dtrain,
                num_boost_round=self.n_boost_rounds,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(20, verbose=False),
                           lgb.log_evaluation(period=0)],
            )
            auc = roc_auc_score(y[va_idx], m.predict(X[va_idx]))
            aucs.append(auc)
        return float(np.mean(aucs)) if aucs else 0.5
