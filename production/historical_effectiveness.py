"""
historical_effectiveness.py
============================
P39 — Monthly Retrospective Report (IN-SAMPLE WARNING)

Train model on all data up to today, then simulate picks for each trading
day of the previous year using point-in-time features.  Aggregate results
into monthly and yearly tables and append today's live picks.

WARNING: Because the model is trained on the full history including the test
period, these results are IN-SAMPLE and will be optimistic.  They are useful
for detecting gross failures, not for claiming live performance.

Usage:
    python production/historical_effectiveness.py
    python production/historical_effectiveness.py --start 2025-01-01 --end 2025-12-31
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

FRICTION_PCT = 0.00469   # 0.469% round-trip (BSE equity delivery)
MAX_PICKS_PER_DAY = 20
THRESHOLD = 0.62
RESULTS_DIR = _ROOT / "stock_picker_data" / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wilson_ci(wins: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0
    try:
        from scipy.stats import proportion_confint
        lo, hi = proportion_confint(wins, n, alpha=0.05, method="wilson")
        return round(lo, 4), round(hi, 4)
    except ImportError:
        p = wins / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        import math
        margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        return round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class HistoricalEffectivenessReport:
    """
    P39 — builds a month-by-month retrospective report.

    Parameters
    ----------
    model_path : str | Path, optional
        Path to LightGBM model file.  Loaded from default location if None.
    feature_cols : list[str], optional
        Canonical feature column list.  Loaded from feature_cols.json if None.
    calibrator : object, optional
        Fitted calibrator with .predict() method.  Loaded from disk if None.
    base_dir : str
        Root data directory (default 'stock_picker_data').
    """

    def __init__(
        self,
        model_path=None,
        feature_cols=None,
        calibrator=None,
        base_dir: str = "stock_picker_data",
    ):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"

        # Model
        self._model = None
        self._model_path = Path(model_path) if model_path else self.models_dir / "lgbm_model.txt"

        # Feature list
        if feature_cols is not None:
            self._feature_cols = list(feature_cols)
        else:
            fc_path = self.models_dir / "feature_cols.json"
            self._feature_cols = json.load(open(fc_path)) if fc_path.exists() else []

        # Calibrator
        self._calibrator = calibrator

        # Storage
        self.trades_df: Optional[pd.DataFrame] = None
        self.monthly_df: Optional[pd.DataFrame] = None
        self.yearly_summary: Optional[dict] = None
        self.today_picks: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Step A — load/train model
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=str(self._model_path))
            logger.info("P39: model loaded from %s.", self._model_path)
        except Exception as exc:
            raise RuntimeError(f"P39: cannot load model from {self._model_path}: {exc}")

    def _ensure_calibrator(self):
        if self._calibrator is not None:
            return
        cal_path = self.models_dir / "calibrator.pkl"
        if cal_path.exists():
            import pickle
            self._calibrator = pickle.load(open(cal_path, "rb"))
            logger.info("P39: calibrator loaded from %s.", cal_path)
        else:
            logger.warning("P39: no calibrator found — raw probabilities used.")

    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        import lightgbm as lgb
        X_aligned = X.reindex(columns=self._feature_cols, fill_value=0).fillna(0)
        probs = self._model.predict(X_aligned.values)
        if self._calibrator is not None:
            try:
                probs = self._calibrator.predict(probs)
            except Exception:
                pass
        return np.clip(probs, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Step B — load bhav data
    # ------------------------------------------------------------------

    def _load_bhav(self, start_date: date, end_date: date) -> pd.DataFrame:
        try:
            from production.data_loader import DataLoader
            dl = DataLoader()
            df = dl.load_range(start_date.strftime("%Y-%m-%d"),
                               end_date.strftime("%Y-%m-%d"))
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            logger.warning("P39: DataLoader failed (%s) — checking cache.", exc)

        cache_dir = self.base_dir / "cache" / "bse"
        frames = []
        for f in sorted(cache_dir.glob("bhav_*.csv")):
            try:
                frames.append(pd.read_csv(f, parse_dates=["DATE"]))
            except Exception:
                pass
        if not frames:
            raise RuntimeError("P39: No bhav data found.")
        df = pd.concat(frames, ignore_index=True)
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
        return df[(df["DATE"] >= start_date) & (df["DATE"] <= end_date)].copy()

    # ------------------------------------------------------------------
    # Step C — build features for a single day (point-in-time)
    # ------------------------------------------------------------------

    def _build_features_pit(
        self, bhav_all: pd.DataFrame, as_of_date: date
    ) -> pd.DataFrame:
        """Build features using ONLY data available at close of as_of_date."""
        slice_df = bhav_all[bhav_all["DATE"] <= as_of_date].copy()
        try:
            from momentum_features import prepare_features_all
            feat = prepare_features_all(slice_df)
            feat["DATE"] = pd.to_datetime(feat["DATE"]).dt.date
            return feat[feat["DATE"] == as_of_date].copy()
        except Exception as exc:
            logger.debug("P39: feature build failed for %s: %s", as_of_date, exc)
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Step D — compute 5-day forward return
    # ------------------------------------------------------------------

    def _fwd_return(
        self, bhav: pd.DataFrame, sc_code, buy_date: date, n: int = 5
    ) -> float:
        fut = bhav[(bhav["SC_CODE"] == sc_code) & (bhav["DATE"] > buy_date)].sort_values("DATE")
        if len(fut) < n:
            return float("nan")
        exit_price = float(fut.iloc[n - 1]["Close"])
        buy_price = float(bhav[(bhav["SC_CODE"] == sc_code) & (bhav["DATE"] == buy_date)]["Close"].iloc[0])
        return (exit_price / buy_price) - 1 if buy_price > 0 else float("nan")

    # ------------------------------------------------------------------
    # Method 1 — run()
    # ------------------------------------------------------------------

    def run(self, start_date: str = None, end_date: str = None):
        today = date.today()

        if end_date is None:
            end_dt = today - timedelta(days=1)
        else:
            end_dt = date.fromisoformat(end_date)

        if start_date is None:
            start_dt = today - timedelta(days=365)
        else:
            start_dt = date.fromisoformat(start_date)

        print(f"Training model on data up to {today} …")
        self._ensure_model()
        self._ensure_calibrator()

        # Load all bhav data for the window plus extra lookback for features
        lookback_start = start_dt - timedelta(days=365)
        print(f"Loading BSE bhav data {lookback_start} → {end_dt} …")
        bhav_all = self._load_bhav(lookback_start, end_dt)
        bhav_all["DATE"] = pd.to_datetime(bhav_all["DATE"]).dt.date
        bhav_all = bhav_all.sort_values(["SC_CODE", "DATE"]).reset_index(drop=True)

        # Get EMA-based regime for date range
        try:
            from production.regime_filter import IndexRegimeFilter
            _ema = IndexRegimeFilter()
        except Exception:
            _ema = None

        trade_dates = sorted([d for d in bhav_all["DATE"].unique()
                              if start_dt <= d <= end_dt])
        n_days = len(trade_dates)
        print(f"Simulating {n_days} trading days ({start_dt} → {end_dt}) …")

        records = []
        for i, d in enumerate(trade_dates):
            if i % 20 == 0:
                logger.info("P39: processing day %d/%d (%s)", i + 1, n_days, d)

            # Regime check (simplified: use EMA from available nifty data)
            regime = "BULL"   # default; full regime requires yfinance
            if _ema is not None:
                try:
                    nifty_slice = bhav_all[(bhav_all["SC_CODE"] == "999920") &
                                           (bhav_all["DATE"] <= d)]
                    if len(nifty_slice) >= 50:
                        nifty_slice = _ema.compute_emas(nifty_slice.rename(columns={"Close": "Close"}))
                        regime = _ema.get_regime(nifty_slice)
                except Exception:
                    regime = "BULL"

            if regime == "BEAR":
                continue

            # Point-in-time features
            feat_df = self._build_features_pit(bhav_all, d)
            if feat_df.empty:
                continue

            # Predict
            try:
                probs = self._predict(feat_df)
            except Exception as exc:
                logger.debug("P39: predict failed for %s: %s", d, exc)
                continue

            feat_df = feat_df.copy()
            feat_df["Probability"] = probs
            picks = feat_df[feat_df["Probability"] >= THRESHOLD].nlargest(MAX_PICKS_PER_DAY, "Probability")

            if picks.empty:
                continue

            for _, row in picks.iterrows():
                sc = row.get("SC_CODE", "")
                gross = self._fwd_return(bhav_all, sc, d)
                if np.isnan(gross):
                    continue
                net = gross - FRICTION_PCT
                records.append({
                    "date":         d,
                    "SC_CODE":      sc,
                    "SC_NAME":      row.get("SC_NAME", ""),
                    "probability":  round(float(row["Probability"]), 4),
                    "gross_return": round(gross, 6),
                    "net_return":   round(net, 6),
                    "positive_gross": int(gross > 0),
                    "positive_net":   int(net > 0),
                })

        self.trades_df = pd.DataFrame(records)
        if self.trades_df.empty:
            logger.warning("P39: no trades generated for the period.")
            return

        self.trades_df["date"] = pd.to_datetime(self.trades_df["date"])

        # Today's picks
        try:
            from production.trade_orchestrator import TradeOrchestrator
            result = TradeOrchestrator().run_daily()
            self.today_picks = result.get("picks", pd.DataFrame())
        except Exception as exc:
            logger.warning("P39: could not generate today's picks: %s", exc)
            self.today_picks = pd.DataFrame()

        self.monthly_df, self.yearly_summary = self.aggregate(self.trades_df)

    # ------------------------------------------------------------------
    # Method 2 — aggregate()
    # ------------------------------------------------------------------

    def aggregate(self, trades_df: pd.DataFrame):
        trades_df = trades_df.copy()
        trades_df["date"] = pd.to_datetime(trades_df["date"])
        trades_df["month"] = trades_df["date"].dt.to_period("M")

        monthly_rows = []
        for month, grp in trades_df.groupby("month"):
            n = len(grp)
            wins_gross = int(grp["positive_gross"].sum())
            wins_net = int(grp["positive_net"].sum())
            avg_net = float(grp["net_return"].mean())
            cum_net = float(grp["net_return"].sum())
            wr_gross = wins_gross / n
            wr_net = wins_net / n
            ci_lo, ci_hi = _wilson_ci(wins_net, n)
            monthly_rows.append({
                "month": str(month),
                "n_picks": n,
                "gross_wr": round(wr_gross, 4),
                "net_wr": round(wr_net, 4),
                "avg_net_return": round(avg_net, 4),
                "cum_net_return": round(cum_net, 4),
                "wr_ci_lo": ci_lo,
                "wr_ci_hi": ci_hi,
            })

        monthly_df = pd.DataFrame(monthly_rows)

        total_n = len(trades_df)
        total_wins_gross = int(trades_df["positive_gross"].sum())
        total_wins_net = int(trades_df["positive_net"].sum())
        ci_lo, ci_hi = _wilson_ci(total_wins_net, total_n)
        yearly_summary = {
            "n_picks": total_n,
            "gross_wr": round(total_wins_gross / total_n, 4) if total_n else 0,
            "net_wr": round(total_wins_net / total_n, 4) if total_n else 0,
            "avg_net_return": round(float(trades_df["net_return"].mean()), 4) if total_n else 0,
            "median_net_return": round(float(trades_df["net_return"].median()), 4) if total_n else 0,
            "cum_net_return": round(float(trades_df["net_return"].sum()), 4) if total_n else 0,
            "wr_ci_lo": ci_lo,
            "wr_ci_hi": ci_hi,
        }
        return monthly_df, yearly_summary

    # ------------------------------------------------------------------
    # Method 3 — print_report()
    # ------------------------------------------------------------------

    def print_report(self, monthly_df=None, yearly_summary=None, today_picks=None):
        monthly_df = monthly_df if monthly_df is not None else self.monthly_df
        yearly_summary = yearly_summary if yearly_summary is not None else self.yearly_summary
        today_picks = today_picks if today_picks is not None else self.today_picks

        sep = "=" * 62
        print(sep)
        print("  MODEL EFFECTIVENESS RETROSPECTIVE [IN-SAMPLE WARNING]")
        print(f"  Model trained on data up to: {date.today()}")
        if monthly_df is not None and not monthly_df.empty:
            print(f"  Test period: {monthly_df['month'].iloc[0]} -> {monthly_df['month'].iloc[-1]}")
        print("  WARNING: Model was trained on this data. Results are optimistic.")
        print(sep)

        if monthly_df is None or monthly_df.empty:
            print("  No trades generated.")
            return

        print()
        print("  MONTHLY BREAKDOWN")
        hdr = f"  {'Month':<10} {'Picks':>5} {'GrossWR':>8} {'NetWR':>7} {'AvgNet':>7} {'CumNet':>7} {'WR_CI':>12}"
        print(hdr)
        print("  " + "-" * 58)
        for _, row in monthly_df.iterrows():
            ci = f"[{row['wr_ci_lo']:.2f},{row['wr_ci_hi']:.2f}]"
            print(
                f"  {row['month']:<10} {row['n_picks']:>5} "
                f"{row['gross_wr']*100:>7.1f}% {row['net_wr']*100:>6.1f}% "
                f"{row['avg_net_return']*100:>6.2f}% {row['cum_net_return']*100:>6.1f}% "
                f"{ci:>12}"
            )

        s = yearly_summary
        print()
        print("  YEARLY SUMMARY")
        print(f"  Total picks: {s['n_picks']} | Gross WR: {s['gross_wr']*100:.1f}% | "
              f"Net WR: {s['net_wr']*100:.1f}% | Avg net: {s['avg_net_return']*100:+.2f}%")
        print(f"  Median net: {s['median_net_return']*100:+.2f}% | "
              f"Cumulative net (equal weight): {s['cum_net_return']*100:+.1f}%")
        print(f"  Net WR 95% CI: [{s['wr_ci_lo']*100:.1f}%, {s['wr_ci_hi']*100:.1f}%]")

        print()
        print("  TODAY'S LIVE PICKS")
        print("  " + "-" * 40)
        if today_picks is not None and not today_picks.empty:
            for _, row in today_picks.iterrows():
                print(f"  {row.get('SC_CODE','?'):>8}  {row.get('SC_NAME','?'):30}  "
                      f"prob={row.get('Probability', 0):.4f}")
        else:
            print("  No picks today (SKIP_DAY or regime gate).")
        print(sep)

    # ------------------------------------------------------------------
    # Method 4 — save()
    # ------------------------------------------------------------------

    def save(self, report_dir: str = None):
        out_dir = Path(report_dir) if report_dir else RESULTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d")

        if self.trades_df is not None and not self.trades_df.empty:
            p1 = out_dir / f"effectiveness_{stamp}.csv"
            self.trades_df.to_csv(p1, index=False)
            logger.info("P39: trade log -> %s", p1)

        if self.monthly_df is not None and not self.monthly_df.empty:
            p2 = out_dir / f"effectiveness_monthly_{stamp}.csv"
            self.monthly_df.to_csv(p2, index=False)
            logger.info("P39: monthly aggregates -> %s", p2)

        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "in_sample_warning": True,
            "yearly_summary": self.yearly_summary,
            "monthly": self.monthly_df.to_dict(orient="records") if self.monthly_df is not None else [],
        }
        p3 = out_dir / f"effectiveness_report_{stamp}.json"
        with open(p3, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        logger.info("P39: JSON report -> %s", p3)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s")
    parser = argparse.ArgumentParser(description="P39 Historical Effectiveness Report")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end",   default=None)
    parser.add_argument("--base-dir", default="stock_picker_data")
    args = parser.parse_args()

    report = HistoricalEffectivenessReport(base_dir=args.base_dir)
    report.run(start_date=args.start, end_date=args.end)
    report.print_report()
    report.save()


if __name__ == "__main__":
    main()
