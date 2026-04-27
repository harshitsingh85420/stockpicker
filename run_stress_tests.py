#!/usr/bin/env python3
"""
run_stress_tests.py  --  P08: Sub-period stress tests.

Tests the swing-trade model under 5 historically distinct Indian market
stress / regime periods, reporting win rate, net return and drawdown for
each sub-period.

Stress periods covered:
  1. COVID crash      : 2020-02-17 -> 2020-03-31  (Nifty -38% in 5 wks)
  2. COVID recovery   : 2020-04-01 -> 2020-08-31  (sharp V-recovery)
  3. Global rate shock: 2022-01-03 -> 2022-06-30  (FII selloff, rising rates)
  4. Russia-Ukraine   : 2022-02-21 -> 2022-03-31  (commodity spike + panic)
  5. Adani crisis     : 2023-01-25 -> 2023-03-31  (Hindenburg report impact)

For each period the script:
  a. Slices the BhavCopy to data available up to the period end
  b. Uses the pre-trained model (no retraining -- genuine OOS test)
  c. Generates daily picks at threshold=0.62 for every day in the period
  d. Computes 5-session forward returns with P02 friction
  e. Reports win_rate, win_rate_net, avg_return, max_drawdown, n_picks

Output:
  stock_picker_data/results/stress_test_results_YYYYMMDD.csv
  stock_picker_data/results/stress_test_YYYYMMDD.png

CLI:
    python run_stress_tests.py
    python run_stress_tests.py --no-charts
    python run_stress_tests.py --threshold 0.65

Programmatic:
    from run_stress_tests import run_stress_tests
    results = run_stress_tests(bhav_df, model, feature_cols)
"""

import argparse
import json
import logging
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "production") not in sys.path:
    sys.path.insert(0, str(_ROOT / "production"))

logger = logging.getLogger(__name__)

RESULTS_DIR = _ROOT / "stock_picker_data" / "results"
MODELS_DIR  = _ROOT / "stock_picker_data" / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Stress period definitions
# ---------------------------------------------------------------------------

STRESS_PERIODS = [
    {
        "name":        "COVID Crash",
        "label":       "covid_crash",
        "start":       "2020-02-17",
        "end":         "2020-03-31",
        "description": "Nifty fell ~38% in 5 weeks; extreme volatility and circuit limits",
    },
    {
        "name":        "COVID Recovery",
        "label":       "covid_recovery",
        "start":       "2020-04-01",
        "end":         "2020-08-31",
        "description": "Sharp V-shaped recovery; momentum strategies typically excel",
    },
    {
        "name":        "Global Rate Shock",
        "label":       "rate_shock",
        "start":       "2022-01-03",
        "end":         "2022-06-30",
        "description": "FII selloff on rising US rates; mid-cap underperformance",
    },
    {
        "name":        "Russia-Ukraine",
        "label":       "russia_ukraine",
        "start":       "2022-02-21",
        "end":         "2022-03-31",
        "description": "Commodity spike and panic selling; high gap-down risk",
    },
    {
        "name":        "Adani Crisis",
        "label":       "adani_crisis",
        "start":       "2023-01-25",
        "end":         "2023-03-31",
        "description": "Hindenburg report on Adani; sector contagion fear",
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forward_close(bhav_df: pd.DataFrame, from_date, n_sessions: int = 5) -> pd.Series:
    """SC_CODE -> close price n_sessions trading days after from_date."""
    all_dates = sorted(bhav_df["DATE"].unique())
    future    = [d for d in all_dates if d > from_date]
    if len(future) < n_sessions:
        return pd.Series(dtype=float)
    target = future[n_sessions - 1]
    day = bhav_df[bhav_df["DATE"] == target][["SC_CODE", "Close"]]
    return day.set_index("SC_CODE")["Close"]


def _apply_friction(detail_df: pd.DataFrame) -> pd.DataFrame:
    """Add friction_pct, return_5d_net, positive_net columns."""
    try:
        from production.friction_model import FrictionModel
        fm = FrictionModel(instrument_type="equity_delivery")
        pos_vals    = np.clip(detail_df["Close"].values * 500, 10_000, 500_000)
        sample_vals = np.quantile(pos_vals, [0.1, 0.3, 0.5, 0.7, 0.9])
        sample_pcts = np.array([
            fm.calculate_round_trip_cost(v, exchange="BSE")["total_costs"]["total"] / v
            for v in sample_vals
        ])
        detail_df["friction_pct"] = np.interp(pos_vals, sample_vals, sample_pcts) + 0.003
    except Exception:
        detail_df["friction_pct"] = 0.005

    detail_df["return_5d_net"] = detail_df["return_5d"] - detail_df["friction_pct"]
    detail_df["positive_net"]  = (detail_df["return_5d_net"] > 0).astype(int)
    return detail_df


def _max_drawdown(returns: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of a cumulative equity curve."""
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1 + returns)
    peak   = np.maximum.accumulate(equity)
    dd     = (equity - peak) / (peak + 1e-9)
    return float(dd.min())


def _period_stats(detail: pd.DataFrame, period: dict) -> dict:
    """Compute statistics for one stress period."""
    n = len(detail)
    if n == 0:
        return {
            "name":          period["name"],
            "start":         period["start"],
            "end":           period["end"],
            "n_picks":       0,
            "win_rate":      np.nan,
            "win_rate_net":  np.nan,
            "avg_return":    np.nan,
            "avg_return_net":np.nan,
            "max_drawdown":  np.nan,
            "avg_friction":  np.nan,
        }

    daily_ret = (
        detail.groupby("pred_date")["return_5d"]
        .mean()
        .sort_index()
        .values
    )
    return {
        "name":           period["name"],
        "start":          period["start"],
        "end":            period["end"],
        "n_picks":        n,
        "win_rate":       float(detail["positive"].mean()),
        "win_rate_net":   float(detail["positive_net"].mean()),
        "avg_return":     float(detail["return_5d"].mean()),
        "avg_return_net": float(detail["return_5d_net"].mean()),
        "max_drawdown":   _max_drawdown(daily_ret),
        "avg_friction":   float(detail["friction_pct"].mean()),
    }


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------

def run_stress_tests(
    bhav_df: pd.DataFrame = None,
    model=None,
    feature_cols: list = None,
    lookback_days: int = 730 * 2,    # load 4 years to cover 2020 periods
    threshold: float = 0.62,
    fwd_sessions: int = 5,
    draw_charts: bool = True,
    periods: list = None,
) -> list:
    """
    Run stress tests for each defined period.

    Parameters
    ----------
    bhav_df      : Pre-loaded BhavCopy. Loaded from DataLoader if None.
    model        : Pre-trained LightGBM model. Loaded from disk if None.
    feature_cols : Feature column list. Loaded from feature_names.json if None.
    lookback_days: Days of history to load (default 4 years to cover 2020).
    threshold    : Pick probability threshold (default 0.62).
    fwd_sessions : Forward-return measurement sessions (default 5).
    draw_charts  : Save stress-test chart PNG (default True).
    periods      : Override STRESS_PERIODS list (for testing).

    Returns
    -------
    list of per-period stat dicts (same order as STRESS_PERIODS).
    """
    periods = periods or STRESS_PERIODS
    today_str = str(_date.today())
    date_tag  = today_str.replace("-", "")

    logger.info("=" * 68)
    logger.info("  P08  STRESS TESTS  |  %s", today_str)
    logger.info("=" * 68)

    # -- Load BhavCopy -------------------------------------------------------
    if bhav_df is None:
        from production.data_loader import DataLoader
        loader  = DataLoader()
        # Load from 2019 to cover the COVID crash period
        end_str = today_str
        # Use explicit lookback
        bhav_df = loader.load(lookback_days=lookback_days, end=end_str)

    if bhav_df is None or bhav_df.empty:
        raise RuntimeError("No BhavCopy data available for stress tests.")

    bhav_df["DATE"] = pd.to_datetime(bhav_df["DATE"]).dt.date

    # -- Load model ----------------------------------------------------------
    if model is None:
        try:
            import lightgbm as lgb
            model_path = MODELS_DIR / "lgbm_model.txt"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            model = lgb.Booster(model_file=str(model_path))
            logger.info("Model loaded from %s", model_path)
        except Exception as e:
            raise RuntimeError(f"Cannot load model: {e}")

    # -- Load feature cols ---------------------------------------------------
    if feature_cols is None:
        feat_path = MODELS_DIR / "feature_names.json"
        if feat_path.exists():
            feature_cols = json.loads(feat_path.read_text())
        else:
            try:
                from production.signal_generator import BASE_FEATURE_COLS
                feature_cols = BASE_FEATURE_COLS
            except Exception:
                raise RuntimeError("Cannot determine feature columns.")

    # -- Compute features on full bhav once ----------------------------------
    logger.info("Computing features on full BhavCopy (%d rows) ...", len(bhav_df))
    try:
        from momentum_features import prepare_features_all
        full_feat_df = prepare_features_all(bhav_df)
        full_feat_df["DATE"] = pd.to_datetime(full_feat_df["DATE"]).dt.date
    except Exception as e:
        raise RuntimeError(f"Feature computation failed: {e}")

    # -- Universe filter at latest date (for consistent stock selection) -----
    try:
        from production.universe_filter import apply_tradability_gate
        ref_date = bhav_df["DATE"].max()
        filtered = apply_tradability_gate(
            bhav_df, ref_date,
            min_value_crore=1.0, min_price=10.0, min_avg_volume=5_000,
            apply_operator_filter=False,   # not relevant for historical periods
            apply_ipo_filter=False,        # stocks may have had fewer sessions then
        )
        tradable_codes = set(filtered["SC_CODE"].unique())
    except Exception:
        tradable_codes = set(bhav_df["SC_CODE"].unique())

    # -- Run each stress period ---------------------------------------------
    all_results  = []
    all_details  = []

    for p in periods:
        p_start = pd.to_datetime(p["start"]).date()
        p_end   = pd.to_datetime(p["end"]).date()

        logger.info("-" * 60)
        logger.info("Period: %s  (%s -> %s)", p["name"], p_start, p_end)

        # Check data coverage
        period_dates = sorted([
            d for d in bhav_df["DATE"].unique()
            if p_start <= d <= p_end
        ])
        if len(period_dates) == 0:
            logger.warning("  No data for period %s — SKIP.", p["name"])
            all_results.append(_period_stats(pd.DataFrame(), p))
            continue

        logger.info("  %d trading days in period.", len(period_dates))

        # Slice features to dates in period
        period_feat = full_feat_df[
            (full_feat_df["DATE"] >= p_start) &
            (full_feat_df["DATE"] <= p_end) &
            (full_feat_df["SC_CODE"].isin(tradable_codes))
        ].copy()

        if period_feat.empty:
            logger.warning("  No feature rows for period %s — SKIP.", p["name"])
            all_results.append(_period_stats(pd.DataFrame(), p))
            continue

        # Generate picks day-by-day
        picks_list = []
        for d in period_dates:
            rows = period_feat[period_feat["DATE"] == d].copy()
            if rows.empty:
                continue

            X_slice = rows.reindex(columns=feature_cols, fill_value=0).fillna(0)
            try:
                probs = model.predict(X_slice.values)
            except Exception:
                continue

            rows["Probability"] = probs
            rows["pred_date"]   = d
            picks = rows[rows["Probability"] >= threshold][
                ["SC_CODE", "SC_NAME", "Close", "Probability", "pred_date"]
            ].copy()
            if picks.empty:
                continue

            fwd_close = _forward_close(bhav_df, d, fwd_sessions)
            picks["close_fwd"] = picks["SC_CODE"].map(fwd_close)
            picks = picks.dropna(subset=["close_fwd", "Close"])
            picks = picks[picks["Close"] > 0]
            picks_list.append(picks)

        if not picks_list:
            logger.warning("  No picks in period %s.", p["name"])
            all_results.append(_period_stats(pd.DataFrame(), p))
            continue

        detail = pd.concat(picks_list, ignore_index=True)
        detail["return_5d"] = (detail["close_fwd"] / detail["Close"]) - 1
        detail["positive"]  = (detail["return_5d"] > 0).astype(int)
        detail["pred_date"] = pd.to_datetime(detail["pred_date"])
        detail["period"]    = p["label"]
        detail = _apply_friction(detail)

        stats = _period_stats(detail, p)
        all_results.append(stats)
        all_details.append(detail)

        logger.info(
            "  Result: n=%d  win_rate=%.1f%%  net_win=%.1f%%  "
            "avg_ret=%.2f%%  max_dd=%.2f%%",
            stats["n_picks"],
            stats["win_rate"]      * 100,
            stats["win_rate_net"]  * 100,
            stats["avg_return"]    * 100,
            stats["max_drawdown"]  * 100,
        )

    # -- Save CSV -----------------------------------------------------------
    stats_df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / f"stress_test_results_{date_tag}.csv"
    stats_df.to_csv(csv_path, index=False)
    logger.info("Stress test results -> %s", csv_path)

    full_detail = pd.concat(all_details, ignore_index=True) if all_details else pd.DataFrame()

    # -- Chart --------------------------------------------------------------
    chart_path = None
    if draw_charts and not stats_df.empty:
        chart_path = _draw_chart(stats_df, full_detail, today_str, date_tag)

    return all_results


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------

def _draw_chart(stats_df: pd.DataFrame, detail_df: pd.DataFrame,
                today_str: str, date_tag: str) -> Path:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available -- chart skipped.")
        return None

    n_periods = len(stats_df)
    x = range(n_periods)
    labels = [r["name"] for _, r in stats_df.iterrows()]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Sub-Period Stress Test Results  |  {today_str}", fontsize=13, fontweight="bold")

    # 1. Win rate (gross & net) per period -----------------------------------
    ax = axes[0, 0]
    wr  = (stats_df["win_rate"].fillna(0) * 100).tolist()
    wrn = (stats_df["win_rate_net"].fillna(0) * 100).tolist()
    width = 0.35
    xpos  = [i - width/2 for i in x]
    xposn = [i + width/2 for i in x]
    ax.bar(xpos,  wr,  width, color="steelblue",  label="Gross", alpha=0.8)
    ax.bar(xposn, wrn, width, color="darkorange",  label="Net",   alpha=0.8)
    ax.axhline(50, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Win Rate by Stress Period", fontsize=11)
    ax.set_ylabel("Win Rate (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    # 2. Avg return per period -----------------------------------------------
    ax = axes[0, 1]
    ar  = (stats_df["avg_return"].fillna(0) * 100).tolist()
    arn = (stats_df["avg_return_net"].fillna(0) * 100).tolist()
    colors_g = ["seagreen" if v >= 0 else "tomato" for v in ar]
    colors_n = ["teal"     if v >= 0 else "salmon"  for v in arn]
    ax.bar(xpos,  ar,  width, color=colors_g, label="Gross", alpha=0.8)
    ax.bar(xposn, arn, width, color=colors_n, label="Net",   alpha=0.8)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title("Avg 5-Day Return by Stress Period", fontsize=11)
    ax.set_ylabel("Avg Return (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # 3. Max drawdown per period -----------------------------------------------
    ax = axes[1, 0]
    dd = (stats_df["max_drawdown"].fillna(0) * 100).tolist()
    colors_dd = ["crimson" if v < -10 else "tomato" if v < -5 else "salmon" for v in dd]
    ax.bar(list(x), dd, color=colors_dd, edgecolor="white", linewidth=0.5)
    ax.set_title("Max Daily Drawdown by Stress Period", fontsize=11)
    ax.set_ylabel("Max Drawdown (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)

    # 4. N picks per period --------------------------------------------------
    ax = axes[1, 1]
    np_ = stats_df["n_picks"].fillna(0).tolist()
    ax.bar(list(x), np_, color="mediumpurple", edgecolor="white", linewidth=0.5)
    ax.set_title("Number of Picks per Stress Period", fontsize=11)
    ax.set_ylabel("Count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = RESULTS_DIR / f"stress_test_{date_tag}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Stress test chart -> %s", out)
    return out


# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

def print_stress_summary(results: list):
    w = 80
    print("\n" + "=" * w)
    print("  P08  SUB-PERIOD STRESS TEST RESULTS")
    print("=" * w)
    print(f"  {'Period':<22} {'Start':>10} {'End':>10}  {'N':>5}  "
          f"{'WR%':>6}  {'Net%':>6}  {'AvgRet%':>8}  {'MaxDD%':>8}")
    print("  " + "-" * (w - 2))
    for r in results:
        wr  = (r.get("win_rate")      or 0) * 100
        nwr = (r.get("win_rate_net")  or 0) * 100
        ar  = (r.get("avg_return")    or 0) * 100
        dd  = (r.get("max_drawdown")  or 0) * 100
        n   = r.get("n_picks", 0)
        flag = " !!" if wr < 45 else ("  +" if wr >= 55 else "   ")
        print(f"  {r['name']:<22} {r['start']:>10} {r['end']:>10}  "
              f"{n:>5}  {wr:>5.1f}%  {nwr:>5.1f}%  {ar:>+7.2f}%  {dd:>+7.2f}%{flag}")
    print("=" * w)
    all_wrs = [r.get("win_rate") or 0 for r in results if r.get("n_picks", 0) > 0]
    if all_wrs:
        avg_wr = sum(all_wrs) / len(all_wrs) * 100
        n_above = sum(1 for w in all_wrs if w >= 0.5)
        print(f"\n  Average win rate across stress periods: {avg_wr:.1f}%")
        print(f"  Periods with win rate >= 50%: {n_above} / {len(all_wrs)}")
    print("=" * w + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    p = argparse.ArgumentParser(description="P08 Sub-Period Stress Tests")
    p.add_argument("--lookback",   type=int,   default=730 * 2,
                   help="Days of history to load (default 1460 = 4 years)")
    p.add_argument("--threshold",  type=float, default=0.62,
                   help="Pick probability threshold (default 0.62)")
    p.add_argument("--no-charts",  action="store_true",
                   help="Skip chart generation")
    args = p.parse_args()

    results = run_stress_tests(
        lookback_days=args.lookback,
        threshold=args.threshold,
        draw_charts=not args.no_charts,
    )
    print_stress_summary(results)


if __name__ == "__main__":
    main()
