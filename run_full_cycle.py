#!/usr/bin/env python3
"""
run_full_cycle.py -- Single-command daily pipeline.

Runs three sequential phases after market close:

    Phase 1 – Train / refresh the LightGBM model on all data up to today.
    Phase 2 – Walk-forward backtest over the past year:
               For every trading day D in [today−1year ... today−5sessions]:
                 • generate picks (prob ≥ threshold) using features up to D
                 • record the actual 5-session forward return
               Saves day-wise CSV, month-wise CSV, detail CSV, and a 3-panel chart.
    Phase 3 – Generate today's swing picks via TradeOrchestrator, save CSV,
               and print a summary table.

Usage:
    python run_full_cycle.py                           # today's date, prob ≥ 0.62
    python run_full_cycle.py --date 2026-04-23         # replay a past date
    python run_full_cycle.py --prob-threshold 0.70     # tighter filter
    python run_full_cycle.py --no-train                # skip retraining (use saved model)
    python run_full_cycle.py --no-backtest             # picks only (fastest)
    python run_full_cycle.py --no-charts               # skip matplotlib output
    python run_full_cycle.py --lookback 365            # 1-year training window

Dependencies (beyond the production/ package):
    pip install matplotlib lightgbm scikit-learn
"""

import argparse
import json
import logging
import sys
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULTS_DIR  = _ROOT / "stock_picker_data" / "results"
MODELS_DIR   = _ROOT / "stock_picker_data" / "models"
FEAT_JSON    = MODELS_DIR / "feature_names.json"   # written by train_model.py
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_expected_cols() -> list:
    """Return the exact feature list the model was trained on."""
    if FEAT_JSON.exists():
        return json.loads(FEAT_JSON.read_text())
    # fallback: BASE_FEATURE_COLS from signal_generator
    from production.signal_generator import BASE_FEATURE_COLS
    return BASE_FEATURE_COLS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Phase 1 -- Train
# ===========================================================================

def phase_train(end_date_str: str, lookback_days: int, calibrate: bool) -> dict:
    """Retrain LightGBM with data up to end_date and return metrics dict."""
    from train_model import run_training
    logger.info("Phase 1 > Training model (data up to %s) ...", end_date_str)
    metrics = run_training(
        lookback_days=lookback_days,
        end=end_date_str,
        calibrate=calibrate,
        apply_ca=True,
    )
    logger.info(
        "Phase 1 OK Training complete -- CV AUC=%.4f (%d features, %d samples)",
        metrics["mean_auc"], metrics["feature_count"], metrics["n_training_samples"],
    )
    return metrics


# ===========================================================================
# Phase 2 -- Backtest
# ===========================================================================

def _forward_close_series(bhav_df: pd.DataFrame, from_date, n_sessions: int = 5) -> pd.Series:
    """
    Return SC_CODE -> close price exactly n_sessions trading days after from_date.
    Uses only data already present in bhav_df (no extra fetch needed).
    """
    all_dates = sorted(bhav_df["DATE"].unique())
    future = [d for d in all_dates if d > from_date]
    if len(future) < n_sessions:
        return pd.Series(dtype=float, name="close_fwd")
    target_date = future[n_sessions - 1]
    day_df = bhav_df[bhav_df["DATE"] == target_date][["SC_CODE", "Close"]]
    return day_df.set_index("SC_CODE")["Close"].rename("close_fwd")


def _predict_for_date(
    target_date,
    feature_df: pd.DataFrame,
    model,
    feat_cols: list,
    threshold: float,
) -> pd.DataFrame:
    """
    Given the precomputed feature_df, return picks for target_date above threshold.
    No look-ahead: feature_df was computed from bhav up to the full history,
    but we only *select* rows where DATE == target_date for prediction.
    Indicators (EMAs, ATR, etc.) already use only past data by construction.

    P01: Feature alignment is enforced -- only the exact columns the model was
    trained on are passed to model.predict(), in the same order.
    """
    date_col = "DATE"
    rows = feature_df[pd.to_datetime(feature_df[date_col]).dt.date == target_date].copy()
    if rows.empty:
        return pd.DataFrame()

    # ── P01 + P29: enforce feature alignment ────────────────────────────
    expected_cols = _load_expected_cols()
    # P29: log WF block col count to catch 33-vs-31 drift
    logger.debug(
        "P29 WF BLOCK COLS (%d): %s", len(expected_cols), sorted(expected_cols)
    )
    # Check BEFORE reindex so we can detect genuine pipeline regressions
    missing = [c for c in expected_cols if c not in rows.columns]
    extra   = [c for c in rows.columns   if c not in expected_cols
               and c not in {"SC_CODE","SC_NAME","DATE","ISIN","Source",
                              "Open","High","Low","Close","Volume","ValueTraded"}]
    if missing:
        logger.warning(
            "P29 [%s]: %d canonical feature(s) missing from WF block: %s",
            target_date, len(missing), missing,
        )
    if extra:
        logger.debug(
            "P01 [%s]: %d extra feature(s) computed (ignored): %s",
            target_date, len(extra), extra[:10],
        )
    # P29: Hard subset — never use X.columns directly
    X_slice = rows.reindex(columns=expected_cols, fill_value=0).fillna(0)
    assert list(X_slice.columns) == expected_cols, (
        f"P29: reindex produced wrong column order on {target_date}"
    )
    assert X_slice.shape[1] == len(expected_cols), (
        f"P29: feature count mismatch in WF block: {X_slice.shape[1]} vs {len(expected_cols)}"
    )
    # ────────────────────────────────────────────────────────────────────

    try:
        probs = model.predict(X_slice.values)
    except Exception as e:
        logger.debug("Predict failed for %s: %s", target_date, e)
        return pd.DataFrame()

    rows = rows.copy()
    rows["Probability"] = probs
    rows["pred_date"]   = target_date
    picks = rows[rows["Probability"] >= threshold][
        ["SC_CODE", "SC_NAME", "Close", "Probability", "pred_date"]
    ].copy()
    return picks


def phase_backtest(
    bhav_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    model,
    feat_cols: list,
    today: _date,
    threshold: float,
    fwd_sessions: int = 5,
) -> tuple:
    """
    Walk-forward backtest over the past year.

    Returns:
        (detail_df, daily_df, monthly_df, yearly_dict)
    """
    logger.info("Phase 2 > Running backtest ...")

    all_dates = sorted(bhav_df["DATE"].unique())
    # Determine backtest window: [today − 1 year ... today − fwd_sessions]
    cutoff_end   = sorted([d for d in all_dates if d < today])
    if len(cutoff_end) < fwd_sessions:
        logger.warning("Not enough historical dates for backtest -- skipping.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    # last valid prediction date = fwd_sessions trading days before today
    last_pred_date = cutoff_end[-fwd_sessions]

    # Start = approximately one year before today
    one_year_ago = pd.Timestamp(today) - pd.DateOffset(years=1)
    backtest_dates = [
        d for d in all_dates
        if pd.Timestamp(d) >= one_year_ago and d <= last_pred_date
    ]
    if not backtest_dates:
        logger.warning("No dates in backtest window.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    logger.info(
        "Backtesting %d trading days (%s -> %s) ...",
        len(backtest_dates), backtest_dates[0], backtest_dates[-1],
    )

    all_picks = []
    for i, d in enumerate(backtest_dates):
        if i % 50 == 0:
            logger.info("  Backtest progress: %d / %d", i, len(backtest_dates))

        picks = _predict_for_date(d, feature_df, model, feat_cols, threshold)
        if picks.empty:
            continue

        fwd_close = _forward_close_series(bhav_df, d, fwd_sessions)
        picks["close_fwd"] = picks["SC_CODE"].map(fwd_close)
        picks = picks.dropna(subset=["close_fwd", "Close"])
        picks = picks[picks["Close"] > 0]
        all_picks.append(picks)

    if not all_picks:
        logger.warning("Backtest produced no picks. Check model and data.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    detail_df = pd.concat(all_picks, ignore_index=True)
    detail_df["return_5d"] = (detail_df["close_fwd"] / detail_df["Close"]) - 1
    detail_df["positive"]  = (detail_df["return_5d"] > 0).astype(int)
    detail_df["pred_date"] = pd.to_datetime(detail_df["pred_date"])

    # ── P02: Wire transaction costs into every backtest trade ────────────
    # Uses FrictionModel (equity_delivery, BSE) for statutory costs and
    # SlippageModel for market-impact cost.
    # Position value is estimated from Close price * assumed 500 shares
    # (floored to Rs.10,000 min, capped at Rs.5,00,000 max -- retail range).
    try:
        from production.friction_model import FrictionModel, SlippageModel
        _fm = FrictionModel(instrument_type="equity_delivery")

        # Vectorised: compute statutory round-trip for a representative position.
        # Position value = Close * 500 shares, clamped to [10k, 5L] INR.
        pos_vals = np.clip(detail_df["Close"].values * 500, 10_000, 500_000)

        # Statutory costs: brokerage(both legs) + STT(sell) + exchange(both) +
        # stamp(buy) + SEBI(both) + GST.  We sample 5 quantile points and
        # interpolate to avoid calling calculate_round_trip_cost N times.
        sample_vals = np.quantile(pos_vals, [0.1, 0.3, 0.5, 0.7, 0.9])
        sample_pcts = np.array([
            _fm.calculate_round_trip_cost(v, exchange="BSE")["total_costs"]["total"] / v
            for v in sample_vals
        ])
        # Linear interpolation over the sampled curve
        statutory_pct = np.interp(pos_vals, sample_vals, sample_pcts)

        # Slippage: 0.15% per leg (entry + exit) = 0.30% round-trip
        # BSE mid-caps have wider spreads than NSE large-caps; 0.15% per leg is conservative
        slippage_pct = 0.003   # 0.3% round-trip

        detail_df["friction_pct"]  = statutory_pct + slippage_pct
        detail_df["return_5d_net"] = detail_df["return_5d"] - detail_df["friction_pct"]
        detail_df["positive_net"]  = (detail_df["return_5d_net"] > 0).astype(int)
        logger.info(
            "P02 friction applied -- avg per trade: %.3f%%",
            detail_df["friction_pct"].mean() * 100,
        )
    except Exception as _e:
        logger.warning("P02 friction model unavailable (%s) -- using gross returns.", _e)
        detail_df["friction_pct"]  = 0.0
        detail_df["return_5d_net"] = detail_df["return_5d"]
        detail_df["positive_net"]  = detail_df["positive"]
    # ─────────────────────────────────────────────────────────────────────

    # ── Day-wise ─────────────────────────────────────────────────────────
    daily_df = (
        detail_df.groupby("pred_date")
        .agg(
            total_picks    = ("positive",     "count"),
            positive_picks = ("positive",     "sum"),
            win_rate       = ("positive",     "mean"),
            win_rate_net   = ("positive_net", "mean"),
            avg_return     = ("return_5d",    "mean"),
            avg_return_net = ("return_5d_net","mean"),
        )
        .reset_index()
    )

    # ── Month-wise ───────────────────────────────────────────────────────
    detail_df["month"] = detail_df["pred_date"].dt.to_period("M").astype(str)
    monthly_df = (
        detail_df.groupby("month")
        .agg(
            total_picks    = ("positive",     "count"),
            positive_picks = ("positive",     "sum"),
            win_rate       = ("positive",     "mean"),
            win_rate_net   = ("positive_net", "mean"),
            avg_return     = ("return_5d",    "mean"),
            avg_return_net = ("return_5d_net","mean"),
        )
        .reset_index()
    )

    # ── Yearly summary ───────────────────────────────────────────────────
    avg_friction = float(detail_df["friction_pct"].mean())
    yearly = {
        "total_trading_days":  len(daily_df),
        "total_picks":         len(detail_df),
        "overall_win_rate":    float(detail_df["positive"].mean()),
        "net_win_rate":        float(detail_df["positive_net"].mean()),
        "avg_friction_pct":    avg_friction,
        "avg_return":          float(detail_df["return_5d"].mean()),
        "avg_return_net":      float(detail_df["return_5d_net"].mean()),
        "median_return":       float(detail_df["return_5d"].median()),
        "median_return_net":   float(detail_df["return_5d_net"].median()),
        "backtest_start":      str(backtest_dates[0]),
        "backtest_end":        str(backtest_dates[-1]),
    }

    logger.info(
        "Phase 2 complete -- gross win rate %.1f%%  |  net win rate %.1f%%  |  avg friction %.3f%%",
        yearly["overall_win_rate"] * 100,
        yearly["net_win_rate"] * 100,
        yearly["avg_friction_pct"] * 100,
    )
    return detail_df, daily_df, monthly_df, yearly


def save_backtest_results(detail_df, daily_df, monthly_df, today_str: str):
    if detail_df.empty:
        return
    detail_df.to_csv(RESULTS_DIR / f"backtest_details_{today_str}.csv", index=False)
    daily_df.to_csv(RESULTS_DIR  / f"backtest_daily_{today_str}.csv",   index=False)
    monthly_df.to_csv(RESULTS_DIR / f"backtest_monthly_{today_str}.csv", index=False)
    logger.info("Backtest CSVs saved to %s/", RESULTS_DIR)


def print_backtest_summary(yearly: dict):
    if not yearly:
        return
    w = 62
    gross_wr = yearly.get("overall_win_rate", 0)
    net_wr   = yearly.get("net_win_rate", gross_wr)
    friction = yearly.get("avg_friction_pct", 0)
    avg_ret  = yearly.get("avg_return", 0)
    avg_net  = yearly.get("avg_return_net", avg_ret)
    med_ret  = yearly.get("median_return", 0)
    med_net  = yearly.get("median_return_net", med_ret)
    print("\n" + "=" * w)
    print(f"  BACKTEST SUMMARY ({yearly['backtest_start']} -> {yearly['backtest_end']})")
    print("=" * w)
    print(f"  Trading days tested         : {yearly['total_trading_days']}")
    print(f"  Total picks made            : {yearly['total_picks']}")
    print(f"  Gross win rate              : {gross_wr:.1%}")
    print(f"  Net win rate (post-friction): {net_wr:.1%}")
    print(f"  Avg friction per trade      : {friction:.3%}")
    print(f"  Avg 5-day return (gross)    : {avg_ret:+.2%}")
    print(f"  Avg 5-day return (net)      : {avg_net:+.2%}")
    print(f"  Median 5-day return (gross) : {med_ret:+.2%}")
    print(f"  Median 5-day return (net)   : {med_net:+.2%}")
    print()
    print(f"  Gross win rate: {gross_wr:.1%}  |  "
          f"Net win rate (post-friction): {net_wr:.1%}  |  "
          f"Avg friction per trade: {friction:.3%}")
    print("=" * w + "\n")


def plot_backtest(detail_df, daily_df, monthly_df, today_str: str):
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend (safe for servers)
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not installed -- skipping charts.")
        return

    fig, axes = plt.subplots(3, 1, figsize=(13, 15))

    # 1. Daily win rate
    ax = axes[0]
    dates    = pd.to_datetime(daily_df["pred_date"])
    win_rate = daily_df["win_rate"] * 100
    ax.plot(dates, win_rate, marker=".", markersize=3, linestyle="-", color="steelblue", linewidth=0.8)
    # Rolling 10-day average
    roll = win_rate.rolling(10, min_periods=1).mean()
    ax.plot(dates, roll, color="navy", linewidth=1.5, label="10-day avg")
    ax.axhline(50, color="red", linestyle="--", alpha=0.6, label="50% line")
    ax.set_title("Daily Win Rate  (5-session forward return)", fontsize=12)
    ax.set_ylabel("Win Rate (%)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # 2. Monthly win-rate bars
    ax = axes[1]
    months    = monthly_df["month"]
    win_rates = monthly_df["win_rate"] * 100
    colors    = ["seagreen" if w >= 50 else "tomato" for w in win_rates]
    x_pos     = range(len(months))
    ax.bar(x_pos, win_rates, color=colors)
    ax.axhline(50, color="black", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_title("Monthly Win Rate", fontsize=12)
    ax.set_ylabel("Win Rate (%)")
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # 3. Return distribution
    ax = axes[2]
    returns = detail_df["return_5d"] * 100
    ax.hist(returns, bins=60, color="mediumpurple", alpha=0.75, edgecolor="white", linewidth=0.3)
    ax.axvline(0,            color="black", linestyle="--", alpha=0.7, linewidth=1)
    ax.axvline(returns.mean(), color="red",  linestyle="-",  alpha=0.8, linewidth=1.2,
               label=f"mean {returns.mean():+.2f}%")
    ax.set_title("Distribution of 5-Day Returns  (all picks)", fontsize=12)
    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.suptitle(f"Backtest Results  |  generated {today_str}", fontsize=10, y=1.01)
    plt.tight_layout()

    plot_path = RESULTS_DIR / f"backtest_charts_{today_str}.png"
    plt.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Charts saved -> %s", plot_path)


# ===========================================================================
# Phase 3 -- Today's picks
# ===========================================================================

def phase_picks(today_str: str, total_capital: float) -> pd.DataFrame:
    """Run TradeOrchestrator for today and return the picks DataFrame."""
    logger.info("Phase 3 > Generating today's swing picks ...")
    from production.trade_orchestrator import TradeOrchestrator
    orch   = TradeOrchestrator()
    result = orch.run_daily(
        reference_date=today_str,
        total_capital=total_capital,
    )
    picks = result.get("picks", pd.DataFrame())
    action = result.get("action", "unknown")
    regime = result.get("regime", "unknown")
    logger.info(
        "Phase 3 OK %d picks generated  |  action=%s  regime=%s",
        len(picks), action, regime,
    )
    return picks


def print_picks_table(picks: pd.DataFrame, today_str: str):
    if picks.empty:
        print("\n  No picks generated today. Check market conditions / circuit breaker.\n")
        return

    w = 72
    print("\n" + "=" * w)
    print(f"  TODAY'S SWING PICKS  --  {today_str}")
    print("=" * w)
    print(f"  {'#':<4} {'SC_NAME':<16} {'CODE':<10} {'Close':>7} {'Prob':>6} {'Stop':>8} {'Sector'}")
    print("  " + "-" * (w - 2))
    for _, r in picks.iterrows():
        rank   = int(r.get("Rank", 0))
        name   = str(r.get("SC_NAME", ""))[:15]
        code   = str(r.get("SC_CODE", ""))
        close  = r.get("Close", float("nan"))
        prob   = r.get("Probability", float("nan"))
        stop   = r.get("Stop_Loss_Price", float("nan"))
        sector = str(r.get("Sector", ""))[:12]
        print(f"  {rank:<4} {name:<16} {code:<10} {close:>7.2f} {prob:>6.3f} {stop:>8.2f}  {sector}")
    print("=" * w + "\n")


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Full daily cycle: train -> backtest -> picks")
    p.add_argument("--date",           type=str,   default=str(_date.today()),
                   help="Reference date YYYY-MM-DD (default: today)")
    p.add_argument("--prob-threshold", type=float, default=0.62,
                   help="Minimum probability for picks (default 0.62)")
    p.add_argument("--lookback",       type=int,   default=730,
                   help="Training window in calendar days (default 730 = 2 years)")
    p.add_argument("--capital",        type=float, default=1_000_000,
                   help="Total capital in INR (default 10,00,000)")
    p.add_argument("--no-train",       action="store_true", help="Skip retraining (use saved model)")
    p.add_argument("--no-backtest",    action="store_true", help="Skip backtest (picks only)")
    p.add_argument("--no-charts",      action="store_true", help="Skip chart generation")
    p.add_argument("--no-calibrate",   action="store_true", help="Skip calibrator fitting")
    return p.parse_args()


def main():
    args     = parse_args()
    today    = pd.Timestamp(args.date).date()
    today_str = str(today)

    logger.info("=" * 60)
    logger.info("run_full_cycle.py  --  %s", today_str)
    logger.info("=" * 60)

    # ── Phase 1: Train ──────────────────────────────────────────────────
    if not args.no_train:
        try:
            phase_train(
                end_date_str=today_str,
                lookback_days=args.lookback,
                calibrate=not args.no_calibrate,
            )
        except Exception as e:
            logger.error("Training failed: %s", e)
            logger.warning("Continuing with previously saved model (if any).")
    else:
        logger.info("Phase 1 > Skipped (--no-train).")

    # ── Phase 2 -- Walk-Forward Backtest (P07) ─────────────────────────
    if not args.no_backtest:
        try:
            from walk_forward_backtest import run_walk_forward, print_walk_forward_summary
            logger.info("Phase 2 > Walk-forward backtest (P07: proper OOS) ...")
            wf_result = run_walk_forward(
                end_date=today_str,
                lookback_days=args.lookback,
                months_per_block=3,
                embargo_sessions=5,
                min_train_months=9,
                threshold=args.prob_threshold,
                draw_charts=not args.no_charts,
            )
            print_walk_forward_summary(wf_result["summary"], wf_result["block_stats"])
            if wf_result.get("chart_path"):
                logger.info("Phase 2 OK Walk-forward chart -> %s", wf_result["chart_path"])
        except Exception as e:
            logger.error("Walk-forward backtest failed: %s", e, exc_info=True)
            logger.warning("Continuing to Phase 3 (picks generation).")
    else:
        logger.info("Phase 2 > Skipped (--no-backtest).")

    # ── Phase 3: Today's picks ──────────────────────────────────────────
    try:
        picks = phase_picks(today_str, total_capital=args.capital)
        print_picks_table(picks, today_str)
    except Exception as e:
        logger.error("Picks generation failed: %s", e, exc_info=True)
        sys.exit(1)

    logger.info("Full cycle complete  --  %s", today_str)


if __name__ == "__main__":
    main()
