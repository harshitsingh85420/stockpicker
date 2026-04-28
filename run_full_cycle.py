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
import logging
import sys
from datetime import date as _date
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

RESULTS_DIR  = _ROOT / "stock_picker_data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
            from production.walk_forward_backtest import run_walk_forward, print_walk_forward_summary
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
