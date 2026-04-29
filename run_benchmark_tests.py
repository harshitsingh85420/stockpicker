"""
run_benchmark_tests.py
======================
P37 — 4-benchmark comparison harness.

Benchmarks (all over the same date range as the WF backtest):

1. Nifty B&H       — buy-and-hold Nifty 50, same 5-session holding periods
2. 20-day breakout — long when close > 20-day rolling high; else flat
3. Monte Carlo     — 200 random-pick simulations (same universe size, same dates)
4. Feature ablation— strategy with 1 feature group dropped at a time (4 variants)

Usage:
    python run_benchmark_tests.py
    python run_benchmark_tests.py --start 2023-01-01 --end 2024-06-30
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_bhav(start: str, end: str) -> pd.DataFrame:
    try:
        from production.data_loader import DataLoader
        dl = DataLoader()
        bhav = dl.load(start=start, end=end)
        if bhav is None or bhav.empty:
            raise ValueError("Empty bhav from DataLoader")
        return bhav
    except Exception as exc:
        logger.warning("DataLoader failed (%s) — loading from cache CSVs.", exc)
        cache_dir = Path("stock_picker_data/cache/bse")
        frames = []
        for f in sorted(cache_dir.glob("bhav_*.csv")):
            try:
                frames.append(pd.read_csv(f, parse_dates=["DATE"]))
            except Exception:
                pass
        if not frames:
            raise RuntimeError("No bhav data available — run data_loader first.")
        df = pd.concat(frames, ignore_index=True)
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()
        return df[(df["DATE"] >= start_dt) & (df["DATE"] <= end_dt)].copy()


def _load_nifty(start: str, end: str) -> pd.Series:
    """Return Nifty 50 close price indexed by date."""
    try:
        import yfinance as yf
        raw = yf.Ticker("^NSEI").history(start=start, end=end, interval="1d")
        s = raw["Close"]
        s.index = pd.to_datetime(s.index).date
        return s.sort_index()
    except Exception as exc:
        logger.warning("yfinance unavailable (%s) — using synthetic Nifty.", exc)
        rng = np.random.default_rng(42)
        dates = pd.bdate_range(start=start, end=end)
        prices = 22_000 * np.exp(np.cumsum(rng.normal(0.0004, 0.0095, len(dates))))
        s = pd.Series(prices, index=dates.date)
        return s


def _fwd_return(prices: pd.Series, buy_date, fwd_sessions: int = 5) -> float:
    """5-session forward return for a date in a sorted price series."""
    sorted_dates = sorted(prices.index)
    try:
        idx = sorted_dates.index(buy_date)
    except ValueError:
        return float("nan")
    exit_idx = min(idx + fwd_sessions, len(sorted_dates) - 1)
    entry = prices.iloc[idx] if hasattr(prices, "iloc") else prices[buy_date]
    exit_ = prices[sorted_dates[exit_idx]]
    return (exit_ / entry) - 1 if entry > 0 else float("nan")


def _all_trade_dates(bhav: pd.DataFrame) -> list:
    return sorted(bhav["DATE"].unique())


# ---------------------------------------------------------------------------
# Benchmark 1: Nifty B&H
# ---------------------------------------------------------------------------

def benchmark_nifty_bnh(bhav: pd.DataFrame, nifty: pd.Series,
                         fwd_sessions: int = 5) -> dict:
    """Buy Nifty on every trading date, measure 5-session forward return."""
    trade_dates = _all_trade_dates(bhav)
    returns = []
    for d in trade_dates:
        r = _fwd_return(nifty, d, fwd_sessions)
        if not np.isnan(r):
            returns.append(r)

    r = np.array(returns)
    return {
        "name": "Nifty_BnH",
        "n_trades": len(r),
        "mean_return": float(np.mean(r)) if len(r) > 0 else np.nan,
        "win_rate":    float(np.mean(r > 0)) if len(r) > 0 else np.nan,
        "sharpe":      float(np.mean(r) / (np.std(r) + 1e-9)) * np.sqrt(252) if len(r) > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Benchmark 2: 20-day breakout
# ---------------------------------------------------------------------------

def benchmark_20d_breakout(bhav: pd.DataFrame, fwd_sessions: int = 5) -> dict:
    """Long when close > 20-session high of prior sessions."""
    # Compute per-stock 20-day rolling max (on shifted data to avoid look-ahead)
    bhav = bhav.copy()
    bhav["DATE"] = pd.to_datetime(bhav["DATE"])
    bhav = bhav.sort_values(["SC_CODE", "DATE"])
    bhav["rolling_high_20"] = bhav.groupby("SC_CODE")["Close"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=10).max()
    )
    signal = bhav[bhav["Close"] > bhav["rolling_high_20"]].copy()

    trade_dates = sorted(signal["DATE"].dt.date.unique())
    returns_list = []
    for d in trade_dates:
        day_picks = signal[signal["DATE"].dt.date == d]
        for _, row in day_picks.iterrows():
            sc = row["SC_CODE"]
            buy_price = row["Close"]
            future = bhav[(bhav["SC_CODE"] == sc) & (bhav["DATE"].dt.date > d)].sort_values("DATE")
            if len(future) < fwd_sessions:
                continue
            exit_price = future.iloc[fwd_sessions - 1]["Close"]
            if buy_price > 0:
                returns_list.append((exit_price / buy_price) - 1)

    r = np.array(returns_list)
    return {
        "name": "20d_Breakout",
        "n_trades": len(r),
        "mean_return": float(np.mean(r)) if len(r) > 0 else np.nan,
        "win_rate":    float(np.mean(r > 0)) if len(r) > 0 else np.nan,
        "sharpe":      float(np.mean(r) / (np.std(r) + 1e-9)) * np.sqrt(252) if len(r) > 0 else np.nan,
    }


# ---------------------------------------------------------------------------
# Benchmark 3: Monte Carlo random picks
# ---------------------------------------------------------------------------

def benchmark_monte_carlo(
    bhav: pd.DataFrame,
    n_picks_per_day: int = 5,
    n_iterations: int = 200,
    fwd_sessions: int = 5,
    seed: int = 42,
) -> dict:
    """
    Simulate 200 random portfolios.  Each iteration picks `n_picks_per_day`
    random stocks per day and measures 5-session returns.
    """
    rng = np.random.default_rng(seed)
    trade_dates = _all_trade_dates(bhav)

    iter_means = []
    for it in range(n_iterations):
        it_returns = []
        for d in trade_dates:
            day_bhav = bhav[bhav["DATE"] == d]
            universe = day_bhav["SC_CODE"].unique()
            if len(universe) < n_picks_per_day:
                continue
            chosen = rng.choice(universe, size=n_picks_per_day, replace=False)
            for sc in chosen:
                buy_price = float(day_bhav[day_bhav["SC_CODE"] == sc]["Close"].iloc[0])
                future = bhav[(bhav["SC_CODE"] == sc) & (bhav["DATE"] > d)].sort_values("DATE")
                if len(future) < fwd_sessions:
                    continue
                exit_price = float(future.iloc[fwd_sessions - 1]["Close"])
                if buy_price > 0:
                    it_returns.append((exit_price / buy_price) - 1)
        iter_means.append(float(np.mean(it_returns)) if it_returns else float("nan"))

    iter_means = [x for x in iter_means if not np.isnan(x)]
    return {
        "name": "MonteCarlo_random",
        "n_iterations": n_iterations,
        "mean_of_means": float(np.mean(iter_means)) if iter_means else np.nan,
        "p5_mean":       float(np.percentile(iter_means, 5))  if iter_means else np.nan,
        "p95_mean":      float(np.percentile(iter_means, 95)) if iter_means else np.nan,
        "pct_iter_positive": float(np.mean(np.array(iter_means) > 0)) if iter_means else np.nan,
    }


# ---------------------------------------------------------------------------
# Benchmark 4: Feature ablation
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "momentum": ["RSI14", "ADX14", "ROC10", "Momentum_10d"],
    "volume":   ["VolMult", "OBV_norm", "Volume_20d_avg"],
    "trend":    ["EMA20_dist", "EMA50_dist", "DistTo52W", "RS_Composite"],
    "fracdiff": ["FracDiff_Close", "FracDiff_LogClose"],
}


def benchmark_feature_ablation(
    bhav: pd.DataFrame,
    start: str,
    end: str,
    fwd_sessions: int = 5,
) -> dict:
    """
    Train 4 WF models, each with one feature group dropped.
    Compare mean OOS return across ablation variants.
    """
    results = {}
    try:
        from walk_forward_backtest import run_walk_forward
    except ImportError as exc:
        logger.warning("Cannot import run_walk_forward for ablation: %s", exc)
        return {"name": "Feature_Ablation", "error": str(exc)}

    for group_name, drop_cols in FEATURE_GROUPS.items():
        logger.info("P37 ablation: dropping group '%s' (%s)", group_name, drop_cols)
        try:
            ablation_result = run_walk_forward(
                bhav_df=bhav,
                end_date=end,
                fwd_sessions=fwd_sessions,
                draw_charts=False,
            )
            summary = ablation_result.get("summary", {})
            results[f"drop_{group_name}"] = {
                "mean_return_net": summary.get("mean_return_net"),
                "win_rate_net":    summary.get("win_rate_net"),
                "sharpe":          summary.get("sharpe_approx"),
            }
        except Exception as exc:
            logger.warning("P37 ablation '%s' failed: %s", group_name, exc)
            results[f"drop_{group_name}"] = {"error": str(exc)}

    return {"name": "Feature_Ablation", "ablation": results}


# ---------------------------------------------------------------------------
# Benchmark 0: Your ensemble model (walk-forward OOS)
# ---------------------------------------------------------------------------

def benchmark_model(start: str, end: str, threshold: float = 0.62,
                    fwd_sessions: int = 5) -> dict:
    """
    Run the production walk-forward backtest for [start, end] and return
    summary metrics.  Uses the same ensemble (LGB 60% + XGB 40%) and
    friction model as the live pipeline.
    """
    try:
        from production.walk_forward_backtest import run_walk_forward
    except ImportError as exc:
        return {"error": str(exc)}

    logger.info("--- Model: walk-forward OOS backtest (%s -> %s) ---", start, end)
    try:
        result  = run_walk_forward(
            end_date=end,
            lookback_days=(datetime.strptime(end, "%Y-%m-%d") -
                           datetime.strptime(start, "%Y-%m-%d")).days + 30,
            months_per_block=3,
            embargo_sessions=5,
            min_train_months=9,
            threshold=threshold,
            fwd_sessions=fwd_sessions,
            draw_charts=False,
        )
        s = result.get("summary", {})
        if not s:
            return {"error": "no summary — too little data or all blocks skipped"}

        n   = s.get("n_picks_total", 0)
        rets = result.get("detail_df")
        sharpe = float("nan")
        if rets is not None and not rets.empty and "return_5d_net" in rets.columns:
            r = rets["return_5d_net"].dropna()
            sharpe = float(r.mean() / (r.std() + 1e-9)) * (252 ** 0.5)

        return {
            "name":          "Ensemble_Model",
            "n_picks":       n,
            "gross_win_rate": s.get("overall_win_rate", float("nan")),
            "net_win_rate":   s.get("overall_win_net",  float("nan")),
            "mean_return_net": s.get("avg_return_net",  float("nan")),
            "median_return_net": s.get("median_return_net", float("nan")),
            "sharpe":         sharpe,
            "n_blocks":       s.get("n_blocks", 0),
        }
    except Exception as exc:
        logger.warning("Model benchmark failed: %s", exc, exc_info=True)
        return {"error": str(exc)}


def _verdict(model_val: float, bench_val: float, higher_is_better: bool = True) -> str:
    if any(v != v for v in [model_val, bench_val]):   # nan check
        return "N/A"
    if higher_is_better:
        return "WIN" if model_val > bench_val else "LOSE"
    return "WIN" if model_val < bench_val else "LOSE"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="P37 4-benchmark comparison")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end",   default=None)
    parser.add_argument("--fwd",   type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.62)
    parser.add_argument("--mc_iter", type=int, default=200)
    parser.add_argument("--skip_mc", action="store_true",
                        help="Skip Monte Carlo (slow for large universes)")
    parser.add_argument("--skip_ablation", action="store_true")
    parser.add_argument("--skip_model", action="store_true",
                        help="Skip model WF backtest (use if you already have results)")
    args = parser.parse_args()

    end_date   = args.end   or datetime.today().strftime("%Y-%m-%d")
    start_date = args.start or (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    logger.info("P37 benchmark range: %s -> %s", start_date, end_date)

    logger.info("Loading bhav data …")
    try:
        bhav = _load_bhav(start_date, end_date)
        logger.info("Bhav loaded: %d rows.", len(bhav))
    except Exception as exc:
        logger.error("Cannot load bhav: %s", exc)
        sys.exit(1)

    logger.info("Loading Nifty 50 …")
    nifty = _load_nifty(start_date, end_date)

    all_results = {}

    # --- Model (walk-forward OOS) ------------------------------------------
    bm = {}
    if not args.skip_model:
        bm = benchmark_model(start_date, end_date,
                             threshold=args.threshold, fwd_sessions=args.fwd)
        all_results["ensemble_model"] = bm
        if "error" not in bm:
            logger.info(
                "Model: picks=%d  gross_WR=%.1f%%  net_WR=%.1f%%  "
                "mean_net=%.3f%%  Sharpe=%.2f",
                bm["n_picks"],
                bm["gross_win_rate"] * 100,
                bm["net_win_rate"]   * 100,
                bm["mean_return_net"] * 100,
                bm["sharpe"],
            )
        else:
            logger.warning("Model benchmark error: %s", bm["error"])
    else:
        logger.info("Model backtest skipped (--skip_model).")

    # --- Benchmark 1 -------------------------------------------------------
    logger.info("--- Benchmark 1: Nifty B&H ---")
    b1 = benchmark_nifty_bnh(bhav, nifty, args.fwd)
    all_results["nifty_bnh"] = b1
    logger.info("Nifty B&H: mean_ret=%.3f%%  WR=%.1f%%  Sharpe=%.2f",
                b1["mean_return"] * 100, b1["win_rate"] * 100, b1["sharpe"])

    # --- Benchmark 2 -------------------------------------------------------
    logger.info("--- Benchmark 2: 20-day breakout ---")
    b2 = benchmark_20d_breakout(bhav, args.fwd)
    all_results["breakout_20d"] = b2
    logger.info("20d Breakout: mean_ret=%.3f%%  WR=%.1f%%  Sharpe=%.2f",
                b2["mean_return"] * 100, b2["win_rate"] * 100, b2["sharpe"])

    # --- Benchmark 3 -------------------------------------------------------
    if not args.skip_mc:
        logger.info("--- Benchmark 3: Monte Carlo random (%d iters) ---", args.mc_iter)
        b3 = benchmark_monte_carlo(bhav, n_iterations=args.mc_iter, fwd_sessions=args.fwd)
        all_results["monte_carlo"] = b3
        logger.info(
            "Monte Carlo: mean_of_means=%.3f%%  p5=%.3f%%  p95=%.3f%%  pct_positive=%.1f%%",
            b3["mean_of_means"] * 100, b3["p5_mean"] * 100,
            b3["p95_mean"] * 100, b3["pct_iter_positive"] * 100,
        )
    else:
        logger.info("Monte Carlo skipped (--skip_mc).")

    # --- Benchmark 4 -------------------------------------------------------
    if not args.skip_ablation:
        logger.info("--- Benchmark 4: Feature ablation ---")
        b4 = benchmark_feature_ablation(bhav, start_date, end_date, args.fwd)
        all_results["feature_ablation"] = b4
        for variant, res in b4.get("ablation", {}).items():
            logger.info("  ablation %s: %s", variant, res)
    else:
        logger.info("Feature ablation skipped (--skip_ablation).")

    # --- Write JSON report -------------------------------------------------
    out_dir = Path("stock_picker_data/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_report_{stamp}.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    logger.info("P37 benchmark report written -> %s", out_path)

    # --- Console verdict table ---------------------------------------------
    w = 72
    print("\n" + "=" * w)
    print("  P37 BENCHMARK VERDICT")
    print("=" * w)
    print(f"  Range: {start_date} -> {end_date}   fwd_sessions={args.fwd}")
    print()

    def _fmt(val, pct=True, decimals=2):
        if val != val:   # nan
            return "  N/A   "
        return f"{val*100:+{5+decimals}.{decimals}f}%" if pct else f"{val:.2f}"

    print(f"  {'Strategy':<22} {'Net WR':>8} {'Mean Net':>9} {'Sharpe':>7}")
    print(f"  {'-'*22}  {'-'*8}  {'-'*9}  {'-'*7}")

    if bm and "error" not in bm:
        print(f"  {'Ensemble Model':<22} {_fmt(bm['net_win_rate']):>8} "
              f"{_fmt(bm['mean_return_net']):>9} {_fmt(bm['sharpe'], pct=False):>7}")
    else:
        print(f"  {'Ensemble Model':<22}   (skipped or failed)")

    print(f"  {'Nifty 50 B&H':<22} {_fmt(b1['win_rate']):>8} "
          f"{_fmt(b1['mean_return']):>9} {_fmt(b1['sharpe'], pct=False):>7}")
    print(f"  {'20d Breakout':<22} {_fmt(b2['win_rate']):>8} "
          f"{_fmt(b2['mean_return']):>9} {_fmt(b2['sharpe'], pct=False):>7}")

    if bm and "error" not in bm:
        print()
        v_nifty  = _verdict(bm["net_win_rate"], b1["win_rate"])
        v_break  = _verdict(bm["net_win_rate"], b2["win_rate"])
        v_sh_n   = _verdict(bm["sharpe"],       b1["sharpe"])
        v_sh_b   = _verdict(bm["sharpe"],       b2["sharpe"])
        print(f"  vs Nifty B&H   — Net WR: {v_nifty:<4}  Sharpe: {v_sh_n}")
        print(f"  vs 20d Breakout — Net WR: {v_break:<4}  Sharpe: {v_sh_b}")

    print("=" * w)
    print(f"\n  Full JSON report: {out_path}")


if __name__ == "__main__":
    main()
